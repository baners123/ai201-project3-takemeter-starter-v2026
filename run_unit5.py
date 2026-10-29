"""Run the TakeMeter Unit 5 training pipeline from the terminal.

This mirrors Sections 1 through 5 of takemeter.ipynb and writes:
    results.json
    test_split.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

NAME = "Jimi Adegoroye"
EMAIL = "adegoroyej24@gmail.com"

LABELS = [
    "supported_take",
    "unsupported_take",
    "discussion_prompt",
]

BASE_MODEL = "distilbert-base-uncased"
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 128
SEED = 42
CSV = "labels.csv"


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU (MPS)"
    return "cpu", "CPU"


def check_dataset(df, labels):
    problems = []

    for column in ("text", "label"):
        if column not in df.columns:
            problems.append(
                f"Missing required column '{column}'. Found: {list(df.columns)}"
            )

    if problems:
        return problems

    blank_text = (
        df["text"].isna().sum()
        + (df["text"].astype(str).str.strip() == "").sum()
    )
    if blank_text:
        problems.append(f"{blank_text} rows have blank text.")

    found = set(df["label"].dropna().astype(str).str.strip().unique())
    expected = set(labels)

    unknown = found - expected
    if unknown:
        problems.append(
            f"Unknown labels in labels.csv: {sorted(unknown)}"
        )

    blank_labels = (
        df["label"].isna().sum()
        + (df["label"].astype(str).str.strip() == "").sum()
    )
    if blank_labels:
        problems.append(f"{blank_labels} rows have blank labels.")

    if len(df) < 150:
        problems.append(
            f"Only {len(df)} rows. Unit 5 requires at least 150."
        )

    counts = df["label"].value_counts()
    for label, count in counts.items():
        share = count / len(df)
        if share > 0.70:
            problems.append(
                f"{label} is {share:.1%} of the dataset, above the 70% cap."
            )

    return problems


def split_dataset(df, seed, test_size=0.15, val_size=0.15):
    rest, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    val_share = val_size / (1 - test_size)

    train, val = train_test_split(
        rest,
        test_size=val_share,
        random_state=seed,
        stratify=rest["label"],
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def to_ids(df, labels):
    lookup = {label: i for i, label in enumerate(labels)}
    out = df.copy()
    out["labels"] = (
        out["label"]
        .astype(str)
        .str.strip()
        .map(lookup)
    )
    return out[["text", "labels"]]


def compute_metrics(eval_pred):
    logits, gold = eval_pred
    predicted = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(gold, predicted),
        "f1_macro": f1_score(
            gold,
            predicted,
            average="macro",
            zero_division=0,
        ),
    }


def confusion_counts(gold, predicted, labels):
    matrix = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }

    for g, p in zip(gold, predicted):
        matrix[labels[int(g)]][labels[int(p)]] += 1

    return matrix


def confidence_report(confidence, gold, predicted):
    right = gold == predicted
    ranked = np.argsort(confidence)
    third = max(1, len(confidence) // 3)

    least = ranked[:third]
    most = ranked[-third:]

    return {
        "mean_when_right": (
            float(confidence[right].mean())
            if right.any()
            else None
        ),
        "mean_when_wrong": (
            float(confidence[~right].mean())
            if (~right).any()
            else None
        ),
        "accuracy_most_confident_third": float(
            right[most].mean()
        ),
        "accuracy_least_confident_third": float(
            right[least].mean()
        ),
        "n_per_third": int(third),
    }


def main():
    repo = Path.cwd()

    print("TakeMeter Unit 5")
    print("-" * 60)
    print("Working in:", repo)

    if not Path(CSV).exists():
        raise SystemExit(
            "\nERROR: labels.csv was not found.\n"
            "Run this script from the root of your TakeMeter repo."
        )

    device, device_name = pick_device()

    print("Name:", NAME)
    print("Device:", device, f"({device_name})")
    print("Model:", BASE_MODEL)
    print()

    df = pd.read_csv(CSV)

    problems = check_dataset(df, LABELS)
    if problems:
        print("Dataset problems:")
        for problem in problems:
            print("  [FIX ME]", problem)
        raise SystemExit("\nFix the dataset problems above, then run again.")

    print("Dataset looks good.")
    print("Rows:", len(df))
    print(df["label"].value_counts().to_string())
    print()

    train_df, val_df, test_df = split_dataset(df, SEED)

    print(
        f"Split: train {len(train_df)}, "
        f"validation {len(val_df)}, "
        f"test {len(test_df)}"
    )
    print()

    set_seed(SEED)

    print("Loading tokenizer and model...")
    print(
        "The first run may download distilbert-base-uncased."
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_ds = Dataset.from_pandas(
        to_ids(train_df, LABELS)
    ).map(tokenize, batched=True)

    val_ds = Dataset.from_pandas(
        to_ids(val_df, LABELS)
    ).map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir="./takemeter_out",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print()
    print("Training...")
    trainer.train()
    print("Training finished.")
    print()

    test_ds = Dataset.from_pandas(
        to_ids(test_df, LABELS)
    ).map(tokenize, batched=True)

    output = trainer.predict(test_ds)

    predicted = np.argmax(output.predictions, axis=-1)
    gold = np.array(output.label_ids)

    scores = output.predictions
    exp = np.exp(
        scores - scores.max(axis=-1, keepdims=True)
    )
    confidence = (
        exp / exp.sum(axis=-1, keepdims=True)
    ).max(axis=-1)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            gold,
            predicted,
            labels=list(range(len(LABELS))),
            zero_division=0,
        )
    )

    results = {
        "name": NAME,
        "email": EMAIL,
        "seed": SEED,
        "n_test": int(len(gold)),
        "accuracy": float(
            accuracy_score(gold, predicted)
        ),
        "f1_macro": float(
            f1_score(
                gold,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "per_label": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion": confusion_counts(
            gold,
            predicted,
            LABELS,
        ),
        "confidence": confidence_report(
            confidence,
            gold,
            predicted,
        ),
        "device": device,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "max_length": MAX_LENGTH,
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    }

    with open(
        "results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    test_df.to_csv(
        "test_split.csv",
        index=False,
    )

    print("=" * 60)
    print("UNIT 5 RESULTS")
    print("=" * 60)
    print(f"accuracy: {results['accuracy']:.3f}")
    print(f"macro F1: {results['f1_macro']:.3f}")
    print()

    for label, metrics in results["per_label"].items():
        print(
            f"{label:<20} "
            f"F1 {metrics['f1']:.3f} "
            f"(n={metrics['support']})"
        )

    print()
    print("Created:")
    print("  results.json")
    print("  test_split.csv")
    print()
    print("Next command:")
    print("  cat results.json")


if __name__ == "__main__":
    main()
