"""TakeMeter Unit 6 required improvement run.

Single change from the before run:
    epochs: 3 -> 5

Everything else stays the same.
Runs seeds 42, 7, 2024 and writes results_three_seeds_after.json.
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
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = [
    "supported_take",
    "unsupported_take",
    "discussion_prompt",
]

BASE_MODEL = "distilbert-base-uncased"
EPOCHS = 5
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 128
SEEDS = [42, 7, 2024]
CSV = "labels.csv"
RUN_LABEL = "after"


def device_info():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU (MPS)"
    return "cpu", "CPU"


def split_dataset(df, seed):
    rest, test = train_test_split(
        df,
        test_size=0.15,
        random_state=seed,
        stratify=df["label"],
    )
    train, val = train_test_split(
        rest,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=rest["label"],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def to_ids(df):
    lookup = {label: i for i, label in enumerate(LABELS)}
    out = df.copy()
    out["labels"] = out["label"].map(lookup)
    return out[["text", "labels"]]


def compute_metrics(eval_pred):
    logits, gold = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(gold, pred),
        "f1_macro": f1_score(
            gold, pred, average="macro", zero_division=0
        ),
    }


def run_seed(df, tokenizer, seed):
    print()
    print("=" * 68)
    print(f"AFTER | SEED {seed} | 5 EPOCHS")
    print("=" * 68)

    set_seed(seed)
    train_df, val_df, test_df = split_dataset(df, seed)

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

    train_ds = Dataset.from_pandas(to_ids(train_df)).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(to_ids(val_df)).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(to_ids(test_df)).map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=f"./takemeter_out_after_{seed}",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    output = trainer.predict(test_ds)

    pred = np.argmax(output.predictions, axis=-1)
    gold = np.array(output.label_ids)

    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )

    cm = confusion_matrix(
        gold,
        pred,
        labels=list(range(len(LABELS))),
    )

    result = {
        "seed": seed,
        "n_test": int(len(gold)),
        "accuracy": float(accuracy_score(gold, pred)),
        "f1_macro": float(
            f1_score(gold, pred, average="macro", zero_division=0)
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
        "confusion_matrix": cm.tolist(),
    }

    print(f"accuracy: {result['accuracy']:.3f}")
    print(f"macro F1: {result['f1_macro']:.3f}")
    for label in LABELS:
        print(
            f"{label:<20} "
            f"F1 {result['per_label'][label]['f1']:.3f}"
        )

    return result


def verdict(values, target):
    return "MET" if all(v >= target for v in values) else "MISSED"


def main():
    if not Path(CSV).exists():
        raise SystemExit("labels.csv not found.")

    df = pd.read_csv(CSV)
    device, name = device_info()

    print(f"Device: {device} ({name})")
    print("Required improvement: epochs 3 -> 5")
    print(f"Seeds: {SEEDS}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    results = [run_seed(df, tokenizer, seed) for seed in SEEDS]

    payload = {
        "run_label": RUN_LABEL,
        "change": "epochs increased from 3 to 5",
        "device": device,
        "device_name": name,
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "seeds": results,
        "labels": LABELS,
    }

    with open("results_three_seeds_after.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    by_seed = {r["seed"]: r for r in results}

    print()
    print("RUN LOG: AFTER")
    print()
    print("| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |")
    print("|---|---:|---:|---:|---:|---|")

    accs = [by_seed[s]["accuracy"] for s in SEEDS]
    macros = [by_seed[s]["f1_macro"] for s in SEEDS]

    print(
        f"| 1. Overall accuracy | >= 0.65 | "
        f"{accs[0]:.3f} | {accs[1]:.3f} | {accs[2]:.3f} | "
        f"{verdict(accs, 0.65)} |"
    )
    print(
        f"| 2. Macro F1 | >= 0.60 | "
        f"{macros[0]:.3f} | {macros[1]:.3f} | {macros[2]:.3f} | "
        f"{verdict(macros, 0.60)} |"
    )

    for num, label in enumerate(LABELS, start=3):
        vals = [by_seed[s]["per_label"][label]["f1"] for s in SEEDS]
        print(
            f"| {num}. `{label}` F1 | >= 0.50 | "
            f"{vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | "
            f"{verdict(vals, 0.50)} |"
        )

    print()
    print("Wrote results_three_seeds_after.json")


if __name__ == "__main__":
    main()
