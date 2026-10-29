"""Run TakeMeter Unit 6 three-seed evaluation from the terminal.

Reads labels.csv and trains the same DistilBERT classifier at seeds:
42, 7, 2024.

Usage:
    python run_unit6_three_seeds.py before
    python run_unit6_three_seeds.py after

Writes:
    results_three_seeds_before.json
or
    results_three_seeds_after.json
"""

import json
import sys
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
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 128
SEEDS = [42, 7, 2024]
CSV = "labels.csv"


def device_info():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)

    if (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
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
    lookup = {
        label: i
        for i, label in enumerate(LABELS)
    }

    out = df.copy()

    out["labels"] = out["label"].map(lookup)

    return out[
        [
            "text",
            "labels",
        ]
    ]


def compute_metrics(eval_pred):
    logits, gold = eval_pred

    pred = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "accuracy": accuracy_score(
            gold,
            pred,
        ),
        "f1_macro": f1_score(
            gold,
            pred,
            average="macro",
            zero_division=0,
        ),
    }


def run_seed(
    df,
    tokenizer,
    seed,
    run_label,
):
    print()

    print("=" * 68)

    print(
        f"{run_label.upper()} | SEED {seed}"
    )

    print("=" * 68)

    set_seed(seed)

    train_df, val_df, test_df = split_dataset(
        df,
        seed,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=len(LABELS),
            id2label={
                i: label
                for i, label in enumerate(LABELS)
            },
            label2id={
                label: i
                for i, label in enumerate(LABELS)
            },
        )
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_ds = Dataset.from_pandas(
        to_ids(train_df)
    ).map(
        tokenize,
        batched=True,
    )

    val_ds = Dataset.from_pandas(
        to_ids(val_df)
    ).map(
        tokenize,
        batched=True,
    )

    test_ds = Dataset.from_pandas(
        to_ids(test_df)
    ).map(
        tokenize,
        batched=True,
    )

    args = TrainingArguments(
        output_dir=(
            f"./takemeter_out_"
            f"{run_label}_{seed}"
        ),
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

    output = trainer.predict(
        test_ds
    )

    pred = np.argmax(
        output.predictions,
        axis=-1,
    )

    gold = np.array(
        output.label_ids
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            gold,
            pred,
            labels=list(
                range(
                    len(LABELS)
                )
            ),
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        gold,
        pred,
        labels=list(
            range(
                len(LABELS)
            )
        ),
    )

    result = {
        "seed": seed,
        "n_test": int(
            len(gold)
        ),
        "accuracy": float(
            accuracy_score(
                gold,
                pred,
            )
        ),
        "f1_macro": float(
            f1_score(
                gold,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "per_label": {
            label: {
                "precision": float(
                    precision[i]
                ),
                "recall": float(
                    recall[i]
                ),
                "f1": float(
                    f1[i]
                ),
                "support": int(
                    support[i]
                ),
            }
            for i, label
            in enumerate(LABELS)
        },
        "confusion_matrix": (
            cm.tolist()
        ),
    }

    print(
        f"accuracy: "
        f"{result['accuracy']:.3f}"
    )

    print(
        f"macro F1: "
        f"{result['f1_macro']:.3f}"
    )

    for label in LABELS:
        print(
            f"{label:<20} "
            f"F1 "
            f"{result['per_label'][label]['f1']:.3f}"
        )

    return result


def print_run_log(results):
    by_seed = {
        result["seed"]: result
        for result in results
    }

    print()

    print("RUN LOG")

    print()

    print(
        "| Criterion | Target | "
        "Seed 42 | Seed 7 | "
        "Seed 2024 |"
    )

    print(
        "|---|---:|---:|---:|---:|"
    )

    print(
        "| 1. Overall accuracy | "
        ">= 0.65 | "
        f"{by_seed[42]['accuracy']:.3f} | "
        f"{by_seed[7]['accuracy']:.3f} | "
        f"{by_seed[2024]['accuracy']:.3f} |"
    )

    print(
        "| 2. Macro F1 | "
        ">= 0.60 | "
        f"{by_seed[42]['f1_macro']:.3f} | "
        f"{by_seed[7]['f1_macro']:.3f} | "
        f"{by_seed[2024]['f1_macro']:.3f} |"
    )

    for num, label in enumerate(
        LABELS,
        start=3,
    ):
        print(
            f"| {num}. `{label}` F1 | "
            ">= 0.50 | "
            f"{by_seed[42]['per_label'][label]['f1']:.3f} | "
            f"{by_seed[7]['per_label'][label]['f1']:.3f} | "
            f"{by_seed[2024]['per_label'][label]['f1']:.3f} |"
        )


def print_matrix(matrix):
    print()

    print("CONFUSION MATRIX")

    print()

    print(
        "| true \\\\ predicted | "
        "`supported_take` | "
        "`unsupported_take` | "
        "`discussion_prompt` |"
    )

    print(
        "|---|---:|---:|---:|"
    )

    for i, label in enumerate(
        LABELS
    ):
        print(
            f"| **`{label}`** | "
            f"{matrix[i][0]} | "
            f"{matrix[i][1]} | "
            f"{matrix[i][2]} |"
        )


def main():
    run_label = (
        sys.argv[1].lower()
        if len(sys.argv) > 1
        else "before"
    )

    if run_label not in {
        "before",
        "after",
    }:
        raise SystemExit(
            "Usage:\n"
            "python run_unit6_three_seeds.py before\n"
            "or:\n"
            "python run_unit6_three_seeds.py after"
        )

    if not Path(CSV).exists():
        raise SystemExit(
            "labels.csv not found "
            "in the current folder."
        )

    df = pd.read_csv(
        CSV
    )

    device, name = (
        device_info()
    )

    print(
        f"Device: {device} "
        f"({name})"
    )

    print(
        f"Run label: {run_label}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL
        )
    )

    results = [
        run_seed(
            df,
            tokenizer,
            seed,
            run_label,
        )
        for seed in SEEDS
    ]

    seed42 = next(
        result
        for result in results
        if result["seed"] == 42
    )

    payload = {
        "run_label": run_label,
        "device": device,
        "device_name": name,
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "seeds": results,
        "confusion_matrix_seed": 42,
        "confusion_matrix": (
            seed42[
                "confusion_matrix"
            ]
        ),
        "labels": LABELS,
    }

    out = (
        f"results_three_seeds_"
        f"{run_label}.json"
    )

    with open(
        out,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print_run_log(
        results
    )

    print_matrix(
        seed42[
            "confusion_matrix"
        ]
    )

    print()

    print(
        f"Wrote {out}"
    )


if __name__ == "__main__":
    main()