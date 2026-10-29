"""Safely create my_staff_labels.csv using only the three valid staff labels.

Allowed inputs:
    a or analysis
    h or hot_take
    r or reaction

This script overwrites the invalid current file only after you complete all 30.
"""

import csv
from pathlib import Path

POSTS = Path("data/staff_posts.csv")
TAXONOMY = Path("data/staff_taxonomy.md")
OUTPUT = Path("my_staff_labels.csv")
BACKUP = Path("my_staff_labels_invalid_backup.csv")

VALID = {
    "a": "analysis",
    "analysis": "analysis",
    "h": "hot_take",
    "hot_take": "hot_take",
    "r": "reaction",
    "reaction": "reaction",
}

if not POSTS.exists():
    raise SystemExit("Missing data/staff_posts.csv")
if not TAXONOMY.exists():
    raise SystemExit("Missing data/staff_taxonomy.md")

print(TAXONOMY.read_text(encoding="utf-8"))
print("\nAllowed labels:")
print("  a = analysis")
print("  h = hot_take")
print("  r = reaction")
print()

with POSTS.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

answers = []

for i, row in enumerate(rows, start=1):
    text = row["text"]
    print("=" * 72)
    print(f"POST {i} OF {len(rows)}")
    print("=" * 72)
    print(text)
    print()

    while True:
        raw = input("Label [a/h/r]: ").strip().lower()
        if raw in VALID:
            label = VALID[raw]
            break
        print("Invalid label. Enter a, h, r, analysis, hot_take, or reaction.")

    answers.append({"text": text, "label": label})

if OUTPUT.exists():
    OUTPUT.replace(BACKUP)
    print(f"\nBacked up the old file as {BACKUP}")

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(answers)

print(f"Wrote {len(answers)} valid labels to {OUTPUT}")
print("Now run the verification command before committing.")
