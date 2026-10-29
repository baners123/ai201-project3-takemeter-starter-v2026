"""Terminal helper for creating my_staff_labels.csv without VS Code shell commands.

This does NOT reveal or generate the staff answer key.
It shows data/staff_taxonomy.md and prompts you to enter your own label
for each post in data/staff_posts.csv.
"""

import csv
from pathlib import Path

TAXONOMY = Path("data/staff_taxonomy.md")
POSTS = Path("data/staff_posts.csv")
OUTPUT = Path("my_staff_labels.csv")

if not TAXONOMY.exists():
    raise SystemExit("Missing data/staff_taxonomy.md")
if not POSTS.exists():
    raise SystemExit("Missing data/staff_posts.csv")

print("=" * 72)
print("STAFF TAXONOMY")
print("=" * 72)
print(TAXONOMY.read_text(encoding="utf-8"))
print("=" * 72)

with POSTS.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows or "text" not in rows[0]:
    raise SystemExit("Expected a text column in data/staff_posts.csv")

existing = {}
if OUTPUT.exists():
    try:
        with OUTPUT.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = row.get("text", "")
                label = (row.get("label") or "").strip()
                if text and label:
                    existing[text] = label
    except Exception:
        pass

answers = []
total = len(rows)

for i, row in enumerate(rows, start=1):
    text = row["text"]

    if text in existing:
        answers.append({"text": text, "label": existing[text]})
        print(f"[{i}/{total}] already labeled: {existing[text]}")
        continue

    print()
    print("-" * 72)
    print(f"POST {i} OF {total}")
    print("-" * 72)
    print(text)
    print()

    while True:
        label = input("Your label: ").strip()
        if label:
            break
        print("Enter one of the labels defined in data/staff_taxonomy.md.")

    answers.append({"text": text, "label": label})

    # Save after every answer so progress is never lost.
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(answers)

print()
print("=" * 72)
print(f"Finished {len(answers)} labels.")
print(f"Saved to {OUTPUT}")
print("=" * 72)
