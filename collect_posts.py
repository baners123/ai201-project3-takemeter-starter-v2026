"""Collect 200 public r/NBATalk posts for TakeMeter."""

import csv
import json
import subprocess
import time
import urllib.parse

SUBREDDIT = "NBATalk"
TARGET = 200
OUTPUT = "labels.csv"
USER_AGENT = "TakeMeter-AI201/1.0"


def fetch_page(before=None):
    """Fetch one page of archived public Reddit posts."""

    params = {
        "subreddit": SUBREDDIT,
        "limit": 100,
        "sort": "desc",
        "fields": "id,title,selftext,created_utc",
    }

    if before is not None:
        params["before"] = str(before)

    base = "https://" + "arctic-shift.photon-reddit.com"
    url = (
        base
        + "/api/posts/search?"
        + urllib.parse.urlencode(params)
    )

    result = subprocess.run(
        [
            "curl",
            "-fsSL",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "-A",
            USER_AGENT,
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Could not retrieve posts.\n"
            f"curl error:\n{result.stderr.strip()}"
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The archive returned something other than JSON.\n"
            f"First 500 characters:\n{result.stdout[:500]}"
        ) from exc


def clean(value):
    """Clean text before saving it to the CSV."""

    return " ".join(
        (value or "")
        .replace("\x00", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    )


def main():
    rows = []
    seen = set()
    before = None
    page_number = 0

    print(f"Collecting public r/{SUBREDDIT} posts...")

    while len(rows) < TARGET:
        page_number += 1

        print(
            f"Fetching page {page_number} "
            f"({len(rows)}/{TARGET} collected)..."
        )

        payload = fetch_page(before)
        posts = payload.get("data", [])

        if not posts:
            print("No more posts were returned.")
            break

        oldest_time = None

        for post in posts:
            title = clean(post.get("title"))
            body = clean(post.get("selftext"))

            if not title:
                continue

            if title.lower() in {"[removed]", "[deleted]"}:
                continue

            if body.lower() in {"[removed]", "[deleted]"}:
                body = ""

            text = title

            if body:
                text = f"{title} {body}"

            key = text.lower()

            if key in seen:
                continue

            seen.add(key)

            row_number = len(rows) + 1

            rows.append(
                {
                    "text": text,
                    "label": "",
                    "note": (
                        "cold"
                        if row_number <= 20
                        else "needs_review"
                    ),
                }
            )

            created = post.get("created_utc")

            if created is not None:
                if oldest_time is None:
                    oldest_time = created
                else:
                    try:
                        if float(created) < float(oldest_time):
                            oldest_time = created
                    except (TypeError, ValueError):
                        oldest_time = created

            if len(rows) >= TARGET:
                break

        if len(rows) >= TARGET:
            break

        if oldest_time is None:
            print("Could not find a timestamp for pagination.")
            break

        try:
            before = int(float(oldest_time)) - 1
        except (TypeError, ValueError):
            before = oldest_time

        time.sleep(1)

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "text",
                "label",
                "note",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {len(rows)} posts to {OUTPUT}.")

    if len(rows) >= TARGET:
        print("Target reached: 200 posts.")
    elif len(rows) >= 150:
        print(
            "The 150-row grading floor was reached, "
            "but the 200-row target was not."
        )
    else:
        print(
            "WARNING: fewer than 150 posts were collected."
        )

    print()
    print(
        "Rows 1-20 are marked 'cold'. "
        "Label those yourself first."
    )

    print(
        "Rows 21 onward are marked 'needs_review'."
    )


if __name__ == "__main__":
    main()