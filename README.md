# TakeMeter: NBA Discussion Classifier

## What This Does

TakeMeter classifies public r/NBATalk posts by how the writer presents an NBA take. It separates evidence-backed basketball arguments, unsupported opinions, and posts whose main purpose is asking the community for input.

## Label Taxonomy

### `supported_take`

Makes a basketball claim and supports it with a specific checkable fact, statistic, historical example, or explained basketball reasoning.

Example 1:

> The Pistons' offense stalled after the first quarter against New York. The post cites 12 points in the second quarter, 13 in the third, and 9-for-37 three-point shooting before asking whether the problem was shooting or New York's defense.

Example 2:

> SGA is 47th in the league this season in free throw attempts. 18 total attempts on the season, while Jayson Tatum attempted 22 in one game.

### `unsupported_take`

Makes a basketball judgment, ranking, prediction, or confident claim without enough supporting evidence or reasoning to establish an argument.

Example 1:

> Lukewarm Take: Kobe Bryant's 81PTS game is more impressive than Bam Adebayo's 83PTS game.

Example 2:

> Yaxel is better than Maxey.

### `discussion_prompt`

Primarily asks the community to choose, rank, predict, compare, explain, or give an opinion rather than making its own developed argument.

Example 1:

> Where would you rank Reaves as a player in the NBA?

Example 2:

> Who is the greatest in-game dunker?

### The hardest boundary

**Which two labels:** `supported_take` and `unsupported_take`

**Decision rule:** If a post gives specific evidence or explains a basketball mechanism that directly supports its main claim, label it `supported_take`. If it mainly states a ranking, prediction, judgment, or conclusion without developing why, label it `unsupported_take`. A number or player name by itself does not automatically make a post supported. If the primary purpose is asking other users for a judgment, label it `discussion_prompt`.

## The Dataset

**Where the posts came from:** Public posts from r/NBATalk.

**How I labelled them:** I used the written taxonomy above to assign one of three labels to each post. The first 20 rows are marked `cold`. Rows 21 through 200 were AI pre-labelled against the same definitions and marked for manual review. Before submission, I reviewed the AI suggestions against the written decision rule and corrected any labels I disagreed with.

**Counts per label:**

| Label | Count | Share |
|---|---:|---:|
| `unsupported_take` | 75 | 37.5% |
| `discussion_prompt` | 70 | 35.0% |
| `supported_take` | 55 | 27.5% |
| **Total** | **200** | **100%** |

No label represents more than 70% of the dataset.

### Three hard cases

1. **Why do NBA fans not like winners anymore?**

   **Could have been:** `discussion_prompt` or `supported_take`

   **I chose:** `discussion_prompt`

   **Because:** The writer gives some reasoning, but the main purpose is asking the community to explain a trend.

2. **If Giannis wins a ring with the Heat would he move above LeBron on the all time list?**

   **Could have been:** `discussion_prompt` or `unsupported_take`

   **I chose:** `discussion_prompt`

   **Because:** The writer gives a personal ranking, but the main purpose is asking other users how Giannis should be ranked.

3. **Am I supposed to feel bad for OKC?**

   **Could have been:** `discussion_prompt` or `supported_take`

   **I chose:** `supported_take`

   **Because:** The opening question is rhetorical. The body argues a position and gives examples involving Lu Dort to support it.

## The Training Run

**Base model:** `distilbert-base-uncased`

**Device:** `mps` on Apple Silicon

**Settings:**

- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16
- Maximum sequence length: 128 tokens
- Seed: 42
- Test size: 30 posts

**Anything I changed from the defaults, and why:** Nothing for the Unit 5 training run. I kept the starter settings so the first result would be a clean reference point.

### Unit 5 results

| Metric | Result |
|---|---:|
| Overall accuracy | 0.567 |
| Macro F1 | 0.488 |
| `supported_take` F1 | 0.588 |
| `unsupported_take` F1 | 0.167 |
| `discussion_prompt` F1 | 0.710 |

The first run did not meet the overall accuracy target of 0.65 or the macro F1 target of 0.60. `supported_take` and `discussion_prompt` both cleared the 0.50 per-label F1 target. `unsupported_take` did not, with an F1 of 0.167.

The confusion counts from the first run were:

| true \ predicted | `supported_take` | `unsupported_take` | `discussion_prompt` |
|---|---:|---:|---:|
| **`supported_take`** | 5 | 0 | 3 |
| **`unsupported_take`** | 4 | 1 | 6 |
| **`discussion_prompt`** | 0 | 0 | 11 |

The clearest issue is `unsupported_take`. Only 1 of 11 true `unsupported_take` posts was predicted correctly. Six were predicted as `discussion_prompt` and four were predicted as `supported_take`. This suggests the model has not learned the middle boundary well enough between an unsupported opinion, a question inviting discussion, and a supported argument.

The confidence check from the same run also shows that confidence had some useful signal. Accuracy on the most confident third was 0.80, compared with 0.30 on the least confident third.

## How I Used AI

### Moment 1

- **What I asked for:** I used AI to stress-test the boundary between a supported basketball argument, an unsupported take, and a discussion prompt.
- **What came back:** The difficult cases were mostly posts that contained a question but also stated an opinion, or posts that mentioned a statistic without using it to support the main claim.
- **What I changed:** I kept a written decision rule that classifies the primary purpose of the post and requires evidence to directly support the claim before using `supported_take`.

### Moment 2

- **What I asked for:** I used AI to pre-label rows after the initial cold-labeling set and to check the dataset for missing labels, duplicate text, and class imbalance.
- **What came back:** The final dataset contains 200 rows, no blank labels, no exact duplicate text, and no class above 37.5%.
- **What I changed:** I reviewed the suggestions against my taxonomy and corrected cases where the post's purpose mattered more than the presence of a question mark or a statistic.

**Pre-labelling disclosure:** AI was used to suggest labels for rows 21 through 200. Those rows are identified in the `note` column. I reviewed the suggestions against the written taxonomy before training.

---

## Baseline vs. Trained

**Prediction written before looking at the baseline:** I expect the fine-tuned model to have its largest advantage on the `supported_take` versus `unsupported_take` boundary because it trained on r/NBATalk examples that use community-specific basketball language and evidence.

| Measure | Baseline | Trained | Difference |
|---|---:|---:|---:|
| Overall accuracy | PENDING | 0.567 | PENDING |
| Macro F1 | PENDING | 0.488 | PENDING |
| F1, `supported_take` | PENDING | 0.588 | PENDING |
| F1, `unsupported_take` | PENDING | 0.167 | PENDING |
| F1, `discussion_prompt` | PENDING | 0.710 | PENDING |

**What the gap actually means:** PENDING AFTER BASELINE RUN.

## Run Log: Before

**Device:** `mps`

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---:|---:|---:|---:|---|
| 1. Overall accuracy | >= 0.65 | 0.567 | PENDING | PENDING | PENDING |
| 2. Macro F1 | >= 0.60 | 0.488 | PENDING | PENDING | PENDING |
| 3. `supported_take` F1 | >= 0.50 | 0.588 | PENDING | PENDING | PENDING |
| 4. `unsupported_take` F1 | >= 0.50 | 0.167 | PENDING | PENDING | PENDING |
| 5. `discussion_prompt` F1 | >= 0.50 | 0.710 | PENDING | PENDING | PENDING |

### Confusion matrix

Rows are true labels and columns are predictions.

| true \ predicted | `supported_take` | `unsupported_take` | `discussion_prompt` |
|---|---:|---:|---:|
| **`supported_take`** | 5 | 0 | 3 |
| **`unsupported_take`** | 4 | 1 | 6 |
| **`discussion_prompt`** | 0 | 0 | 11 |

**My biggest off-diagonal number, and what it means:** In the seed 42 run, 6 true `unsupported_take` posts were predicted as `discussion_prompt`. This is the largest confusion so far and suggests that the model often interprets unsupported opinions as posts whose main purpose is starting a discussion. This matrix will be updated after the full three-seed Unit 6 run.

## Verdicts and Diagnoses

| # | Criterion | Target | Verdict | How I decided |
|---|---|---:|---|---|
| 1 | Overall accuracy | >= 0.65 | PENDING | Compare all three seed accuracies with 0.65. A single seed below the target means MISSED. |
| 2 | Macro F1 | >= 0.60 | PENDING | Compare all three macro F1 values with 0.60. |
| 3 | `supported_take` F1 | >= 0.50 | PENDING | Compare the label F1 across all three seeds with 0.50. |
| 4 | `unsupported_take` F1 | >= 0.50 | PENDING | Compare the label F1 across all three seeds with 0.50. |
| 5 | `discussion_prompt` F1 | >= 0.50 | PENDING | Compare the label F1 across all three seeds with 0.50. |

### Diagnoses

PENDING. For every MISSED criterion, connect the miss to evidence from the three-seed results, confusion matrix, and agreement report.

## Agreement Report

**Agreement rate:** PENDING / 30 = PENDING%

### Disagreements

PENDING AFTER `python agreement.py`.

For every disagreement, record:

1. **The post:** exact text
2. **Staff said / I said:** both labels
3. **My call, and why:** which interpretation is stronger under `data/staff_taxonomy.md`
4. **Which pile:** rule covered it but I applied it loosely; rule does not say; or rule is ambiguous and my reading is defensible

**What the pattern in my disagreements tells me:** PENDING.

## The Improvement

**What I changed:** PENDING UNTIL THE DIAGNOSIS IDENTIFIES ONE FAILURE.

**Which diagnosis pointed at it:** PENDING.

Only one change should be made for the required improvement. Use the same three seeds afterward.

### Run Log: After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---:|---:|---:|---:|---|
| 1. Overall accuracy | >= 0.65 | 0.567 | PENDING | PENDING | PENDING |
| 2. Macro F1 | >= 0.60 | 0.488 | PENDING | PENDING | PENDING |
| 3. `supported_take` F1 | >= 0.50 | 0.588 | PENDING | PENDING | PENDING |
| 4. `unsupported_take` F1 | >= 0.50 | 0.167 | PENDING | PENDING | PENDING |
| 5. `discussion_prompt` F1 | >= 0.50 | 0.710 | PENDING | PENDING | PENDING |

**Did it help, and how do I know:** PENDING. Compare the before and after runs across the same three seeds. A change that makes performance worse is still a valid measured result if reported accurately.

## What's Still Broken

PENDING AFTER THE IMPROVEMENT RUN.

For every criterion still missed, state what you would try next and why you stopped.

**The gap between what I meant my labels to capture and what the model learned:** PENDING. Use the confusion matrix as evidence.
