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

**Prediction written before looking at the baseline:** I expected the fine-tuned model to have its largest advantage on the `supported_take` versus `unsupported_take` boundary because it trained on r/NBATalk examples that use community-specific basketball language and evidence.

| Measure | Baseline | Trained | Difference |
|---|---:|---:|---:|
| Overall accuracy | 0.300 | 0.567 | +0.267 |
| Macro F1 | 0.224 | 0.488 | +0.264 |
| F1, `discussion_prompt` | 0.000 | 0.710 | +0.710 |
| F1, `supported_take` | 0.438 | 0.588 | +0.151 |
| F1, `unsupported_take` | 0.235 | 0.167 | -0.069 |

The fine-tuned model clearly beat the zero-shot baseline overall. Accuracy improved by 0.267 and macro F1 improved by 0.264 on the same 30 held-out posts. The largest gain was on `discussion_prompt`, which improved from 0.000 F1 to 0.710. Fine-tuning also improved `supported_take` from 0.438 to 0.588. The one regression was `unsupported_take`, which fell from 0.235 to 0.167. That result suggests the fine-tuned model learned the prompt and supported-take patterns better than the unsupported-take boundary.

## Run Log: Before

**Device:** `mps` on Apple Silicon

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---:|---:|---:|---:|---|
| 1. Overall accuracy | >= 0.65 | 0.567 | 0.567 | 0.600 | MISSED |
| 2. Macro F1 | >= 0.60 | 0.488 | 0.442 | 0.583 | MISSED |
| 3. `supported_take` F1 | >= 0.50 | 0.588 | 0.000 | 0.706 | MISSED |
| 4. `unsupported_take` F1 | >= 0.50 | 0.167 | 0.625 | 0.375 | MISSED |
| 5. `discussion_prompt` F1 | >= 0.50 | 0.710 | 0.700 | 0.667 | MET |

### Confusion matrix

The required typed matrix below is from seed 42, the same seed used for the original Unit 5 run.

| true \ predicted | `supported_take` | `unsupported_take` | `discussion_prompt` |
|---|---:|---:|---:|
| **`supported_take`** | 5 | 0 | 3 |
| **`unsupported_take`** | 4 | 1 | 6 |
| **`discussion_prompt`** | 0 | 0 | 11 |

The biggest off-diagonal count in seed 42 is 6 true `unsupported_take` posts predicted as `discussion_prompt`. The other major miss is 4 unsupported takes predicted as supported takes. This shows that `unsupported_take` sits between the other two labels and is the least stable boundary.

The same pattern changes direction across seeds. In seed 7, all 8 true `supported_take` posts were missed, with 7 predicted as `unsupported_take`. In seed 2024, 7 of 11 true `unsupported_take` posts were predicted as `discussion_prompt`. The overall accuracy changed only from 0.567 to 0.600, but the per-label behavior changed much more. That means the model is fairly stable in total accuracy while still being unstable about which side of the label boundary individual posts belong on.

## Verdicts and Diagnoses

### Criterion 1: Overall accuracy >= 0.65

**Verdict: MISSED**

The three accuracies were 0.567, 0.567, and 0.600. All three were below 0.65, so the target was missed.

**Diagnosis:** The model is consistently below the target rather than failing because of one unlucky split. The spread is only 0.033, which suggests the overall task is difficult for this dataset size and taxonomy. The lower macro F1 values show that the model's errors are concentrated in specific labels rather than being evenly distributed.

### Criterion 2: Macro F1 >= 0.60

**Verdict: MISSED**

The macro F1 values were 0.488, 0.442, and 0.583. All three were below 0.60.

**Diagnosis:** Macro F1 is being pulled down by label instability. `discussion_prompt` stays strong across all three seeds, but `supported_take` falls to 0.000 on seed 7 and `unsupported_take` falls below 0.50 on seeds 42 and 2024. The model is therefore not learning all three labels equally well.

### Criterion 3: `supported_take` F1 >= 0.50

**Verdict: MISSED**

The F1 values were 0.588, 0.000, and 0.706. Seed 7 missed the target completely.

**Diagnosis:** This label is highly split-sensitive. In seed 7, all 8 true `supported_take` posts were misclassified. Seven were called `unsupported_take` and one was called `discussion_prompt`. Because the same label performs well on seeds 42 and 2024, the problem looks less like the model being unable to learn `supported_take` at all and more like an unstable boundary between supported and unsupported claims. `supported_take` is also the smallest class in the dataset at 55 examples, so its training examples change more noticeably when the random split changes.

### Criterion 4: `unsupported_take` F1 >= 0.50

**Verdict: MISSED**

The F1 values were 0.167, 0.625, and 0.375. Seeds 42 and 2024 both missed the target.

**Diagnosis:** `unsupported_take` is the clearest weak boundary. In seed 42, only 1 of 11 unsupported takes was classified correctly. Six were called `discussion_prompt` and four were called `supported_take`. In seed 2024, 7 of 11 were called `discussion_prompt`. The baseline also scored 0.235 F1 on this label, while the trained model scored 0.167 in the original seed 42 run. This suggests the label is difficult to separate consistently and that the training data likely contains borderline posts where a question, opinion, or small amount of evidence changes the interpretation.

### Criterion 5: `discussion_prompt` F1 >= 0.50

**Verdict: MET**

The F1 values were 0.710, 0.700, and 0.667. All three exceeded 0.50.

**Diagnosis:** This is the most stable label. Its F1 stayed above 0.66 across every split. The model learned question and community-response language more consistently than it learned the difference between supported and unsupported claims.

### Overall diagnosis before the agreement check

The strongest current evidence points to a label-boundary problem rather than a simple model-capacity problem. Total accuracy is fairly stable across seeds, but `supported_take` and `unsupported_take` trade errors depending on which examples land in the training set. The staff agreement check is the next piece of evidence needed to decide whether the required improvement should relabel one boundary or change something else.

## Agreement Report

The official comparison could not be completed yet because `data/staff_labels.csv` was not available in the repository. The assignment specifies that this file must come from the TF after my own 30 labels are saved.

My first attempt at `my_staff_labels.csv` is not valid for agreement scoring because I accidentally entered arbitrary short values instead of the three required staff taxonomy labels. I will replace those entries using only `analysis`, `hot_take`, and `reaction` before running `agreement.py`.

I did not create or invent a staff answer key or agreement rate.

## The Improvement

**What I changed:** I increased training from 3 epochs to 5 epochs. I kept the dataset, base model, learning rate, batch size, maximum sequence length, and the same three random seeds unchanged.

**Why I changed it:** The validation metrics were still improving at the end of the original 3-epoch runs. For seed 42, validation macro F1 increased from about 0.383 at epoch 1 to about 0.461 at epoch 3. For seed 7, it increased from about 0.186 to about 0.481. That suggested the model might still be learning when the original run stopped.

### Run Log: After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---:|---:|---:|---:|---|
| 1. Overall accuracy | >= 0.65 | 0.600 | 0.633 | 0.800 | MISSED |
| 2. Macro F1 | >= 0.60 | 0.568 | 0.615 | 0.798 | MISSED |
| 3. `supported_take` F1 | >= 0.50 | 0.588 | 0.545 | 0.800 | MET |
| 4. `unsupported_take` F1 | >= 0.50 | 0.400 | 0.609 | 0.762 | MISSED |
| 5. `discussion_prompt` F1 | >= 0.50 | 0.714 | 0.692 | 0.833 | MET |

### Did the change help?

Yes. The five-epoch run improved the model overall.

Average accuracy across the three seeds increased from about **0.578** before to **0.678** after. Average macro F1 increased from about **0.504** to **0.660**.

The per-label averages also improved:

| Label | Before average F1 | After average F1 | Change |
|---|---:|---:|---:|
| `supported_take` | 0.431 | 0.645 | +0.214 |
| `unsupported_take` | 0.389 | 0.590 | +0.201 |
| `discussion_prompt` | 0.692 | 0.747 | +0.055 |

The change was especially useful for the two unstable argument labels. `supported_take` moved from failing one seed completely at 0.000 to clearing 0.50 on all three seeds. `unsupported_take` improved substantially, although seed 42 still remained below the 0.50 target at 0.400.

The change did not make every criterion pass, but it made the model more stable and improved both overall performance and the weakest label boundary.

## What's Still Broken

Three criteria still miss their targets after the improvement.

**Overall accuracy:** Seed 42 reached 0.600 and seed 7 reached 0.633, both below the 0.65 target. The average improved, but the criterion requires the target to hold across all three seeds.

**Macro F1:** Seed 42 reached 0.568, below the 0.60 target. Seeds 7 and 2024 passed. I would next focus on the remaining boundary errors rather than simply training longer again.

**`unsupported_take` F1:** Seed 42 reached 0.400, while seeds 7 and 2024 reached 0.609 and 0.762. This label remains the least stable one. The confusion matrices show that unsupported takes are often pulled toward either `supported_take` or `discussion_prompt`, depending on the split.

The gap between what I meant the taxonomy to capture and what the model learned is clearest around `unsupported_take`. I intended it to represent a confident basketball claim that lacks enough support to become an argument. The model often learned surface cues instead. Posts phrased as questions could be pulled toward `discussion_prompt`, while longer opinionated posts could be pulled toward `supported_take` even when their evidence was weak. Five epochs helped the model learn the distinction more consistently, but the boundary is still not fully stable.

