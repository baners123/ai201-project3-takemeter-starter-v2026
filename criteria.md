# Acceptance Criteria: TakeMeter

These criteria were set before viewing any training or evaluation results.

## 1. Overall accuracy is at least 65%

Across each held-out test split, TakeMeter should correctly classify at least 65% of posts.

**Why this target:** With three labels, 65% is meaningfully above chance and would show that the model learned useful differences between supported takes, unsupported takes, and discussion prompts.

## 2. Macro F1 is at least 0.60

Across each held-out test split, TakeMeter should achieve a macro F1 score of at least 0.60.

**Why this target:** The r/NBATalk dataset is not perfectly balanced, so macro F1 gives all three labels equal weight instead of letting the largest class dominate the result.

## 3. `supported_take` F1 is at least 0.50

Across each held-out test split, the F1 score for `supported_take` should be at least 0.50.

**Why this target:** `supported_take` is the smallest class in the dataset with 55 examples, and it requires the model to recognize when evidence or basketball reasoning actually supports a claim.

## 4. `unsupported_take` F1 is at least 0.50

Across each held-out test split, the F1 score for `unsupported_take` should be at least 0.50.

**Why this target:** The hardest boundary in this taxonomy is between `supported_take` and `unsupported_take`. A 0.50 F1 floor tests whether the model can separate an argued claim from a confident claim with little support.

## 5. `discussion_prompt` F1 is at least 0.50

Across each held-out test split, the F1 score for `discussion_prompt` should be at least 0.50.

**Why this target:** r/NBATalk contains many posts that ask the community to rank, compare, predict, or explain. The model should identify the post's primary purpose instead of treating every question as an argument.
