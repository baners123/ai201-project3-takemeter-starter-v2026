# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---


Overall accuracy: 65%
Macro F1: 0.60
Minimum per-label F1: 0.50
Human agreement: 75%
Maximum class share: 50%

# TakeMeter Acceptance Criteria

These criteria were written before training results were viewed.

# Acceptance Criteria: TakeMeter

These criteria were set before training results were viewed.

## 1. Overall accuracy is at least 65%

On the held-out test set, TakeMeter should correctly classify at least 65% of posts.

**Why this target:**

There are three labels, so 65% would show that the classifier learned useful differences between supported takes, unsupported takes, and discussion prompts instead of relying on simple guessing.

## 2. Macro F1 is at least 0.60

TakeMeter should achieve a macro F1 score of at least 0.60 on the held-out test set.

**Why this target:**

The dataset contains three labels with different counts. Macro F1 gives each label equal importance, so a score of at least 0.60 would show that performance is not coming from only predicting the most common NBA discussion type.

## 3. Every label has an F1 score of at least 0.50

The F1 score for `supported_take`, `unsupported_take`, and `discussion_prompt` should each be at least 0.50 on the held-out test set.

**Why this target:**

The three labels represent different types of posts in r/NBATalk. `supported_take` is the smallest class with 55 examples, so I expect it to be harder to learn than the larger classes. I still want every label to reach at least 0.50 F1.

## 4. Human label agreement is at least 75%

When the same held-out posts are independently labeled by another person using the written taxonomy, at least 75% of their labels should match my labels.

**Why this target:**

The hardest distinction in this dataset is between `supported_take` and `unsupported_take`. A 75% agreement rate would show that the definitions and decision rule are clear enough for another person to apply without needing my interpretation.

## 5. No label represents more than 50% of the dataset

Before training, no single label should account for more than 50% of the 200 labeled examples.

**Why this target:**

A dominant label could let the classifier perform well by repeatedly predicting the same class. The NBA dataset should contain enough supported takes, unsupported takes, and discussion prompts for the model to learn all three categories.
---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 6 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath, like this:

         ## 2. Every label performs acceptably

         The model performs well on all labels.

         **Why this target:** ...

         > **Revised in unit 6:** Every label has an F1 of at least 0.60 on
         > the held-out set.
         >
         > **Why revised:** "performs well" gave me nothing to check. I
         > couldn't produce a verdict from it at all.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "Overall accuracy of at least 0.65" → "at least 0.55", because
            0.65 turned out to be optimistic for 200 examples.

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.
     ───────────────────────────────────────────────────────────────────────── -->
