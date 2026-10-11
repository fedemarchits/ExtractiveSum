# Summary of results

The main numbers of the project on one page. The reasoning behind them is in
[`docs/`](docs/), and every figure here can be recomputed from the CSV files in
`extsum/reports/`.

**How to read the tables.** F1 measures how well the sentences a model picked
match the reference sentences (1 is a perfect match). "Local models" are the
six run on a GPU with every prompt variant; the two largest models (27B, 31B)
were run online on the five best techniques of each dataset, so they are
compared on those shared variants only.

## The findings in brief

| | Finding |
| --- | --- |
| Model size | Results improve up to about 4B parameters and then flatten. On CNN/DailyMail size makes no difference at all. |
| Model family | Gemma4 is ahead of Qwen3.5 on ACLSum and clearly ahead on BillSum; the two are level on CNN/DailyMail. |
| Prompting technique | No technique is best everywhere. Reasoning techniques help a little on ACLSum and not at all on whole documents. |
| TF-IDF scores | Showing a score next to each sentence is by far the best technique on BillSum, and makes no reliable difference elsewhere. |
| Limiting the answer | Without a limit, the smaller Qwen3.5 models return most of the document. Gemma4 limits itself. |
| Position | CNN/DailyMail rewards picking the opening sentences; BillSum does not. This is why CNN/DailyMail cannot tell models apart. |
| Labelling rule (Part B) | The simplest rule, `local_score`, is the best way to turn a written summary into sentence labels. |

## Part A — one aspect of a paper (ACLSum, human labels)

100 papers; the model selects the sentences for one aspect at a time.

| Model | F1, all 54 variants | F1, 5 best techniques | challenge | approach | outcome |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.461 | 0.472 | 0.35 | 0.42 | 0.50 |
| Qwen3.5-4B | 0.609 | 0.612 | 0.48 | 0.61 | 0.70 |
| Qwen3.5-9B | 0.623 | 0.625 | 0.49 | 0.64 | 0.71 |
| Qwen3.5-27B | n/a | 0.633 | 0.50 | 0.66 | 0.71 |
| Gemma4-E2B | 0.567 | 0.571 | 0.43 | 0.61 | 0.61 |
| Gemma4-E4B | 0.611 | 0.629 | 0.51 | 0.64 | 0.65 |
| Gemma4-12B | 0.631 | 0.633 | 0.50 | 0.66 | 0.71 |
| Gemma4-31B | n/a | 0.641 | 0.50 | 0.68 | 0.71 |

- The jump is from 2B to 4B; from 4B to 31B the gain is 0.01 to 0.02.
- _Challenge_ is the hard aspect for every model, about 0.20 below _outcome_.
- Thirteen of the sixteen techniques are within 0.012 F1 of each other (mean
  over the local models). The plain prompt is last (0.558); the best is 0.596.
- Limiting the number of sentences costs about 0.015 F1 and gives a summary
  closer to the written one.
- A worked example hurts the two smallest models (−0.04) and helps the larger
  ones (up to +0.04).

Best five techniques: `self_ask`, `negative_aware`, `salience_inference`,
`self_critique`, `contrastive_joint`.

## Part B — turning a written summary into sentence labels (ACLSum)

Which rule best recovers the sentences a human marked, given only the written
summary?

| Rule | Precision | Recall | F1 |
| --- | :-: | :-: | :-: |
| **`local_score`**: score each sentence on its own, keep the top K | 0.78 | **0.79** | **0.77** |
| Greedy search | 0.81 | 0.56 | 0.65 |
| Beam search | 0.81 | 0.56 | 0.64 |
| Global search | 0.81 | 0.57 | 0.64 |

- `local_score` wins because it does not avoid repetition, and the human
  labels contain repetition: annotators marked every sentence that expresses
  the aspect.
- More elaborate methods (a neural scorer, embedding-based MMR, MoverScore)
  do not beat it and cost far more.
- Its best K on ACLSum is 4 sentences per aspect.

Part C uses `local_score` to build its labels.

## Part C — a whole document (silver labels)

100 documents per dataset. The number of sentences to select, K, comes from how
long the written summaries are compared with the documents.

| | CNN/DailyMail | BillSum |
| --- | :-: | :-: |
| Documents | news articles | US Congressional bills |
| Sentences per document | 31 | 51 |
| K | 3 | 6 |
| F1 with no model, taking the first K sentences | 0.360 | 0.163 |
| Share of the labels in the first fifth of the document | 52% | 24% |

### CNN/DailyMail

| Model | F1, all 52 variants | F1 with the limit | F1, 11 shared variants |
| --- | :-: | :-: | :-: |
| Qwen3.5-2B | 0.302 | 0.378 | 0.330 |
| Qwen3.5-4B | 0.354 | 0.390 | 0.370 |
| Qwen3.5-9B | 0.331 | 0.363 | 0.344 |
| Qwen3.5-27B | n/a | 0.345 | 0.339 |
| Gemma4-E2B | 0.350 | 0.347 | 0.358 |
| Gemma4-E4B | 0.360 | 0.362 | 0.360 |
| Gemma4-12B | 0.344 | 0.341 | 0.344 |
| Gemma4-31B | n/a | 0.342 | 0.344 |
| _No model: first 3 sentences_ | _0.360_ | _0.360_ | _0.360_ |

- No model beats "take the first 3 sentences", and no model is reliably better
  than another.
- No technique makes a reliable difference. The plain prompt ranks fifth of
  twenty.
- The models that pick most from the opening score highest.

Best five techniques: `tool_no_roleplay`, `tool_augmented`, `tool_no_meta`,
`vanilla`, `salience_inference`.

### BillSum

| Model | F1, all 52 variants | F1 with the limit | F1 without | F1, 12 shared variants | Best single variant |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.211 | 0.179 | 0.242 | 0.215 | 0.284 |
| Qwen3.5-4B | 0.312 | 0.329 | 0.295 | 0.337 | 0.414 |
| Qwen3.5-9B | 0.316 | 0.333 | 0.300 | 0.351 | 0.473 |
| Qwen3.5-27B | n/a | n/a | n/a | 0.376 | 0.467 |
| Gemma4-E2B | 0.288 | 0.294 | 0.282 | 0.337 | 0.484 |
| Gemma4-E4B | 0.387 | 0.391 | 0.383 | 0.403 | 0.495 |
| Gemma4-12B | 0.389 | 0.387 | 0.391 | 0.405 | 0.491 |
| Gemma4-31B | n/a | n/a | n/a | 0.412 | 0.450 |
| _No model: first 6 sentences_ | _0.163_ | _0.163_ | | _0.163_ | |

- Every model beats the baseline; Qwen3.5-2B only barely.
- Gemma4 beats Qwen3.5 at every size. Gemma4-E4B beats Qwen3.5-27B.
- Size helps up to 4B, then little.
- The best variant of every model is a prompt with TF-IDF scores.

**The effect of the TF-IDF scores** (no example, with the limit):

| Model | Plain prompt | Same prompt family, scores removed | With TF-IDF scores |
| --- | :-: | :-: | :-: |
| Qwen3.5-2B | 0.165 | 0.165 | 0.223 |
| Qwen3.5-4B | 0.323 | 0.322 | 0.410 |
| Qwen3.5-9B | 0.296 | 0.328 | 0.469 |
| Gemma4-E2B | 0.233 | 0.253 | 0.453 |
| Gemma4-E4B | 0.380 | 0.396 | 0.495 |
| Gemma4-12B | 0.349 | 0.375 | 0.479 |

The gain disappears when the scores are removed, so it comes from the scores.
Part of it may be specific to silver labels, which are themselves based on
word overlap.

Best five techniques: `tool_no_roleplay`, `tool_augmented`, `least_to_most`,
`negative_aware`, `explanation_based`.

## Position: how much each model relies on the opening

Share of a model's selected sentences that come from the first fifth of the
document, when the answer is limited to K sentences:

| | CNN/DailyMail | BillSum |
| --- | :-: | :-: |
| **Reference labels** | **52%** | **24%** |
| Qwen3.5-2B | 89% | 79% |
| Qwen3.5-4B | 64% | 41% |
| Qwen3.5-9B | 63% | 42% |
| Qwen3.5-27B | 55% | 34% |
| Gemma4-E2B | 60% | 40% |
| Gemma4-E4B | 58% | 33% |
| Gemma4-12B | 52% | 33% |
| Gemma4-31B | 48% | 31% |

- Every model favours the opening. On CNN/DailyMail the labels do too, so it
  goes unnoticed; on BillSum every model is 8 to 18 points above the labels.
- Larger models rely on position less, and Gemma4 less than Qwen3.5.
- On CNN/DailyMail relying on the opening goes with a _higher_ score; on
  BillSum with a _lower_ one.

## Without a limit: how many sentences each model returns

| Model | CNN/DailyMail (31 per document) | BillSum (51 per document) |
| --- | :-: | :-: |
| Qwen3.5-2B | 29 | 49 |
| Qwen3.5-4B | 16 | 32 |
| Qwen3.5-9B | 17 | 33 |
| Qwen3.5-27B | 7 | 16 |
| Gemma4-E2B | 5 | 6 |
| Gemma4-E4B | 5 | 10 |
| Gemma4-12B | 6 | 12 |
| Gemma4-31B | 5 | 8 |

## What to keep in mind

- In Part C the labels come from a heuristic, not from people. A model that
  picks different but equally good sentences is scored as wrong.
- Each dataset has 100 documents, so differences of about 0.03 F1 or less
  could be chance.
- F1 levels are not comparable between datasets: the labels and K differ.
- The two largest models ran through an online service whose answers are not
  exactly repeatable.

## Where to find more

| For | See |
| --- | --- |
| The analysis of each part | [`docs/analysis_part_a_aclsum.md`](docs/analysis_part_a_aclsum.md), [`docs/analysis_part_b_heuristics.md`](docs/analysis_part_b_heuristics.md), [`docs/analysis_part_c_whole_document.md`](docs/analysis_part_c_whole_document.md) |
| Positional bias in detail | [`docs/analysis_positional_bias.md`](docs/analysis_positional_bias.md) |
| What each technique does | [`docs/techniques.md`](docs/techniques.md) |
| Every number, per model and variant | `extsum/reports/<dataset>/summary.csv` and `tables/` |
