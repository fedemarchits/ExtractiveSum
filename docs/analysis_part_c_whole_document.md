# Part C — Whole-document extractive summarization on silver labels

**Question.** Do the Part A findings hold when the model summarizes a whole
document, in other domains, against labels produced by local search?

**Answer.** It depends on the dataset.

- **CNN/DailyMail cannot tell models apart.** Its reference sentences sit at
  the start of the article, so "take the first three sentences" scores as well
  as any model. All eight models land within 0.04 F1 of each other.
- **BillSum can.** The same trick scores very low there. Models spread from
  0.21 to 0.41, Gemma4 beats Qwen3.5, and one technique is far ahead of the
  rest: showing a TF-IDF score next to each sentence.

## Setup

|                         | CNN/DailyMail                                                              | BillSum                    |
| ----------------------- | -------------------------------------------------------------------------- | -------------------------- |
| Documents               | 100 news articles                                                          | 100 US Congressional bills |
| Sentences per document  | 31                                                                         | 51                         |
| Sentences to select (K) | 3                                                                          | 6                          |
| Labels                  | silver: `local_score` picks the K sentences closest to the written summary | same                       |
| Models                  | all 8                                                                      | all 8                      |
| Prompt variants         | 52 (local models), 11 (two large models)                                   | 52 (local), 12 (large)     |

**Where K comes from.** For each document: how long is the written summary
compared with the document, and how many of the document's sentences is that?
K is the typical (median) answer, computed on documents that are not
evaluated. News summaries are about 9% of the article, which is 3 sentences;
bill summaries are about 13% of the bill, which is 6.

**How to read the numbers.**

- **F1** measures how well the model's sentences match the silver labels (1 is
  a perfect match). It is agreement with a heuristic, not with a human.
- **The baseline** is what you score with no model at all, by returning the
  first K sentences. A model has to beat it to show it is doing more than
  that.
- **"Significant"** means we compared two results document by document and one
  was better consistently enough that it is unlikely to be chance. "Not
  significant" means the difference could be noise; with 100 documents that is
  typically anything under about 0.03 F1.
- The two datasets use different K, so their F1 levels cannot be compared with
  each other.

The two large models (Qwen3.5-27B, Gemma4-31B) ran each dataset's five best
techniques only. Every technique is described in
[`techniques.md`](techniques.md).

## 1. The two datasets are very different

| With no model                                          | CNN/DailyMail | BillSum |
| ------------------------------------------------------ | :-----------: | :-----: |
| F1 of "the first K sentences"                          |     0.360     |  0.163  |
| Share of the labels in the first fifth of the document |      52%      |   24%   |

![Position of selected sentences](../extsum/reports/figures/position_bias.png)

News puts the key facts first, so the labels cluster at the start. Bills do
not, so their labels are spread through the text. On CNN/DailyMail a model can
score well by position alone; on BillSum it cannot. The
[positional-bias analysis](analysis_positional_bias.md) covers this in detail.

## 2. CNN/DailyMail: nothing separates

![F1 against model size](../extsum/reports/figures/scaling.png)

| Model                         | F1, same 11 variants for all | F1 when capped at 3 | Sentences picked with no cap |
| ----------------------------- | :--------------------------: | :-----------------: | :--------------------------: |
| Qwen3.5-2B                    |            0.330             |        0.378        |              29              |
| Qwen3.5-4B                    |            0.370             |        0.390        |              16              |
| Qwen3.5-9B                    |            0.344             |        0.363        |              17              |
| Qwen3.5-27B                   |            0.339             |        0.345        |              7               |
| Gemma4-E2B                    |            0.358             |        0.347        |              5               |
| Gemma4-E4B                    |            0.360             |        0.362        |              5               |
| Gemma4-12B                    |            0.344             |        0.341        |              6               |
| Gemma4-31B                    |            0.344             |        0.342        |              5               |
| _Baseline: first 3 sentences_ |           _0.360_            |       _0.360_       |                              |

- **No model beats the baseline.** All eight are between 0.33 and 0.37.
- **Bigger is not better.** Gemma4-31B scores the same as 12B, and
  Qwen3.5-27B is significantly _below_ Qwen3.5-4B.
- **No technique stands out.** The TF-IDF prompt is first by 0.02, which is
  not significant. The plain prompt ranks fifth of twenty.
- **The models that copy the opening most score best.** Qwen3.5-2B answers
  exactly "sentences 1, 2, 3" in 58% of its capped answers, and is second best
  when capped.

Why size does not show: larger models rely less on position (89% of
Qwen3.5-2B's picks come from the first fifth of the article, 48% of
Gemma4-31B's), and on this dataset relying on position is what pays. So this
result says the dataset cannot measure the difference, not that large models
are no better at summarizing news.

**Conclusion.** CNN/DailyMail is useful as a control: it shows what a result
looks like when position explains it.

## 3. BillSum: models and techniques separate

### 3.1 Models

| Model                         | F1, all 52 variants | F1, same 12 variants for all | Best single variant |
| ----------------------------- | :-----------------: | :--------------------------: | :-----------------: |
| Qwen3.5-2B                    |        0.211        |            0.215             |        0.284        |
| Qwen3.5-4B                    |        0.312        |            0.337             |        0.414        |
| Qwen3.5-9B                    |        0.316        |            0.351             |        0.473        |
| Qwen3.5-27B                   |         n/a         |            0.376             |        0.467        |
| Gemma4-E2B                    |        0.288        |            0.337             |        0.484        |
| Gemma4-E4B                    |        0.387        |            0.403             |        0.495        |
| Gemma4-12B                    |        0.389        |            0.405             |        0.491        |
| Gemma4-31B                    |         n/a         |            0.412             |        0.450        |
| _Baseline: first 6 sentences_ |       _0.163_       |           _0.163_            |                     |

- **Every model beats the baseline**, most by a wide margin. Qwen3.5-2B is the
  exception: it is barely above it, because it mostly returns the opening
  sentences.
- **Gemma4 beats Qwen3.5 at every size.** Gemma4-E4B (0.403) even beats
  Qwen3.5-27B (0.376), a model seven times larger.
- **Size helps up to 4B, then little.** For Gemma4, the 4B, 12B and 31B models
  are level (differences not significant). For Qwen3.5, 27B is better than 9B
  by 0.025, which is significant.

### 3.2 The technique that matters: TF-IDF scores

![BillSum F1 for four versions of the tool prompt](../extsum/reports/figures/billsum_tfidf.png)

The `tool_augmented` prompt prints, next to each sentence, a score for how
central that sentence is to the document (computed from word overlap, with no
model). Four versions of the prompt show what causes the gain:

| What the prompt contains                         | Result                                                       |
| ------------------------------------------------ | ------------------------------------------------------------ |
| Plain prompt                                     | the reference point                                          |
| Tool prompt **with** the scores                  | **+0.06 to +0.25 F1 for every local model**, all significant |
| Tool prompt with the scores **removed**          | back to the plain prompt's level                             |
| Scores kept, "you are an expert" framing removed | no change                                                    |

So the gain comes from the scores, not from the wording around them. It helps
weak models most: Gemma4-E2B goes from 0.23 to 0.48, level with models six
times its size. A prompt with the scores is the best variant for every one of
the eight models.

**A caution.** The score measures word overlap inside the document, and the
silver labels are also chosen by word overlap (with the summary). The two are
related, so part of this gain may come from handing the model a hint about how
the labels were made. On human labels (Part A) the same technique was second
from last.

### 3.3 Every other technique is close to the plain prompt

| Technique                                  | F1 (mean over the six local models) |
| ------------------------------------------ | :---------------------------------: |
| `tool_no_roleplay` (TF-IDF scores)         |                0.440                |
| `tool_augmented` (TF-IDF scores)           |                0.421                |
| Best reasoning technique (`least_to_most`) |                0.327                |
| Plain prompt                               |                0.309                |
| Worst (`self_ask`)                         |                0.290                |

Eighteen techniques sit between 0.29 and 0.33. No reasoning technique adds
more than 0.02 over asking plainly, and `self_ask`, the best technique in
Part A, is last.

### 3.4 Capping

![Sentences selected without a cap](../extsum/reports/figures/overselection.png)

Asked to select with no limit, the smaller Qwen3.5 models return most of the
bill (32 to 49 of 51 sentences). Gemma4 stops at 6 to 12 on its own. So a cap
is essential for Qwen3.5 (+0.03 F1 and a much better summary) and makes little
difference for Gemma4. Qwen3.5-27B still selects 16.

### 3.5 Examples and reasoning traces

- **A worked example (one-shot)** helps the models that are weak or select too
  much (Gemma4-E2B, and Qwen3.5 4B and 9B without a cap: about +0.05). It
  slightly hurts the two strongest local models. It seems to work mainly as a
  hint about how many sentences to pick.
- **Showing the reasoning in the example** does not help: five of six models
  are slightly worse with it. Gemma4-E2B is the one model it helps, here and
  on the other datasets.

### 3.6 Position

Even on BillSum, where the labels are spread out, every model picks too much
from the opening: 31–42% of their sentences come from the first fifth of the
bill, against 24% for the labels (79% for Qwen3.5-2B). The models that do this
least score highest.

## 5. The three datasets side by side

![Gain of each technique over the plain prompt](../extsum/reports/figures/techniques.png)

|                                        | ACLSum (Part A)       | CNN/DailyMail   | BillSum               |
| -------------------------------------- | --------------------- | --------------- | --------------------- |
| Models beat "first K sentences"        | n/a                   | no              | yes                   |
| Bigger models are better               | up to 4B, then barely | no              | up to 4B, then barely |
| Gemma4 vs Qwen3.5                      | Gemma slightly ahead  | level           | Gemma clearly ahead   |
| Best technique                         | reasoning techniques  | none stands out | TF-IDF scores, by far |
| Reasoning techniques vs plain prompt   | small gain            | none            | none                  |
| Reasoning traces in the example        | no effect             | no effect       | no effect             |
| Qwen3.5 selects too much without a cap | yes                   | yes             | yes                   |

**What holds everywhere:** gains from size stop at about 4B; Qwen3.5
over-selects without a cap; reasoning traces do not help; Qwen3.5-2B selects
by position.

**What does not transfer:** the ranking of techniques. No technique is in the
top five of all three datasets.

## Conclusions

1. **Choose the dataset before the model.** CNN/DailyMail alone would suggest
   that a 4B model is as good as a 31B one and that answering "sentences 1, 2,
   3" is a strong strategy. BillSum shows otherwise.
2. **On BillSum, one cheap signal beats every prompting strategy.** A TF-IDF
   score per sentence adds up to 0.25 F1; no reasoning technique adds more
   than 0.02.
3. **That result is partly about the silver labels** and needs human-labelled
   documents to confirm.
4. **Cap the smaller Qwen3.5 models.** Without a cap they return most of the
   document.
5. **The model family matters more than size** on BillSum.
