# Part B — Heuristics for building extractive labels

**Question.** Most summarization datasets come with a _written_ summary, not
with labels saying which source sentences belong in a summary. Which heuristic best
turns a written summary into sentence labels?

**Short answer.** The simplest one. `local_score` scores every sentence on its
own against the summary with ROUGE and keeps the top K. It beats greedy, beam
and global search, an embedding-based method (MMR), and MoverScore used as the
objective, and it is the cheapest.

Part C uses this rule to label CNN/DailyMail and BillSum.

## Setup

ACLSum is the one dataset here with both a written summary and human sentence
labels, so a heuristic can be judged on it: build labels from the written
summary, then compare the selected sentences with the human selection.

|       |                                                                                                                                                       |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data  | ACLSum test split, three written aspect summaries per paper                                                                                           |
| Modes | **singular**: each aspect on its own, scores averaged over the three. **all_together**: the three summaries concatenated, one selection per paper     |
| Score | ROUGE-1, ROUGE-2, ROUGE-L of the selected sentences against the human extractive selection; `rouge_avg` is their mean. MoverScore as a semantic check |
| K     | swept from 1 to 32 sentences                                                                                                                          |

Four families were compared:

| Family                   | What it does                                  | Objective            | Cost           |
| ------------------------ | --------------------------------------------- | -------------------- | -------------- |
| ROUGE sweep ("fast")     | greedy, beam (size 4), `local_score`, global  | ROUGE-1 / 2 / L      | low, CPU       |
| Transformer sweep ("hf") | the same four heuristics                      | a transformer scorer | high, GPU      |
| MMR                      | relevance minus redundancy, λ from 0.3 to 0.9 | SBERT cosine         | medium, GPU    |
| MoverScore-extract       | the same four heuristics                      | MoverScore           | very high, GPU |

The four heuristics differ in one respect. `local_score` judges each sentence
**independently**. Greedy, beam and global search judge the **selection so
far**, so a sentence that repeats content already selected adds little and is
skipped.

## 1. `local_score` wins, and it wins on recall

At the best overall setting (all_together, extraction by ROUGE-2, K = 14,
scored by ROUGE-1):

| Heuristic       | Precision |  Recall   |    F1     |
| --------------- | :-------: | :-------: | :-------: |
| greedy_search   |   0.810   |   0.564   |   0.645   |
| beam_search     |   0.806   |   0.564   |   0.644   |
| global_score    |   0.806   |   0.565   |   0.644   |
| **local_score** |   0.777   | **0.785** | **0.766** |

- The three joint-search heuristics are indistinguishable from each other
  (F1 0.644–0.645).
- `local_score` gives up 3 points of precision and gains 22 points of recall.
  That is the whole margin.
- **Why.** The joint heuristics avoid redundancy, and the human selection is
  redundant. This is not a flaw in the annotators: they were not writing a
  summary. Their task was to mark every source sentence that expresses the
  aspect, and the same people also wrote a separate, concise summary. The two
  are very different in size and overlap (ACLSum test split, measured for this
  document):

  | Aspect    | Gold sentences marked | Words in the marked sentences | Words in the written summary | Word overlap between two marked sentences | Between two random sentences | Marked sentences with a marked neighbour | By chance |
  | --------- | :-------------------: | :---------------------------: | :--------------------------: | :---------------------------------------: | :--------------------------: | :--------------------------------------: | :-------: |
  | challenge |          4.1          |              119              |              23              |                   0.21                    |             0.15             |                   60%                    |    18%    |
  | approach  |          6.0          |              163              |              22              |                   0.30                    |             0.15             |                   58%                    |    26%    |
  | outcome   |          4.3          |              118              |              21              |                   0.31                    |             0.15             |                   48%                    |    19%    |

  The marked sentences are five to seven times longer than the summary written
  for the same aspect, overlap with each other up to twice as much as random
  sentences do (ROUGE-1), and about half of them sit next to another marked
  sentence. They are passages, not a minimal set. A rule that keeps one
  sentence per topic and skips the rest therefore misses gold sentences, which
  is the recall gap above.

Across every combination of extraction objective, evaluation metric and mode
that was scored, `local_score` is first in all but one, and that one is a tie:

| Extraction objective | Evaluated by | Mode         | Winner      |    Margin    |
| -------------------- | ------------ | ------------ | ----------- | :----------: |
| ROUGE                | rouge_avg    | singular     | local_score |    +0.08     |
| ROUGE                | rouge_avg    | all_together | local_score |    +0.04     |
| ROUGE                | MoverScore   | singular     | local_score |    +0.04     |
| ROUGE                | MoverScore   | all_together | local_score |    +0.02     |
| MoverScore           | rouge_avg    | singular     | local_score |    +0.06     |
| MoverScore           | rouge_avg    | all_together | beam_search | +0.006 (tie) |
| MoverScore           | MoverScore   | singular     | local_score |    +0.02     |
| MoverScore           | MoverScore   | all_together | local_score |    +0.004    |

## 2. How many sentences: the best K is 4

![rouge_avg against K for the four heuristics](../extsum/reports/figures/heuristics_k.png)

Singular mode, extraction by ROUGE-1, mean over the three aspects
(`extsum/reports/heuristics/k_sweep_singular_rouge1.csv`):

|   K   | local_score ROUGE-1 |  ROUGE-2  | ROUGE-L | rouge_avg | greedy rouge_avg |
| :---: | :-----------------: | :-------: | :-----: | :-------: | :--------------: |
|   1   |        0.371        |   0.347   |  0.364  |   0.361   |      0.361       |
|   2   |        0.543        |   0.490   |  0.465  |   0.499   |      0.429       |
|   3   |        0.628        |   0.556   |  0.488  |   0.557   |      0.444       |
| **4** |      **0.651**      | **0.569** |  0.487  | **0.569** |      0.447       |
|   5   |        0.644        |   0.554   |  0.465  |   0.554   |      0.451       |
|   6   |        0.629        |   0.543   |  0.447  |   0.540   |      0.449       |
|   7   |        0.611        |   0.531   |  0.427  |   0.523   |      0.443       |
|   8   |        0.588        |   0.515   |  0.409  |   0.504   |      0.436       |

- **`local_score` has a clear peak at K = 4** (rouge_avg 0.569) and falls on
  both sides. K = 3 is best by ROUGE-L and by MoverScore; K = 4 by ROUGE-1,
  ROUGE-2 and their average.
- **The joint heuristics are flat.** Greedy moves between 0.43 and 0.45 from
  K = 2 to K = 8. Once they have covered the summary's content they add
  sentences that contribute little, so a larger budget does not help them.
- **At K = 1 all four are identical** (0.361): with one sentence there is no
  joint effect, and they pick the same one.
- The gap between `local_score` and the rest opens at K = 2 and is 0.12 at
  the peak.

For all_together mode the best K is 14 by rouge_avg and 13 by MoverScore,
about three times the per-aspect value, as expected for three aspects.

## 3. Costlier objectives do not help

Best configuration of each family, by rouge_avg:

**all_together**

| Family             | Best configuration           | ROUGE-1 | ROUGE-2 | ROUGE-L | rouge_avg | MoverScore |
| ------------------ | ---------------------------- | :-----: | :-----: | :-----: | :-------: | :--------: |
| **ROUGE sweep**    | local_score, ROUGE-2, K = 14 |  0.766  |  0.670  |  0.391  | **0.609** |   0.674    |
| Transformer sweep  | local_score, ROUGE-2, K = 19 |  0.750  |  0.660  |  0.370  |   0.593   |    n/a     |
| MoverScore-extract | beam_search, K = 21          |  0.738  |  0.629  |  0.362  |   0.576   |   0.663    |
| MMR                | λ = 0.9, K = 13              |  0.723  |  0.605  |  0.392  |   0.573   |    n/a     |

**singular**

| Family             | Best configuration          | ROUGE-1 | ROUGE-2 | ROUGE-L | rouge_avg | MoverScore |
| ------------------ | --------------------------- | :-----: | :-----: | :-----: | :-------: | :--------: |
| **ROUGE sweep**    | local_score, ROUGE-1, K = 4 |  0.651  |  0.569  |  0.487  | **0.569** |   0.659    |
| Transformer sweep  | local_score, ROUGE-1, K = 4 |  0.650  |  0.570  |  0.490  |   0.570   |    n/a     |
| MoverScore-extract | local_score, K = 6          |  0.636  |  0.548  |  0.450  |   0.545   |   0.651    |
| MMR                | λ = 0.7, K = 4              |  0.594  |  0.485  |  0.445  |   0.508   |    n/a     |

- **The transformer scorer** matches the ROUGE sweep to within 0.01 in
  singular mode and is slightly worse in all_together, at about fifty times
  the compute.
- **MMR is the weakest family**, 0.04–0.06 below `local_score`. It is the
  method built most explicitly around penalizing redundancy, which is
  consistent with section 1. Its best λ is the highest tested (0.9 in
  all_together), the setting in the tested range closest to plain relevance
  ranking.
- **Optimizing MoverScore directly does not win even on MoverScore.** Selecting
  with ROUGE and scoring with MoverScore gives 0.674; selecting with MoverScore
  gives 0.668. A semantic objective at selection time buys nothing here.

Within a family, `local_score` is the best heuristic in every row but one.

## 4. Cost

`local_score` is one ROUGE computation per sentence. Greedy search takes about
twice as long. Beam and global search take up to ten times as long, for equal
or worse results. The transformer and MoverScore families need a GPU and are
one to two orders of magnitude slower again.

## Conclusions

1. **Use `local_score`.** It is best or tied on every metric and mode, and it
   is the cheapest. Nothing more elaborate earned its cost.
2. **Do not penalize redundancy when the target is human sentence labels.**
   This is the mechanism behind the result, and it separates the task of
   _labelling_ from the task of _summarizing_.
3. **K matters and has a clear optimum**: 4 sentences per aspect on ACLSum.
4. **ROUGE is a sufficient objective.** Transformer and MoverScore scoring do
   not change which sentences are best.
