# Summary of results — Parts A, B and C

LLM extractive summarization by prompting, and heuristics for building silver extractive labels.

- **Part A** — aspect-based extractive summarization on ACLSum (code: `extsum/`).
- **Part B** — heuristics that turn abstractive summaries into extractive labels, on ACLSum (code and full report in the `Extractive-Summarization-with-LLMs-and-Heuristics` project).
- **Part C** — whole-document extractive summarization on silver labels: CNN/DailyMail and BillSum (code: `extsum/`).

**Models:** Qwen3.5 {2B, 4B, 9B, 27B} and Gemma4 {E2B, E4B, 12B, 31B}. The six smaller models run locally at bf16; the two largest run through OpenRouter on the five best techniques of each part.

## Headline findings

| Part | Finding |
| --- | --- |
| A | Scaling is monotonic in both families and flattens past the medium tier (9B→27B and 12B→31B both +0.008 F1). Gemma4 leads Qwen3.5 at every tier. |
| A | Capping barely changes F1 but clearly improves ROUGE-L. One-shot hurts the smallest models and helps larger ones. |
| A | Best techniques: `self_ask`, `negative_aware`, `salience_inference`, `self_critique`, `contrastive_joint`. `tool_augmented` is second-last. |
| B | `local_score` (independent per-sentence ROUGE, top K) beats greedy, beam, global, MMR and MoverScore-based extraction, and is the cheapest. |
| C, CNN/DM | All eight models score 0.33–0.37, around the "first 3 sentences" baseline (0.360); size brings nothing, and the models that pick most from the opening score highest: the dataset rewards position. |
| C, BillSum | "First 6" scores 0.163, so the dataset separates selecting from copying positions. Best variants reach 0.28–0.50. |
| C, BillSum | Showing TF-IDF centrality scores adds 0.06–0.25 F1 over the plain prompt for every local model; no reasoning technique comes close. The opposite of Part A, where the tool technique ranked near the bottom. |
| C, BillSum | Gemma4 beats Qwen3.5 at every size. Scaling is flat at the top (Gemma 12B→31B +0.007, not significant; Qwen 9B→27B +0.025, significant). |

---

## Part A — Aspect-based extractive summarization (ACLSum)

File paths in this part are relative to `extsum/`.

### Results

Aggregated from `reports/aclsum/summary.csv` (one row per model × variant × aspect).
Metrics: per-sentence **F1 / P / R** on selected indices, **ROUGE-L** of the built
summary, **oracle gap** (ROUGE-L below the best achievable selection; lower is
better). Aspect means include the per-doc `union` selection alongside the three
aspects. **All 8 grid models now have results**; the two largest (`qwen35_27b`,
`gemma4_31b`) are endpoint-only and were run on a **partial grid — the 5 best
techniques only** (budget-limited).

### Where each model was computed

| Model        | GPU                       | Backend         | Precision | Status              |
| ------------ | ------------------------- | --------------- | --------- | ------------------- |
| `qwen35_2b`  | RTX 3090 (faretra)        | HF transformers | bf16      | ✅ done             |
| `qwen35_4b`  | RTX 3090 (faretra)        | HF transformers | bf16      | ✅ done             |
| `qwen35_9b`  | A100 40GB (vast.ai)       | **vLLM**        | bf16      | ✅ done             |
| `gemma4_e2b` | RTX 3090 (faretra)        | HF transformers | bf16      | ✅ done             |
| `gemma4_e4b` | A100 40GB (vast.ai)       | **vLLM**        | bf16      | ✅ done             |
| `gemma4_12b` | A100 40GB (vast.ai)       | **vLLM**        | bf16      | ✅ done             |
| `qwen35_27b` | — (endpoint / OpenRouter) | endpoint        | full      | ⚠️ partial (5 tech) |
| `gemma4_31b` | — (endpoint / OpenRouter) | endpoint        | full      | ⚠️ partial (5 tech) |

gemma-4 is too new for the CUDA-12 cluster's vLLM, so `9b / e4b / 12b` were run
on a rented A100 (CUDA-13 → modern vLLM). All six local models are bf16, so
quality is comparable; **latency is not** (vLLM batches, HF does not; the endpoint
latency is round-trip wall-time, not compute).

The two large models were served via **OpenRouter** (`qwen/qwen3.5-27b`,
`google/gemma-4-31b-it`) at full precision, on a ~$10 budget that covered only the
**5 best techniques** (18 of 54 variants each). See
[§ Large tier](#large-tier--partial-run-5-best-techniques).

### Prompt variants computed (per model)

Each technique × **shot** (`zero`/`one`) × **cap** (`capped`/`uncapped`) — the
4-cell grid, except `negative_aware` and the `_trace` ablations (one-shot only).
**54 variants per model** — for the **6 local models**. The **2 large endpoint
models ran a partial grid: the 5 best techniques only = 18 variants each**
(`self_ask`, `salience_inference`, `self_critique`, `contrastive_joint`,
`negative_aware`; no `_trace` ablations). The full 54-cell grid is preserved in
`configs/grids/aspect_full.yaml`; the partial grid is `configs/grids/aspect_large.yaml`.

| Technique                | zero·cap | zero·unc | one·cap | one·unc |
| ------------------------ | :------: | :------: | :-----: | :-----: |
| vanilla                  |    ✅    |    ✅    |   ✅    |   ✅    |
| chain_of_thought         |    ✅    |    ✅    |   ✅    |   ✅    |
| least_to_most            |    ✅    |    ✅    |   ✅    |   ✅    |
| self_ask                 |    ✅    |    ✅    |   ✅    |   ✅    |
| explanation_based        |    ✅    |    ✅    |   ✅    |   ✅    |
| salience_inference       |    ✅    |    ✅    |   ✅    |   ✅    |
| scoring_based            |    ✅    |    ✅    |   ✅    |   ✅    |
| tool_augmented           |    ✅    |    ✅    |   ✅    |   ✅    |
| simulated_tool_augmented |    ✅    |    ✅    |   ✅    |   ✅    |
| self_critique            |    ✅    |    ✅    |   ✅    |   ✅    |
| contrastive_joint        |    ✅    |    ✅    |   ✅    |   ✅    |
| negative_aware           |    —     |    —     |   ✅    |   ✅    |
| chain_of_thought_trace   |    —     |    —     |   ✅    |   ✅    |
| salience_inference_trace |    —     |    —     |   ✅    |   ✅    |
| scoring_based_trace      |    —     |    —     |   ✅    |   ✅    |
| self_ask_trace           |    —     |    —     |   ✅    |   ✅    |

### Results by model

Mean over all 54 variants × 3 aspects. `latency` = mean s/doc (backend-dependent,
not a cross-model comparison).

| Model        |  F1   | Precision | Recall | ROUGE-L | Oracle gap | Latency (s) |
| ------------ | :---: | :-------: | :----: | :-----: | :--------: | :---------: |
| `qwen35_2b`  | 0.461 |   0.438   | 0.674  |  0.137  |   0.263    |    4.48     |
| `qwen35_4b`  | 0.609 |   0.626   | 0.661  |  0.176  |   0.225    |    3.34     |
| `qwen35_9b`  | 0.623 |   0.641   | 0.678  |  0.177  |   0.223    |    0.84     |
| `gemma4_e2b` | 0.567 |   0.626   | 0.570  |  0.185  |   0.216    |    3.51     |
| `gemma4_e4b` | 0.611 |   0.647   | 0.639  |  0.179  |   0.222    |    0.43     |
| `gemma4_12b` | 0.631 |   0.642   | 0.681  |  0.179  |   0.222    |    1.18     |
| `qwen35_27b` |   †   |     †     |   †    |    †    |     †      |      †      |
| `gemma4_31b` |   †   |     †     |   †    |    †    |     †      |      †      |

† Large models ran only the 5 best techniques, so a 54-variant mean isn't
comparable — see [§ Large tier](#large-tier--partial-run-5-best-techniques) for
their stats and a like-for-like 5-technique comparison across all 8 models.

### Comparison 1 — capped vs. uncapped (per model)

| Model        | F1 capped | F1 uncapped | ROUGE-L capped | ROUGE-L uncapped |
| ------------ | :-------: | :---------: | :------------: | :--------------: |
| `qwen35_2b`  |   0.486   |    0.437    |   **0.178**    |      0.097       |
| `qwen35_4b`  |   0.602   |    0.616    |   **0.191**    |      0.160       |
| `qwen35_9b`  |   0.620   |    0.627    |   **0.194**    |      0.161       |
| `gemma4_e2b` |   0.558   |    0.576    |   **0.192**    |      0.177       |
| `gemma4_e4b` |   0.603   |    0.619    |   **0.187**    |      0.170       |
| `gemma4_12b` |   0.623   |    0.638    |   **0.192**    |      0.166       |

F1 is roughly flat (uncapped slightly higher except the weak 2B), but **ROUGE-L
is consistently better capped** — a fixed sentence budget stops over-selection
from diluting the summary.

### Comparison 2 — zero-shot vs. one-shot (per model, F1)

| Model        | zero-shot | one-shot |   Δ    |
| ------------ | :-------: | :------: | :----: |
| `qwen35_2b`  |   0.483   |  0.446   | −0.037 |
| `qwen35_4b`  |   0.605   |  0.612   | +0.007 |
| `qwen35_9b`  |   0.615   |  0.629   | +0.014 |
| `gemma4_e2b` |   0.589   |  0.552   | −0.037 |
| `gemma4_e4b` |   0.589   |  0.626   | +0.037 |
| `gemma4_12b` |   0.625   |  0.635   | +0.010 |

The exemplar **hurts the smallest models** (`2b`, `e2b`) and **helps as size
grows** — capacity to use a worked example appears with scale.

### How the rationale (trace) exemplars were built

The `_trace` variants show a one-shot example whose reasoning was produced
**once** by a strong reference model — **Qwen3.5-397B** (`qwen35_397b`, via
endpoint) — not by the grid model being evaluated. Built by
`extsum rationales`:

1. For each reasoning technique × aspect, draw candidate exemplars from the
   **TRAIN** split only (never test).
2. Ask the 397B to solve the exemplar **and reveal its step-by-step reasoning**,
   ending with a parseable `Selected indices: [...]` line.
3. **Verify against human gold**: score the model's picks (F1). Try up to 10
   exemplars — early-accept at F1 ≥ 0.7, otherwise keep the best if it clears
   0.4; below that, that cell falls back to an answer-only one-shot.
4. Cache the `(exemplar, reasoning trace, gold)` once; **every grid model reuses
   the same frozen trace**.

So a trace is a _verified, gold-aligned reasoning demonstration from a large
model_, shown identically to every smaller model — which isolates the effect of
"showing **how** to reason" from each model's own capability.

### Comparison 3 — one-shot: answer-only vs. rationale (trace)

For the four techniques with a `_trace` variant (`chain_of_thought`,
`salience_inference`, `scoring_based`, `self_ask`): one-shot exemplar as a plain
answer vs. the 397B reasoning trace.

| Model        | answer-only F1 | trace F1 |     Δ      |
| ------------ | :------------: | :------: | :--------: |
| `qwen35_2b`  |     0.450      |  0.450   |   −0.000   |
| `qwen35_4b`  |     0.608      |  0.617   |   +0.008   |
| `qwen35_9b`  |     0.629      |  0.630   |   +0.001   |
| `gemma4_e2b` |     0.542      |  0.571   | **+0.029** |
| `gemma4_e4b` |     0.630      |  0.638   |   +0.008   |
| `gemma4_12b` |     0.635      |  0.632   |   −0.004   |

Rationale exemplars help most models slightly (largest gain on `gemma4_e2b`), but
the effect is small and not universal — no gain on the weakest (`2b`) or the
largest (`12b`).

### Large tier — partial run (5 best techniques)

The large tier is endpoint-only (OpenRouter, full precision). A ~$10 credit budget
covered the **5 best techniques by mean F1** — `self_ask`, `salience_inference`,
`self_critique`, `contrastive_joint`, `negative_aware` — i.e. **18 of 54 variants
each** (no `_trace` ablations). Both models completed all 18 with zero errors
(100 test docs × 3 aspects per variant).

**Full stats for the two large models** (mean over their 18 variants; incl. `union`):

| Model        |  F1   | Precision | Recall | ROUGE-L | Oracle gap | Latency (s)\* |
| ------------ | :---: | :-------: | :----: | :-----: | :--------: | :-----------: |
| `qwen35_27b` | 0.633 |   0.645   | 0.688  |  0.178  |   0.222    |     7.88      |
| `gemma4_31b` | 0.641 |   0.655   | 0.684  |  0.181  |   0.220    |     9.96      |

\* Endpoint round-trip wall-time (queueing + network), **not** compute — not
comparable to the local vLLM/HF latencies above.

**Like-for-like scaling** — mean F1 on the **same 5 techniques**, all 8 models, so
the large tier is directly comparable to the smaller ones:

| Tier      | Qwen3.5 model |    F1     | Gemma4 model |    F1     |
| --------- | ------------- | :-------: | ------------ | :-------: |
| small     | `qwen35_2b`   |   0.472   | `gemma4_e2b` |   0.571   |
| small·med | `qwen35_4b`   |   0.612   | `gemma4_e4b` |   0.629   |
| medium    | `qwen35_9b`   |   0.625   | `gemma4_12b` |   0.633   |
| **large** | `qwen35_27b`  | **0.633** | `gemma4_31b` | **0.641** |

**Scaling is monotonic in both families** — every size step gains F1, no
inversions. Gemma4 leads Qwen3.5 at every tier. Returns diminish sharply past the
medium tier: Qwen `9b→27b` is **+0.008**, Gemma `12b→31b` is **+0.008**, vs the
huge low-end jump (Qwen `2b→4b` = +0.140). The extra parameters and full precision
of the large tier buy very little on this extractive task.

The two earlier per-model patterns **hold at the large tier** (5-technique means):

- **Capped vs. uncapped** — F1 flat/slightly higher uncapped (`27b` 0.628→0.639,
  `31b` 0.631→0.650), but **ROUGE-L clearly better capped** (`27b` 0.192 vs 0.164,
  `31b` 0.193 vs 0.169). Same story as the local models.
- **Zero- vs. one-shot** — one-shot ≥ zero-shot (`27b` 0.630→0.636, `31b`
  0.630→0.649), consistent with "exemplars help as size grows."

### Top 5 prompts (mean F1 over the 6 local models)

This ranking is what **selected the 5 techniques** run on the large tier (the
`_trace` winners collapse to their base technique, since no `_trace` was run on
the endpoint models): `self_ask`, `salience_inference`, `self_critique`,
`contrastive_joint`, `negative_aware`.

| Rank | Technique                  |  F1   | Precision | Recall | ROUGE-L |
| :--: | -------------------------- | :---: | :-------: | :----: | :-----: |
|  1   | `self_ask_trace`           | 0.596 |   0.622   | 0.657  |  0.176  |
|  2   | `negative_aware`           | 0.593 |   0.626   | 0.639  |  0.178  |
|  3   | `salience_inference_trace` | 0.591 |   0.636   | 0.625  |  0.181  |
|  4   | `salience_inference`       | 0.591 |   0.631   | 0.629  |  0.180  |
|  5   | `self_critique`            | 0.590 |   0.601   | 0.662  |  0.171  |

Winners are the **reasoning-trace** and **precision-oriented** techniques (note
the high precision, 0.62–0.64, they probably win by cutting false positives?).

For reference:: the baseline `vanilla` is last at **0.558** and `tool_augmented`
second-last at 0.571.

---

## Part B — Heuristics for silver extractive labels (ACLSum)

Source: the original Part B report; the K sweep was re-run with `extsum heuristics sweep`. ACLSum test split (~250 documents), three section summaries per document (`challenge`, `approach`, `outcome`). Two modes: **singular** (each section on its own, scores averaged) and **all_together** (the three sections concatenated).

**Finding: `local_score` is the best heuristic and the cheapest.** It scores each sentence independently against the reference (ROUGE) and takes the top K. Greedy, beam and global search score the joint selection and penalise redundancy, which costs recall.

### Methods compared

| Method | Parameters | Scoring objective | Cost |
| --- | --- | --- | --- |
| fast-sweep | greedy / beam / local / global, K ∈ [1, 32] | ROUGE-1/2/L | low (CPU) |
| hf-sweep | same 4 heuristics, K ∈ {4..8} / {19..23} | HF transformer | high (GPU) |
| MMR | λ ∈ {0.3, 0.5, 0.7, 0.9}, K ∈ [1, 32] | SBERT cosine + redundancy | medium (GPU) |
| MoverScore-extract | same 4 heuristics, K = 6 (singular), 21 (all_together) | MoverScore | very high (GPU) |

`beam_size = 4` for `beam_search` and `global_score`.

### Heuristics at the best slice (K=14, all_together, extract=ROUGE-2, eval=ROUGE-1)

| Heuristic | Precision | Recall | F1 |
| --- | :-: | :-: | :-: |
| greedy_search | 0.810 | 0.564 | 0.645 |
| beam_search | 0.806 | 0.564 | 0.644 |
| **local_score** | 0.777 | **0.785** | **0.766** |
| global_score | 0.806 | 0.565 | 0.644 |

`local_score` gives up 3 points of precision for 22 points of recall.

### Which heuristic wins each slice

| Extract method | Eval | Mode | Winner | Margin |
| --- | --- | --- | --- | :-: |
| fast (ROUGE) | rouge_avg | singular | **local_score** | +0.08 |
| fast (ROUGE) | rouge_avg | all_together | **local_score** | +0.04 |
| fast (ROUGE) | moverscore | singular | **local_score** | +0.04 |
| fast (ROUGE) | moverscore | all_together | **local_score** | +0.02 |
| MoverScore | rouge_avg | singular | **local_score** | +0.06 |
| MoverScore | rouge_avg | all_together | beam_search | tie (+0.006) |
| MoverScore | moverscore | singular | **local_score** | +0.02 |
| MoverScore | moverscore | all_together | **local_score** | +0.004 |

### Best instance per family, by `rouge_avg`

**all_together**

| Family | Best | ROUGE-1 | ROUGE-2 | ROUGE-L | rouge_avg | MoverScore |
| --- | --- | :-: | :-: | :-: | :-: | :-: |
| **fast** | local_score, extract=rouge2, K=14 | 0.7660 | 0.6699 | 0.3907 | **0.6089** | 0.6735 |
| hf | local_score, extract=rouge2, K=19 | 0.7500 | 0.6600 | 0.3700 | 0.5933 | n/a |
| MMR | λ=0.9, K=13 | 0.7234 | 0.6045 | 0.3919 | 0.5733 | n/a |
| MoverScore-extract | beam_search, K=21 | 0.7377 | 0.6285 | 0.3620 | 0.5761 | 0.6633 |

**singular**

| Family | Best | ROUGE-1 | ROUGE-2 | ROUGE-L | rouge_avg | MoverScore |
| --- | --- | :-: | :-: | :-: | :-: | :-: |
| **fast** | local_score, extract=rouge1, K=4 | 0.6514 | 0.5688 | 0.4871 | **0.5691** | 0.6586 |
| hf | local_score, extract=rouge1, K=4 | 0.6500 | 0.5700 | 0.4900 | 0.5700 | n/a |
| MMR | λ=0.7, K=4 | 0.5940 | 0.4848 | 0.4445 | 0.5078 | n/a |
| MoverScore-extract | local_score, K=6 | 0.6361 | 0.5483 | 0.4504 | 0.5449 | 0.6506 |

### Best instance per family, by MoverScore

| Mode | Family | Best | MoverScore | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | --- | :-: | :-: | :-: | :-: |
| all_together | **fast** | local_score, extract=rouge2, K=13 | **0.6739** | 0.7643 | 0.6692 | 0.3933 |
| all_together | MoverScore-extract | local_score, K=21 | 0.6679 | 0.7105 | 0.6429 | 0.3564 |
| singular | **fast** | local_score, extract=rouge1, K=3 | **0.6608** | 0.6281 | 0.5558 | 0.4881 |
| singular | MoverScore-extract | local_score, K=6 | 0.6506 | 0.6361 | 0.5483 | 0.4504 |

- **Transformer scoring (hf-sweep)** lands within 0.01 of the ROUGE sweep at about 50× the compute.
- **MMR** peaks at `rouge_avg` 0.5733 (λ=0.9, K=13), about 0.036 below `local_score`.
- **MoverScore as the extraction objective** does not beat ROUGE-based `local_score`, even when evaluated with MoverScore.
- **Compute:** `local_score` is the cheapest by an order of magnitude; greedy takes about twice as long, beam and global up to ten times.

### What Part C takes from Part B

Part C uses `local_score` with ROUGE-1 to build its silver labels. It does not reuse K from ACLSum: K is derived for each dataset from its own compression ratio (3 for CNN/DailyMail, 6 for BillSum).

On ACLSum itself, the K sweep in singular mode (extract = ROUGE-1, scored against the extractive gold; re-run with `extsum heuristics sweep`) peaks at K = 4:

| K | ROUGE-1 | ROUGE-2 | ROUGE-L | rouge_avg |
| :-: | :-: | :-: | :-: | :-: |
| 3 | 0.6281 | 0.5558 | 0.4881 | 0.5573 |
| **4** | **0.6514** | **0.5688** | 0.4871 | **0.5691** |
| 5 | 0.6435 | 0.5542 | 0.4647 | 0.5542 |
| 6 | 0.6287 | 0.5433 | 0.4469 | 0.5396 |

### Caveats (Part B)

- ACLSum test split only.
- hf-sweep K range limited to [4..8] (singular) and [19..23] (all_together).
- MMR and hf-sweep were not scored under MoverScore.

---

## Part C — Whole-document extractive summarization

Code and reports are in `extsum/` (experiments `cnndm`, `billsum`).

Silver labels from `local_score` (ROUGE-1). 52 prompt variants (12 techniques + ablations + trace, × shot × cap) on the six local models; the two large models ran each dataset's five best techniques. Metrics: sentence-index Precision / Recall / F1 against silver, ROUGE-L of the reconstructed summary against the abstractive reference, oracle gap (ROUGE-L below the greedy oracle).

K is derived per dataset from its compression ratio on a split that is not evaluated (`extsum compute-k`):

| | CNN/DailyMail (validation, 13,368 articles) | BillSum (train, 18,949 bills) |
| --- | :-: | :-: |
| Compression ratio, median | 0.092 | 0.130 |
| Sentences per document, median | 28 | 46 |
| Sentences per summary, median | 2.55 | 6.06 |
| **K** | **3** | **6** |

- **CNN/DailyMail:** 100 random test articles (seed 42), all eight models, K = 3.
- **BillSum:** 100 random test bills (seed 42), all eight models, K = 6. Section below.

### CNN/DailyMail (K = 3)

No-model baselines: first 3 sentences 0.360 F1; every sentence 0.217.

#### Mean over all 52 variants — six local models

| Model | P | R | F1 | ROUGE-L | Oracle gap |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.256 | 0.677 | 0.302 | 0.185 | 0.215 |
| Qwen3.5-4B | 0.307 | 0.572 | 0.354 | 0.218 | 0.182 |
| Qwen3.5-9B | 0.282 | 0.563 | 0.331 | 0.210 | 0.190 |
| Gemma4-E2B | 0.331 | 0.398 | 0.350 | 0.234 | 0.166 |
| Gemma4-E4B | 0.334 | 0.419 | 0.360 | 0.239 | 0.161 |
| Gemma4-12B | 0.309 | 0.426 | 0.344 | 0.236 | 0.164 |

#### Capped vs uncapped

| Model | F1 capped | F1 uncapped | ROUGE-L capped | ROUGE-L uncapped | Sentences picked, uncapped |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.378 | 0.227 | 0.251 | 0.119 | 29.4 |
| Qwen3.5-4B | 0.390 | 0.317 | 0.259 | 0.176 | 16.1 |
| Qwen3.5-9B | 0.363 | 0.299 | 0.249 | 0.171 | 16.9 |
| Gemma4-E2B | 0.347 | 0.353 | 0.242 | 0.227 | 4.9 |
| Gemma4-E4B | 0.362 | 0.358 | 0.248 | 0.230 | 5.5 |
| Gemma4-12B | 0.341 | 0.348 | 0.247 | 0.224 | 6.4 |

#### Zero-shot vs one-shot (mean F1, nine techniques with all four variants)

| Model | zero·capped | one·capped | zero·uncapped | one·uncapped |
| --- | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.373 | 0.380 | 0.225 | 0.225 |
| Qwen3.5-4B | 0.394 | 0.390 | 0.312 | 0.331 |
| Qwen3.5-9B | 0.373 | 0.352 | 0.284 | 0.314 |
| Gemma4-E2B | 0.347 | 0.327 | 0.355 | 0.340 |
| Gemma4-E4B | 0.353 | 0.368 | 0.343 | 0.367 |
| Gemma4-12B | 0.319 | 0.351 | 0.326 | 0.370 |

#### Technique ranking (mean F1 over six local models)

| Rank | Technique | F1 | P | R |
| :-: | --- | :-: | :-: | :-: |
| 1 | tool_no_roleplay | 0.375 | 0.377 | 0.374 |
| 2 | tool_augmented | 0.370 | 0.372 | 0.369 |
| 3 | salience_inference_trace | 0.365 | 0.331 | 0.510 |
| 4 | tool_no_meta | 0.357 | 0.357 | 0.357 |
| 5 | vanilla | 0.349 | 0.320 | 0.472 |
| 6 | chain_of_thought_trace | 0.348 | 0.310 | 0.516 |
| 7 | negative_aware | 0.345 | 0.315 | 0.493 |
| 8 | scoring_based_trace | 0.344 | 0.306 | 0.512 |
| 9 | self_ask_trace | 0.343 | 0.306 | 0.528 |
| 10 | contrastive_joint | 0.343 | 0.306 | 0.509 |
| 11 | salience_inference | 0.340 | 0.302 | 0.513 |
| 12 | least_to_most | 0.340 | 0.305 | 0.487 |
| 13 | chain_of_thought | 0.337 | 0.299 | 0.516 |
| 14 | simulated_tool_augmented | 0.337 | 0.299 | 0.525 |
| 15 | explanation_based | 0.335 | 0.298 | 0.498 |
| 16 | self_ask | 0.332 | 0.294 | 0.525 |
| 17 | self_critique | 0.332 | 0.293 | 0.510 |
| 18 | scoring_no_length | 0.306 | 0.223 | 0.674 |
| 19 | scoring_no_redundancy | 0.305 | 0.219 | 0.689 |
| 20 | scoring_based | 0.305 | 0.220 | 0.681 |

With `_trace` folded into its base technique, the five best are `tool_no_roleplay`, `tool_augmented`, `tool_no_meta`, `vanilla`, `salience_inference`; these are the techniques the large tier ran (11 variants).

#### TF-IDF ablation (zero-shot, capped, F1)

| Model | vanilla | tool_no_meta (no scores) | tool_augmented | tool_no_roleplay |
| --- | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.373 | 0.377 | 0.387 | 0.390 |
| Qwen3.5-4B | 0.380 | 0.373 | 0.355 | 0.388 |
| Qwen3.5-9B | 0.367 | 0.373 | 0.340 | 0.347 |
| Gemma4-E2B | 0.357 | 0.367 | 0.413 | 0.390 |
| Gemma4-E4B | 0.340 | 0.340 | 0.357 | 0.363 |
| Gemma4-12B | 0.321 | 0.310 | 0.370 | 0.375 |

None of the 198 paired comparisons involving a tool prompt is significant (Wilcoxon, Benjamini–Hochberg).

#### Large tier — same 11 variants, all eight models

| Model | P | R | F1 | ROUGE-L | Sentences (uncapped) |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.298 | 0.592 | 0.330 | 0.205 | 28.5 |
| Qwen3.5-4B | 0.339 | 0.494 | 0.370 | 0.233 | 12.5 |
| Qwen3.5-9B | 0.309 | 0.497 | 0.344 | 0.223 | 15.3 |
| **Qwen3.5-27B** | 0.316 | 0.400 | 0.339 | 0.239 | 6.9 |
| Gemma4-E2B | 0.346 | 0.385 | 0.358 | 0.238 | 4.5 |
| Gemma4-E4B | 0.345 | 0.393 | 0.360 | 0.241 | 4.9 |
| Gemma4-12B | 0.324 | 0.389 | 0.344 | 0.239 | 5.8 |
| **Gemma4-31B** | 0.334 | 0.372 | 0.344 | 0.251 | 4.9 |

Paired Wilcoxon over the 100 articles: Qwen 27B − 9B −0.005 (p = 0.58); Qwen 27B − 4B −0.031 (p = 0.002); Gemma 31B − 12B 0.000 (p = 0.88); Gemma 31B − Qwen 27B +0.005 (p = 0.90).

#### Best-F1 variant per model

| Model | Best F1 | 95% CI | Technique | shot | cap |
| --- | :-: | :-: | --- | :-: | :-: |
| Qwen3.5-2B | 0.397 | 0.343–0.450 | contrastive_joint | zero_shot | capped |
| Qwen3.5-4B | 0.417 | 0.370–0.463 | contrastive_joint | one_shot | capped |
| Qwen3.5-9B | 0.390 | 0.343–0.437 | simulated_tool_augmented | zero_shot | capped |
| Qwen3.5-27B | 0.373 | 0.330–0.420 | tool_augmented | zero_shot | capped |
| Gemma4-E2B | 0.413 | 0.360–0.470 | tool_augmented | zero_shot | capped |
| Gemma4-E4B | 0.392 | 0.350–0.435 | salience_inference_trace | one_shot | uncapped |
| Gemma4-12B | 0.385 | 0.339–0.431 | salience_inference_trace | one_shot | capped |
| Gemma4-31B | 0.358 | 0.316–0.403 | vanilla | zero_shot | uncapped |

#### Position of selections (capped variants, as answered)

| | Share of picks in the first fifth | Answer is exactly sentences 1–3 |
| --- | :-: | :-: |
| Silver labels | 52% | — |
| Qwen3.5-2B | 89% | 58% |
| Qwen3.5-4B | 64% | 8% |
| Qwen3.5-9B | 63% | 12% |
| Qwen3.5-27B | 55% | 4% |
| Gemma4-E2B | 60% | 2% |
| Gemma4-E4B | 58% | 1% |
| Gemma4-12B | 52% | 2% |
| Gemma4-31B | 48% | 3% |

---

### BillSum

Whole-bill extractive summarization on **BillSum** (US Congressional bills), added because CNN/DailyMail rewards position: on CNN/DM 52% of the silver sentences sit in the first fifth of the article, on BillSum 24% (an even spread is 20%).

**Run:** 100 random bills from the `test` split (seed 42), identical across models. Same 52-variant grid. Six models at bf16 on vLLM: Qwen3.5 {2B, 4B, 9B}, Gemma4 {E2B, E4B, 12B}. The two large models (Qwen3.5-27B, Gemma4-31B) ran a 12-variant subset through OpenRouter; see "Large tier" below. Tables headed "52 variants" cover the six local models only.

**Setup**

- **Prompts:** the bill-specific prompt set (`extsum/src/extsum/prompts/sets/bill/`). Same layout, rules and return format as the generic `document` set; only the domain wording changes. The generic prompts were not run on BillSum.
- **Sentence splitting:** structure-aware (`extsum/src/extsum/data/segmentation.py::bill_sentence_split`): each enumerated provision is a unit and section headers attach to the sentence that follows. Bills average 50.9 units.
- **K = 6**, derived from the compression ratio on the BillSum `train` split (`extsum compute-k`, `extsum/configs/heuristics/billsum.json`). BillSum has no validation split, so `train` plays that role, for K and for the one-shot exemplar pool (100 bills, seed 42).
- **Parsing:** `gemma4_e2b` often returns indices as strings (`["1", "13"]`) or as labels (`["Sentence 13"]`). The parser now accepts both, and the BillSum rows were re-scored from the saved raw responses (`extsum rescore`); only `gemma4_e2b` changed (mean F1 0.196 → 0.288). The CNN/DailyMail run already used the fixed parser.

Aggregates are in `extsum/reports/billsum/` (`summary.csv`, `tables/`, `figures/`).

### No-model baselines

Computed on the same 100 documents, against the same silver labels.

| Baseline | BillSum F1 (K = 6) | CNN/DM F1 (K = 3) |
| --- | :-: | :-: |
| First K sentences | 0.163 | 0.360 |
| Every sentence | 0.244 | 0.217 |

On CNN/DM no model's capped mean beats "first 3" by more than 0.03 (Qwen3.5-4B 0.390; Qwen3.5-2B 0.378, which returns exactly sentences 1–3 in 58% of its capped answers). On BillSum the same habit scores 0.163, so the dataset separates selecting from copying positions.

### Mean over all 52 variants — BILLSUM

| Model | P | R | F1 | ROUGE-L | Oracle gap |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.161 | 0.574 | 0.211 | 0.210 | 0.193 |
| Qwen3.5-4B | 0.277 | 0.537 | 0.312 | 0.261 | 0.143 |
| Qwen3.5-9B | 0.274 | 0.559 | 0.316 | 0.261 | 0.143 |
| Gemma4-E2B | 0.308 | 0.285 | 0.288 | 0.256 | 0.148 |
| Gemma4-E4B | 0.379 | 0.432 | 0.387 | 0.299 | 0.104 |
| Gemma4-12B | 0.357 | 0.474 | 0.389 | 0.292 | 0.111 |

### Capped vs uncapped — BILLSUM

| Model | F1 capped | F1 uncapped | ROUGE-L capped | ROUGE-L uncapped | Sentences picked, uncapped |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.179 | 0.242 | 0.253 | 0.168 | 49.2 |
| Qwen3.5-4B | 0.329 | 0.295 | 0.304 | 0.217 | 31.8 |
| Qwen3.5-9B | 0.333 | 0.300 | 0.302 | 0.220 | 32.5 |
| Gemma4-E2B | 0.294 | 0.282 | 0.259 | 0.253 | 5.8 |
| Gemma4-E4B | 0.391 | 0.383 | 0.306 | 0.292 | 9.8 |
| Gemma4-12B | 0.387 | 0.391 | 0.305 | 0.279 | 11.9 |

Without a cap, Qwen selects most of the bill (32–49 of ~51 sentences); Gemma limits itself to 6–12. The cap is what makes Qwen's output a summary.

### Zero-shot vs one-shot — BILLSUM (mean F1)

| Model | zero·capped | one·capped | zero·uncapped | one·uncapped |
| --- | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.179 | 0.179 | 0.241 | 0.243 |
| Qwen3.5-4B | 0.324 | 0.334 | 0.270 | 0.316 |
| Qwen3.5-9B | 0.335 | 0.330 | 0.276 | 0.320 |
| Gemma4-E2B | 0.271 | 0.313 | 0.249 | 0.310 |
| Gemma4-E4B | 0.400 | 0.383 | 0.388 | 0.379 |
| Gemma4-12B | 0.393 | 0.381 | 0.396 | 0.388 |

The exemplar helps where the model over-selects or is weak (Qwen 4B/9B uncapped, Gemma E2B); it slightly hurts Gemma E4B/12B.

### Best-F1 variant per model — BILLSUM

| Model | Best F1 | 95% CI | Technique | shot | cap |
| --- | :-: | :-: | --- | :-: | :-: |
| Qwen3.5-2B | 0.284 | 0.234–0.332 | tool_no_roleplay | zero_shot | capped |
| Qwen3.5-4B | 0.414 | 0.375–0.453 | tool_no_roleplay | zero_shot | capped |
| Qwen3.5-9B | 0.473 | 0.437–0.510 | tool_no_roleplay | zero_shot | capped |
| Gemma4-E2B | 0.484 | 0.444–0.525 | tool_no_roleplay | zero_shot | capped |
| Gemma4-E4B | 0.495 | 0.457–0.532 | tool_augmented | zero_shot | capped |
| Gemma4-12B | 0.491 | 0.455–0.528 | tool_no_roleplay | zero_shot | capped |

CIs are bootstrap intervals over the 100 bills (`extsum/reports/billsum/tables/bootstrap_confidence_intervals.csv`).

### TF-IDF centrality is what helps — BILLSUM (zero-shot, capped, F1)

| Model | vanilla | tool_no_meta (no scores) | tool_augmented | tool_no_roleplay |
| --- | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.165 | 0.165 | 0.223 | 0.284 |
| Qwen3.5-4B | 0.323 | 0.322 | 0.410 | 0.414 |
| Qwen3.5-9B | 0.296 | 0.328 | 0.469 | 0.473 |
| Gemma4-E2B | 0.233 | 0.253 | 0.453 | 0.484 |
| Gemma4-E4B | 0.380 | 0.396 | 0.495 | 0.493 |
| Gemma4-12B | 0.349 | 0.375 | 0.479 | 0.491 |

Showing a TF-IDF centrality score next to each sentence adds 0.06–0.25 F1 over `vanilla` for every model (paired Wilcoxon, Benjamini–Hochberg adjusted p ≤ 0.0025 in all 12 comparisons). Removing the scores (`tool_no_meta`) returns the technique to `vanilla` level; removing the expert role-play changes nothing. No reasoning technique comes close: the next-best technique averages 0.327.

### Technique ranking — BILLSUM (mean F1 over 6 models)

| Rank | Technique | F1 | P | R |
| :-: | --- | :-: | :-: | :-: |
| 1 | tool_no_roleplay | 0.440 | 0.451 | 0.432 |
| 2 | tool_augmented | 0.421 | 0.428 | 0.417 |
| 3 | least_to_most | 0.327 | 0.309 | 0.451 |
| 4 | negative_aware | 0.323 | 0.316 | 0.424 |
| 5 | explanation_based | 0.321 | 0.298 | 0.465 |
| 6 | salience_inference | 0.321 | 0.294 | 0.484 |
| … | vanilla | 0.309 | 0.294 | 0.438 |
| last | self_ask | 0.290 | 0.267 | 0.455 |

The two `tool_*` techniques run in one variant only (zero-shot capped); the others are means over up to four variants, so ranks 1–2 are not strictly like-for-like with the rest. Ranks 3–20 span only 0.04 F1.

### Reasoning traces — BILLSUM (one-shot F1; chain_of_thought, salience_inference, self_ask)

| Model | answer-only | trace | Δ |
| --- | :-: | :-: | :-: |
| Qwen3.5-2B | 0.213 | 0.209 | −0.004 |
| Qwen3.5-4B | 0.325 | 0.312 | −0.013 |
| Qwen3.5-9B | 0.331 | 0.300 | −0.032 |
| Gemma4-E2B | 0.301 | 0.312 | +0.011 |
| Gemma4-E4B | 0.381 | 0.373 | −0.008 |
| Gemma4-12B | 0.389 | 0.375 | −0.014 |

Traces do not help on BillSum; five of six models are slightly worse with them.

### Position of selections — BILLSUM (capped variants)

| | Share of picks in the first fifth of the bill | Answer is exactly sentences 1–6 |
| --- | :-: | :-: |
| Silver labels | 24% | — |
| Qwen3.5-2B | 94% | 74% |
| Qwen3.5-4B | 42% | 3% |
| Qwen3.5-9B | 44% | 3% |
| Gemma4-E2B | 40% | 0% |
| Gemma4-E4B | 34% | 0% |
| Gemma4-12B | 34% | 0% |

Every model over-picks the opening relative to the silver labels; Qwen3.5-2B does almost nothing else. Per-model curves: `extsum/reports/billsum/figures/position_bias_billsum_<model>.pdf`.

### Large tier — BILLSUM (5 best techniques, OpenRouter)

`qwen35_27b` and `gemma4_31b` are endpoint-only (OpenRouter, reasoning disabled). They ran the **5 best BillSum techniques by mean F1 over the six local models** (ranking above, `_trace` collapsed to its base technique): `tool_no_roleplay`, `tool_augmented`, `least_to_most`, `negative_aware`, `explanation_based`. With the cells of the full grid that is **12 variants** per model (`extsum/configs/grids/billsum_large.yaml`), on the same 100 bills. This is a different five from Part A's (`self_ask`, `salience_inference`, `self_critique`, `contrastive_joint`, `negative_aware`): only `negative_aware` is shared, and Part A's five would exclude both TF-IDF variants.

Both models completed all 1,200 calls. `gemma4_31b` has no empty answers; `qwen35_27b` has 6 (`least_to_most` one-shot), where it wrote out its reasoning and hit the 512-token limit before the JSON. Cost: $0.52 (`gemma4_31b`) and $1.10 (`qwen35_27b`).

**Like-for-like scaling** — mean over the same 12 variants, all 8 models:

| Model | P | R | F1 | ROUGE-L | Oracle gap | Sentences (uncapped) |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.174 | 0.514 | 0.215 | 0.219 | 0.184 | 48.8 |
| Qwen3.5-4B | 0.316 | 0.468 | 0.337 | 0.278 | 0.125 | 24.0 |
| Qwen3.5-9B | 0.325 | 0.493 | 0.351 | 0.278 | 0.125 | 23.8 |
| **Qwen3.5-27B** | 0.350 | 0.471 | 0.376 | 0.294 | 0.109 | 16.1 |
| Gemma4-E2B | 0.364 | 0.326 | 0.337 | 0.283 | 0.121 | 5.7 |
| Gemma4-E4B | 0.407 | 0.426 | 0.403 | 0.298 | 0.105 | 8.6 |
| Gemma4-12B | 0.385 | 0.453 | 0.405 | 0.295 | 0.109 | 10.0 |
| **Gemma4-31B** | 0.401 | 0.441 | 0.412 | 0.310 | 0.093 | 8.5 |

Split by technique type (mean F1):

| Model | 3 reasoning techniques (10 variants) | 2 TF-IDF variants |
| --- | :-: | :-: |
| Qwen3.5-9B | 0.327 | 0.471 |
| Qwen3.5-27B | 0.358 | 0.465 |
| Gemma4-E4B | 0.385 | 0.494 |
| Gemma4-12B | 0.389 | 0.485 |
| Gemma4-31B | 0.404 | 0.449 |

- **Scaling is monotonic but flat at the top.** Qwen 9B→27B is +0.025 F1 (paired Wilcoxon over the 100 bills, p = 0.004); Gemma 12B→31B is +0.007 (p = 0.36, not significant) and E4B→31B +0.008 (p = 0.75).
- **Gemma4 still leads Qwen3.5 at the large tier:** +0.036 F1 (p < 0.001).
- **The larger models gain on the reasoning prompts and not on the TF-IDF ones.** `gemma4_31b` is 0.036 *worse* than `gemma4_12b` on the two TF-IDF variants (p = 0.005) while slightly better on the reasoning ones (+0.015, p = 0.07). The TF-IDF variants remain the best two for every model, but the gap to plain reasoning shrinks with size (Gemma: 0.109 at E4B, 0.045 at 31B).
- **Qwen's over-selection eases with size** (48.8 → 24.0 → 23.8 → 16.1 sentences uncapped) but 27B still picks about twice what Gemma does.

### Known limitations — BILLSUM

- **Gemma4-E2B still fails to answer in 13% of cases**: it copies out sentence text instead of sentence numbers. Those are scored as empty selections.
- **Capped answers longer than K are cut to their first K indices**, which favours early sentences. It rarely triggers except for Qwen3.5-2B (22% of its capped answers).
- **Silver summaries are about twice the reference length** (≈320 vs ≈155–175 words): the local scorer favours long sentences, so K = 6 sentences is longer in words than the compression ratio implies.
- **100 bills.** Differences below roughly 0.04 F1 are within the bootstrap intervals.
