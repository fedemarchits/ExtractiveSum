# extsum

Extractive summarization with LLM prompting, across model sizes and prompting
techniques. One engine runs every dataset:

| Experiment | Task | Labels | Prompt set |
| --- | --- | --- | --- |
| `aclsum` | one selection per aspect (challenge / approach / outcome) | human gold | `aspect` |
| `cnndm` | one selection per document | silver (`local_score`, K=3) | `document` |
| `billsum` | one selection per bill | silver (`local_score`, K=6) | `bill` |

Results so far are summarised in [`../summary_result.md`](../summary_result.md).

## How it is organised

Three things define a run, and each is one small file:

- **Experiment** (`configs/experiments/<name>.yaml`): the *task* (what the model
  selects), the *prompt set* (how it is asked) and the data.
- **Grid** (`configs/grids/<name>.yaml`): which technique × zero/one-shot ×
  capped/uncapped variants run.
- **Model** (`configs/models.yaml`): each model once. A *profile* (`vast`,
  `cluster`) decides the backend for local models; the large tier is
  endpoint-only.

```
src/extsum/
  cli.py            the `extsum` command
  config.py         experiment configs          paths.py   project-root resolution
  runner.py         model x variant x document -> JSONL (resume-safe)
  tasks/            aspect.py (ACLSum), document.py (silver datasets)
  prompts/          base types, registry, TF-IDF centrality, rationale cache
    sets/aspect|document|bill/   the wording: shared.py + techniques/
  backends/         hf.py, vllm.py, endpoint.py, chosen by profile
  data/             aclsum.py, silver_io.py, segmentation.py (news + bills)
  silver/           local_score labelling, K from the compression ratio
  rationales/       one-shot reasoning traces from a reference model
  postprocess.py    answer parsing      metrics.py   P/R/F1, ROUGE, oracle
  report.py  rescore.py  preflight.py
  analysis/         bootstrap CIs, significance, ablations, position bias
  heuristics/       the study that selected local_score (greedy/beam/MMR/...)
configs/            models.yaml, experiments/, grids/, heuristics/
data/rationales/    cached traces (tracked)     data/silver/   rebuilt on demand
results/<experiment>/<model>/<variant>.jsonl    raw answers (not in git)
reports/<experiment>/summary.csv, tables/, figures/
deploy/vast/        setup.sh, sweep.sh          deploy/cluster/   docker + SLURM
tests/              no GPU or network needed
```

## Install

```bash
cd extsum
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,endpoint]"      # laptop: analysis, tests, OpenRouter runs
cp .env.example .env                  # then fill in the tokens; never committed
pytest -q
```

Extras: `vllm` (rented GPU), `hf` (Transformers backend), `moverscore`,
`heuristics`.

## Everyday commands

```bash
extsum list                          # experiments
extsum list models --profile vast    # where each model would run
extsum list variants -e billsum -g billsum_large

extsum preflight -e billsum -m qwen35_9b     # is everything in place?
extsum run -e billsum -m qwen35_9b           # full grid, resume-safe
extsum run -e billsum -m gemma4_31b -g billsum_large   # large tier, 12 variants
extsum run -e billsum -m qwen35_9b --limit 5 # smoke test on 5 documents

extsum report -e billsum cnndm       # -> reports/<experiment>/summary.csv
extsum analyze -e billsum            # -> reports/billsum/tables, figures
extsum figures                       # -> reports/figures (charts in the root README)
extsum rescore -e billsum --dry-run  # effect of a parser change, no GPU
```

Data preparation, each needed once per dataset:

```bash
extsum compute-k -e billsum          # K from the compression ratio (dev split)
extsum silver -e billsum             # eval + exemplar silver files
extsum rationales -e billsum         # reasoning traces (reference model, endpoint)
```

## Running on a rented GPU (vast.ai)

Rent one A100 40GB or larger with the PyTorch template and about 100 GB disk.

```bash
# on the box
git clone <repo-url> && cd <repo>/extsum
cp .env.example .env && nano .env            # HF_TOKEN
bash deploy/vast/setup.sh                    # installs + prints the GPU

extsum run -e billsum -m qwen35_4b --limit 5 # 1-minute smoke test
mkdir -p logs
nohup env EXPERIMENTS="billsum" bash deploy/vast/sweep.sh > logs/sweep.out 2>&1 &
tail -f logs/sweep.out
```

`sweep.sh` builds the silver files, runs every model in `MODELS` (default: the
six local ones) on every experiment in `EXPERIMENTS`, purges each model's
weights before the next, and writes one log per model. It skips finished
variants, so re-running it after an interruption continues.

```bash
# on your laptop, before destroying the instance
bash deploy/fetch_results.sh <ssh-port> root@<host> /workspace/<repo>/extsum
```

This copies `results/`, `reports/` and `logs/` and re-checks checksums; destroy
the instance only when it prints `0 files differ`.

The large tier needs no GPU: with `OPENAI_BASE_URL` and `OPENAI_API_KEY` in
`.env`, run it from the laptop.

## Adding things

- **A dataset**: add its loader to `silver/labelling.py::load_abstractive_dataset`
  and a `configs/experiments/<name>.yaml`; then `compute-k`, `silver`,
  `rationales`.
- **A technique**: add a file under each prompt set's `techniques/` that
  registers a `Technique`. Absent techniques default to their full grid, so
  partial grids must set the new name to `[]`.
- **A model**: one entry in `configs/models.yaml`.
- **A machine**: one entry under `profiles:`.

## Things to know before trusting a number

- **Prompt wording is frozen.** `tests/test_prompt_golden.py` holds hashes of
  every prompt recorded from the original code. A changed prompt is a new
  experiment.
- **Segmentation is part of the dataset.** Silver labels are sentence indices;
  editing `data/segmentation.py` invalidates the silver files and every result
  computed on them.
- **Endpoint runs are not exactly reproducible.** At temperature 0 with a fixed
  seed, two identical OpenRouter runs of `gemma4_31b` agreed on 14 of 24
  selections (mean Jaccard 0.88). Local vLLM runs are deterministic.
- **Answer parsing.** Indices given as `"13"` or `"Sentence 13"` are accepted.
  BillSum and CNN/DM results were re-scored with this parser; the stored ACLSum
  results were not (see `extsum rescore -e aclsum --dry-run`).
- **ACLSum raw answers are incomplete.** `gemma4_e2b`, `qwen35_2b` and
  `qwen35_4b` have summary rows only; `extsum report` carries them over.

