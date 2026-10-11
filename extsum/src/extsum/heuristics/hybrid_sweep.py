"""Sweep new hybrid heuristics (RRF + stage-gate) over K=[1..32] and evaluate
under ROUGE-1/2/L and MoverScore, for both singular and all_together modes.

Extraction is fast (CPU, ROUGE-1 scorer). MoverScore eval is batched on GPU.

Output JSONs live under outputs_xmetric/hybrid/ with the same shape as the
existing fast-sweep aggregates: {K: {heuristic_name: score}}.

  outputs_xmetric/hybrid/avg_singular_eval_<m>.json
  outputs_xmetric/hybrid/avg_all_tog_eval_<m>.json
  outputs_xmetric/hybrid/picks_K_<K>.pkl     # raw picks per K (for downstream)
"""

import argparse
import json
import os
import time

from joblib import Parallel, delayed

from .data import load_aclsum, build_summary_lists
from .heuristics_fast import (
    rrf_extractive_summary,
    stage_gate_extractive_summary,
)
from .io_utils import save_pickle
from .metrics import rouge_score
from .metrics_semantic import mover_score_batch
from .runner import SINGULAR_SUMM_TYPES


HYBRIDS = {
    "hybrid_rrf": rrf_extractive_summary,
    "hybrid_stage": stage_gate_extractive_summary,
}


def _extract_one_doc(sentences, ref, max_sent, scorer):
    """Run both hybrids for a single doc at given K. Returns {h: [picks]}."""
    return {
        name: fn(sentences, ref, max_sent=max_sent, scorer=scorer)
        for name, fn in HYBRIDS.items()
    }


def extract_all(data, ext, K, scorer, n_jobs=-1):
    """Extract picks for all docs at given K. Returns:
    res_sing[summ][heuristic] = [picks per doc]
    res_all[heuristic]        = [picks per doc]
    """
    list_source_sentences = data["list_source_sentences"]
    list_summaries = data["list_summaries"]

    res_sing = {summ: {h: [] for h in HYBRIDS} for summ in SINGULAR_SUMM_TYPES}
    res_all = {h: [] for h in HYBRIDS}

    for summ in SINGULAR_SUMM_TYPES:
        refs = list_summaries[summ]
        out = Parallel(n_jobs=n_jobs)(
            delayed(_extract_one_doc)(s, r, K, scorer)
            for s, r in zip(list_source_sentences, refs)
        )
        for d in out:
            for h, picks in d.items():
                res_sing[summ][h].append(picks)

    refs_all = list_summaries["all_together"]
    out = Parallel(n_jobs=n_jobs)(
        delayed(_extract_one_doc)(s, r, K, scorer)
        for s, r in zip(list_source_sentences, refs_all)
    )
    for d in out:
        for h, picks in d.items():
            res_all[h].append(picks)

    return res_sing, res_all


def eval_singular(res_sing, dataset_extractive, eval_metric, use_mover=False):
    """Returns {heuristic: avg_score} averaged across summ_types and docs."""
    out = {h: 0.0 for h in HYBRIDS}
    for h in HYBRIDS:
        per_summ_sum = 0.0
        for summ in SINGULAR_SUMM_TYPES:
            picks = res_sing[summ][h]
            n = len(picks)
            cands = [" ".join(picks[i]) for i in range(n)]
            refs = [
                " ".join(dataset_extractive[i][summ + "_sentences"])
                for i in range(n)
            ]
            if use_mover:
                scores = mover_score_batch(cands, refs)
                per_summ_sum += sum(scores) / len(scores)
            else:
                s = sum(rouge_score(c, r, eval_metric) for c, r in zip(cands, refs))
                per_summ_sum += s / n
        out[h] = round(per_summ_sum / len(SINGULAR_SUMM_TYPES), 4)
    return out


def eval_all_tog(res_all, list_all_extractive, eval_metric, use_mover=False):
    out = {h: 0.0 for h in HYBRIDS}
    for h in HYBRIDS:
        picks = res_all[h]
        n = len(picks)
        cands = [" ".join(picks[i]) for i in range(n)]
        refs = list_all_extractive[:n]
        if use_mover:
            scores = mover_score_batch(cands, refs)
            out[h] = round(sum(scores) / len(scores), 4)
        else:
            s = sum(rouge_score(c, r, eval_metric) for c, r in zip(cands, refs))
            out[h] = round(s / n, 4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="reports/heuristics/hybrid")
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=32)
    p.add_argument("--scorer", default="rouge1",
                   help="Inner scorer used during extraction (rouge1/2/L).")
    p.add_argument("--eval-metrics", nargs="+",
                   default=["rouge1", "rouge2", "rougeL"])
    p.add_argument("--with-moverscore", action="store_true", default=True,
                   help="Also evaluate under MoverScore (slower).")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[hybrid] out_dir={args.out_dir}", flush=True)
    print(f"[hybrid] heuristics={list(HYBRIDS)}", flush=True)
    print(f"[hybrid] scorer={args.scorer}  K=[{args.k_min}..{args.k_max}]",
          flush=True)

    print("[load] ACLSum", flush=True)
    abst, ext = load_aclsum()
    data = build_summary_lists(abst, ext)
    list_all = data["list_all_summaries_extractive"]

    eval_metrics_full = list(args.eval_metrics)
    if args.with_moverscore:
        eval_metrics_full = eval_metrics_full + ["moverscore"]

    agg_sing = {ev: {} for ev in eval_metrics_full}
    agg_all = {ev: {} for ev in eval_metrics_full}

    for K in range(args.k_min, args.k_max + 1):
        t0 = time.perf_counter()
        res_sing, res_all = extract_all(
            data, ext, K, args.scorer, n_jobs=args.n_jobs
        )
        t_extract = time.perf_counter() - t0

        # Save raw picks for downstream
        save_pickle(
            {"singular": res_sing, "all_tog": res_all},
            args.out_dir, f"picks_K_{K:02d}.pkl",
        )

        t_eval0 = time.perf_counter()
        for ev in eval_metrics_full:
            use_mover = (ev == "moverscore")
            agg_sing[ev][K] = eval_singular(res_sing, ext, ev, use_mover=use_mover)
            agg_all[ev][K] = eval_all_tog(res_all, list_all, ev, use_mover=use_mover)
        t_eval = time.perf_counter() - t_eval0

        sing_summary = {h: agg_sing["rouge1"][K][h] for h in HYBRIDS}
        all_summary = {h: agg_all["rouge1"][K][h] for h in HYBRIDS}
        print(f"[hybrid] K={K:02d} extract={t_extract:.1f}s eval={t_eval:.1f}s  "
              f"r1_sing={sing_summary}  r1_all={all_summary}",
              flush=True)

        # Checkpoint JSONs every K so progress is durable
        for ev in eval_metrics_full:
            with open(
                os.path.join(args.out_dir, f"avg_singular_eval_{ev}.json"), "w"
            ) as f:
                json.dump({str(k): v for k, v in agg_sing[ev].items()}, f, indent=2)
            with open(
                os.path.join(args.out_dir, f"avg_all_tog_eval_{ev}.json"), "w"
            ) as f:
                json.dump({str(k): v for k, v in agg_all[ev].items()}, f, indent=2)

    print("[hybrid] done", flush=True)


if __name__ == "__main__":
    main()
