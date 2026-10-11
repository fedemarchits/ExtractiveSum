"""Score fast-sweep picks under MoverScore for every K.

Reads outputs_xmetric/<m>/fast_sweep/<m>/fast_K_<K>.pkl, scores each
(heuristic, summ_type) under MoverScore, and saves two JSONs per extract
metric — same shape as the existing `avg_<mode>_eval_<rouge>.json` files so the
report code can fold MoverScore eval into the K-sweep comparison.

Outputs (per extract metric `<m>`):
  outputs_xmetric/<m>/fast_sweep/<m>/avg_singular_eval_moverscore.json
  outputs_xmetric/<m>/fast_sweep/<m>/avg_all_tog_eval_moverscore.json

Each JSON: {K: {heuristic: score}} matching the existing ROUGE eval format.
"""

import argparse
import json
import os
import time

# Ensure repo root on path when run as a script from anywhere
from .data import load_aclsum, build_summary_lists
from .io_utils import load_pickle
from .metrics_semantic import mover_score_batch
from .runner import SINGULAR_SUMM_TYPES


def _eval_singular_batched(res, dataset_extractive):
    """Per-summ_type, per-heuristic MoverScore averaged over docs.

    Returns {summ_type: {heuristic: avg_score}}.
    """
    heurs = list(res[SINGULAR_SUMM_TYPES[0]].keys())
    out = {summ: {} for summ in SINGULAR_SUMM_TYPES}
    for summ in SINGULAR_SUMM_TYPES:
        for h in heurs:
            picks = res[summ][h]["result"]
            n = len(picks)
            cands = [" ".join(picks[i]) for i in range(n)]
            refs = [
                " ".join(dataset_extractive[i][summ + "_sentences"])
                for i in range(n)
            ]
            scores = mover_score_batch(cands, refs)
            out[summ][h] = sum(scores) / len(scores)
    return out


def _eval_all_tog_batched(res, list_all_summaries_extractive):
    """Per-heuristic MoverScore averaged over docs for all_together mode."""
    heurs = list(res["all_together"].keys())
    out = {}
    for h in heurs:
        picks = res["all_together"][h]["result"]
        n = len(picks)
        cands = [" ".join(picks[i]) for i in range(n)]
        refs = list_all_summaries_extractive[:n]
        scores = mover_score_batch(cands, refs)
        out[h] = sum(scores) / len(scores)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="path to outputs_xmetric directory ")
    p.add_argument("--metrics", nargs="+",
                   default=["rouge1", "rouge2", "rougeL"])
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=32)
    args = p.parse_args()

    print("[load] ACLSum", flush=True)
    abst, ext = load_aclsum()
    data = build_summary_lists(abst, ext)
    list_all = data["list_all_summaries_extractive"]

    for em in args.metrics:
        sub = os.path.join(args.base, em, "fast_sweep", em)
        if not os.path.isdir(sub):
            print(f"[skip] {sub} not found", flush=True)
            continue

        avg_sing_by_K = {}    # singular averaged across summ_types
        avg_per_summ_by_K = {}  # full per-summ breakdown (bonus)
        avg_all_by_K = {}

        for K in range(args.k_min, args.k_max + 1):
            pkl = os.path.join(sub, f"fast_K_{K:02d}.pkl")
            if not os.path.exists(pkl):
                print(f"[skip] {pkl} missing", flush=True)
                continue
            t0 = time.perf_counter()
            res = load_pickle(pkl)

            per_summ = _eval_singular_batched(res, ext)
            heurs = list(next(iter(per_summ.values())).keys())
            avg_sing_by_K[K] = {
                h: sum(per_summ[s][h] for s in per_summ) / len(per_summ)
                for h in heurs
            }
            avg_per_summ_by_K[K] = per_summ

            avg_all_by_K[K] = _eval_all_tog_batched(res, list_all)

            print(f"[mover-eval:{em}] K={K:02d} done in "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)

        out_sing = {str(k): {h: round(v, 4) for h, v in avg_sing_by_K[k].items()}
                    for k in avg_sing_by_K}
        out_per_summ = {
            str(k): {s: {h: round(v, 4) for h, v in avg_per_summ_by_K[k][s].items()}
                     for s in avg_per_summ_by_K[k]}
            for k in avg_per_summ_by_K
        }
        out_all = {str(k): {h: round(v, 4) for h, v in avg_all_by_K[k].items()}
                   for k in avg_all_by_K}

        with open(os.path.join(sub, "avg_singular_eval_moverscore.json"), "w") as f:
            json.dump(out_sing, f, indent=2)
        with open(os.path.join(sub, "avg_singular_per_summ_eval_moverscore.json"), "w") as f:
            json.dump(out_per_summ, f, indent=2)
        with open(os.path.join(sub, "avg_all_tog_eval_moverscore.json"), "w") as f:
            json.dump(out_all, f, indent=2)
        print(f"[mover-eval:{em}] wrote 3 JSONs in {sub}", flush=True)


if __name__ == "__main__":
    main()
