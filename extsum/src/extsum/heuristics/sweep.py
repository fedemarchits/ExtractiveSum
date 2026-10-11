"""Run the full Track B evaluation suite end-to-end on the cluster.

Sections (each gated by --skip-*):
  1. Fast sweep (notebook reproduction): K=1..32, 4 heuristics, real HF rouge1.
  2. HF sweep:                            K in [4..8] singular, [19..23] all_tog.
  3. R2 single point:                     R2 evaluation at chosen K.
  4. PRF: precision / recall / F1 at chosen best K (rouge1 + rouge2 + rougeL).
  5. MMR: heuristic with embedding similarity, K range + lambda sweep.
  6. MoverScore: 4 base heuristics scored with MoverScore at chosen K only.

All outputs in --out-dir under per-section subfolders:
  fast_sweep/  hf_sweep/  r2/  prf/  mmr/  moverscore/
"""

import argparse
import copy
import os
import time

from .data import build_summary_lists, load_aclsum
from .evaluate_results import (
    avg_score_across_all_tog,
    avg_score_across_heuristics_all_tog,
    avg_score_across_heuristics_singular,
    avg_score_across_summaries,
    avg_score_generic,
    avg_score_generic_all_tog,
    avg_score_per_singular,
    avg_score_slow_all_tog,
    avg_score_slow_single,
    prf_at_K_all_tog,
    prf_at_K_singular,
    prf_sweep_all_tog,
    prf_sweep_singular,
)
from .heuristics_fast import (
    beam_extractive_summary,
    global_extractive_summary,
    greedy_extractive_summary,
    local_extractive_summary,
)
from .heuristics_hf import (
    beam_extractive_summary_hf,
    global_extractive_summary_hf,
    greedy_extractive_summary_hf,
    local_extractive_summary_hf,
)
from .heuristics_mmr import mmr_extractive_summary
from .io_utils import save_json, save_pickle
from .plotting import (
    plot_avg_score,
    plot_histogram_with_values,
    plot_prf_grouped,
    plot_prf_per_summary,
    plot_total_time,
)
from .runner import (
    HEURISTICS,
    SINGULAR_SUMM_TYPES,
    SUMM_TYPES,
    apply_heuristic_to_dataset_hf,
    apply_heuristic_to_dataset_parallel,
    apply_heuristic_to_dataset_sequential,
    make_all_together_template,
    make_results_template,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="reports/heuristics")
    p.add_argument("--max-sent-start", type=int, default=1)
    p.add_argument("--max-sent-end", type=int, default=33,
                   help="exclusive end, like notebook range(1, 33)")
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--max-sent-singular", type=int, default=6)
    p.add_argument("--max-sent-all", type=int, default=21)
    p.add_argument("--range-single-summaries", type=int, nargs="+",
                   default=[4, 5, 6, 7, 8])
    p.add_argument("--range-all-tog-summ", type=int, nargs="+",
                   default=[19, 20, 21, 22, 23])
    p.add_argument("--extract-metrics", nargs="+",
                   default=["rouge1", "rouge2", "rougeL"],
                   help="ROUGE variants to use as the extraction target. "
                        "Pass a single one (e.g. 'rouge2') to parallelize across SLURM jobs.")
    p.add_argument("--eval-metrics", nargs="+",
                   default=["rouge1", "rouge2", "rougeL"],
                   help="ROUGE variants used for cross-evaluation of extracted picks.")
    p.add_argument("--extract-roots", nargs="*", default=[],
                   help="Stage-2 only: directories from prior stage-1 runs. Used by MoverScore "
                        "cross-eval and final aggregation when this job didn't extract itself.")
    p.add_argument("--mmr-lambdas", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--mmr-embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--mmr-n-jobs", type=int, default=1,
                   help="Workers for MMR. Each worker loads SBERT on GPU; >1 risks CUDA OOM.")
    p.add_argument("--moverscore-on-summaries", nargs="+",
                   default=["challenge", "approach", "outcome", "all_together"])
    p.add_argument("--skip-fast-sweep", action="store_true")
    p.add_argument("--skip-hf-sweep", action="store_true")
    p.add_argument("--skip-prf", action="store_true")
    p.add_argument("--skip-mmr", action="store_true")
    p.add_argument("--skip-moverscore", action="store_true")
    return p.parse_args()


# ============================ Section 1: Fast sweep =========================

def run_fast_sweep_one_metric(args, data, out_dir, extract_metric):
    list_source_sentences = data["list_source_sentences"]
    list_summaries = data["list_summaries"]

    template = make_results_template()
    sweep_range = range(args.max_sent_start, args.max_sent_end)
    all_results = []

    print(f"[fast-sweep:{extract_metric}] max_sent in {list(sweep_range)}", flush=True)
    for sum_sent in sweep_range:
        t0 = time.perf_counter()
        res = copy.deepcopy(template)
        for summary in SUMM_TYPES:
            res[summary]["greedy_search"]["result"], res[summary]["greedy_search"]["time"] = \
                apply_heuristic_to_dataset_parallel(
                    greedy_extractive_summary, list_source_sentences,
                    list_summaries[summary], n_jobs=args.n_jobs,
                    max_sent=sum_sent, scorer=extract_metric,
                )
            res[summary]["beam_search"]["result"], res[summary]["beam_search"]["time"] = \
                apply_heuristic_to_dataset_parallel(
                    beam_extractive_summary, list_source_sentences,
                    list_summaries[summary], n_jobs=args.n_jobs,
                    max_sent=sum_sent, beam_size=args.beam_size, scorer=extract_metric,
                )
            res[summary]["local_score"]["result"], res[summary]["local_score"]["time"] = \
                apply_heuristic_to_dataset_parallel(
                    local_extractive_summary, list_source_sentences,
                    list_summaries[summary], n_jobs=args.n_jobs,
                    max_sent=sum_sent, scorer=extract_metric,
                )
            res[summary]["global_score"]["result"], res[summary]["global_score"]["time"] = \
                apply_heuristic_to_dataset_parallel(
                    global_extractive_summary, list_source_sentences,
                    list_summaries[summary], n_jobs=args.n_jobs,
                    max_sent=sum_sent, beam_size=args.beam_size, scorer=extract_metric,
                )
        all_results.append(res)
        save_pickle(res, out_dir, f"fast_K_{sum_sent:02d}.pkl")
        save_pickle(all_results, out_dir, "fast_all_results.pkl")
        print(f"[fast-sweep:{extract_metric}] sum_sent={sum_sent} done in "
              f"{time.perf_counter() - t0:.1f}s (checkpoint saved)", flush=True)

    return all_results


def run_fast_sweep(args, data, out_dir):
    """Run fast-sweep once per --extract-metrics. Returns {metric: all_results}."""
    results_per_metric = {}
    for em in args.extract_metrics:
        sub = os.path.join(out_dir, em)
        os.makedirs(sub, exist_ok=True)
        results_per_metric[em] = run_fast_sweep_one_metric(args, data, sub, em)
    return results_per_metric


def run_aggregations_one_pair(args, all_results, data, dataset_extractive,
                              out_dir, extract_metric, eval_metric):
    """Aggregate avg-score curves + plots for one (extract, eval) cell."""
    n_runs = args.max_sent_end - args.max_sent_start
    K_values = list(range(args.max_sent_start, args.max_sent_end))

    avg_singular = avg_score_across_summaries(
        all_results, dataset_extractive, n_runs=n_runs, rouge_type=eval_metric
    )
    save_json(avg_singular, out_dir, f"avg_singular_eval_{eval_metric}.json")
    plot_avg_score(
        avg_singular,
        f"Avg Singular  (extract={extract_metric}, eval={eval_metric})",
        out_dir, f"avg_singular_eval_{eval_metric}.png",
    )

    avg_all_tog = avg_score_across_all_tog(
        all_results, data["list_all_summaries_extractive"],
        n_runs=n_runs, rouge_type=eval_metric,
    )
    save_json(avg_all_tog, out_dir, f"avg_all_tog_eval_{eval_metric}.json")
    plot_avg_score(
        avg_all_tog,
        f"Avg All-Together  (extract={extract_metric}, eval={eval_metric})",
        out_dir, f"avg_all_tog_eval_{eval_metric}.png",
    )

    for summ in SINGULAR_SUMM_TYPES:
        d = avg_score_per_singular(
            all_results, dataset_extractive, summ, n_runs=n_runs, rouge_type=eval_metric
        )
        save_json(d, out_dir, f"avg_{summ}_eval_{eval_metric}.json")

    plot_histogram_with_values(
        avg_singular[args.max_sent_singular],
        out_dir, f"hist_singular_eval_{eval_metric}.png",
        title=f"Singular  K={args.max_sent_singular}  "
              f"(extract={extract_metric}, eval={eval_metric})",
    )
    plot_histogram_with_values(
        avg_all_tog[args.max_sent_all],
        out_dir, f"hist_all_tog_eval_{eval_metric}.png",
        title=f"All-Together  K={args.max_sent_all}  "
              f"(extract={extract_metric}, eval={eval_metric})",
    )

    prf_sing = prf_sweep_singular(all_results, dataset_extractive, K_values, eval_metric)
    save_json(prf_sing, out_dir, f"prf_sweep_singular_eval_{eval_metric}.json")
    prf_all = prf_sweep_all_tog(
        all_results, data["list_all_summaries_extractive"], K_values, eval_metric
    )
    save_json(prf_all, out_dir, f"prf_sweep_all_tog_eval_{eval_metric}.json")

    best_res = all_results[args.max_sent_singular - 1]
    best_res_all = all_results[args.max_sent_all - 1]

    avg_per_summary = avg_score_across_heuristics_singular(
        best_res, dataset_extractive, rouge_type=eval_metric
    )
    avg_per_summary["all_together"] = avg_score_across_heuristics_all_tog(
        best_res_all, data["list_all_summaries_extractive"], rouge_type=eval_metric
    )
    save_json(avg_per_summary, out_dir, f"avg_per_summary_eval_{eval_metric}.json")
    plot_histogram_with_values(
        avg_per_summary, out_dir, f"hist_per_summary_eval_{eval_metric}.png",
        y_label=eval_metric,
        title=f"Per-summary  (extract={extract_metric}, eval={eval_metric})",
    )

    return best_res, best_res_all


def run_aggregations_and_plots(args, results_per_metric, data, dataset_extractive, out_dir):
    """Cross-eval grid: each extract metric × each eval metric. Plots time + gold lengths once."""
    gold_lens = {"approach": 0, "challenge": 0, "outcome": 0}
    for app, chal, outc in zip(
        dataset_extractive["approach_sentences"],
        dataset_extractive["challenge_sentences"],
        dataset_extractive["outcome_sentences"],
    ):
        gold_lens["approach"] += len(app)
        gold_lens["challenge"] += len(chal)
        gold_lens["outcome"] += len(outc)
    n_docs = len(dataset_extractive["approach_sentences"])
    gold_avg = {k: v / n_docs for k, v in gold_lens.items()}
    save_json(gold_avg, out_dir, "gold_avg_lengths.json")

    best_res_per_metric = {}
    best_res_all_per_metric = {}
    for em, all_results in results_per_metric.items():
        sub = os.path.join(out_dir, em)
        os.makedirs(sub, exist_ok=True)
        plot_total_time(
            all_results[args.max_sent_singular - 1],
            f"Time Singular  (extract={em})",
            sub, "time_singular.png",
        )
        plot_total_time(
            all_results[args.max_sent_all - 1],
            f"Time All-Together  (extract={em})",
            sub, "time_all_tog.png",
        )
        for ev in args.eval_metrics:
            best_res, best_res_all = run_aggregations_one_pair(
                args, all_results, data, dataset_extractive, sub, em, ev
            )
        best_res_per_metric[em] = best_res
        best_res_all_per_metric[em] = best_res_all

    return best_res_per_metric, best_res_all_per_metric


# ============================ Section 2: HF sweep ===========================

def run_hf_sweep_one_metric(args, data, dataset_extractive, out_dir, extract_metric):
    list_source_sentences = data["list_source_sentences"]
    list_summaries = data["list_summaries"]
    template = make_results_template()

    print(f"[hf-sweep:{extract_metric}] singular range={args.range_single_summaries}",
          flush=True)
    slow_results = []
    for sum_sent in args.range_single_summaries:
        t0 = time.perf_counter()
        res = copy.deepcopy(template)
        for summary in SINGULAR_SUMM_TYPES:
            res[summary]["greedy_search"]["result"], res[summary]["greedy_search"]["time"] = \
                apply_heuristic_to_dataset_hf(
                    greedy_extractive_summary_hf, list_source_sentences,
                    list_summaries[summary], extract_metric, max_sent=sum_sent,
                )
            res[summary]["beam_search"]["result"], res[summary]["beam_search"]["time"] = \
                apply_heuristic_to_dataset_hf(
                    beam_extractive_summary_hf, list_source_sentences,
                    list_summaries[summary], extract_metric, max_sent=sum_sent,
                    beam_size=args.beam_size,
                )
            res[summary]["local_score"]["result"], res[summary]["local_score"]["time"] = \
                apply_heuristic_to_dataset_hf(
                    local_extractive_summary_hf, list_source_sentences,
                    list_summaries[summary], extract_metric, max_sent=sum_sent,
                )
            res[summary]["global_score"]["result"], res[summary]["global_score"]["time"] = \
                apply_heuristic_to_dataset_hf(
                    global_extractive_summary_hf, list_source_sentences,
                    list_summaries[summary], extract_metric, max_sent=sum_sent,
                    beam_size=args.beam_size,
                )
        slow_results.append(res)
        save_pickle(res, out_dir, f"slow_singular_K_{sum_sent:02d}.pkl")
        save_pickle(slow_results, out_dir, "slow_results.pkl")
        print(f"[hf-sweep:{extract_metric}] singular sum_sent={sum_sent} done in "
              f"{time.perf_counter() - t0:.1f}s (checkpoint saved)", flush=True)

    for ev in args.eval_metrics:
        avg = avg_score_slow_single(
            slow_results, args.range_single_summaries, dataset_extractive, ev
        )
        save_json(avg, out_dir, f"avg_slow_singular_eval_{ev}.json")
        plot_avg_score(
            avg, f"HF Singular  (extract={extract_metric}, eval={ev})",
            out_dir, f"avg_slow_singular_eval_{ev}.png",
        )

    print(f"[hf-sweep:{extract_metric}] all_together range={args.range_all_tog_summ}",
          flush=True)
    template_all_tog = make_all_together_template()
    slow_results_all_tog = []
    last_sum_sent = None
    for sum_sent in args.range_all_tog_summ:
        t0 = time.perf_counter()
        res = copy.deepcopy(template_all_tog)
        res["all_together"]["greedy_search"]["result"], res["all_together"]["greedy_search"]["time"] = \
            apply_heuristic_to_dataset_hf(
                greedy_extractive_summary_hf, list_source_sentences,
                list_summaries["all_together"], extract_metric, max_sent=sum_sent,
            )
        res["all_together"]["beam_search"]["result"], res["all_together"]["beam_search"]["time"] = \
            apply_heuristic_to_dataset_hf(
                beam_extractive_summary_hf, list_source_sentences,
                list_summaries["all_together"], extract_metric, max_sent=sum_sent,
                beam_size=args.beam_size,
            )
        res["all_together"]["local_score"]["result"], res["all_together"]["local_score"]["time"] = \
            apply_heuristic_to_dataset_hf(
                local_extractive_summary_hf, list_source_sentences,
                list_summaries["all_together"], extract_metric, max_sent=sum_sent,
            )
        res["all_together"]["global_score"]["result"], res["all_together"]["global_score"]["time"] = \
            apply_heuristic_to_dataset_hf(
                global_extractive_summary_hf, list_source_sentences,
                list_summaries["all_together"], extract_metric, max_sent=sum_sent,
                beam_size=args.beam_size,
            )
        slow_results_all_tog.append(res)
        last_sum_sent = sum_sent
        save_pickle(res, out_dir, f"slow_all_tog_K_{sum_sent:02d}.pkl")
        save_pickle(slow_results_all_tog, out_dir, "slow_results_all_tog.pkl")
        print(f"[hf-sweep:{extract_metric}] all_together sum_sent={sum_sent} done in "
              f"{time.perf_counter() - t0:.1f}s (checkpoint saved)", flush=True)

    for ev in args.eval_metrics:
        avg = avg_score_slow_all_tog(
            slow_results_all_tog, args.range_all_tog_summ,
            data["list_all_summaries_extractive"], ev,
        )
        save_json(avg, out_dir, f"avg_slow_all_tog_eval_{ev}.json")
        plot_avg_score(
            avg, f"HF All-Together  (extract={extract_metric}, eval={ev})",
            out_dir, f"avg_slow_all_tog_eval_{ev}.png",
        )

    return slow_results, slow_results_all_tog, last_sum_sent


def run_hf_sweep(args, data, dataset_extractive, out_dir):
    last_sum_sent = None
    for em in args.extract_metrics:
        sub = os.path.join(out_dir, em)
        os.makedirs(sub, exist_ok=True)
        _, _, last_sum_sent = run_hf_sweep_one_metric(
            args, data, dataset_extractive, sub, em
        )
    return last_sum_sent


# ============================ Section 4: PRF ================================

def run_prf(args, best_res_per_metric, best_res_all_per_metric, data,
            dataset_extractive, out_dir):
    """Precision / Recall / F1 at chosen K. Runs the full extract×eval grid."""
    if not best_res_per_metric or not best_res_all_per_metric:
        print("[prf] skipped — fast-sweep results unavailable", flush=True)
        return

    for em, best_res in best_res_per_metric.items():
        best_res_all = best_res_all_per_metric[em]
        sub = os.path.join(out_dir, em)
        os.makedirs(sub, exist_ok=True)
        for rt in args.eval_metrics:
            prf_singular = prf_at_K_singular(best_res, dataset_extractive, rt)
            save_json(prf_singular, sub, f"prf_singular_eval_{rt}.json")
            plot_prf_per_summary(
                prf_singular, sub, f"prf_singular_eval_{rt}.png",
                title=f"P/R/F1  extract={em}, eval={rt}, K={args.max_sent_singular}",
            )

            prf_all = prf_at_K_all_tog(
                best_res_all, data["list_all_summaries_extractive"], rt
            )
            save_json(prf_all, sub, f"prf_all_tog_eval_{rt}.json")
            plot_prf_grouped(
                prf_all, sub, f"prf_all_tog_eval_{rt}.png",
                title=f"P/R/F1  extract={em}, eval={rt}, K={args.max_sent_all}",
            )


# ============================ Section 5: MMR ================================

def run_mmr(args, data, dataset_extractive, out_dir):
    list_source_sentences = data["list_source_sentences"]
    list_summaries = data["list_summaries"]

    print(f"[mmr] sweep K={args.max_sent_start}..{args.max_sent_end - 1}, "
          f"lambdas={args.mmr_lambdas}", flush=True)

    sweep_range = list(range(args.max_sent_start, args.max_sent_end))

    from .metrics import rouge_score as _rs

    def _scorer(metric):
        return lambda c, r: _rs(c, r, metric)

    avg_per_lambda = {}
    for lam in args.mmr_lambdas:
        avg_per_K = {}
        all_runs = []
        for sum_sent in sweep_range:
            t0 = time.perf_counter()
            res = make_results_template(heuristics=["mmr"])
            for summary in SUMM_TYPES:
                res[summary]["mmr"]["result"], res[summary]["mmr"]["time"] = \
                    apply_heuristic_to_dataset_parallel(
                        mmr_extractive_summary, list_source_sentences,
                        list_summaries[summary], n_jobs=args.mmr_n_jobs,
                        max_sent=sum_sent, lambda_=lam,
                        embed_model=args.mmr_embed_model,
                    )
            all_runs.append(res)

            per_metric = {}
            for ev in args.eval_metrics:
                f1_singular = sum(
                    avg_score_generic(res, dataset_extractive, summ, _scorer(ev))["mmr"]
                    for summ in SINGULAR_SUMM_TYPES
                ) / len(SINGULAR_SUMM_TYPES)
                f1_all_tog = avg_score_generic_all_tog(
                    res, data["list_all_summaries_extractive"], _scorer(ev),
                )["mmr"]
                per_metric[ev] = {
                    "singular_avg": round(f1_singular, 4),
                    "all_tog": round(f1_all_tog, 4),
                }
            avg_per_K[sum_sent] = per_metric
            save_pickle(res, out_dir, f"mmr_lambda_{lam}_K_{sum_sent:02d}.pkl")
            ev0 = args.eval_metrics[0]
            print(
                f"[mmr] lambda={lam} K={sum_sent} "
                f"{ev0}_sing={per_metric[ev0]['singular_avg']:.3f} "
                f"{ev0}_all={per_metric[ev0]['all_tog']:.3f} "
                f"({time.perf_counter() - t0:.1f}s) (checkpoint saved)",
                flush=True,
            )

        avg_per_lambda[str(lam)] = avg_per_K
        save_pickle(all_runs, out_dir, f"mmr_all_runs_lambda_{lam}.pkl")
        save_json(avg_per_lambda, out_dir, "mmr_avg_scores.json")

    save_json(avg_per_lambda, out_dir, "mmr_avg_scores.json")

    for ev in args.eval_metrics:
        sing_curve = {
            K: {f"lam={lam}": avg_per_lambda[str(lam)][K][ev]["singular_avg"]
                for lam in args.mmr_lambdas}
            for K in sweep_range
        }
        plot_avg_score(sing_curve, f"MMR Singular  (eval={ev})",
                       out_dir, f"mmr_singular_curve_eval_{ev}.png")

        all_curve = {
            K: {f"lam={lam}": avg_per_lambda[str(lam)][K][ev]["all_tog"]
                for lam in args.mmr_lambdas}
            for K in sweep_range
        }
        plot_avg_score(all_curve, f"MMR All-Together  (eval={ev})",
                       out_dir, f"mmr_all_tog_curve_eval_{ev}.png")


# ============================ Section 6: MoverScore =========================

def run_moverscore(args, data, dataset_extractive, out_dir):
    """Run 4 base heuristics with MoverScore as the scorer at chosen K only.

    Sequential per document — MoverScore loads a transformer model per process,
    so joblib parallelism is counterproductive. Embedding model is GPU-bound.
    """
    list_source_sentences = data["list_source_sentences"]
    list_summaries = data["list_summaries"]

    print("[moverscore] running heuristics with scorer='moverscore' "
          f"at K_singular={args.max_sent_singular}, K_all={args.max_sent_all}",
          flush=True)

    # Singular summaries
    res_sing = make_results_template(summ_types=SINGULAR_SUMM_TYPES)
    for summary in SINGULAR_SUMM_TYPES:
        if summary not in args.moverscore_on_summaries:
            continue
        for fn, name, kwargs in [
            (greedy_extractive_summary, "greedy_search",
             {"max_sent": args.max_sent_singular, "scorer": "moverscore"}),
            (beam_extractive_summary, "beam_search",
             {"max_sent": args.max_sent_singular, "beam_size": args.beam_size,
              "scorer": "moverscore"}),
            (local_extractive_summary, "local_score",
             {"max_sent": args.max_sent_singular, "scorer": "moverscore"}),
            (global_extractive_summary, "global_score",
             {"max_sent": args.max_sent_singular, "beam_size": args.beam_size,
              "scorer": "moverscore"}),
        ]:
            t0 = time.perf_counter()
            sel, _ = apply_heuristic_to_dataset_sequential(
                fn, list_source_sentences, list_summaries[summary], **kwargs,
            )
            res_sing[summary][name]["result"] = sel
            res_sing[summary][name]["time"] = time.perf_counter() - t0
            print(f"[moverscore] {summary}/{name} {time.perf_counter() - t0:.1f}s",
                  flush=True)
    save_pickle(res_sing, out_dir, "moverscore_singular.pkl")

    # All_together
    res_all = make_all_together_template()
    if "all_together" in args.moverscore_on_summaries:
        for fn, name, kwargs in [
            (greedy_extractive_summary, "greedy_search",
             {"max_sent": args.max_sent_all, "scorer": "moverscore"}),
            (beam_extractive_summary, "beam_search",
             {"max_sent": args.max_sent_all, "beam_size": args.beam_size,
              "scorer": "moverscore"}),
            (local_extractive_summary, "local_score",
             {"max_sent": args.max_sent_all, "scorer": "moverscore"}),
            (global_extractive_summary, "global_score",
             {"max_sent": args.max_sent_all, "beam_size": args.beam_size,
              "scorer": "moverscore"}),
        ]:
            t0 = time.perf_counter()
            sel, _ = apply_heuristic_to_dataset_sequential(
                fn, list_source_sentences, list_summaries["all_together"], **kwargs,
            )
            res_all["all_together"][name]["result"] = sel
            res_all["all_together"][name]["time"] = time.perf_counter() - t0
            print(f"[moverscore] all_together/{name} {time.perf_counter() - t0:.1f}s",
                  flush=True)
    save_pickle(res_all, out_dir, "moverscore_all_tog.pkl")

    from .metrics import rouge_score as _rs

    def _scorer(metric):
        if metric == "moverscore":
            from .metrics_semantic import mover_score
            return lambda c, r: mover_score(c, r)
        return lambda c, r: _rs(c, r, metric)

    eval_metrics = list(args.eval_metrics) + ["moverscore"]

    # Eval the moverscore-extracted picks with all metrics + moverscore-self
    for ev in eval_metrics:
        avg_sing = {}
        for summ in SINGULAR_SUMM_TYPES:
            if summ in args.moverscore_on_summaries:
                avg_sing[summ] = avg_score_generic(
                    res_sing, dataset_extractive, summ, _scorer(ev)
                )
        save_json(avg_sing, out_dir, f"moverscore_singular_eval_{ev}.json")
        if avg_sing:
            flat = {h: sum(avg_sing[s][h] for s in avg_sing) / len(avg_sing)
                    for h in next(iter(avg_sing.values())).keys()}
            plot_histogram_with_values(
                flat, out_dir, f"hist_moverscore_singular_eval_{ev}.png",
                title=f"MoverScore-extracted, eval={ev} (singular avg)",
            )

        if "all_together" in args.moverscore_on_summaries:
            avg_all = avg_score_generic_all_tog(
                res_all, data["list_all_summaries_extractive"], _scorer(ev),
            )
            save_json(avg_all, out_dir, f"moverscore_all_tog_eval_{ev}.json")
            plot_histogram_with_values(
                avg_all, out_dir, f"hist_moverscore_all_tog_eval_{ev}.png",
                title=f"MoverScore-extracted, eval={ev} (all_together)",
            )

    # Cross-eval: rescore picks from prior --extract-roots runs with MoverScore
    if args.extract_roots:
        from .io_utils import load_pickle
        ms_score = _scorer("moverscore")
        cross_grid = {}
        for root in args.extract_roots:
            label = os.path.basename(os.path.normpath(root))
            cross_grid[label] = {}
            for em in args.extract_metrics:
                pkl_sing = os.path.join(
                    root, "fast_sweep", em, f"fast_K_{args.max_sent_singular:02d}.pkl"
                )
                pkl_all = os.path.join(
                    root, "fast_sweep", em, f"fast_K_{args.max_sent_all:02d}.pkl"
                )
                if not (os.path.exists(pkl_sing) and os.path.exists(pkl_all)):
                    print(f"[moverscore-cross] missing {pkl_sing} or {pkl_all}, skip",
                          flush=True)
                    continue
                picks_sing = load_pickle(pkl_sing)
                picks_all = load_pickle(pkl_all)
                sing = {}
                for summ in SINGULAR_SUMM_TYPES:
                    sing[summ] = avg_score_generic(
                        picks_sing, dataset_extractive, summ, ms_score
                    )
                allt = avg_score_generic_all_tog(
                    picks_all, data["list_all_summaries_extractive"], ms_score,
                )
                cross_grid[label][em] = {"singular": sing, "all_together": allt}
                print(f"[moverscore-cross] root={label} extract={em} done",
                      flush=True)
        save_json(cross_grid, out_dir, "moverscore_cross_eval.json")


# ============================ Main ==========================================

def _load_fast_sweep_from_disk(args, root):
    """Stage 2: rebuild results_per_metric dict from a prior fast_sweep dir."""
    from .io_utils import load_pickle
    out = {}
    for em in args.extract_metrics:
        pkl = os.path.join(root, "fast_sweep", em, "fast_all_results.pkl")
        if os.path.exists(pkl):
            out[em] = load_pickle(pkl)
    return out


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run_all] out_dir={out_dir}", flush=True)
    print(f"[run_all] extract_metrics={args.extract_metrics} "
          f"eval_metrics={args.eval_metrics}", flush=True)

    print("[load] ACLSum", flush=True)
    dataset_abstractive, dataset_extractive = load_aclsum()
    data = build_summary_lists(dataset_abstractive, dataset_extractive)

    best_res_per_metric = {}
    best_res_all_per_metric = {}

    if not args.skip_fast_sweep:
        sub = os.path.join(out_dir, "fast_sweep")
        results_per_metric = run_fast_sweep(args, data, sub)
        best_res_per_metric, best_res_all_per_metric = run_aggregations_and_plots(
            args, results_per_metric, data, dataset_extractive, sub
        )
    elif args.extract_roots:
        for root in args.extract_roots:
            rpm = _load_fast_sweep_from_disk(args, root)
            for em, all_results in rpm.items():
                if em not in best_res_per_metric:
                    best_res_per_metric[em] = all_results[args.max_sent_singular - 1]
                    best_res_all_per_metric[em] = all_results[args.max_sent_all - 1]

    if not args.skip_hf_sweep:
        sub = os.path.join(out_dir, "hf_sweep")
        run_hf_sweep(args, data, dataset_extractive, sub)

    if not args.skip_prf:
        sub = os.path.join(out_dir, "prf")
        run_prf(args, best_res_per_metric, best_res_all_per_metric,
                data, dataset_extractive, sub)

    if not args.skip_mmr:
        sub = os.path.join(out_dir, "mmr")
        run_mmr(args, data, dataset_extractive, sub)

    if not args.skip_moverscore:
        sub = os.path.join(out_dir, "moverscore")
        run_moverscore(args, data, dataset_extractive, sub)

    print("[run_all] done", flush=True)


if __name__ == "__main__":
    main()
