"""Aggregators over result dicts produced by runner.apply_heuristic_*."""

from .metrics import rouge_pr_f, rouge_score
from .runner import HEURISTICS, SINGULAR_SUMM_TYPES


def _heur_keys(res, summ_key):
    """Pull heuristic keys from the actual res dict (supports custom heuristic sets)."""
    return list(res[summ_key].keys())


# ------------------ original notebook aggregators (F1 only) -----------------

def avg_score_across_summaries(all_results, dataset_extractive, n_runs=32, rouge_type="rouge1"):
    out = {i: {heur: 0 for heur in HEURISTICS} for i in range(1, n_runs + 1)}
    for i in range(n_runs):
        res = all_results[i]
        for heur in HEURISTICS:
            sum_score = 0
            for summ in SINGULAR_SUMM_TYPES:
                avg_score_doc = 0
                n_docs = len(res[summ][heur]["result"])
                for doc in range(n_docs):
                    candidate = " ".join(res[summ][heur]["result"][doc])
                    reference = " ".join(dataset_extractive[doc][summ + "_sentences"])
                    avg_score_doc += rouge_score(candidate, reference, rouge_type)
                sum_score += avg_score_doc / n_docs
            out[i + 1][heur] = round(sum_score / len(SINGULAR_SUMM_TYPES), 4)
    return out


def avg_score_across_all_tog(all_results, list_all_summaries_extractive, n_runs=32, rouge_type="rouge1"):
    out = {i: {heur: 0 for heur in HEURISTICS} for i in range(1, n_runs + 1)}
    for i in range(n_runs):
        res = all_results[i]
        for heur in HEURISTICS:
            sum_score = 0
            avg_score_doc = 0
            n_docs = len(res["all_together"][heur]["result"])
            for doc in range(n_docs):
                candidate = " ".join(res["all_together"][heur]["result"][doc])
                reference = list_all_summaries_extractive[doc]
                avg_score_doc += rouge_score(candidate, reference, rouge_type)
            sum_score += avg_score_doc / n_docs
            out[i + 1][heur] = round(sum_score, 4)
    return out


def avg_score_per_singular(all_results, dataset_extractive, summ_key, n_runs=32, rouge_type="rouge1"):
    out = {i: {heur: 0 for heur in HEURISTICS} for i in range(1, n_runs + 1)}
    for i in range(n_runs):
        res = all_results[i]
        for heur in HEURISTICS:
            sum_score = 0
            avg_score_doc = 0
            n_docs = len(res[summ_key][heur]["result"])
            for doc in range(n_docs):
                candidate = " ".join(res[summ_key][heur]["result"][doc])
                reference = " ".join(dataset_extractive[doc][summ_key + "_sentences"])
                avg_score_doc += rouge_score(candidate, reference, rouge_type)
            sum_score += avg_score_doc / n_docs
            out[i + 1][heur] = round(sum_score, 4)
    return out


def avg_score_across_heuristics_singular(best_res, dataset_extractive, rouge_type="rouge1"):
    out = {"challenge": 0, "approach": 0, "outcome": 0, "all_together": 0}
    heurs = _heur_keys(best_res, SINGULAR_SUMM_TYPES[0])
    for summ in SINGULAR_SUMM_TYPES:
        avg_score = 0
        for heur in heurs:
            avg_score_doc = 0
            n_docs = len(best_res[summ][heur]["result"])
            for doc in range(n_docs):
                candidate = " ".join(best_res[summ][heur]["result"][doc])
                reference = " ".join(dataset_extractive[doc][summ + "_sentences"])
                avg_score_doc += rouge_score(candidate, reference, rouge_type)
            avg_score += avg_score_doc / n_docs
        avg_score = avg_score / len(heurs)
        out[summ] = round(avg_score, 4)
    return out


def avg_score_across_heuristics_all_tog(best_res_all, list_all_summaries_extractive, rouge_type="rouge1"):
    heurs = _heur_keys(best_res_all, "all_together")
    avg_score = 0
    for heur in heurs:
        avg_score_doc = 0
        n_docs = len(best_res_all["all_together"][heur]["result"])
        for doc in range(n_docs):
            candidate = " ".join(best_res_all["all_together"][heur]["result"][doc])
            reference = list_all_summaries_extractive[doc]
            avg_score_doc += rouge_score(candidate, reference, rouge_type)
        avg_score += avg_score_doc / n_docs
    avg_score = avg_score / len(heurs)
    return round(avg_score, 4)


def avg_score_slow_single(slow_results, range_single, dataset_extractive, eval_criteria="rouge1"):
    out = {i: {heur: 0 for heur in HEURISTICS} for i in range_single}
    for idx, val in enumerate(range_single):
        res = slow_results[idx]
        for heur in HEURISTICS:
            sum_score = 0
            for summ in SINGULAR_SUMM_TYPES:
                avg_score_doc = 0
                n_docs = len(res[summ][heur]["result"])
                for doc in range(n_docs):
                    candidate = " ".join(res[summ][heur]["result"][doc])
                    reference = " ".join(dataset_extractive[doc][summ + "_sentences"])
                    avg_score_doc += rouge_score(candidate, reference, eval_criteria)
                sum_score += avg_score_doc / n_docs
            out[val][heur] = round(sum_score / len(SINGULAR_SUMM_TYPES), 2)
    return out


def avg_score_slow_all_tog(
    slow_results_all_tog, range_all_tog, list_all_summaries_extractive, eval_criteria="rouge1"
):
    out = {i: {heur: 0 for heur in HEURISTICS} for i in range_all_tog}
    for idx, val in enumerate(range_all_tog):
        res = slow_results_all_tog[idx]
        for heur in HEURISTICS:
            sum_score = 0
            avg_score_doc = 0
            n_docs = len(res["all_together"][heur]["result"])
            for doc in range(n_docs):
                candidate = " ".join(res["all_together"][heur]["result"][doc])
                reference = list_all_summaries_extractive[doc]
                score = rouge_score(candidate, reference, eval_criteria)
                avg_score_doc += score
            sum_score += avg_score_doc / n_docs
            out[val][heur] = round(sum_score, 2)
    return out


def avg_score_singular_point(single_res, dataset_extractive, eval_criteria):
    out = {heur: 0 for heur in HEURISTICS}
    for heur in HEURISTICS:
        sum_score = 0
        for summ in SINGULAR_SUMM_TYPES:
            avg_score_doc = 0
            n_docs = len(single_res[summ][heur]["result"])
            for doc in range(n_docs):
                candidate = " ".join(single_res[summ][heur]["result"][doc])
                reference = " ".join(dataset_extractive[doc][summ + "_sentences"])
                avg_score_doc += rouge_score(candidate, reference, eval_criteria)
            sum_score += avg_score_doc / n_docs
        out[heur] = round(sum_score / len(SINGULAR_SUMM_TYPES), 2)
    return out


def avg_score_all_tog_point(all_tog_res, list_all_summaries_extractive, eval_criteria):
    out = {heur: 0 for heur in HEURISTICS}
    for heur in HEURISTICS:
        sum_score = 0
        avg_score_doc = 0
        n_docs = len(all_tog_res["all_together"][heur]["result"])
        for doc in range(n_docs):
            candidate = " ".join(all_tog_res["all_together"][heur]["result"][doc])
            reference = list_all_summaries_extractive[doc]
            score = rouge_score(candidate, reference, eval_criteria)
            avg_score_doc += score
        sum_score += avg_score_doc / n_docs
        out[heur] = round(sum_score, 2)
    return out


# ------------------ Precision / Recall / F1 aggregators ---------------------

def prf_at_K_singular(best_res, dataset_extractive, rouge_type="rouge1"):
    """Per-summary, per-heuristic dict of {precision, recall, fmeasure}."""
    heurs = _heur_keys(best_res, SINGULAR_SUMM_TYPES[0])
    out = {}
    for summ in SINGULAR_SUMM_TYPES:
        out[summ] = {}
        for heur in heurs:
            n_docs = len(best_res[summ][heur]["result"])
            p_sum = r_sum = f_sum = 0.0
            for doc in range(n_docs):
                candidate = " ".join(best_res[summ][heur]["result"][doc])
                reference = " ".join(dataset_extractive[doc][summ + "_sentences"])
                d = rouge_pr_f(candidate, reference, rouge_type)
                p_sum += d["precision"]
                r_sum += d["recall"]
                f_sum += d["fmeasure"]
            out[summ][heur] = {
                "precision": round(p_sum / n_docs, 4),
                "recall": round(r_sum / n_docs, 4),
                "fmeasure": round(f_sum / n_docs, 4),
            }
    return out


def prf_at_K_all_tog(best_res_all, list_all_summaries_extractive, rouge_type="rouge1"):
    heurs = _heur_keys(best_res_all, "all_together")
    out = {}
    for heur in heurs:
        n_docs = len(best_res_all["all_together"][heur]["result"])
        p_sum = r_sum = f_sum = 0.0
        for doc in range(n_docs):
            candidate = " ".join(best_res_all["all_together"][heur]["result"][doc])
            reference = list_all_summaries_extractive[doc]
            d = rouge_pr_f(candidate, reference, rouge_type)
            p_sum += d["precision"]
            r_sum += d["recall"]
            f_sum += d["fmeasure"]
        out[heur] = {
            "precision": round(p_sum / n_docs, 4),
            "recall": round(r_sum / n_docs, 4),
            "fmeasure": round(f_sum / n_docs, 4),
        }
    return out


# ------------------ PRF over K-sweep ----------------------------------------

def prf_sweep_singular(all_results, dataset_extractive, K_values, rouge_type="rouge1"):
    """For each K in K_values, return per-summary, per-heuristic PRF dict.

    K_values lines up index-wise with all_results (i.e. all_results[i] corresponds
    to K_values[i]). Use list(range(...)) to match what run_fast_sweep produced.
    """
    out = {}
    for idx, K in enumerate(K_values):
        out[K] = prf_at_K_singular(all_results[idx], dataset_extractive, rouge_type)
    return out


def prf_sweep_all_tog(all_results, list_all_summaries_extractive, K_values, rouge_type="rouge1"):
    out = {}
    for idx, K in enumerate(K_values):
        out[K] = prf_at_K_all_tog(all_results[idx], list_all_summaries_extractive, rouge_type)
    return out


# ------------------ MMR / generic-scorer aggregators ------------------------

def avg_score_generic(res, dataset_extractive, summ_key, scorer_fn):
    """Average scorer_fn(candidate, reference) over docs for one summary type."""
    heurs = _heur_keys(res, summ_key)
    out = {}
    for heur in heurs:
        n_docs = len(res[summ_key][heur]["result"])
        if n_docs == 0:
            out[heur] = 0.0
            continue
        total = 0.0
        for doc in range(n_docs):
            candidate = " ".join(res[summ_key][heur]["result"][doc])
            reference = " ".join(dataset_extractive[doc][summ_key + "_sentences"])
            total += scorer_fn(candidate, reference)
        out[heur] = round(total / n_docs, 4)
    return out


def avg_score_generic_all_tog(res, list_all_summaries_extractive, scorer_fn):
    heurs = _heur_keys(res, "all_together")
    out = {}
    for heur in heurs:
        n_docs = len(res["all_together"][heur]["result"])
        if n_docs == 0:
            out[heur] = 0.0
            continue
        total = 0.0
        for doc in range(n_docs):
            candidate = " ".join(res["all_together"][heur]["result"][doc])
            reference = list_all_summaries_extractive[doc]
            total += scorer_fn(candidate, reference)
        out[heur] = round(total / n_docs, 4)
    return out
