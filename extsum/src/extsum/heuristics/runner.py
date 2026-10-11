"""Drivers that apply heuristics across documents."""

import time

from joblib import Parallel, delayed


HEURISTICS = ["greedy_search", "beam_search", "local_score", "global_score"]
HEURISTICS_WITH_MMR = HEURISTICS + ["mmr"]
SUMM_TYPES = ["challenge", "approach", "outcome", "all_together"]
SINGULAR_SUMM_TYPES = ["challenge", "approach", "outcome"]


def make_results_template(heuristics=None, summ_types=None):
    heuristics = heuristics or HEURISTICS
    summ_types = summ_types or SUMM_TYPES
    return {
        summ: {h: {"result": [], "time": None} for h in heuristics}
        for summ in summ_types
    }


def make_all_together_template(heuristics=None):
    heuristics = heuristics or HEURISTICS
    return {"all_together": {h: {"result": None, "time": None} for h in heuristics}}


def apply_heuristic_to_dataset_parallel(
    heuristic_fn, docs_sentences, abstractive_summaries, n_jobs=-1, **fn_kwargs
):
    """Forward fn_kwargs (max_sent, beam_size, scorer, lambda_, ...) to heuristic_fn."""
    start = time.perf_counter()

    def process_one(s, abs_s):
        return heuristic_fn(s, abs_s, **fn_kwargs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_one)(s, abs_s)
        for s, abs_s in zip(docs_sentences, abstractive_summaries)
    )
    return list(results), time.perf_counter() - start


def apply_heuristic_to_dataset_sequential(
    heuristic_fn, docs_sentences, abstractive_summaries, **fn_kwargs
):
    """Single-process variant. Use for MoverScore where per-worker model load is too costly."""
    start = time.perf_counter()
    selected_list = []
    for s, abs_s in zip(docs_sentences, abstractive_summaries):
        selected_list.append(heuristic_fn(s, abs_s, **fn_kwargs))
    return selected_list, time.perf_counter() - start


def apply_heuristic_to_dataset_hf(
    heuristic_fn,
    docs_sentences,
    abstractive_summaries,
    eval_criteria,
    max_sent=6,
    beam_size=0,
):
    """Sequential evaluation of HF-rouge legacy heuristics (notebook reproduction)."""
    selected_list = []
    start = time.perf_counter()
    for sentences, abs_summary in zip(docs_sentences, abstractive_summaries):
        if beam_size > 0:
            selected = heuristic_fn(sentences, abs_summary, eval_criteria, max_sent, beam_size)
        else:
            selected = heuristic_fn(sentences, abs_summary, eval_criteria, max_sent)
        selected_list.append(selected)
    return selected_list, time.perf_counter() - start
