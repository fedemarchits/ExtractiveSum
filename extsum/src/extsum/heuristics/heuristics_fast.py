"""Generic-scorer extractive heuristics.

Each heuristic takes a `scorer` string (passed to scorers.get_scorer) so the
same code paths drive ROUGE-1/2/L and MoverScore.
"""

from .scorers import get_scorer


def greedy_extractive_summary(sentences, abstractive_summary, max_sent=6, scorer="rouge1"):
    score_fn = get_scorer(scorer)
    sentences = [str(s) for s in sentences]
    abstractive_summary = str(abstractive_summary)

    selected = []
    remaining = sentences[:]

    while remaining and max_sent > 0:
        best_sentence = None
        best_score = -float("inf")
        for sent in remaining:
            cand = " ".join(selected + [sent])
            s = score_fn(cand, abstractive_summary)
            if s > best_score:
                best_score = s
                best_sentence = sent

        selected.append(best_sentence)
        remaining.remove(best_sentence)
        max_sent -= 1

    return selected


def beam_extractive_summary(
    sentences, abstractive_summary, max_sent=6, beam_size=4, scorer="rouge1"
):
    score_fn = get_scorer(scorer)
    sentences = [str(s) for s in sentences]
    abstractive_summary = str(abstractive_summary)

    beams = [([], 0.0)]

    for _ in range(min(max_sent, len(sentences))):
        new_beams = []
        for selected, _ in beams:
            remaining = [s for s in sentences if s not in selected]
            for sent in remaining:
                cand = " ".join(selected + [sent])
                s = score_fn(cand, abstractive_summary)
                new_beams.append((selected + [sent], s))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    return beams[0][0]


def local_extractive_summary(sentences, abstractive_summary, max_sent=3, scorer="rouge1"):
    score_fn = get_scorer(scorer)
    sentences = [str(s) for s in sentences]
    abstractive_summary = str(abstractive_summary)

    sent_with_score = []
    for sent in sentences:
        s = score_fn(sent, abstractive_summary)
        sent_with_score.append({"sentence": sent, "r_score": s})

    sent_with_score.sort(key=lambda x: x["r_score"], reverse=True)
    return [s["sentence"] for s in sent_with_score[:max_sent]]


def global_extractive_summary(
    sentences, abstractive_summary, max_sent=3, beam_size=4, scorer="rouge1"
):
    score_fn = get_scorer(scorer)
    sentences = [str(s) for s in sentences]
    abstractive_summary = str(abstractive_summary)

    beams = [([], 0.0)]
    for _ in range(min(max_sent, len(sentences))):
        new_beams = []
        for selected, _ in beams:
            remaining = [s for s in sentences if s not in selected]
            for sent in remaining:
                cand = " ".join(selected + [sent])
                s = score_fn(cand, abstractive_summary)
                new_beams.append((selected + [sent], s))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    candidates = beams
    unique_sent_in_candidates = list(
        set([s for cand_sents, _ in candidates for s in cand_sents])
    )

    scored = []
    for sent in unique_sent_in_candidates:
        total = 0.0
        for cand_sents, score in candidates:
            if sent in cand_sents:
                total += score
        scored.append((sent, total))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:max_sent]]


# --------------------------- Hybrids ---------------------------------------
# Combine local_score (recall-rich, independent scoring) with greedy_search
# (precision-rich, joint scoring with redundancy penalty).

def _local_ranking(sentences, abstractive_summary, scorer="rouge1"):
    """Full ranking of sentences by independent score vs summary (local_score logic)."""
    score_fn = get_scorer(scorer)
    sentences = [str(s) for s in sentences]
    abstractive_summary = str(abstractive_summary)
    scored = [(sent, score_fn(sent, abstractive_summary)) for sent in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


def _greedy_ranking(sentences, abstractive_summary, max_rank, scorer="rouge1"):
    """Greedy iterative selection up to max_rank. Order = greedy rank."""
    score_fn = get_scorer(scorer)
    sentences = [str(s) for s in sentences]
    abstractive_summary = str(abstractive_summary)
    selected = []
    remaining = sentences[:]
    while remaining and len(selected) < max_rank:
        best_sent = None
        best_score = -float("inf")
        for sent in remaining:
            cand = " ".join(selected + [sent])
            s = score_fn(cand, abstractive_summary)
            if s > best_score:
                best_score = s
                best_sent = sent
        selected.append(best_sent)
        remaining.remove(best_sent)
    return selected


def rrf_extractive_summary(
    sentences, abstractive_summary, max_sent=6, scorer="rouge1", k_rrf=60,
):
    """Reciprocal Rank Fusion of local-score ranking and greedy ranking.

    score(sent) = 1/(k_rrf + rank_local) + 1/(k_rrf + rank_greedy).
    Sentences not picked by greedy get a fixed worst rank == len(sentences).
    """
    local_rank = {s: r for r, s in enumerate(
        _local_ranking(sentences, abstractive_summary, scorer)
    )}
    # Greedy ranking only over max_sent picks is cheaper but lossy. Use full N
    # so unpicked sentences contribute via the unranked fallback.
    greedy_top = _greedy_ranking(
        sentences, abstractive_summary, max_rank=max_sent, scorer=scorer
    )
    greedy_rank = {s: r for r, s in enumerate(greedy_top)}
    fallback = len(sentences)

    fused = []
    for sent in sentences:
        rl = local_rank.get(sent, fallback)
        rg = greedy_rank.get(sent, fallback)
        score = 1.0 / (k_rrf + rl) + 1.0 / (k_rrf + rg)
        fused.append((sent, score))
    fused.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in fused[:max_sent]]


def stage_gate_extractive_summary(
    sentences, abstractive_summary, max_sent=6, scorer="rouge1", pool_mult=2,
):
    """Stage-gate hybrid: local picks top (pool_mult * max_sent) by raw similarity,
    then greedy selects max_sent from that pool with redundancy penalty.

    Aims for local's recall (large candidate pool) + greedy's precision (diversity).
    """
    pool_size = min(pool_mult * max_sent, len(sentences))
    pool = _local_ranking(sentences, abstractive_summary, scorer)[:pool_size]
    # Greedy iterative selection inside the pool only.
    return _greedy_ranking(pool, abstractive_summary, max_rank=max_sent, scorer=scorer)
