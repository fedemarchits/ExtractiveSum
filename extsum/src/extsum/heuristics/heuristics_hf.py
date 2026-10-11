from .metrics import rouge_score


def _as_ref(x):
    return " ".join(x) if isinstance(x, (list, tuple)) else x


def greedy_extractive_summary_hf(sentences, abstractive_summaries, eval_criteria, max_sent):
    abstractive_summaries = _as_ref(abstractive_summaries)
    selected = []
    remaining = sentences[:]

    while remaining and max_sent != 0:
        best_sentence = None
        best_score = -1
        for sent in remaining:
            candidate_summary = " ".join(selected + [sent])
            r1 = rouge_score(candidate_summary, abstractive_summaries, eval_criteria)
            if r1 > best_score:
                best_score = r1
                best_sentence = sent

        selected.append(best_sentence)
        remaining.remove(best_sentence)
        max_sent -= 1

    return selected


def beam_extractive_summary_hf(
    sentences, abstractive_summary, eval_criteria, max_sent, beam_size=4, with_score=False
):
    abstractive_summary = _as_ref(abstractive_summary)
    beams = [([], 0.0)]

    for _ in range(min(max_sent, len(sentences))):
        new_beams = []
        for selected, _ in beams:
            remaining = [s for s in sentences if s not in selected]
            for sent in remaining:
                candidate_summary = " ".join(selected + [sent])
                r1 = rouge_score(candidate_summary, abstractive_summary, eval_criteria)
                new_beams.append((selected + [sent], r1))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
    if with_score:
        return beams
    return beams[0][0]


def local_extractive_summary_hf(sentences, abstractive_summaries, eval_criteria, max_sent):
    abstractive_summaries = _as_ref(abstractive_summaries)
    sent_with_score = []
    for sent in sentences:
        r1 = rouge_score(sent, abstractive_summaries, eval_criteria)
        sent_with_score.append({"sentence": sent, "r_score": r1})
    sent_with_score.sort(key=lambda x: x["r_score"], reverse=True)
    return [s["sentence"] for s in sent_with_score[:max_sent]]


def global_extractive_summary_hf(
    sentences, abstractive_summary, eval_criteria, max_sent, beam_size=4
):
    candidates = beam_extractive_summary_hf(
        sentences, abstractive_summary, eval_criteria, max_sent, beam_size, True
    )
    unique_sent_in_candidates = list(
        set([sent for candidate in [sents for sents, _ in candidates] for sent in candidate])
    )
    unique_sent_with_scores = []

    for sent in unique_sent_in_candidates:
        final_score = 0
        for cand, score in candidates:
            if sent in cand:
                final_score += score
        unique_sent_with_scores.append((sent, final_score))

    sorted_unique_sent_with_scores = sorted(
        unique_sent_with_scores, key=lambda x: x[1], reverse=True
    )
    return [sent for sent, _ in sorted_unique_sent_with_scores][:max_sent]
