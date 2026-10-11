"""Maximum Marginal Relevance (MMR) extractive heuristic.

Selection rule per step:
  argmax_i [ lambda * sim(s_i, ref) - (1 - lambda) * max_{j in selected} sim(s_i, s_j) ]
"""

from .embedder import cosine_sim_matrix, encode


def mmr_extractive_summary(
    sentences,
    abstractive_summary,
    max_sent=6,
    lambda_=0.7,
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
):
    sentences = [str(s) for s in sentences]
    abstractive_summary = str(abstractive_summary)
    if not sentences:
        return []

    sent_emb = encode(sentences, embed_model)
    ref_emb = encode([abstractive_summary], embed_model)

    relevance = cosine_sim_matrix(sent_emb, ref_emb).flatten()
    sent_sim = cosine_sim_matrix(sent_emb, sent_emb)

    selected_idx = []
    remaining = set(range(len(sentences)))

    for _ in range(min(max_sent, len(sentences))):
        if not remaining:
            break
        best_idx = None
        best_score = -float("inf")
        for i in remaining:
            if not selected_idx:
                redundancy = 0.0
            else:
                redundancy = max(sent_sim[i, j] for j in selected_idx)
            score = lambda_ * relevance[i] - (1 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    return [sentences[i] for i in selected_idx]
