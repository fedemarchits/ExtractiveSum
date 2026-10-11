from datasets import load_dataset


def load_aclsum():
    """Load ACLSum abstractive + extractive test splits."""
    abstractive = load_dataset("sobamchan/aclsum", "abstractive", split="test")
    extractive = load_dataset("sobamchan/aclsum", "extractive", split="test")
    return abstractive, extractive


def build_summary_lists(dataset_abstractive, dataset_extractive):
    """Build the abstractive + extractive summary lookup dicts used in notebook."""
    list_source_sentences = list(dataset_extractive["source_sentences"])

    list_all_summaries_abstractive = [
        [c, a, o]
        for c, a, o in zip(
            dataset_abstractive["challenge"],
            dataset_abstractive["approach"],
            dataset_abstractive["outcome"],
        )
    ]

    list_summaries = {
        "challenge": list(dataset_abstractive["challenge"]),
        "approach": list(dataset_abstractive["approach"]),
        "outcome": list(dataset_abstractive["outcome"]),
        "all_together": list_all_summaries_abstractive,
    }

    list_all_summaries_extractive = [
        " ".join([" ".join(c), " ".join(a), " ".join(o)])
        for c, a, o in zip(
            dataset_extractive["challenge_sentences"],
            dataset_extractive["approach_sentences"],
            dataset_extractive["outcome_sentences"],
        )
    ]

    joined_gold_labels_extractive = {
        "challenge": [" ".join(s) for s in dataset_extractive["challenge_sentences"]],
        "approach": [" ".join(s) for s in dataset_extractive["approach_sentences"]],
        "outcome": [" ".join(s) for s in dataset_extractive["outcome_sentences"]],
        "all_together": list_all_summaries_extractive,
    }

    return {
        "list_source_sentences": list_source_sentences,
        "list_summaries": list_summaries,
        "list_all_summaries_abstractive": list_all_summaries_abstractive,
        "list_all_summaries_extractive": list_all_summaries_extractive,
        "joined_gold_labels_extractive": joined_gold_labels_extractive,
    }
