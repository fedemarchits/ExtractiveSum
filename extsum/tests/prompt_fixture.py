"""Fixed inputs for the prompt golden test (see test_prompt_golden.py)."""
from __future__ import annotations

SENTENCES = [
    "The committee met on Tuesday to review the proposal.",
    "It would raise the fuel levy by two cents per litre.",
    "Opponents argue the measure hurts rural commuters.",
    "A similar plan was rejected in 2019.",
    "The revenue is earmarked for road maintenance.",
    "Members also discussed the agenda for next month.",
    "A final vote is expected before the summer recess.",
    "The minister declined to comment on the timing.",
]

EXEMPLAR_SENTENCES = [
    "Researchers trained a model on ten thousand annotated abstracts.",
    "Prior systems required hand-written rules for every domain.",
    "The new approach learns the rules from examples instead.",
    "It improved accuracy by four points on the benchmark.",
    "Funding was provided by a national science agency.",
]
EXEMPLAR_GOLD = [2, 3, 4]
EXEMPLAR_NEAR_MISS = [1, 5]

CAP_K = 3

#: prompt set -> (rationale cache it renders with, units). The document set
#: uses a frozen copy kept with the tests, so rebuilding an experiment's traces
#: does not change what the golden hashes are checked against.
SETS = {
    "aspect": ("data/rationales/aclsum", ("challenge", "approach", "outcome")),
    "document": ("tests/fixtures/rationales_document", ("summary",)),
    "bill": ("data/rationales/billsum", ("summary",)),
}
