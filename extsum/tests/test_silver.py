"""Segmentation and the local_score labelling heuristic."""
from extsum.data.segmentation import bill_sentence_split, sentence_split
from extsum.silver.labelling import local_score_select

BILL = """SECTION 1. SHORT TITLE.

    This Act may be cited as the ``Fuel Levy Act''.

SEC. 2. SUSPENSION OF TAX.

    (a) In General.--The tax imposed by section 4081 is suspended.
    (b) Period.--The suspension applies for 90 days after enactment.
"""


def test_bill_headers_attach_to_the_following_sentence():
    units = bill_sentence_split(BILL)
    assert not any(u.strip().rstrip(".") in ("SECTION 1. SHORT TITLE", "SEC. 2. SUSPENSION OF TAX") for u in units)
    assert any("SHORT TITLE" in u and "Fuel Levy Act" in u for u in units)
    assert sum("suspended" in u for u in units) == 1
    assert sum("90 days" in u for u in units) == 1


def test_only_billsum_uses_the_bill_splitter():
    assert sentence_split(BILL, dataset_name="billsum") == bill_sentence_split(BILL)
    news = "Rates rose on Tuesday. Analysts expected it. Markets fell."
    assert sentence_split(news, dataset_name="cnndm") == [
        "Rates rose on Tuesday.", "Analysts expected it.", "Markets fell.",
    ]


def test_local_score_picks_the_top_k_in_document_order():
    sentences = [
        "The weather was mild.",
        "The bank raised interest rates.",
        "Shoppers visited the market.",
        "Interest rates may rise again next year.",
    ]
    picked = local_score_select(sentences, "The bank raised interest rates and may raise them again.", 2)
    assert picked == [2, 4]
