"""Answer parsing: JSON extraction, index cleaning and capping."""
from extsum.postprocess import (
    cap_indices_in_order, clean_indices, parse_selection, safe_extract_json,
)


def test_json_direct_and_fenced():
    assert safe_extract_json('{"selected_sentences": [1,2]}')["selected_sentences"] == [1, 2]
    fenced = "```json\n{\"selected_sentences\": [3]}\n```"
    assert safe_extract_json(fenced)["selected_sentences"] == [3]


def test_json_last_object_and_bare_array():
    txt = "noise {\"x\":1} then {\"selected_sentences\": [2, 4]}"
    assert safe_extract_json(txt)["selected_sentences"] == [2, 4]
    assert safe_extract_json("here:\n[1, 5]\n")["selected_sentences"] == [1, 5]


def test_clean_and_cap():
    assert clean_indices([2, 2, 1, 9, 0, "x"], n_sentences=5) == [1, 2]
    assert cap_indices_in_order([1, 2, 3, 4], 2) == [1, 2]
    assert cap_indices_in_order([1, 2, 3], None) == [1, 2, 3]


def test_indices_given_as_strings_or_sentence_labels_are_accepted():
    # gemma-4-E2B answers this way; both name a sentence number unambiguously.
    assert clean_indices(["1", " 13 "], n_sentences=20) == [1, 13]
    assert clean_indices(["Sentence 3", "sentence 7"], n_sentences=20) == [3, 7]


def test_sentence_text_and_bools_are_not_indices():
    assert clean_indices(["The committee met on Tuesday.", True, None, "3rd"], 20) == []


def test_parse_selection_end_to_end():
    out = 'The answer is {"selected_sentences": [1, 3, 7, 3]}'
    assert parse_selection(out, n_sentences=5, cap=None) == [1, 3]
    assert parse_selection(out, n_sentences=5, cap=1) == [1]


def test_parse_selection_garbage_is_empty():
    assert parse_selection("no json here", n_sentences=4) == []
