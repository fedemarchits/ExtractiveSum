"""Experiment configs, grids and model/profile resolution."""
import pytest

from extsum.backends import model_aliases, resolve_model
from extsum.config import load_experiment
from extsum.grid import resolve_variants
from extsum.paths import named_config
from extsum.prompts import Cap, Shot, load_prompt_set

EXPECTED_VARIANTS = {
    ("aclsum", "aspect_full"): 54,
    ("aclsum", "aspect_large"): 18,
    ("cnndm", "document_full"): 52,
    ("billsum", "document_full"): 52,
    ("billsum", "billsum_large"): 12,
}


@pytest.mark.parametrize("experiment,grid", sorted(EXPECTED_VARIANTS))
def test_shipped_grids_resolve_to_the_published_variant_counts(experiment, grid):
    exp = load_experiment(experiment)
    variants = resolve_variants(named_config("grids", grid), load_prompt_set(exp.prompt_set))
    assert len(variants) == EXPECTED_VARIANTS[(experiment, grid)]


def test_experiment_default_grid_exists():
    for name in ("aclsum", "cnndm", "billsum"):
        assert named_config("grids", load_experiment(name).grid).is_file()


def test_empty_grid_means_every_technique_in_full(tmp_path):
    grid = tmp_path / "grid.yaml"
    grid.write_text("techniques: {}\n")
    assert len(resolve_variants(grid, load_prompt_set("aspect"))) == 54


def test_grid_cells_are_intersected_with_what_the_technique_allows(tmp_path):
    # a zero-shot negative_aware cell isn't valid -> dropped
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "techniques:\n"
        "  negative_aware:\n"
        "    - {shot: zero_shot, cap: uncapped}\n"
        "    - {shot: one_shot, cap: capped}\n"
    )
    cells = {
        (v.shot, v.cap)
        for v in resolve_variants(grid, load_prompt_set("aspect"))
        if v.technique == "negative_aware"
    }
    assert cells == {(Shot.ONE, Cap.CAPPED)}


def test_grid_naming_an_unknown_technique_fails_loudly(tmp_path):
    # tool_no_meta exists in the document sets but not in the aspect set
    grid = tmp_path / "grid.yaml"
    grid.write_text("techniques:\n  tool_no_meta: all\n")
    with pytest.raises(KeyError):
        resolve_variants(grid, load_prompt_set("aspect"))


def test_unknown_experiment_lists_the_available_ones():
    with pytest.raises(FileNotFoundError) as err:
        load_experiment("nope")
    assert "billsum" in str(err.value)


def test_profile_decides_where_local_models_run():
    assert resolve_model("qwen35_4b", profile="vast")["backend"] == "vllm"
    assert resolve_model("qwen35_4b", profile="vast")["max_model_len"] == 16384
    assert resolve_model("qwen35_4b", profile="cluster")["backend"] == "hf"


def test_model_can_override_its_profile():
    m = resolve_model("gemma4_12b", profile="cluster")
    assert (m["backend"], m["quantization"], m["max_model_len"]) == ("vllm", "bitsandbytes", 4096)
    assert "quantization" not in resolve_model("gemma4_12b", profile="vast")


def test_endpoint_models_ignore_the_profile():
    for profile in ("vast", "cluster"):
        m = resolve_model("gemma4_31b", profile=profile)
        assert m["backend"] == "endpoint" and m["endpoint_model"] == "google/gemma-4-31b-it"


def test_unknown_model_or_profile_fails_loudly():
    with pytest.raises(KeyError):
        resolve_model("nope")
    with pytest.raises(KeyError):
        resolve_model("qwen35_4b", profile="nope")


def test_every_alias_resolves():
    for alias in model_aliases():
        assert resolve_model(alias)["backend"] in ("hf", "vllm", "endpoint")
