"""Safety and reproducibility checks for Path C preparation and evaluation."""
import json

import pytest

from src.training.repair_data import (
    article_split, filter_verified_candidates, generate_candidates, is_held_out,
    load_holdouts,
)
from src.training.repair_model import format_repair_prompt, generate_edit
from src.training.repair_training import (
    assert_disjoint_splits, evaluate_predictions, example_prompt, token_f1,
)
from scripts.build_repair_data import main as build_repair_main


def _source(article="Ada_Lovelace", sentence="Ada Lovelace was born in 1815 in London."):
    return {"article_id": article, "source_id": article + ":0", "sentence": sentence,
            "evidence": f"{article.replace('_', ' ')}. {sentence}"}


def _holdouts():
    return {"article_ids": {"unrelated article"}, "entities": {"grace hopper"},
            "prompt_ids": set(), "prompts": set()}


def test_holdout_entities_block_context_and_empty_manifest_fails(tmp_path):
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps({"entities": []}))
    with pytest.raises(ValueError, match="empty"):
        load_holdouts(holdout_path)
    holdout_path.write_text(json.dumps({"entities": ["Grace Hopper"]}))
    holdouts = load_holdouts(holdout_path)
    assert is_held_out(_source(article="Grace_Hopper"), holdouts)
    assert is_held_out(_source(sentence="Ada Lovelace met Grace Hopper in 1815."), holdouts)


def test_candidates_are_reproducible_and_article_split_stable():
    first = list(generate_candidates([_source()], _holdouts(), seed=42))
    second = list(generate_candidates([_source()], _holdouts(), seed=42))
    assert first == second
    assert {r["split"] for r in first} == {article_split("Ada Lovelace", 42)}
    assert {r["error_type"] for r in first} & {"year_shift", "negation"}
    assert first[0]["sentence"] == first[0]["original_sentence"]


def test_unverified_or_mislabelled_corruption_never_enters_training():
    rows = list(generate_candidates([_source()], _holdouts(), seed=42))
    split = rows[0]["split"]
    kept, counts = filter_verified_candidates(
        rows, lambda row: ("Supported", "Supported"), unchanged_fraction=0.2
    )
    assert kept[split] == []
    assert counts["corruption_not_verified"] == 1
    kept, counts = filter_verified_candidates(
        rows, lambda row: ("Supported", "Supported" if row["error_type"] == "unchanged" else "Contradicted"),
        unchanged_fraction=0.5,
    )
    assert len(kept[split]) == 2
    assert {r["error_type"] for r in kept[split]} == {"unchanged", rows[1]["error_type"]}
    assert all(r["target"] == rows[0]["original_sentence"] for r in kept[split])
    with pytest.raises(ValueError, match="Invalid verifier"):
        filter_verified_candidates(rows, lambda _: ("Supported", "unknown"))


def test_training_prompt_and_inference_short_circuit(tmp_path):
    row = {"sentence": "Ada was born in 1816.", "flagged_claim": "Ada was born in 1816.",
           "flagged_label": "Contradicted", "evidence": [{"title": "Ada", "text": "Ada was born in 1815."}]}
    prompt = example_prompt(row)
    assert "Contradicted" in prompt and "Ada was born in 1815" in prompt
    assert prompt == format_repair_prompt(row["sentence"],
                                          [{"text": row["flagged_claim"], "label": row["flagged_label"]}],
                                          row["evidence"])
    assert generate_edit("All good.", [], [], str(tmp_path / "missing")) == "All good."
    with pytest.raises(FileNotFoundError):
        generate_edit(row["sentence"], [{"text": row["flagged_claim"]}], row["evidence"],
                      str(tmp_path / "missing"))


def test_split_leakage_checks_article_and_prompt():
    train = [{"article_id": "Ada", "prompt_id": "p1", "sentence": "Ada was born in 1815."}]
    with pytest.raises(ValueError, match="article_id"):
        assert_disjoint_splits(train, [{"article_id": "ada", "sentence": "different"}])
    with pytest.raises(ValueError, match="prompt_id"):
        assert_disjoint_splits(train, [{"article_id": "Other", "prompt_id": "p1", "sentence": "different"}])


def test_exact_and_token_metrics_are_distinct():
    rows = [{"target": "Ada was born in 1815.", "sentence": "Ada was born in 1815.",
             "error_type": "unchanged"}]
    assert token_f1(rows[0]["target"], "Ada was born in 1816.") > 0.5
    scores = evaluate_predictions(rows, ["Ada was born in 1816."])
    assert scores["exact_match"] == 0
    assert scores["unchanged_accuracy"] == 0


def test_builder_cli_filters_only_verifier_approved_rows(tmp_path):
    sources = tmp_path / "sources.jsonl"
    sources.write_text(json.dumps(_source()) + "\n", encoding="utf-8")
    holdouts = tmp_path / "holdouts.json"
    holdouts.write_text(json.dumps({"entities": ["Grace Hopper"]}), encoding="utf-8")
    candidates = tmp_path / "candidates.jsonl"
    assert build_repair_main(["candidates", "--sources", str(sources),
                              "--holdout-manifest", str(holdouts),
                              "--output", str(candidates)]) == 0
    rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text("\n".join(json.dumps({
        "id": row["id"], "original_label": "Supported",
        "candidate_label": "Supported" if row["error_type"] == "unchanged" else "Contradicted",
    }) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    assert build_repair_main(["filter", "--candidates", str(candidates),
                              "--verdicts", str(verdicts),
                              "--output-dir", str(output),
                              "--unchanged-fraction", "0.5"]) == 0
    result_file = output / f"repair_{rows[0]['split']}.jsonl"
    examples = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines()]
    assert len(examples) == 2
    assert {row["error_type"] for row in examples} == {"unchanged", rows[1]["error_type"]}


def test_wiki_candidates_require_article_or_entity_holdouts(tmp_path, capsys):
    wiki = tmp_path / "wiki-pages"
    wiki.mkdir()
    holdouts = tmp_path / "holdouts.json"
    holdouts.write_text(json.dumps({"prompt_ids": ["held-out-prompt"]}), encoding="utf-8")
    output = tmp_path / "candidates.jsonl"

    with pytest.raises(SystemExit) as exc:
        build_repair_main(["candidates", "--wiki-pages-dir", str(wiki),
                           "--holdout-manifest", str(holdouts),
                           "--output", str(output)])

    assert exc.value.code == 2
    assert "requires article_ids or entities" in capsys.readouterr().err
    assert not output.exists()
