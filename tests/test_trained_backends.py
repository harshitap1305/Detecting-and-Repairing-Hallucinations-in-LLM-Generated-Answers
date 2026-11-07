"""Integration checks for the optional Path A and Path C backends."""
from unittest.mock import patch

import pytest

from src.repair import repair
from src.schema import AnswerRecord, Claim, EvidencePassage
from src.verify import verify


def _record(label="Contradicted"):
    passage = EvidencePassage(
        doc_id="bell_0", title="Alexander Graham Bell",
        text="Alexander Graham Bell was born in Edinburgh in 1847.",
    )
    return AnswerRecord(
        id="bell", source_dataset="manual", question="Where was Bell born?",
        answer="Alexander Graham Bell was born in London in 1847.",
        claims=[Claim(
            id="bell_c1", text="Bell was born in London.", sentence_id=0,
            label=label, evidence=[passage],
        )],
    )


def test_trained_verifier_uses_checkpoint_probabilities():
    record = _record(label=None)
    with patch(
        "src.training.verifier_model.predict_verdict",
        return_value=("Contradicted", [0.01, 0.96, 0.03]),
    ) as predict:
        result = verify(record, verifier="trained", trained_model_path="/model/verifier")
    assert result.claims[0].label == "Contradicted"
    assert result.reliability_score_before == 0.0
    assert "0.960" in result.claims[0].label_rationale
    predict.assert_called_once_with(
        ["Title: Alexander Graham Bell\nAlexander Graham Bell was born in Edinburgh in 1847."],
        "Bell was born in London.", "/model/verifier",
    )


def test_trained_repair_keeps_existing_guards():
    record = _record()
    with (
        patch("src.training.repair_model.generate_edit", return_value="Alexander Graham Bell was born in Edinburgh in 1847.") as generate,
        patch("src.repair.validate_repair_guards", return_value=(True, "")) as guards,
        patch("src.repair.llm.call") as prompted,
    ):
        result = repair(record, backend="trained", trained_model_path="/model/repair")
    assert result.repaired_answer == "Alexander Graham Bell was born in Edinburgh in 1847."
    generate.assert_called_once()
    guards.assert_called_once()
    prompted.assert_not_called()


def test_trained_repair_cannot_delete_under_hedge_policy():
    record = _record()
    with patch("src.training.repair_model.generate_edit", return_value="REMOVE"):
        result = repair(record, backend="trained", trained_model_path="/model/repair")
    assert result.repaired_answer.startswith("Reportedly,")


def test_trained_backends_require_checkpoint_paths():
    with pytest.raises(ValueError, match="trained_model_path"):
        verify(_record(), verifier="trained")
    with pytest.raises(ValueError, match="trained_model_path"):
        repair(_record(), backend="trained")
