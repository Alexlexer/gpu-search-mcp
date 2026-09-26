from __future__ import annotations

import json
from pathlib import Path

from gpu_service.decision_model import (
    DecisionRequest, DeterministicDecisionModel, LocalDecisionModel,
)
from gpu_service import server_config


def _request() -> DecisionRequest:
    return DecisionRequest("fix service", "fix service", (
        {"id": "candidate-0", "path": "Service.cs", "score": 1.0},
        {"id": "candidate-1", "path": "Tests.cs", "score": 0.8},
    ), 8000)


def test_deterministic_decisions_are_replayable() -> None:
    model = DeterministicDecisionModel()
    first = model.choose_evidence(_request())
    assert first == model.choose_evidence(_request())
    assert first.selected_ids == ("candidate-0", "candidate-1")
    assert model.choose_next_action(_request()).decision == "BUILD_CONTEXT"


def test_deterministic_fixture_replay() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "decision_request.json").read_text())
    request = DecisionRequest(
        fixture["task"], fixture["query"], tuple(fixture["candidates"]), fixture["token_budget"]
    )
    result = DeterministicDecisionModel().choose_evidence(request)
    assert result.decision == fixture["expected"]["decision"]
    assert list(result.selected_ids) == fixture["expected"]["selected_ids"]


def test_local_adapter_rejects_unknown_candidate_ids(monkeypatch) -> None:
    class Response:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({
                "decision": "SELECT_EVIDENCE", "confidence": 0.99,
                "selected_ids": ["invented-path"],
            })}}]}).encode()

    monkeypatch.setattr("gpu_service.decision_model.urlopen", lambda *args, **kwargs: Response())
    result = LocalDecisionModel("http://localhost:1234/v1", "test").choose_evidence(_request())
    assert result.fallback_used is True
    assert result.reason_code == "INVALID_RESPONSE"


def test_local_adapter_accepts_bounded_json_response(monkeypatch) -> None:
    class Response:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({
                "decision": "BUILD_CONTEXT", "confidence": 0.91,
            })}}]}).encode()

    monkeypatch.setattr("gpu_service.decision_model.urlopen", lambda *args, **kwargs: Response())
    result = LocalDecisionModel("http://localhost:1234/v1", "test").choose_next_action(_request())
    assert result.decision == "BUILD_CONTEXT"
    assert result.fallback_used is False


def test_settings_save_never_persists_api_key(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.json"
    monkeypatch.setattr(server_config, "CONFIG_PATH", path)
    saved = server_config.save_decision_model_config(
        "local", "http://localhost:1234/v1", "small-model", 0.8
    )
    persisted = json.loads(path.read_text())
    assert saved["provider"] == "local"
    assert persisted["decisionModel"] == {
        "provider": "local", "baseUrl": "http://localhost:1234/v1",
        "model": "small-model", "confidenceThreshold": 0.8,
    }
    assert "api_key" not in persisted["decisionModel"]
