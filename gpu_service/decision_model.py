"""Bounded, optional decision backends for context planning.

Decision backends receive only retrieval metadata.  They cannot create paths or
evidence; callers validate every returned candidate id and retain deterministic
fallback behaviour.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen


NEXT_ACTIONS = frozenset({"CONTINUE_RETRIEVAL", "BUILD_CONTEXT", "ESCALATE"})


@dataclass(frozen=True, slots=True)
class DecisionResult:
    decision: str
    confidence: float
    provider: str
    model: str | None
    fallback_used: bool
    reason_code: str
    latency_ms: float
    selected_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "decision": self.decision, "confidence": self.confidence,
            "provider": self.provider, "model": self.model,
            "fallback_used": self.fallback_used, "reason_code": self.reason_code,
            "latency_ms": self.latency_ms, "selected_ids": list(self.selected_ids),
        }


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    task: str
    query: str
    candidates: tuple[dict, ...]
    token_budget: int


class DecisionModel(Protocol):
    """Small interface deliberately limited to bounded control decisions."""
    provider: str

    def choose_evidence(self, request: DecisionRequest) -> DecisionResult: ...
    def choose_next_action(self, request: DecisionRequest) -> DecisionResult: ...
    def score_retention(self, request: DecisionRequest) -> dict[str, float]: ...


class DeterministicDecisionModel:
    provider = "deterministic"

    def choose_evidence(self, request: DecisionRequest) -> DecisionResult:
        # Existing planner ordering remains authoritative; IDs are telemetry.
        return DecisionResult("SELECT_EVIDENCE", 1.0, self.provider, None, False,
                              "DETERMINISTIC_RANKING", 0.0,
                              tuple(item["id"] for item in request.candidates))

    def choose_next_action(self, request: DecisionRequest) -> DecisionResult:
        action = "BUILD_CONTEXT" if request.candidates else "CONTINUE_RETRIEVAL"
        return DecisionResult(action, 1.0, self.provider, None, False,
                              "DETERMINISTIC_SUFFICIENT_EVIDENCE", 0.0)

    def score_retention(self, request: DecisionRequest) -> dict[str, float]:
        return {item["id"]: float(item.get("score", 0.0)) for item in request.candidates}


class LocalDecisionModel:
    """OpenAI-compatible chat-completions adapter using JSON-schema responses."""
    provider = "local"

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout_seconds: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _request(self, request: DecisionRequest, kind: str, choices: list[str]) -> DecisionResult:
        started = time.perf_counter()
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "decision": {"type": "string", "enum": choices},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "selected_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            }, "required": ["decision", "confidence"],
        }
        payload = {"model": self.model, "temperature": 0,
            "messages": [{"role": "system", "content": "Return only the supplied JSON schema. Never invent evidence IDs."},
                         {"role": "user", "content": json.dumps({"kind": kind, "state": asdict(request)}, separators=(",", ":"))}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "bounded_decision", "strict": True, "schema": schema}}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            raw = urlopen(Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(), headers=headers, method="POST"), timeout=self.timeout_seconds).read()
            content = json.loads(raw)["choices"][0]["message"]["content"]
            value = json.loads(content)
            decision, confidence = value["decision"], float(value["confidence"])
            if decision not in choices or not 0 <= confidence <= 1:
                raise ValueError("invalid bounded decision")
            allowed = {item["id"] for item in request.candidates}
            selected = tuple(item for item in value.get("selected_ids", []) if item in allowed)
            if any(item not in allowed for item in value.get("selected_ids", [])):
                raise ValueError("unknown candidate id")
            return DecisionResult(decision, confidence, self.provider, self.model, False,
                                  "MODEL_RESPONSE", round((time.perf_counter() - started) * 1000, 3), selected)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, URLError, OSError) as exc:
            return DecisionResult("INVALID", 0.0, self.provider, self.model, True,
                                  "INVALID_RESPONSE", round((time.perf_counter() - started) * 1000, 3))

    def choose_evidence(self, request: DecisionRequest) -> DecisionResult:
        return self._request(request, "select evidence ids to expand first", ["SELECT_EVIDENCE"])

    def choose_next_action(self, request: DecisionRequest) -> DecisionResult:
        return self._request(request, "choose the next retrieval control action", sorted(NEXT_ACTIONS))

    def score_retention(self, request: DecisionRequest) -> dict[str, float]:
        # Retention is intentionally advisory until an explicit destructive mode exists.
        return DeterministicDecisionModel().score_retention(request)


class TypeSafeJevDecisionModel:
    """Minimal stdlib adapter for TypeSafe's documented /v1/systemone API."""
    provider = "typesafe"

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.typesafe.ai",
                 timeout_seconds: float = 3.0) -> None:
        self.model, self.api_key = model, api_key
        self.base_url, self.timeout_seconds = base_url.rstrip("/"), timeout_seconds

    def _choice(self, request: DecisionRequest, name: str, criteria: dict) -> DecisionResult:
        started = time.perf_counter()
        try:
            payload = {"model": self.model, "state": {"task": request.task, "query": request.query, "candidates": request.candidates, "token_budget": request.token_budget},
                       "questions": {name: {"type": "choice", "instructions": "Choose only from the provided criteria.", "criteria": criteria}}}
            raw = urlopen(Request(self.base_url + "/v1/systemone", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}, method="POST"), timeout=self.timeout_seconds).read()
            response = json.loads(raw)
            answer = response["answers"][name]
            choice, confidence = answer["choice"], float(answer["confidence"])
            if choice not in criteria or not 0 <= confidence <= 1:
                raise ValueError("invalid TypeSafe response")
            return DecisionResult(choice, confidence, self.provider, response.get("model", self.model), False,
                                  "TYPESAFE_CHOICE", round((time.perf_counter() - started) * 1000, 3),
                                  (choice,) if choice.startswith("candidate-") else ())
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, URLError, OSError):
            return DecisionResult("INVALID", 0.0, self.provider, self.model, True,
                                  "INVALID_RESPONSE", round((time.perf_counter() - started) * 1000, 3))

    def choose_evidence(self, request: DecisionRequest) -> DecisionResult:
        criteria = {item["id"]: "Retrieved repository evidence." for item in request.candidates[:255]}
        return self._choice(request, "evidence", criteria) if criteria else DeterministicDecisionModel().choose_evidence(request)

    def choose_next_action(self, request: DecisionRequest) -> DecisionResult:
        return self._choice(request, "action", {action: action.replace("_", " ").title() for action in NEXT_ACTIONS})

    def score_retention(self, request: DecisionRequest) -> dict[str, float]:
        return DeterministicDecisionModel().score_retention(request)
