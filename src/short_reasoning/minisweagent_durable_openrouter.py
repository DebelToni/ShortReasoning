"""Durable OpenRouter transport adapter for upstream mini-SWE-agent."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from functools import partial
from pathlib import Path
from typing import Any

from minisweagent.models.openrouter_model import OpenRouterModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from . import (
    BudgetFloorReached,
    ChatAttemptError,
    openrouter_key,
    remaining_openrouter_credits,
    rewrite_fidelity,
)
from .durable import DurableChat, DurableChatError

_RESERVATION_LOCK = threading.Lock()
_IN_FLIGHT = 0

_COMPACTION_SYSTEM = """Rewrite one historical assistant reasoning block into a shorter faithful state record for the same coding agent. Output only the state record as plain text, without labels, preamble, JSON, or fences.

Preserve every decision-relevant observation, file or symbol name, number, conclusion, correction, unresolved uncertainty, rejected alternative that may matter later, and the intended next action. The pending tool result is unavailable: never predict it. Remove repeated commands, narration, rhetoric, headings, and redundant restatement. Prefer compact clauses. Fidelity outranks brevity, but the result must be shorter than the source."""


def concurrent_guarded_post_raw(
    payload: dict[str, Any],
    *,
    key_variable: str = "OPENROUTER_API_KEY",
    floor_variable: str = "OPENROUTER_MIN_BALANCE_USD",
    max_call_cost_variable: str = "OPENROUTER_MAX_CALL_COST_USD",
    forbid_paid_variable: str = "OPENROUTER_FORBID_PAID_CALLS",
) -> tuple[bytes, float]:
    """Send one request while reserving enough balance for concurrent calls."""
    forbid_paid = os.environ.get(forbid_paid_variable, "").lower() in {
        "1",
        "true",
        "yes",
    }
    if forbid_paid and not str(payload.get("model", "")).endswith(":free"):
        raise BudgetFloorReached(
            f"paid OpenRouter model forbidden by {forbid_paid_variable}"
        )
    floor = float(os.environ.get(floor_variable, "2.0"))
    max_call_cost = float(os.environ.get(max_call_cost_variable, "0.12"))
    global _IN_FLIGHT
    with _RESERVATION_LOCK:
        remaining = remaining_openrouter_credits(key_variable)
        if remaining - max_call_cost * (_IN_FLIGHT + 1) <= floor:
            raise BudgetFloorReached(
                f"OpenRouter balance ${remaining:.4f} cannot preserve the ${floor:.2f} "
                f"floor with {_IN_FLIGHT + 1} concurrent ${max_call_cost:.2f} reservations"
            )
        _IN_FLIGHT += 1

    started = time.perf_counter()
    try:
        body = json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {openrouter_key(key_variable)}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local.short-reasoning/",
                "X-Title": "ShortReasoning mini-SWE-agent clean baseline",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.read(), time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        response_body = exc.read()
        status = exc.code
        exc.close()
        raise ChatAttemptError(
            f"OpenRouter HTTP {status}: {response_body.decode(errors='replace')}",
            wall_seconds=time.perf_counter() - started,
            http_status=status,
            response_body_bytes=response_body,
        ) from exc
    except Exception as exc:
        if isinstance(exc, BudgetFloorReached):
            raise
        raise ChatAttemptError(
            f"{type(exc).__name__}: {exc}",
            wall_seconds=time.perf_counter() - started,
        ) from exc
    finally:
        with _RESERVATION_LOCK:
            _IN_FLIGHT -= 1


class DurableOpenRouterModel(OpenRouterModel):
    """Use mini-SWE-agent's native OpenRouter model with durable raw transport."""

    abort_exceptions = [DurableChatError, KeyboardInterrupt]

    def __init__(self, **kwargs: Any) -> None:
        model_kwargs = copy.deepcopy(kwargs.get("model_kwargs", {}))
        self._drop_reasoning_details = bool(
            model_kwargs.pop("short_reasoning_drop_reasoning_details", False)
        )
        self._drop_content_with_reasoning = bool(
            model_kwargs.pop("short_reasoning_drop_content_with_reasoning", False)
        )
        self._history_representation = model_kwargs.pop(
            "short_reasoning_history_representation", "native_reasoning"
        )
        kwargs["model_kwargs"] = model_kwargs
        super().__init__(**kwargs)
        root = Path(os.environ["SHORT_REASONING_DURABLE_JOURNAL_ROOT"])
        session = os.environ["SHORT_REASONING_DURABLE_SESSION"]
        self._journal_dir = root / session / f"model-{uuid.uuid4().hex}"
        self._durable_chat = DurableChat(
            self._journal_dir,
            max_returned_retries=2,
            network=concurrent_guarded_post_raw,
        )

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        prepared_messages = copy.deepcopy(messages)
        for message in prepared_messages:
            if self._drop_reasoning_details:
                message.pop("reasoning_details", None)
            if self._drop_content_with_reasoning and message.get("reasoning"):
                message["content"] = None
            if self._history_representation == "tagged_visible":
                raw_reasoning = message.pop("reasoning", None)
                if isinstance(raw_reasoning, str) and raw_reasoning.strip():
                    state = raw_reasoning.strip()
                    existing = message.get("content")
                    suffix = existing if isinstance(existing, str) else ""
                    message["content"] = (
                        "<compact_reasoning_state>\n"
                        + state
                        + "\n</compact_reasoning_state>"
                        + ("\n" + suffix if suffix else "")
                    )
            elif self._history_representation != "native_reasoning":
                raise ValueError(
                    f"unknown history representation: {self._history_representation}"
                )
        payload = {
            "model": self.config.model_name,
            "messages": prepared_messages,
            "tools": [BASH_TOOL],
            "usage": {"include": True},
            **(self.config.model_kwargs | kwargs),
        }
        response, _ = self._durable_chat(payload)
        if not isinstance(response.get("choices"), list):
            raise ValueError(
                "OpenRouter returned a response without choices: "
                + json.dumps(response.get("error"), ensure_ascii=False)
            )
        return response

    def serialize(self) -> dict[str, Any]:
        result = super().serialize()
        result["info"]["config"]["durable_journal_dir"] = str(self._journal_dir)
        return result


class ContinuousCompactingOpenRouterModel(DurableOpenRouterModel):
    """Compact each prior reasoning block once before subsequent target calls."""

    def __init__(self, **kwargs: Any) -> None:
        model_kwargs = copy.deepcopy(kwargs.get("model_kwargs", {}))
        profile = model_kwargs.pop("short_reasoning_compaction", None)
        if not isinstance(profile, dict):
            raise ValueError("short_reasoning_compaction profile is required")
        kwargs["model_kwargs"] = model_kwargs
        super().__init__(**kwargs)
        self._compaction_profile = profile
        self._compaction_cache: dict[str, str] = {}
        self._compaction_events: list[dict[str, Any]] = []
        self._compaction_attempts: list[dict[str, Any]] = []
        self._compaction_cost = 0.0
        compressor_key_variable = os.environ.get(
            "SHORT_REASONING_COMPRESSOR_OPENROUTER_KEY_ENV"
        )
        compressor_network = concurrent_guarded_post_raw
        if compressor_key_variable:
            compressor_network = partial(
                concurrent_guarded_post_raw,
                key_variable=compressor_key_variable,
                floor_variable="OPENROUTER_COMPRESSOR_MIN_BALANCE_USD",
                max_call_cost_variable="OPENROUTER_COMPRESSOR_MAX_CALL_COST_USD",
                forbid_paid_variable="OPENROUTER_COMPRESSOR_FORBID_PAID_CALLS",
            )
        self._compressor_chat = DurableChat(
            self._journal_dir / "compressor",
            max_returned_retries=2,
            network=compressor_network,
        )

    @staticmethod
    def _action_name(message: dict[str, Any]) -> str | None:
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return None
        function = calls[0].get("function") if isinstance(calls[0], dict) else None
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        return name if isinstance(name, str) else None

    def _compressor_profile(self) -> dict[str, Any]:
        profile = self._compaction_profile
        if profile.get("mode") == "self":
            return {
                "model": self.config.model_name,
                "provider": copy.deepcopy(self.config.model_kwargs.get("provider")),
                "reasoning_effort": profile["reasoning_effort"],
                "expected_provider": profile["expected_provider"],
            }
        return {
            "model": profile["model"],
            "provider": copy.deepcopy(profile["provider"]),
            "reasoning_effort": profile["reasoning_effort"],
            "expected_provider": profile["expected_provider"],
        }

    @staticmethod
    def _plain_state(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("compressor response must have one choice")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("compressor response has no plain text state")
        state = content.strip()
        if state.startswith("```") or state.endswith("```"):
            raise ValueError("compressor response used a code fence")
        return state

    @staticmethod
    def _route_matches(response: dict[str, Any], profile: dict[str, Any]) -> bool:
        returned_model = response.get("model")
        returned_provider = response.get("provider")
        model_ok = isinstance(returned_model, str) and (
            returned_model == profile["model"]
            or returned_model.startswith(profile["model"] + "-")
        )
        normalize = lambda value: "".join(  # noqa: E731
            character for character in str(value).lower() if character.isalnum()
        )
        return model_ok and normalize(returned_provider) == normalize(
            profile["expected_provider"]
        )

    def _compact(self, raw: str, action_name: str | None) -> str:
        key = hashlib.sha256(
            json.dumps(
                {"raw": raw, "action_name": action_name},
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if key in self._compaction_cache:
            return self._compaction_cache[key]

        raw_words = len(raw.split())
        if raw_words <= 12:
            self._compaction_cache[key] = raw
            self._compaction_events.append(
                {
                    "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "state_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "raw_words": raw_words,
                    "state_words": raw_words,
                    "word_ratio": 1.0,
                    "missing_identifiers": [],
                    "missing_numeric_literals": [],
                    "requested_action_name": action_name,
                    "compressor_model": None,
                    "compressor_provider": None,
                    "provider_reported_cost_usd": 0.0,
                    "compacted": False,
                    "identity_reason": "short_source",
                }
            )
            return raw

        profile = self._compressor_profile()
        payload = {
            "model": profile["model"],
            "messages": [
                {"role": "system", "content": _COMPACTION_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "raw_reasoning": raw,
                            "requested_action_name": action_name,
                            "requested_action_arguments_preserved_outside_reasoning": True,
                            "pending_result": "withheld",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "reasoning": {
                "effort": profile["reasoning_effort"],
                "exclude": True,
            },
            "include_reasoning": False,
            "provider": profile["provider"],
            "usage": {"include": True},
        }
        response, _ = self._compressor_chat(payload)
        cost = float(response.get("usage", {}).get("cost") or 0)
        self._compaction_cost += cost
        attempt_record = {
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "compressor_model": response.get("model"),
            "compressor_provider": response.get("provider"),
            "provider_reported_cost_usd": cost,
            "accepted": False,
        }
        self._compaction_attempts.append(attempt_record)
        if not self._route_matches(response, profile):
            attempt_record["error"] = "wrong_route"
            raise ValueError(
                "compressor returned wrong route: "
                f"{response.get('model')!r} / {response.get('provider')!r}"
            )
        try:
            state = self._plain_state(response)
        except ValueError as exc:
            attempt_record["error"] = str(exc)
            raise
        state_words = len(state.split())
        if state_words >= raw_words:
            attempt_record["error"] = "not_shorter"
            state = raw
            state_words = raw_words
            identity_reason = "returned_state_not_shorter"
        else:
            attempt_record["accepted"] = True
            identity_reason = None
        fidelity = rewrite_fidelity(raw, state)
        self._compaction_events.append(
            {
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                "raw_words": raw_words,
                "state_words": state_words,
                "word_ratio": state_words / raw_words,
                "missing_identifiers": fidelity["missing_identifiers"],
                "missing_numeric_literals": fidelity["missing_numeric_literals"],
                "requested_action_name": action_name,
                "compressor_model": response.get("model"),
                "compressor_provider": response.get("provider"),
                "provider_reported_cost_usd": cost,
                "compacted": identity_reason is None,
                "identity_reason": identity_reason,
            }
        )
        self._compaction_cache[key] = state
        return state

    def _compacted_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        result = copy.deepcopy(messages)
        representation = self._compaction_profile["representation"]
        for message in result:
            if message.get("role") != "assistant":
                continue
            raw = message.get("reasoning")
            if not isinstance(raw, str) or not raw.strip():
                continue
            state = self._compact(raw.strip(), self._action_name(message))
            message.pop("reasoning_details", None)
            if representation == "native_reasoning":
                message["reasoning"] = state
            elif representation == "tagged_visible":
                message.pop("reasoning", None)
                existing = message.get("content")
                suffix = existing if isinstance(existing, str) else ""
                message["content"] = (
                    "<compact_reasoning_state>\n"
                    + state
                    + "\n</compact_reasoning_state>"
                    + ("\n" + suffix if suffix else "")
                )
            else:
                raise ValueError(f"unknown compaction representation: {representation}")
        return result

    def _query(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        return super()._query(self._compacted_messages(messages), **kwargs)

    def serialize(self) -> dict[str, Any]:
        result = super().serialize()
        result["info"]["compaction"] = {
            "profile": self._compaction_profile,
            "unique_blocks": len(self._compaction_cache),
            "provider_reported_cost_usd": self._compaction_cost,
            "attempts": self._compaction_attempts,
            "events": self._compaction_events,
        }
        return result
