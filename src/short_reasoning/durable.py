"""Exactly-once durable transport for paid chat calls."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from . import (
    BudgetFloorReached,
    ChatAttemptError,
    post_chat_once_raw,
    strict_json_loads,
)

RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
TERMINAL_FAILURES = frozenset(
    {"ambiguous_unreturned", "budget_floor", "invalid_response", "returned_http_error"}
)


class DurableChatError(RuntimeError):
    """A durable logical call ended without a usable response."""


class AmbiguousRequest(DurableChatError):
    """A persisted request has no known returned body and may not be resent."""


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def returned_http_error_is_retryable(http_status: Any, raw_body: bytes) -> bool:
    if http_status not in RETRYABLE_HTTP_STATUSES or not raw_body:
        return False
    try:
        response = strict_json_loads(raw_body)
    except (ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(response, dict)
        and isinstance(response.get("error"), dict)
        and response["error"].get("code") == http_status
        and "choices" not in response
    )


class DurableChat:
    """Persist each request and raw response before exposing parsed content."""

    def __init__(
        self,
        directory: Path,
        *,
        max_returned_retries: int = 2,
        network: Callable[[dict[str, Any]], tuple[bytes, float]] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_returned_retries < 0:
            raise ValueError("max_returned_retries must be nonnegative")
        self.directory = directory
        self.max_attempts = max_returned_retries + 1
        self.network = network
        self.sleeper = sleeper
        self.call_index = 0

    def __call__(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        self.call_index += 1
        call_dir = self.directory / f"call-{self.call_index:03d}"
        request_hash = canonical_hash(payload)
        metadata_path = call_dir / "request.json"
        if metadata_path.exists():
            metadata = strict_json_loads(metadata_path.read_bytes())
            if metadata.get("request_sha256") != request_hash or metadata.get("request") != payload:
                raise DurableChatError(f"persisted request changed: {call_dir}")
        else:
            write_json_atomic(
                metadata_path,
                {
                    "schema_version": 1,
                    "request": payload,
                    "request_sha256": request_hash,
                    "max_attempts": self.max_attempts,
                },
            )

        for attempt_index in range(1, self.max_attempts + 1):
            attempt_path = call_dir / f"attempt-{attempt_index:02d}.json"
            if attempt_path.exists():
                record = strict_json_loads(attempt_path.read_bytes())
                response = self._resume_attempt(attempt_path, record)
                if response is not None:
                    return response, self._total_wall(call_dir, attempt_index)
                if record.get("status") == "returned_http_error" and attempt_index < self.max_attempts:
                    continue
                raise DurableChatError(
                    f"logical call failed with {record.get('status')}: {call_dir}"
                )

            record = {
                "schema_version": 1,
                "attempt": attempt_index,
                "status": "request_persisted",
                "request_sha256": request_hash,
            }
            write_json_atomic(attempt_path, record)
            network = self.network or post_chat_once_raw
            try:
                raw_body, wall_seconds = network(payload)
            except BudgetFloorReached as exc:
                record.update(status="budget_floor", error=str(exc))
                write_json_atomic(attempt_path, record)
                raise DurableChatError(str(exc)) from exc
            except ChatAttemptError as exc:
                record["wall_seconds"] = exc.wall_seconds
                record["error"] = str(exc)
                if exc.response_body_bytes is None:
                    record["status"] = "ambiguous_unreturned"
                    write_json_atomic(attempt_path, record)
                    raise AmbiguousRequest(str(exc)) from exc
                self._store_raw(record, exc.response_body_bytes)
                record["status"] = "returned_http_error"
                record["http_status"] = exc.http_status
                record["retryable"] = returned_http_error_is_retryable(
                    exc.http_status, exc.response_body_bytes
                )
                write_json_atomic(attempt_path, record)
                if record["retryable"] and attempt_index < self.max_attempts:
                    self.sleeper(float(attempt_index))
                    continue
                raise DurableChatError(str(exc)) from exc

            record["status"] = "response_received"
            record["wall_seconds"] = wall_seconds
            self._store_raw(record, raw_body)
            write_json_atomic(attempt_path, record)
            response = self._resume_attempt(attempt_path, record)
            if response is None:
                raise DurableChatError(f"response parsing failed: {call_dir}")
            return response, self._total_wall(call_dir, attempt_index)
        raise AssertionError("unreachable")

    @staticmethod
    def _store_raw(record: dict[str, Any], raw_body: bytes) -> None:
        record["raw_response_base64"] = base64.b64encode(raw_body).decode("ascii")
        record["raw_response_sha256"] = hashlib.sha256(raw_body).hexdigest()

    def _resume_attempt(
        self, path: Path, record: dict[str, Any]
    ) -> dict[str, Any] | None:
        status = record.get("status")
        if status == "request_persisted":
            record["status"] = "ambiguous_unreturned"
            record["error"] = (
                "restart found a persisted request without a persisted response; not resent"
            )
            write_json_atomic(path, record)
            raise AmbiguousRequest(record["error"])
        if status == "response_received":
            try:
                raw_body = base64.b64decode(record["raw_response_base64"], validate=True)
                if hashlib.sha256(raw_body).hexdigest() != record["raw_response_sha256"]:
                    raise ValueError("raw response hash changed")
                response = strict_json_loads(raw_body)
                if not isinstance(response, dict):
                    raise TypeError("chat response must be an object")
            except Exception as exc:
                record["status"] = "invalid_response"
                record["error"] = f"{type(exc).__name__}: {exc}"
                write_json_atomic(path, record)
                raise DurableChatError(record["error"]) from exc
            record["status"] = "complete"
            write_json_atomic(path, record)
            return response
        if status == "complete":
            raw_body = base64.b64decode(record["raw_response_base64"], validate=True)
            if hashlib.sha256(raw_body).hexdigest() != record["raw_response_sha256"]:
                raise DurableChatError("raw response hash changed")
            response = strict_json_loads(raw_body)
            if not isinstance(response, dict):
                raise DurableChatError("chat response must be an object")
            return response
        if status in TERMINAL_FAILURES:
            if status == "ambiguous_unreturned":
                raise AmbiguousRequest(record.get("error", status))
            return None
        raise DurableChatError(f"unknown durable attempt status: {status}")

    @staticmethod
    def _total_wall(call_dir: Path, through_attempt: int) -> float:
        total = 0.0
        for index in range(1, through_attempt + 1):
            path = call_dir / f"attempt-{index:02d}.json"
            if path.exists():
                record = strict_json_loads(path.read_bytes())
                total += float(record.get("wall_seconds") or 0)
        return total
