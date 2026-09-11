#!/usr/bin/env python3
"""Generate, freeze, and continue a five-arm historical-reasoning dose study."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import os
import platform
import random
import secrets
import ssl
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_hosted_history_action import payload_for as delivery_payload_for  # noqa: E402
from short_reasoning import (  # noqa: E402
    BudgetFloorReached,
    ChatAttemptError,
    aggregate_metrics,
    append_tool_cycle,
    assistant_history_message,
    build_tools,
    evaluate_tool_response,
    final_is_correct,
    initial_messages,
    load_dotenv,
    post_chat_once_raw,
    reasoning_text,
    response_metrics,
    strict_json_loads,
    target_payload,
)
from short_reasoning.frozen import frozen_source_hash, reasoning_only_fork_audit  # noqa: E402
from short_reasoning.history_tiers import (  # noqa: E402
    ARMS,
    FOUR_FRESH_TIERS,
    SELF_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TIER_INSTRUCTIONS,
    automatic_tier_audit,
    automatic_turn_audit,
    build_tier_variant,
    generated_tiers_for,
    parse_tier_response,
    source_turns,
    tier_payload,
    turn_word_intervals,
    verify_tier_variant,
    word_interval,
)

DEFAULT_PARENT = ROOT / "results" / "20260724-v2-12-task-two-route-screen"
EXPECTED_PARENT_MANIFEST_SHA256 = "fddc5629275d8275ef2a3cf7414e32758c2cbcab542403d0b1bfb731316645bd"
EXPECTED_PARENT_TASKS_SHA256 = "8ff6de3790beb350b1f8d64465311e7f593d098e6efb812b830c86f2f2eb7421"
EXPECTED_PARENT_MODELS_SHA256 = "a873e2c17de35efaf7fac0d6ee2fbb73a06a2bf909260469e4a3dc3a62c242da"
TERMINAL_JOURNAL_STATUSES = {
    "complete",
    "automatic_gate_failure",
    "transport_failure",
    "route_failure",
}
EXPECTED_PROVIDERS = {
    "deepseek-v4-flash": "DeepInfra",
    "laguna-s-2.1": "Poolside",
}
SEMANTIC_CHECKS = {
    "facts",
    "numbers_and_units",
    "comparators",
    "uncertainty",
    "evidence_status",
    "corrections",
    "rejected_branches",
    "dependencies",
    "conclusions",
    "intended_action",
    "future_result_absent",
}
STYLE_CHECKS = {"tier_compliance", "dose_compliance", "no_semantic_padding"}
MAX_REWRITER_DRAFTS = 6
MAX_REWRITER_ATTEMPTS = 12
REWRITER_CANARY_SEED = 84003
RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
SELF_REWRITE_DEVELOPMENT_TASKS = {
    "hospital-decision-v2",
    "manufacturing-recall-v2",
    "wildfire-evacuation-v2",
}
SELF_REWRITE_EXPANSION_TASKS = {
    "cargo-decision-v2",
    "data-center-recovery-v2",
    "flood-decision-v2",
    "network-decision-v2",
    "orbital-observation-v2",
    "vaccine-decision-v2",
    "water-decision-v2",
}
DOSE_ORDER = (
    "clean",
    "relaxed_full_sentence_compact",
    "full_sentence_compact",
    "telegraphic_compact",
    "ultra_telegraphic",
)
NONCLEAN_ARMS = DOSE_ORDER[1:]
EVALUATION_TASKS = {
    "cargo-decision-v2",
    "data-center-recovery-v2",
    "flood-decision-v2",
    "hospital-decision-v2",
    "manufacturing-recall-v2",
    "network-decision-v2",
    "orbital-observation-v2",
    "vaccine-decision-v2",
    "water-decision-v2",
    "wildfire-evacuation-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("init", "canary", "generate", "unblind", "finalize", "delivery", "continue"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--min-balance", type=float, default=2.0)
    parser.add_argument("--call-reserve", type=float, default=0.10)
    parser.add_argument("--no-budget-guard", action="store_true")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--replicate-limit", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--tier-mode",
        choices=("shared_telegraphic", "four_fresh"),
        default="shared_telegraphic",
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--generation-tasks")
    parser.add_argument("--canary-task", default="hospital-decision-v2")
    parser.add_argument(
        "--protocol-profile",
        choices=("legacy", "deepseek_self_development", "deepseek_self_expansion"),
        default="legacy",
    )
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--selection-run", type=Path)
    parser.add_argument(
        "--rewriter-profile",
        choices=(
            "prototype",
            "prototype_long",
            "prototype_unbounded",
            "sol_unbounded",
            "deepseek_self",
            "deepseek_self_examples",
        ),
        default="prototype",
    )
    return parser.parse_args()


def expected_rewriter_provider(rewriter: dict[str, Any]) -> str:
    routes = tuple(rewriter.get("provider", {}).get("only", []))
    expected = {("openai",): "OpenAI", ("deepinfra",): "DeepInfra"}.get(routes)
    if expected is None:
        raise ValueError(f"unsupported exact rewriter route: {routes}")
    return expected


def selected_names(requested: str, available: set[str], label: str) -> tuple[str, ...]:
    selected = available if requested == "all" else set(requested.split(","))
    missing = selected - available
    if missing or not selected:
        raise ValueError(f"unknown or empty {label} selection: {sorted(missing)}")
    return tuple(sorted(selected))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resolved_repo_file(relative: str, expected_sha256: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"example provenance path is not one regular repository file: {relative}")
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"example provenance file hash changed: {relative}")
    return path


def _verified_style_example(profile_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    examples_path = ROOT / "configs" / "history_tier_examples.json"
    examples = strict_json_loads(examples_path.read_bytes())
    if not isinstance(examples, dict) or profile_name not in examples:
        raise RuntimeError(f"unknown history-tier example profile: {profile_name}")
    example = examples[profile_name]
    provenance = example.get("provenance", {})
    task_id = provenance.get("task_id")
    turn_index = provenance.get("historical_turn")
    if (
        task_id != "rescue-decision-v2"
        or task_id in EVALUATION_TASKS
        or turn_index != 2
        or provenance.get("active_evaluation_cohort_excluded") is not True
    ):
        raise RuntimeError("style example is not bound to one excluded source turn")
    parent_path = _resolved_repo_file(
        provenance["parent_source_file"], provenance["parent_source_file_sha256"]
    )
    parent = strict_json_loads(parent_path.read_bytes())
    if (
        parent.get("task_id") != task_id
        or parent.get("source_sha256") != provenance.get("parent_source_sha256")
        or len(parent.get("shared", [])) < turn_index
    ):
        raise RuntimeError("style example parent source binding changed")
    rewrite = parent["shared"][turn_index - 1]["rewrite"]
    if (
        example.get("raw_reasoning") != rewrite.get("raw_reasoning")
        or example.get("requested_action") != rewrite.get("requested_tool")
    ):
        raise RuntimeError("style example raw turn/action differs from its parent")
    tier_order = tuple(example.get("tier_order", []))
    expected_order = (
        "relaxed_full_sentence_compact",
        "full_sentence_compact",
        "telegraphic_compact",
        "ultra_telegraphic",
    )
    states = example.get("states_by_tier", {})
    tier_sources = provenance.get("tier_sources", {})
    if (
        tier_order != expected_order
        or set(states) != set(FOUR_FRESH_TIERS)
        or set(tier_sources) != set(FOUR_FRESH_TIERS)
    ):
        raise RuntimeError("style example does not freeze the exact four-tier ordering")
    for tier in FOUR_FRESH_TIERS:
        source_record = tier_sources[tier]
        source_path = _resolved_repo_file(
            source_record["file"], source_record["file_sha256"]
        )
        if tier == "telegraphic_compact":
            if (
                source_path != parent_path
                or source_record.get("source_sha256") != parent["source_sha256"]
                or states[tier] != rewrite.get("rewritten")
            ):
                raise RuntimeError("telegraphic example is not the exact parent Sol state")
            continue
        variant = strict_json_loads(source_path.read_bytes())
        generated = next(
            (
                item
                for item in variant.get("generated_turns", [])
                if item.get("turn") == turn_index
            ),
            None,
        )
        if (
            variant.get("task_id") != task_id
            or variant.get("tier") != tier
            or variant.get("parent_source_sha256") != parent["source_sha256"]
            or variant.get("variant_sha256") != source_record.get("variant_sha256")
            or not isinstance(generated, dict)
            or generated.get("state") != states[tier]
        ):
            raise RuntimeError(f"{tier} example differs from its immutable variant")
    semantic = example.get("semantic_audit", {})
    checks_by_tier = semantic.get("checks_by_tier", {})
    if (
        semantic.get("future_results_inspected") is not False
        or set(checks_by_tier) != set(FOUR_FRESH_TIERS)
        or any(
            set(checks) != SEMANTIC_CHECKS
            or any(value is not True for value in checks.values())
            for checks in checks_by_tier.values()
        )
        or not isinstance(semantic.get("notes"), str)
        or not semantic["notes"]
    ):
        raise RuntimeError("style example lacks a complete frozen semantic audit")
    prompt_value = {
        "scope": "unrelated formatting demonstration; facts must never transfer",
        "tier_order": list(tier_order),
        "raw_reasoning": example["raw_reasoning"],
        "requested_action": copy.deepcopy(example["requested_action"]),
        "rewrites": [
            {"tier": tier, "state": states[tier]} for tier in tier_order
        ],
    }
    return prompt_value, copy.deepcopy(example)


def load_rewriter_profile(name: str) -> dict[str, Any]:
    profiles_path = ROOT / "configs" / "rewriters.json"
    profiles = strict_json_loads(profiles_path.read_bytes())
    if not isinstance(profiles, dict) or name not in profiles:
        raise RuntimeError(f"unknown rewriter profile: {name}")
    original = profiles[name]
    if not isinstance(original, dict):
        raise RuntimeError("rewriter profile must be a JSON object")
    rewriter = copy.deepcopy(original)
    rewriter["profile_name"] = name
    rewriter["profile_config_sha256"] = canonical_hash(original)
    reference_mode = rewriter.get("reference_mode", "compact_anchor")
    prompt = SELF_SYSTEM_PROMPT if reference_mode == "raw_only" else SYSTEM_PROMPT
    rewriter["prompt_profile_sha256"] = canonical_hash(
        {"system": prompt, "tier_instructions": TIER_INSTRUCTIONS}
    )
    example_profile = rewriter.get("example_profile")
    if example_profile is not None:
        style_example, frozen_example = _verified_style_example(example_profile)
        rewriter["style_example"] = style_example
        rewriter["example_profile_sha256"] = canonical_hash(frozen_example)
        rewriter["example_provenance"] = frozen_example["provenance"]
    if name == "deepseek_self" or name.startswith("deepseek_self_"):
        if (
            rewriter.get("model") != "deepseek/deepseek-v4-flash"
            or rewriter.get("provider")
            != {
                "only": ["deepinfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            or rewriter.get("reasoning_effort") != "xhigh"
            or rewriter.get("reference_mode") != "raw_only"
            or rewriter.get("temperature") != 0
            or rewriter.get("max_tokens") != 8192
        ):
            raise RuntimeError("DeepSeek self-rewriter profile differs from its frozen route")
    elif "temperature" in rewriter:
        raise RuntimeError("temperature is permitted only for DeepSeek self-rewriter profiles")
    expected_rewriter_provider(rewriter)
    return rewriter


def build_protocol_context(
    name: str,
    rewriter: dict[str, Any],
    model_names: tuple[str, ...],
    task_ids: tuple[str, ...],
    tier_mode: str,
    workers: int,
    budget_guard: bool,
    baseline_run: Path | None = None,
    selection_run: Path | None = None,
) -> dict[str, Any] | None:
    if name == "legacy":
        if (
            baseline_run is not None
            or selection_run is not None
            or tier_mode != "shared_telegraphic"
            or rewriter.get("profile_name")
            in {"deepseek_self", "deepseek_self_examples"}
        ):
            raise ValueError(
                "legacy protocol permits only shared-telegraphic non-self rewriter runs"
            )
        return None
    if (
        model_names != ("deepseek-v4-flash",)
        or tier_mode != "four_fresh"
        or rewriter.get("profile_name")
        not in {"deepseek_self", "deepseek_self_examples"}
        or expected_rewriter_provider(rewriter) != "DeepInfra"
        or workers < 1
        or budget_guard
    ):
        raise ValueError(
            "DeepSeek self-rewrite requires its exact target/profile, four-fresh tiers, "
            "positive workers, and the authorized parallel no-budget-guard mode"
        )
    context: dict[str, Any] = {
        "name": name,
        "version": 1,
        "rewriter_profile": rewriter["profile_name"],
    }
    if name == "deepseek_self_development":
        if set(task_ids) != SELF_REWRITE_DEVELOPMENT_TASKS or selection_run is not None:
            raise ValueError("self-rewrite development requires the exact three-task cohort")
        if rewriter["profile_name"] == "deepseek_self":
            if baseline_run is not None:
                raise ValueError("baseline self-rewrite cannot name a prior baseline run")
            context["prompt_selection"] = "baseline_raw_only"
        else:
            if baseline_run is None:
                raise ValueError("example retry requires one frozen failed baseline run")
            resolved_baseline = baseline_run.resolve()
            context.update(
                prompt_selection="single_example_retry_after_baseline_failure",
                baseline_failure=verify_failed_baseline_run(resolved_baseline),
                required_retry_run_dir=str(
                    resolved_baseline.parent
                    / f"{resolved_baseline.name}-example-retry"
                ),
            )
        return context
    if name == "deepseek_self_expansion":
        if set(task_ids) != SELF_REWRITE_EXPANSION_TASKS or baseline_run is not None:
            raise ValueError("self-rewrite expansion requires the exact untouched seven-task cohort")
        if selection_run is None:
            raise ValueError("self-rewrite expansion requires a passed development selection run")
        context.update(
            prompt_selection="frozen_from_passed_development",
            development_selection=verify_passed_development_run(
                selection_run.resolve(), rewriter
            ),
        )
        return context
    raise ValueError(f"unknown protocol profile: {name}")


def invocation_command() -> list[str]:
    return [sys.executable, *sys.argv]


def record_invocation(
    run_path: Path, run: dict[str, Any], stage: str
) -> None:
    commands = run.setdefault("commands", {})
    values = commands.setdefault(stage, [])
    command = invocation_command()
    if not any(item.get("command") == command for item in values):
        values.append({"recorded_at": now(), "command": command})
        atomic_json(run_path, run)


def request_body_record(payload: dict[str, Any]) -> tuple[str, str]:
    body = json.dumps(payload, ensure_ascii=False)
    return body, hashlib.sha256(body.encode()).hexdigest()


def persist_raw_response(attempt: dict[str, Any], raw_body: bytes, wall: float) -> None:
    attempt.update(
        status="response_received",
        response_received_at=now(),
        response_body_base64=base64.b64encode(raw_body).decode("ascii"),
        response_body_sha256=hashlib.sha256(raw_body).hexdigest(),
        wall_seconds=wall,
    )


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


def persist_returned_http_error(
    attempt: dict[str, Any], exc: ChatAttemptError
) -> None:
    if exc.http_status is None or exc.response_body_bytes is None:
        raise ValueError("returned HTTP error requires a status and raw response body")
    persist_raw_response(attempt, exc.response_body_bytes, exc.wall_seconds)
    attempt.update(
        status="returned_http_error",
        completed_at=now(),
        http_status=exc.http_status,
        error=str(exc),
        response_body=exc.response_body,
        retryable=returned_http_error_is_retryable(
            exc.http_status, exc.response_body_bytes
        ),
    )


def persisted_raw_response(attempt: dict[str, Any]) -> bytes:
    raw_body = base64.b64decode(attempt["response_body_base64"], validate=True)
    if hashlib.sha256(raw_body).hexdigest() != attempt["response_body_sha256"]:
        raise RuntimeError("persisted raw response body digest changed")
    return raw_body


def parse_persisted_response(attempt: dict[str, Any]) -> dict[str, Any]:
    parsed = strict_json_loads(persisted_raw_response(attempt))
    if not isinstance(parsed, dict):
        raise ValueError("OpenRouter response body is not a JSON object")
    return parsed


def _durable_replace_text(path: Path, temporary: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _durable_replace_text(
        path,
        temporary,
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
    )


def run_bounded_jobs(
    jobs: list[Any], workers: int, execute: Any
) -> list[Any]:
    """Execute with at most `workers` submitted futures and preserve input order."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        return [execute(job) for job in jobs]
    results: list[Any] = [None] * len(jobs)
    iterator = iter(enumerate(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[concurrent.futures.Future[Any], int] = {}
        for _ in range(min(workers, len(jobs))):
            index, job = next(iterator)
            pending[executor.submit(execute, job)] = index
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                index = pending.pop(future)
                results[index] = future.result()
                try:
                    next_index, next_job = next(iterator)
                except StopIteration:
                    continue
                pending[executor.submit(execute, next_job)] = next_index
    return results


def reconcile_immutable_json(path: Path, value: Any) -> None:
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"refusing to replace differing immutable JSON: {path}")
        if temporary.exists():
            temporary.unlink()
        return
    atomic_json(path, value)


def _manifestable_files(directory: Path, manifest: Path) -> dict[str, Path]:
    result = {}
    for path in directory.rglob("*"):
        if path == manifest:
            continue
        if path.is_symlink():
            raise RuntimeError(f"manifest directory contains a symlink: {path}")
        if path.is_file():
            relative = str(path.relative_to(directory))
            if relative in result:
                raise RuntimeError(f"duplicate manifest relative path: {relative}")
            result[relative] = path
    return result


def verify_manifest_exact(directory: Path, manifest: Path) -> None:
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if (
            name in entries
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not name
        ):
            raise RuntimeError(f"invalid or duplicate manifest entry: {name}")
        entries[name] = digest
    actual = _manifestable_files(directory, manifest)
    if set(entries) != set(actual):
        raise RuntimeError(f"manifest path set differs from files in {directory}")
    for name, digest in entries.items():
        if file_sha256(actual[name]) != digest:
            raise RuntimeError(f"manifest digest mismatch: {actual[name]}")


def load_parent_sources(
    parent_run: Path,
    model_names: tuple[str, ...] | None = None,
    task_ids: tuple[str, ...] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    source_manifest = parent_run / "sources" / "SHA256SUMS"
    task_path = parent_run / "tasks.snapshot.json"
    model_path = parent_run / "models.snapshot.json"
    if (
        file_sha256(source_manifest) != EXPECTED_PARENT_MANIFEST_SHA256
        or file_sha256(task_path) != EXPECTED_PARENT_TASKS_SHA256
        or file_sha256(model_path) != EXPECTED_PARENT_MODELS_SHA256
    ):
        raise RuntimeError("parent cohort files differ from the prespecified immutable screen")
    verify_manifest_exact(parent_run / "sources", source_manifest)
    models = json.loads(model_path.read_text())["targets"]
    tasks = {task["id"]: task for task in json.loads(task_path.read_text())}
    result = []
    for path in sorted((parent_run / "sources").glob("*.json")):
        source = json.loads(path.read_text())
        if (
            source.get("status") != "selected"
            or source["model_name"] not in EXPECTED_PROVIDERS
            or source["model"] != models[source["model_name"]]
            or source["task_id"] not in tasks
            or source.get("fork_after") != 3
            or len(source.get("shared", [])) != 3
        ):
            raise RuntimeError(f"parent source target/cohort configuration mismatch: {path}")
        task = tasks[source["task_id"]]
        tools = build_tools(task)
        reconstructed_clean = initial_messages(task)
        reconstructed_compact = copy.deepcopy(reconstructed_clean)
        for turn_index, (step, phase) in enumerate(
            zip(source["shared"], task["phases"][:3]), start=1
        ):
            expected_request = target_payload(
                source["model"],
                reconstructed_clean,
                tools,
                source["source_seed"] + turn_index - 1,
            )
            choices = step.get("response", {}).get("choices")
            message = (
                choices[0].get("message")
                if isinstance(choices, list)
                and len(choices) == 1
                and isinstance(choices[0], dict)
                else None
            )
            action = evaluate_tool_response(message, phase)
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            rewrite = step.get("rewrite", {})
            rewrite_attempts = rewrite.get("attempts")
            final_rewrite_attempt = (
                rewrite_attempts[-1]
                if isinstance(rewrite_attempts, list)
                and rewrite_attempts
                and isinstance(rewrite_attempts[-1], dict)
                else {}
            )
            rewrite_choices = rewrite.get("response", {}).get("choices")
            rewritten_content = (
                rewrite_choices[0].get("message", {}).get("content")
                if isinstance(rewrite_choices, list)
                and len(rewrite_choices) == 1
                and isinstance(rewrite_choices[0], dict)
                else None
            )
            if (
                step.get("request") != expected_request
                or step.get("turn") != turn_index
                or step.get("phase") != phase["name"]
                or step.get("action") != action
                or not action["correct"]
                or step["response"].get("model") != source["model"]["model"]
                or step["response"].get("provider")
                != EXPECTED_PROVIDERS[source["model_name"]]
                or step.get("metrics")
                != response_metrics(step["response"], step["wall_seconds"])
                or not isinstance(calls, list)
                or len(calls) != 1
                or rewrite.get("raw_reasoning") != reasoning_text(message)
                or rewrite.get("requested_tool") != calls[0].get("function")
                or rewrite.get("request") != final_rewrite_attempt.get("request")
                or rewrite.get("response") != final_rewrite_attempt.get("response")
                or rewrite.get("wall_seconds") != final_rewrite_attempt.get("wall_seconds")
                or rewritten_content != rewrite.get("rewritten")
            ):
                raise RuntimeError(
                    f"parent shared turn {turn_index} does not replay: {path}"
                )
            append_tool_cycle(
                reconstructed_clean, message, rewrite["raw_reasoning"], phase
            )
            append_tool_cycle(
                reconstructed_compact, message, rewrite["rewritten"], phase
            )
        clean = source["histories"]["clean"]
        compact = source["histories"]["rewritten"]
        recomputed_fork = reasoning_only_fork_audit(clean, compact)
        if (
            clean != reconstructed_clean
            or compact != reconstructed_compact
            or recomputed_fork.get("reasoning_message_indices") != [2, 4, 6]
            or recomputed_fork != source["fork_audit"]
        ):
            raise RuntimeError(f"parent source histories do not replay: {path}")
        expected = frozen_source_hash(
            source["task_id"], source["model_name"], source["model"], clean, compact
        )
        if expected != source.get("source_sha256"):
            raise RuntimeError(f"parent source hash does not recompute: {path}")
        result.append((path, source))
    matrix = [(source["model_name"], source["task_id"]) for _, source in result]
    expected_matrix = {
        (model_name, task_id)
        for model_name in EXPECTED_PROVIDERS
        for task_id in tasks
    }
    if len(matrix) != len(set(matrix)) or set(matrix) != expected_matrix:
        raise RuntimeError("parent sources do not form the exact unique 2x12 model/task matrix")
    selected_models = set(model_names or EXPECTED_PROVIDERS)
    selected_tasks = set(task_ids or tasks)
    if selected_models - set(EXPECTED_PROVIDERS) or selected_tasks - set(tasks):
        raise RuntimeError("selected model/task cohort is outside the immutable parent matrix")
    filtered = [
        item
        for item in result
        if item[1]["model_name"] in selected_models
        and item[1]["task_id"] in selected_tasks
    ]
    if len(filtered) != len(selected_models) * len(selected_tasks):
        raise RuntimeError("selected parent cohort is not a complete model/task product")
    return filtered


def initialize(
    run_dir: Path,
    parent_run: Path,
    rewriter: dict[str, Any],
    replicates: int,
    model_names: tuple[str, ...] | None = None,
    task_ids: tuple[str, ...] | None = None,
    tier_mode: str = "shared_telegraphic",
    workers: int = 1,
    budget_guard: bool = True,
    protocol_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if replicates != 3:
        raise ValueError("the tier protocol requires exactly three replicates")
    if workers < 1:
        raise ValueError("workers must be at least one")
    generated_tiers = generated_tiers_for(tier_mode)
    expected_rewriter_provider(rewriter)
    self_protocol = (
        protocol_context.get("name")
        if isinstance(protocol_context, dict)
        else None
    )
    if (
        tier_mode == "four_fresh"
        or rewriter.get("profile_name") in {"deepseek_self", "deepseek_self_examples"}
    ) and self_protocol not in {
        "deepseek_self_development",
        "deepseek_self_expansion",
    }:
        raise ValueError("four-fresh/self rewriter initialization requires its named protocol")
    if (
        protocol_context is not None
        and protocol_context.get("prompt_selection")
        == "single_example_retry_after_baseline_failure"
        and run_dir.resolve()
        != Path(protocol_context["required_retry_run_dir"]).resolve()
    ):
        raise ValueError("example fallback must use its one canonical retry run directory")
    parent_run = parent_run.resolve()
    manifest = parent_run / "sources" / "SHA256SUMS"
    sources = load_parent_sources(parent_run, model_names, task_ids)
    task_source = parent_run / "tasks.snapshot.json"
    model_source = parent_run / "models.snapshot.json"
    all_tasks = json.loads(task_source.read_text())
    selected_task_ids = {source["task_id"] for _, source in sources}
    tasks = [task for task in all_tasks if task["id"] in selected_task_ids]
    task_hash = canonical_hash(tasks)
    selected_model_names = tuple(sorted({source["model_name"] for _, source in sources}))
    if self_protocol in {
        "deepseek_self_development",
        "deepseek_self_expansion",
    }:
        expected_tasks = (
            SELF_REWRITE_DEVELOPMENT_TASKS
            if self_protocol == "deepseek_self_development"
            else SELF_REWRITE_EXPANSION_TASKS
        )
        if (
            selected_model_names != ("deepseek-v4-flash",)
            or selected_task_ids != expected_tasks
            or tier_mode != "four_fresh"
            or rewriter.get("profile_name")
            not in {"deepseek_self", "deepseek_self_examples"}
            or protocol_context.get("rewriter_profile") != rewriter.get("profile_name")
        ):
            raise ValueError("named self-rewrite initialization differs from its exact cohort/profile")
    source_counts_by_model = {
        model: sum(source["model_name"] == model for _, source in sources)
        for model in selected_model_names
    }
    custom_selection = model_names is not None or task_ids is not None
    launch_gate = {
        "minimum_total_sources": len(sources) if custom_selection else 20,
        "minimum_sources_by_model": (
            source_counts_by_model
            if custom_selection
            else {model: 10 for model in selected_model_names}
        ),
    }
    harness_paths = (
        ROOT / "scripts" / "run_history_tiers.py",
        ROOT / "scripts" / "review_history_tier_packets.py",
        ROOT / "scripts" / "analyze_history_tier_sweep.py",
        ROOT / "scripts" / "audit_hosted_history_action.py",
        ROOT / "src" / "short_reasoning" / "history_tiers.py",
        ROOT / "src" / "short_reasoning" / "history_controls.py",
        ROOT / "src" / "short_reasoning" / "__init__.py",
        ROOT / "src" / "short_reasoning" / "frozen.py",
        ROOT / "configs" / "models.json",
        ROOT / "configs" / "rewriters.json",
        ROOT / "configs" / "history_tier_examples.json",
        ROOT / "docs" / "plans" / "2026-07-25-history-tier-dose.md",
        ROOT / "docs" / "plans" / "2026-07-26-deepseek-five-tier-sweep.md",
        ROOT / "docs" / "plans" / "2026-07-26-deepseek-self-rewrite.md",
    )
    harness_sha256 = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in harness_paths
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    tracked_diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "openssl": ssl.OPENSSL_VERSION,
    }
    executable_manifest_sha256 = canonical_hash(
        {
            "harness_sha256": harness_sha256,
            "parent_task_file_sha256": file_sha256(task_source),
            "parent_model_file_sha256": file_sha256(model_source),
            "runtime": runtime,
        }
    )
    snapshot = {
        "parent_run": str(parent_run),
        "parent_source_manifest_sha256": file_sha256(manifest),
        "parent_task_file_sha256": file_sha256(task_source),
        "parent_model_file_sha256": file_sha256(model_source),
        "task_snapshot_sha256": task_hash,
        "harness_sha256": harness_sha256,
        "runtime": runtime,
        "executable_manifest_sha256": executable_manifest_sha256,
        "audit_packet_salt": secrets.token_hex(32),
        "rewriter": copy.deepcopy(rewriter),
        "selected_models": list(selected_model_names),
        "selected_tasks": sorted(selected_task_ids),
        "launch_gate": launch_gate,
        "tier_mode": tier_mode,
        "generated_tiers": list(generated_tiers),
        "arms": list(ARMS),
        "replicates": replicates,
        "git_commit": commit,
        "tracked_diff_sha256_at_initialization": hashlib.sha256(tracked_diff).hexdigest(),
        "execution": {
            "workers": workers,
            "budget_guard": "enabled" if budget_guard else "disabled_by_user_authorization",
            "parallel_jobs": "source-tier cells and block-replicate continuations only",
        },
        "intended_sources": [
            {
                "file": path.name,
                "file_sha256": file_sha256(path),
                "source_sha256": source["source_sha256"],
                "model_name": source["model_name"],
                "task_id": source["task_id"],
            }
            for path, source in sources
        ],
        "outcome_blinding": (
            "Each request receives one immutable raw reasoning turn and its already-requested action only; "
            "the self-rewriter does not receive the parent compact anchor. It never receives another tier, a "
            "pending result, later turn, final answer, or prior outcome."
            if rewriter.get("reference_mode") == "raw_only"
            else
            "Each request receives one immutable reasoning turn, its accepted compact anchor, and its already-"
            "requested action only. It never receives a pending result, later turn, final answer, or prior outcome."
        ),
        "retry_policy": {
            "rewriter_returned_drafts_per_turn": MAX_REWRITER_DRAFTS,
            "rewriter_total_attempts_per_turn": MAX_REWRITER_ATTEMPTS,
            "target_total_attempts_per_slot": 20,
            "in_flight_attempt_after_restart": "stop as ambiguous; never resend automatically",
        },
    }
    if protocol_context is not None:
        snapshot["protocol"] = copy.deepcopy(protocol_context)
    run_path = run_dir / "run.json"
    task_snapshot = run_dir / "tasks.snapshot.json"
    if run_path.exists():
        run = json.loads(run_path.read_text())
        recorded = run.get("snapshot", {})
        stable_keys = (
            "git_commit",
            "parent_run",
            "parent_source_manifest_sha256",
            "parent_task_file_sha256",
            "parent_model_file_sha256",
            "task_snapshot_sha256",
            "harness_sha256",
            "runtime",
            "executable_manifest_sha256",
            "rewriter",
            "selected_models",
            "selected_tasks",
            "launch_gate",
            "tier_mode",
            "generated_tiers",
            "arms",
            "replicates",
            "execution",
            "intended_sources",
            "outcome_blinding",
            "retry_policy",
        ) + (("protocol",) if "protocol" in snapshot or "protocol" in recorded else ())
        if any(recorded.get(key) != snapshot.get(key) for key in stable_keys):
            raise RuntimeError("tier-run inputs differ from the existing immutable initialization")
        if canonical_hash(json.loads(task_snapshot.read_text())) != task_hash:
            raise RuntimeError("frozen tier task snapshot changed")
        return run
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    unexpected_status = [
        line for line in status.splitlines() if line != "?? paper/main.pdf"
    ]
    if tracked_diff or unexpected_status:
        raise RuntimeError("commit every executable input before initializing paid experimental data")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("refusing to adopt a nonempty directory without an immutable run.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(task_snapshot, tasks)
    run = {
        "created_at": now(),
        "status": "initialized_before_variant_calls",
        "commands": {
            "init": [{"recorded_at": now(), "command": invocation_command()}]
        },
        "outcomes_launched": False,
        "snapshot": snapshot,
    }
    atomic_json(run_path, run)
    return run


def _repair_message(audit: dict[str, Any] | None, parse_error: str | None) -> str:
    if parse_error:
        return f"Return valid schema-conforming JSON. Prior parse error: {parse_error}"
    assert audit is not None
    issues = []
    words = audit["variant_words"]
    lower, upper = audit["target_word_interval_inclusive"]
    if not lower <= words <= upper:
        issues.append(f"word count {words}, required {lower}..{upper}")
    for key in (
        "missing_anchor_identifiers",
        "missing_anchor_numeric_literals",
        "added_identifiers",
        "added_numeric_literals",
    ):
        if audit.get(key):
            issues.append(f"{key}: {audit[key]}")
    if audit.get("forbidden_template_tag"):
        issues.append(f"forbidden tag: {audit['forbidden_template_tag']}")
    if audit.get("unsupported_result_status"):
        issues.append(
            "remove requested-action result-status prose: "
            f"{audit['unsupported_result_status']}"
        )
    action_audit = audit.get("requested_action_audit")
    if isinstance(action_audit, dict) and not action_audit.get("passed"):
        for key in (
            "missing_function_name",
            "missing_argument_keys",
            "missing_string_argument_values",
            "missing_numeric_argument_values",
        ):
            if action_audit.get(key):
                issues.append(f"requested_action_{key}: {action_audit[key]}")
        if action_audit.get("arguments_parseable_object") is not True:
            issues.append("requested action arguments are not one parseable JSON object")
    if not audit.get("differs_from_raw", True) or not audit.get("differs_from_compact", True):
        issues.append("state must be genuinely rewritten")
    return "Repair only these automatic-gate failures while preserving meaning: " + "; ".join(issues)


def _attempt_counts(turn: dict[str, Any]) -> tuple[int, int]:
    counted = [attempt for attempt in turn["attempts"] if attempt["status"] != "budget_pause"]
    drafts = sum(
        attempt["status"] in {"accepted_draft", "rejected_draft"}
        for attempt in counted
    )
    return drafts, len(counted)


def _process_returned_generation_attempt(
    journal_path: Path,
    journal: dict[str, Any],
    turn: dict[str, Any],
    attempt: dict[str, Any],
    rewriter: dict[str, Any],
) -> bool:
    try:
        response = parse_persisted_response(attempt)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        attempt.update(
            status="returned_error",
            route_valid=None,
            parse_error=f"{type(exc).__name__}: {exc}",
        )
        _, total = _attempt_counts(turn)
        if total >= MAX_REWRITER_ATTEMPTS:
            turn.update(status="transport_failure", completed_at=now())
            turn.pop("next_request", None)
            journal.update(status="transport_failure", completed_at=now())
        atomic_json(journal_path, journal)
        return False
    attempt["response"] = response
    expected_provider = expected_rewriter_provider(rewriter)
    explicit_route_mismatch = (
        response.get("provider") is not None
        and response.get("provider") != expected_provider
    ) or (
        response.get("model") is not None and response.get("model") != rewriter["model"]
    )
    if explicit_route_mismatch:
        attempt.update(status="route_failure", route_valid=False)
        turn["status"] = "route_failure"
        turn.pop("next_request", None)
        journal.update(status="route_failure", completed_at=now())
        atomic_json(journal_path, journal)
        return False
    choices = response.get("choices")
    usable_shape = (
        isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict)
        and choices[0].get("finish_reason") == "stop"
        and isinstance(choices[0].get("message"), dict)
        and choices[0]["message"].get("role") == "assistant"
    )
    if not usable_shape:
        attempt.update(
            status="returned_error",
            route_valid=None,
            parse_error="returned response contains no single usable choice/message",
        )
        _, total = _attempt_counts(turn)
        if total >= MAX_REWRITER_ATTEMPTS:
            turn.update(status="transport_failure", completed_at=now())
            turn.pop("next_request", None)
            journal.update(status="transport_failure", completed_at=now())
        atomic_json(journal_path, journal)
        return False
    route_valid = (
        response.get("provider") == expected_provider
        and response.get("model") == rewriter["model"]
    )
    attempt["route_valid"] = route_valid
    if not route_valid:
        attempt["status"] = "route_failure"
        turn["status"] = "route_failure"
        turn.pop("next_request", None)
        journal.update(status="route_failure", completed_at=now())
        atomic_json(journal_path, journal)
        return False
    parse_error = None
    state = None
    audit = None
    try:
        state = parse_tier_response(response)
        audit = automatic_turn_audit(
            turn["source"],
            state,
            tuple(turn["word_interval"]),
            require_exact_action=rewriter.get("reference_mode") == "raw_only",
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    attempt.update(
        parsed_state=state,
        parse_error=parse_error,
        automatic_audit=audit,
        status="accepted_draft" if audit is not None and audit["passed"] else "rejected_draft",
    )
    if audit is not None and audit["passed"]:
        turn.update(status="complete", state=state, automatic_audit=audit, completed_at=now())
        turn.pop("next_request", None)
        atomic_json(journal_path, journal)
        return True
    drafts, total = _attempt_counts(turn)
    if drafts >= MAX_REWRITER_DRAFTS or total >= MAX_REWRITER_ATTEMPTS:
        turn.update(status="automatic_gate_failure", completed_at=now())
        journal.update(status="automatic_gate_failure", completed_at=now())
        turn.pop("next_request", None)
        atomic_json(journal_path, journal)
        return False
    prior = state if state is not None else "unparseable structured response"
    turn["next_request"] = tier_payload(
        rewriter,
        journal["tier"],
        turn["source"],
        turn["turn"],
        tuple(turn["word_interval"]),
        prior_draft=prior,
        repair_instruction=_repair_message(audit, parse_error),
    )
    turn["next_request"]["seed"] += drafts
    atomic_json(journal_path, journal)
    return False


def _validate_generation_journal(
    journal: dict[str, Any],
    source: dict[str, Any],
    tier: str,
    rewriter: dict[str, Any],
) -> None:
    expected_turns = source_turns(source)
    expected_intervals = turn_word_intervals(
        tier,
        [len(turn["raw_reasoning"].split()) for turn in expected_turns],
        [len(turn["compact_reasoning"].split()) for turn in expected_turns],
    )
    if (
        journal.get("tier") != tier
        or journal.get("parent_source_sha256") != source.get("source_sha256")
        or journal.get("model_name") != source.get("model_name")
        or journal.get("task_id") != source.get("task_id")
        or [turn.get("turn") for turn in journal.get("turns", [])]
        != list(range(1, len(expected_turns) + 1))
        or [turn.get("source") for turn in journal.get("turns", [])]
        != expected_turns
        or [turn.get("word_interval") for turn in journal.get("turns", [])]
        != [list(interval) for interval in expected_intervals]
    ):
        raise RuntimeError("generation journal source binding changed")
    expected_provider = expected_rewriter_provider(rewriter)
    require_exact_action = rewriter.get("reference_mode") == "raw_only"
    if journal.get("require_exact_action", False) is not require_exact_action:
        raise RuntimeError("generation journal exact-action mode changed")
    for turn in journal["turns"]:
        interval = tuple(turn["word_interval"])
        expected_request = tier_payload(
            rewriter, tier, turn["source"], turn["turn"], interval
        )
        accepted: tuple[str, dict[str, Any]] | None = None
        draft_count = 0
        for attempt_index, attempt in enumerate(turn["attempts"]):
            body, digest = request_body_record(expected_request)
            if (
                attempt.get("request") != expected_request
                or attempt.get("request_body") != body
                or attempt.get("request_body_sha256") != digest
            ):
                raise RuntimeError("generation attempt request does not replay")
            status = attempt.get("status")
            if status in {
                "in_flight",
                "response_received",
                "route_failure",
                "ambiguous_after_restart",
                "ambiguous_transport_error",
            } and attempt_index + 1 != len(turn["attempts"]):
                raise RuntimeError("generation attempts continue after an unresolved/terminal attempt")
            if status in {"accepted_draft", "rejected_draft"}:
                draft_count += 1
            if status == "returned_http_error":
                if (
                    not isinstance(attempt.get("http_status"), int)
                    or attempt["http_status"] < 400
                ):
                    raise RuntimeError("returned generation HTTP error lacks an error status")
                retryable = returned_http_error_is_retryable(
                    attempt["http_status"], persisted_raw_response(attempt)
                )
                if attempt.get("retryable") is not retryable:
                    raise RuntimeError("returned generation HTTP retry decision changed")
                if not retryable and attempt_index + 1 != len(turn["attempts"]):
                    raise RuntimeError("generation attempts continue after terminal HTTP error")
                continue
            if status in {
                "response_received",
                "accepted_draft",
                "rejected_draft",
                "returned_error",
                "route_failure",
            }:
                try:
                    response = parse_persisted_response(attempt)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    if status != "returned_error":
                        raise RuntimeError("generation attempt status disagrees with its raw response")
                    continue
                if attempt.get("response") is not None and attempt["response"] != response:
                    raise RuntimeError("parsed generation response differs from its raw body")
                choices = response.get("choices")
                usable_shape = (
                    isinstance(choices, list)
                    and len(choices) == 1
                    and isinstance(choices[0], dict)
                    and choices[0].get("finish_reason") == "stop"
                    and isinstance(choices[0].get("message"), dict)
                    and choices[0]["message"].get("role") == "assistant"
                )
                route_valid = (
                    response.get("provider") == expected_provider
                    and response.get("model") == rewriter["model"]
                )
                if status == "returned_error":
                    if usable_shape and route_valid:
                        raise RuntimeError("usable generation response was mislabeled returned_error")
                    continue
                if status == "route_failure":
                    explicit_mismatch = (
                        response.get("provider") is not None
                        and response.get("provider") != expected_provider
                    ) or (
                        response.get("model") is not None
                        and response.get("model") != rewriter["model"]
                    )
                    if not explicit_mismatch and not (usable_shape and not route_valid):
                        raise RuntimeError("generation route failure lacks a usable wrong/absent route")
                    continue
                if not usable_shape or not route_valid:
                    if status == "response_received":
                        continue
                    raise RuntimeError("generation draft lacks its required route/shape")
                if status == "response_received":
                    continue
                try:
                    state = parse_tier_response(response)
                    audit = automatic_turn_audit(
                        turn["source"],
                        state,
                        interval,
                        require_exact_action=require_exact_action,
                    )
                    parse_error = None
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    state = None
                    audit = None
                    parse_error = f"{type(exc).__name__}: {exc}"
                expected_status = (
                    "accepted_draft" if audit is not None and audit["passed"] else "rejected_draft"
                )
                if (
                    status != expected_status
                    or attempt.get("parsed_state") != state
                    or attempt.get("parse_error") != parse_error
                    or attempt.get("automatic_audit") != audit
                ):
                    raise RuntimeError("generation draft parse/audit does not recompute")
                if expected_status == "accepted_draft":
                    if accepted is not None or attempt_index + 1 != len(turn["attempts"]):
                        raise RuntimeError("generation attempts continue after acceptance")
                    accepted = (state, audit)
                else:
                    expected_request = tier_payload(
                        rewriter,
                        tier,
                        turn["source"],
                        turn["turn"],
                        interval,
                        prior_draft=(
                            state if state is not None else "unparseable structured response"
                        ),
                        repair_instruction=_repair_message(audit, parse_error),
                    )
                    expected_request["seed"] += draft_count
            elif status not in {
                "in_flight",
                "ambiguous_after_restart",
                "budget_pause",
                "transport_error",
                "ambiguous_transport_error",
            }:
                raise RuntimeError(f"unknown generation attempt status: {status}")
        counted = [
            attempt for attempt in turn["attempts"] if attempt.get("status") != "budget_pause"
        ]
        if (
            len(counted) > MAX_REWRITER_ATTEMPTS
            or draft_count > MAX_REWRITER_DRAFTS
        ):
            raise RuntimeError("generation journal exceeds its frozen retry caps")
        route_failed = any(
            attempt.get("status") == "route_failure" for attempt in turn["attempts"]
        )
        terminal_http_error = any(
            attempt.get("status") == "returned_http_error"
            and attempt.get("retryable") is False
            for attempt in turn["attempts"]
        )
        terminal_status = None
        if route_failed:
            terminal_status = "route_failure"
        elif accepted is None and terminal_http_error:
            terminal_status = "transport_failure"
        elif accepted is None and draft_count >= MAX_REWRITER_DRAFTS:
            terminal_status = "automatic_gate_failure"
        elif accepted is None and len(counted) >= MAX_REWRITER_ATTEMPTS:
            terminal_status = "transport_failure"
        if accepted is not None:
            state, audit = accepted
            if turn["status"] == "complete":
                if turn.get("state") != state or turn.get("automatic_audit") != audit:
                    raise RuntimeError("completed generation turn differs from accepted response")
            elif turn["status"] != "pending":
                raise RuntimeError("accepted generation response has an invalid turn status")
        elif turn["status"] == "complete":
            raise RuntimeError("generation turn completed without an accepted raw response")
        if terminal_status is not None and turn["status"] not in {
            terminal_status,
            "pending",
        }:
            raise RuntimeError("generation terminal status does not derive from attempts")
        if terminal_status is None and accepted is None and turn["status"] != "pending":
            raise RuntimeError("generation turn was excluded before a retry cap/route failure")
        if turn["status"] == "pending" and turn.get("next_request") != expected_request:
            raise RuntimeError("generation next_request does not replay")
        if turn["status"] != "pending" and turn.get("next_request") is not None:
            raise RuntimeError("terminal generation turn retained a mutable next request")
    first_noncomplete = next(
        (turn for turn in journal["turns"] if turn["status"] != "complete"), None
    )
    if first_noncomplete is None:
        if journal.get("status") not in {"pending", "complete"}:
            raise RuntimeError("generation journal status disagrees with three complete turns")
    else:
        first_index = journal["turns"].index(first_noncomplete)
        if any(
            turn["attempts"] or turn["status"] != "pending"
            for turn in journal["turns"][first_index + 1 :]
        ):
            raise RuntimeError("generation attempted a later turn after an incomplete turn")
        expected_journal_status = (
            first_noncomplete["status"]
            if first_noncomplete["status"] in TERMINAL_JOURNAL_STATUSES
            else "pending"
        )
        if journal.get("status") != expected_journal_status:
            raise RuntimeError("generation journal terminal state does not derive from turns")


def _audit_packet_id(
    audit_salt: str, variant: dict[str, Any], generated: dict[str, Any]
) -> str:
    return hashlib.sha256(
        f"{audit_salt}:{variant['variant_sha256']}:{generated['turn']}".encode()
    ).hexdigest()[:24]


def _audit_packet(packet_id: str, generated: dict[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet_id,
        "auditor_blinding": (
            "Condition, model, task, interval, pending result, later history, final answer, and continuation "
            "outcomes are withheld. This packet contains exactly one historical turn."
        ),
        "raw_reasoning": generated["source"]["raw_reasoning"],
        "accepted_compact_anchor": generated["source"]["compact_reasoning"],
        "requested_action": generated["source"]["requested_tool"],
        "candidate_state": generated["state"],
    }


def _write_audit_packets(
    run_dir: Path, variant: dict[str, Any], audit_salt: str
) -> None:
    for generated in variant["generated_turns"]:
        packet_id = _audit_packet_id(audit_salt, variant, generated)
        packet = _audit_packet(packet_id, generated)
        path = run_dir / "audit-packets" / f"{packet_id}.json"
        if path.exists() and json.loads(path.read_text()) != packet:
            raise RuntimeError(f"refusing to overwrite a differing audit packet: {path}")
        if not path.exists():
            atomic_json(path, packet)


def _packet_map_from_variants(
    run_dir: Path, variants: list[tuple[Path, dict[str, Any]]], audit_salt: str
) -> dict[str, Any]:
    entries = []
    for variant_path, variant in variants:
        for generated in variant["generated_turns"]:
            packet_id = _audit_packet_id(audit_salt, variant, generated)
            packet_path = run_dir / "audit-packets" / f"{packet_id}.json"
            if json.loads(packet_path.read_text()) != _audit_packet(packet_id, generated):
                raise RuntimeError("masked packet does not recompute from its variant turn")
            entries.append(
                {
                    "packet_id": packet_id,
                    "packet_file": str(packet_path.relative_to(run_dir)),
                    "packet_sha256": file_sha256(packet_path),
                    "variant_file": str(variant_path.relative_to(run_dir)),
                    "variant_sha256": variant["variant_sha256"],
                    "tier": variant["tier"],
                    "model_name": variant["model_name"],
                    "task_id": variant["task_id"],
                    "parent_source_sha256": variant["parent_source_sha256"],
                    "turn": generated["turn"],
                }
            )
    if len({entry["packet_id"] for entry in entries}) != len(entries):
        raise RuntimeError("opaque semantic-audit packet ID collision")
    expected_paths = {
        run_dir / entry["packet_file"] for entry in entries
    }
    if set((run_dir / "audit-packets").glob("*.json")) != expected_paths:
        raise RuntimeError("masked packet path set is not bijective with variant turns")
    return {"version": 1, "entries": sorted(entries, key=lambda item: item["packet_id"])}


def _variant_from_generation_journal(
    source_path: Path,
    source: dict[str, Any],
    tier: str,
    journal: dict[str, Any],
) -> dict[str, Any]:
    if any(turn["status"] != "complete" for turn in journal["turns"]):
        raise RuntimeError("cannot construct a variant from incomplete generation turns")
    generated_turns = [
        {
            "turn": turn["turn"],
            "source": copy.deepcopy(turn["source"]),
            "word_interval": copy.deepcopy(turn["word_interval"]),
            "state": turn["state"],
            "automatic_audit": copy.deepcopy(turn["automatic_audit"]),
        }
        for turn in journal["turns"]
    ]
    variant = build_tier_variant(
        source,
        tier,
        generated_turns,
        require_exact_action=journal.get("require_exact_action", False),
    )
    variant.update(
        parent_file=source_path.name,
        parent_file_sha256=file_sha256(source_path),
    )
    return variant


def prepare_generation_cell(
    run_dir: Path,
    source_path: Path,
    source: dict[str, Any],
    tier: str,
    rewriter: dict[str, Any],
    code_manifest_sha256: str,
) -> Path:
    journal_path = run_dir / "generation-journals" / tier / source_path.name
    variant_path = run_dir / "variants" / tier / source_path.name
    if variant_path.exists() and not journal_path.exists():
        raise RuntimeError(f"generated variant lacks its request journal: {variant_path}")
    if journal_path.exists():
        return journal_path
    turns = source_turns(source)
    raw_words = [len(turn["raw_reasoning"].split()) for turn in turns]
    compact_words = [len(turn["compact_reasoning"].split()) for turn in turns]
    intervals = turn_word_intervals(tier, raw_words, compact_words)
    journal_turns = [
        {
            "turn": index,
            "source": turn,
            "word_interval": list(interval),
            "status": "pending",
            "attempts": [],
            "next_request": tier_payload(rewriter, tier, turn, index, interval),
        }
        for index, (turn, interval) in enumerate(zip(turns, intervals), 1)
    ]
    journal = {
        "created_at": now(),
        "status": "pending",
        "tier": tier,
        "require_exact_action": rewriter.get("reference_mode") == "raw_only",
        "parent_file": source_path.name,
        "parent_file_sha256": file_sha256(source_path),
        "parent_source_sha256": source["source_sha256"],
        "model_name": source["model_name"],
        "task_id": source["task_id"],
        "code_manifest_sha256": code_manifest_sha256,
        "turns": journal_turns,
    }
    atomic_json(journal_path, journal)
    return journal_path


def generate_cell(
    run_dir: Path,
    source_path: Path,
    source: dict[str, Any],
    tier: str,
    rewriter: dict[str, Any],
    audit_salt: str,
    code_manifest_sha256: str,
) -> None:
    journal_path = run_dir / "generation-journals" / tier / source_path.name
    variant_path = run_dir / "variants" / tier / source_path.name
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        expected_turns = source_turns(source)
        expected_intervals = turn_word_intervals(
            tier,
            [len(turn["raw_reasoning"].split()) for turn in expected_turns],
            [len(turn["compact_reasoning"].split()) for turn in expected_turns],
        )
        if (
            journal.get("tier") != tier
            or journal.get("parent_file") != source_path.name
            or journal.get("parent_file_sha256") != file_sha256(source_path)
            or journal.get("parent_source_sha256") != source["source_sha256"]
            or journal.get("model_name") != source["model_name"]
            or journal.get("task_id") != source["task_id"]
            or journal.get("code_manifest_sha256") != code_manifest_sha256
            or journal.get("require_exact_action", False)
            is not (rewriter.get("reference_mode") == "raw_only")
            or [turn.get("source") for turn in journal.get("turns", [])] != expected_turns
            or [turn.get("word_interval") for turn in journal.get("turns", [])]
            != [list(interval) for interval in expected_intervals]
        ):
            raise RuntimeError("generation journal does not bind to its immutable source cell")
        _validate_generation_journal(journal, source, tier, rewriter)
        if journal.get("status") == "complete" or variant_path.exists():
            expected_variant = _variant_from_generation_journal(
                source_path, source, tier, journal
            )
            if variant_path.exists():
                variant = json.loads(variant_path.read_text())
                if variant != expected_variant:
                    raise RuntimeError("persisted variant does not reconstruct from its journal")
            else:
                variant = expected_variant
                atomic_json(variant_path, variant)
            if journal.get("status") != "complete":
                journal.update(
                    status="complete",
                    completed_at=now(),
                    variant_sha256=variant["variant_sha256"],
                    variant_file=str(variant_path.relative_to(run_dir)),
                )
                atomic_json(journal_path, journal)
        if journal.get("status") in TERMINAL_JOURNAL_STATUSES:
            if journal["status"] == "complete":
                if (
                    journal.get("variant_sha256") != variant["variant_sha256"]
                    or journal.get("variant_file")
                    != str(variant_path.relative_to(run_dir))
                ):
                    raise RuntimeError("complete generation journal has wrong variant binding")
                _write_audit_packets(run_dir, variant, audit_salt)
            return
    else:
        prepare_generation_cell(
            run_dir,
            source_path,
            source,
            tier,
            rewriter,
            code_manifest_sha256,
        )
        journal = json.loads(journal_path.read_text())

    for turn in journal["turns"]:
        if turn["status"] == "complete":
            continue
        if turn["status"] in TERMINAL_JOURNAL_STATUSES:
            journal.update(status=turn["status"], completed_at=now())
            atomic_json(journal_path, journal)
            return
        if turn["attempts"] and turn["attempts"][-1]["status"] == "in_flight":
            turn["attempts"][-1].update(
                status="ambiguous_after_restart",
                ambiguity=(
                    "Process ended after pre-call journal write; the request may or may not have reached the "
                    "provider and will not be resent automatically."
                ),
            )
            atomic_json(journal_path, journal)
            raise RuntimeError(f"ambiguous generation request requires adjudication: {journal_path}")
        if turn["attempts"] and turn["attempts"][-1]["status"] in {
            "ambiguous_after_restart",
            "ambiguous_transport_error",
        }:
            raise RuntimeError(f"ambiguous generation request requires adjudication: {journal_path}")
        if (
            turn["attempts"]
            and turn["attempts"][-1]["status"] == "returned_http_error"
            and turn["attempts"][-1].get("retryable") is False
        ):
            turn.update(status="transport_failure", completed_at=now())
            turn.pop("next_request", None)
            journal.update(status="transport_failure", completed_at=now())
            atomic_json(journal_path, journal)
            return
        if turn["attempts"] and turn["attempts"][-1]["status"] == "response_received":
            if _process_returned_generation_attempt(
                journal_path, journal, turn, turn["attempts"][-1], rewriter
            ):
                continue
            if journal["status"] in TERMINAL_JOURNAL_STATUSES:
                return
        if turn["attempts"] and turn["attempts"][-1]["status"] == "accepted_draft":
            accepted_attempt = turn["attempts"][-1]
            audit = automatic_turn_audit(
                turn["source"],
                accepted_attempt["parsed_state"],
                tuple(turn["word_interval"]),
                require_exact_action=rewriter.get("reference_mode") == "raw_only",
            )
            if not audit["passed"] or audit != accepted_attempt["automatic_audit"]:
                raise RuntimeError("accepted generation draft no longer verifies")
            turn.update(
                status="complete",
                state=accepted_attempt["parsed_state"],
                automatic_audit=audit,
                completed_at=now(),
            )
            turn.pop("next_request", None)
            atomic_json(journal_path, journal)
            continue
        while turn["status"] == "pending":
            drafts, total = _attempt_counts(turn)
            if drafts >= MAX_REWRITER_DRAFTS or total >= MAX_REWRITER_ATTEMPTS:
                status = (
                    "automatic_gate_failure"
                    if drafts >= MAX_REWRITER_DRAFTS
                    else "transport_failure"
                )
                turn.update(status=status, completed_at=now())
                turn.pop("next_request", None)
                journal.update(status=status, completed_at=now())
                atomic_json(journal_path, journal)
                return
            request = copy.deepcopy(turn["next_request"])
            request_body, request_digest = request_body_record(request)
            attempt = {
                "status": "in_flight",
                "created_at": now(),
                "request": request,
                "request_body": request_body,
                "request_body_sha256": request_digest,
            }
            turn["attempts"].append(attempt)
            atomic_json(journal_path, journal)
            print(
                f"generate {tier} {source['model_name']} {source['task_id']} turn={turn['turn']} "
                f"draft={drafts + 1} transport={total + 1}",
                flush=True,
            )
            try:
                raw_body, wall = post_chat_once_raw(request)
            except BudgetFloorReached as exc:
                attempt.update(status="budget_pause", completed_at=now(), error=str(exc))
                atomic_json(journal_path, journal)
                raise
            except ChatAttemptError as exc:
                if exc.http_status is not None and exc.response_body_bytes is not None:
                    persist_returned_http_error(attempt, exc)
                    atomic_json(journal_path, journal)
                    if not attempt["retryable"]:
                        turn.update(status="transport_failure", completed_at=now())
                        turn.pop("next_request", None)
                        journal.update(status="transport_failure", completed_at=now())
                        atomic_json(journal_path, journal)
                        return
                    _, total = _attempt_counts(turn)
                    if total >= MAX_REWRITER_ATTEMPTS:
                        turn.update(status="transport_failure", completed_at=now())
                        turn.pop("next_request", None)
                        journal.update(status="transport_failure", completed_at=now())
                        atomic_json(journal_path, journal)
                        return
                    tier_offset = (
                        FOUR_FRESH_TIERS.index(tier)
                        if tier in FOUR_FRESH_TIERS
                        else 0
                    )
                    time.sleep(min(30 * total, 120) + 10 * tier_offset)
                    continue
                attempt.update(
                    status="ambiguous_transport_error",
                    completed_at=now(),
                    ambiguity="No response body returned; the paid request will not be resent automatically.",
                    **exc.record(),
                )
                atomic_json(journal_path, journal)
                raise RuntimeError(
                    f"ambiguous generation transport requires adjudication: {journal_path}"
                ) from exc
            persist_raw_response(attempt, raw_body, wall)
            attempt["response_received_at"] = now()
            atomic_json(journal_path, journal)
            if _process_returned_generation_attempt(
                journal_path, journal, turn, attempt, rewriter
            ):
                break
            if journal["status"] in TERMINAL_JOURNAL_STATUSES:
                return

    variant = _variant_from_generation_journal(source_path, source, tier, journal)
    atomic_json(variant_path, variant)
    journal.update(
        status="complete",
        completed_at=now(),
        variant_sha256=variant["variant_sha256"],
        variant_file=str(variant_path.relative_to(run_dir)),
    )
    atomic_json(journal_path, journal)
    _write_audit_packets(run_dir, variant, audit_salt)


def _validate_rewriter_canary_response(
    response: dict[str, Any],
    rewriter: dict[str, Any],
    source_turn: dict[str, Any],
    interval: tuple[int, int],
) -> tuple[str, dict[str, Any]]:
    if (
        response.get("model") != rewriter["model"]
        or response.get("provider") != expected_rewriter_provider(rewriter)
    ):
        raise RuntimeError("rewriter canary returned the wrong model/provider")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("rewriter canary did not return exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if (
        choice.get("finish_reason") != "stop"
        or not isinstance(message, dict)
        or message.get("reasoning") not in (None, "")
        or message.get("reasoning_content") not in (None, "")
        or message.get("reasoning_details") not in (None, [])
    ):
        raise RuntimeError("rewriter canary finish/message reasoning exclusion failed")
    try:
        state = parse_tier_response(response)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("rewriter canary strict state schema failed") from exc
    audit = automatic_turn_audit(
        source_turn,
        state,
        interval,
        require_exact_action=True,
    )
    if not audit["passed"]:
        raise RuntimeError("rewriter canary automatic dose/fidelity audit failed")
    return state, audit


def verify_completed_rewriter_canary(
    run_dir: Path, run: dict[str, Any]
) -> str:
    journal_path = run_dir / "rewriter-canary-journal.json"
    if (
        run.get("rewriter_canary_passed") is not True
        or not journal_path.exists()
        or run.get("rewriter_canary_journal_sha256") != file_sha256(journal_path)
    ):
        raise RuntimeError("baseline lacks its exact passed rewriter canary")
    journal = strict_json_loads(journal_path.read_bytes())
    snapshot = run["snapshot"]
    if snapshot.get("rewriter") != load_rewriter_profile("deepseek_self"):
        raise RuntimeError("baseline rewriter profile changed before canary verification")
    parent_file = journal.get("parent_file")
    if not isinstance(parent_file, str) or Path(parent_file).name != parent_file:
        raise RuntimeError("rewriter canary parent binding is invalid")
    source_path = Path(snapshot["parent_run"]) / "sources" / parent_file
    source = strict_json_loads(source_path.read_bytes())
    turn_index = journal.get("historical_turn")
    if (
        journal.get("status") != "complete"
        or journal.get("task_id") != "hospital-decision-v2"
        or source.get("task_id") != "hospital-decision-v2"
        or source.get("model_name") != "deepseek-v4-flash"
        or journal.get("parent_file_sha256") != file_sha256(source_path)
        or journal.get("parent_source_sha256") != source.get("source_sha256")
        or turn_index != 1
    ):
        raise RuntimeError("completed rewriter canary provenance changed")
    turns = source_turns(source)
    if turn_index > len(turns):
        raise RuntimeError("rewriter canary historical turn is invalid")
    source_turn = turns[turn_index - 1]
    tier = journal.get("tier")
    if tier != "full_sentence_compact":
        raise RuntimeError("rewriter canary tier is invalid")
    interval = word_interval(
        tier,
        len(source_turn["raw_reasoning"].split()),
        len(source_turn["compact_reasoning"].split()),
    )
    request = tier_payload(
        snapshot["rewriter"], tier, source_turn, turn_index, interval
    )
    request["seed"] = REWRITER_CANARY_SEED
    request_body, request_digest = request_body_record(request)
    if (
        journal.get("word_interval") != list(interval)
        or journal.get("request") != request
        or journal.get("request_body") != request_body
        or journal.get("request_body_sha256") != request_digest
    ):
        raise RuntimeError("rewriter canary request binding changed")
    response = parse_persisted_response(journal)
    state, audit = _validate_rewriter_canary_response(
        response, snapshot["rewriter"], source_turn, interval
    )
    if (
        journal.get("response") != response
        or journal.get("state") != state
        or journal.get("automatic_audit") != audit
    ):
        raise RuntimeError("completed rewriter canary does not recompute")
    return file_sha256(journal_path)


def run_rewriter_canary(
    run_dir: Path,
    parent_run: Path,
    rewriter: dict[str, Any],
    canary_task: str,
) -> None:
    run_path = run_dir / "run.json"
    run = strict_json_loads(run_path.read_bytes())
    if run.get("outcomes_launched") or run.get("variants_frozen"):
        raise RuntimeError("rewriter canary must precede variant/outcome freeze")
    generation_started = any((run_dir / "generation-journals").glob("*/*.json"))
    if (
        run["snapshot"]["tier_mode"] != "four_fresh"
        or rewriter.get("reference_mode") != "raw_only"
        or rewriter != run["snapshot"]["rewriter"]
    ):
        raise RuntimeError("rewriter canary requires the frozen four-fresh self profile")
    selected_sources = load_parent_sources(
        parent_run,
        tuple(run["snapshot"]["selected_models"]),
        tuple(run["snapshot"]["selected_tasks"]),
    )
    matches = [item for item in selected_sources if item[1]["task_id"] == canary_task]
    if len(matches) != 1:
        raise ValueError("canary task must identify one frozen source")
    if not generation_started:
        record_invocation(run_path, run, "rewriter_canary")
    source_path, source = matches[0]
    turn_index = 1
    canary_tier = "full_sentence_compact"
    turn = source_turns(source)[turn_index - 1]
    interval = word_interval(
        canary_tier,
        len(turn["raw_reasoning"].split()),
        len(turn["compact_reasoning"].split()),
    )
    request = tier_payload(
        rewriter, canary_tier, turn, turn_index, interval
    )
    request["seed"] = REWRITER_CANARY_SEED
    body, body_sha256 = request_body_record(request)
    journal_path = run_dir / "rewriter-canary-journal.json"
    expected_binding = {
        "task_id": canary_task,
        "parent_file": source_path.name,
        "parent_file_sha256": file_sha256(source_path),
        "parent_source_sha256": source["source_sha256"],
        "historical_turn": turn_index,
        "tier": canary_tier,
        "word_interval": list(interval),
        "request": request,
        "request_body": body,
        "request_body_sha256": body_sha256,
    }
    if journal_path.exists():
        journal = strict_json_loads(journal_path.read_bytes())
        if any(journal.get(key) != value for key, value in expected_binding.items()):
            raise RuntimeError("rewriter canary journal binding changed")
    else:
        journal = {
            "status": "prepared",
            "prepared_at": now(),
            **expected_binding,
            "attempt_policy": "exactly one request; no resend after ambiguous send",
        }
        atomic_json(journal_path, journal)
    status = journal.get("status")
    if status == "complete":
        response = parse_persisted_response(journal)
        state, audit = _validate_rewriter_canary_response(
            response, rewriter, turn, interval
        )
        if (
            journal.get("response") != response
            or journal.get("state") != state
            or journal.get("automatic_audit") != audit
        ):
            raise RuntimeError("completed rewriter canary does not recompute")
    else:
        if status == "in_flight":
            journal.update(
                status="ambiguous_after_restart",
                completed_at=now(),
                ambiguity="The canary may have reached the provider and will not be resent.",
            )
            atomic_json(journal_path, journal)
            raise RuntimeError("ambiguous rewriter canary requires adjudication")
        if status in {
            "ambiguous_after_restart",
            "ambiguous_transport_error",
            "returned_http_error",
            "failed",
        }:
            raise RuntimeError(f"terminal rewriter canary status: {status}")
        if status == "response_received":
            response = parse_persisted_response(journal)
        elif status in {"prepared", "budget_pause"}:
            journal.update(status="in_flight", sent_at=now())
            atomic_json(journal_path, journal)
            try:
                raw_body, wall = post_chat_once_raw(request)
            except BudgetFloorReached as exc:
                journal.update(status="budget_pause", completed_at=now(), error=str(exc))
                atomic_json(journal_path, journal)
                raise
            except ChatAttemptError as exc:
                if exc.http_status is not None and exc.response_body_bytes is not None:
                    persist_returned_http_error(journal, exc)
                    atomic_json(journal_path, journal)
                    raise RuntimeError("rewriter canary returned an HTTP provider error") from exc
                journal.update(
                    status="ambiguous_transport_error",
                    completed_at=now(),
                    ambiguity="No response body returned; the canary will not be resent.",
                    **exc.record(),
                )
                atomic_json(journal_path, journal)
                raise RuntimeError("ambiguous rewriter canary transport") from exc
            persist_raw_response(journal, raw_body, wall)
            journal["response_received_at"] = now()
            atomic_json(journal_path, journal)
            response = parse_persisted_response(journal)
        else:
            raise RuntimeError(f"unknown rewriter canary status: {status}")
        try:
            state, audit = _validate_rewriter_canary_response(
                response, rewriter, turn, interval
            )
        except RuntimeError as exc:
            journal.update(
                status="failed",
                completed_at=now(),
                response=response,
                validation_error=str(exc),
            )
            atomic_json(journal_path, journal)
            raise
        journal.update(
            status="complete",
            completed_at=now(),
            response=response,
            state=state,
            automatic_audit=audit,
        )
        atomic_json(journal_path, journal)
    canary_values = {
        "rewriter_canary_passed": True,
        "rewriter_canary_completed_at": journal["completed_at"],
        "rewriter_canary_journal_sha256": file_sha256(journal_path),
    }
    if generation_started:
        if any(run.get(key) != value for key, value in canary_values.items()):
            raise RuntimeError("rewriter canary binding changed after generation started")
        return
    run.update(
        status="rewriter_canary_passed_before_variant_calls",
        **canary_values,
    )
    atomic_json(run_path, run)


def _completed_source_doses(
    run_dir: Path,
    selected_sources: list[tuple[Path, dict[str, Any]]],
    generated_tiers: tuple[str, ...],
    tier_mode: str,
) -> dict[str, dict[str, int]]:
    doses: dict[str, dict[str, int]] = {}
    for source_path, source in selected_sources:
        paths = {
            tier: run_dir / "variants" / tier / source_path.name
            for tier in generated_tiers
        }
        if not all(path.exists() for path in paths.values()):
            continue
        generated = {
            tier: strict_json_loads(path.read_bytes()) for tier, path in paths.items()
        }
        histories = block_histories(source, generated, tier_mode)
        words = {arm: _history_reasoning_words(history) for arm, history in histories.items()}
        dose_order = (
            "clean",
            "relaxed_full_sentence_compact",
            "full_sentence_compact",
            "telegraphic_compact",
            "ultra_telegraphic",
        )
        if any(
            words[left] <= words[right]
            for left, right in zip(dose_order, dose_order[1:])
        ):
            raise RuntimeError(
                f"achieved source history dose is not strictly ordered: "
                f"{source['task_id']}: {words}"
            )
        doses[source["task_id"]] = words
    return doses


def _validate_development_canary_audits(
    run_dir: Path, run: dict[str, Any], task_id: str
) -> dict[str, str]:
    import review_history_tier_packets as review_runner

    generated_tiers = tuple(run["snapshot"]["generated_tiers"])
    entries = []
    for path in sorted((run_dir / "variants").glob("*/*.json")):
        variant = strict_json_loads(path.read_bytes())
        if variant.get("task_id") != task_id:
            continue
        verify_tier_variant(variant)
        if variant.get("require_exact_action") is not True:
            raise RuntimeError("development canary variant lacks exact-action gating")
        for generated in variant["generated_turns"]:
            packet_id = _audit_packet_id(
                run["snapshot"]["audit_packet_salt"], variant, generated
            )
            packet_path = run_dir / "audit-packets" / f"{packet_id}.json"
            if strict_json_loads(packet_path.read_bytes()) != _audit_packet(
                packet_id, generated
            ):
                raise RuntimeError("development canary packet does not reconstruct")
            entries.append(
                {
                    "packet_id": packet_id,
                    "packet_file": str(packet_path.relative_to(run_dir)),
                    "packet_sha256": file_sha256(packet_path),
                    "variant_file": str(path.relative_to(run_dir)),
                    "variant_sha256": variant["variant_sha256"],
                    "tier": variant["tier"],
                    "turn": generated["turn"],
                }
            )
    entries = sorted(entries, key=lambda entry: entry["packet_id"])
    if (
        len(entries) != len(generated_tiers) * 3
        or {entry["tier"] for entry in entries} != set(generated_tiers)
    ):
        raise RuntimeError("development canary lacks its exact four-tier packet set")
    semantic_path = run_dir / f"semantic-canary-audit-{task_id}.json"
    style_path = run_dir / f"style-canary-audit-{task_id}.json"
    semantic = strict_json_loads(semantic_path.read_bytes())
    style = strict_json_loads(style_path.read_bytes())
    workers = run["snapshot"]["execution"]["workers"]
    semantic_decisions = []
    for index, entry in enumerate(entries):
        packet = strict_json_loads((run_dir / entry["packet_file"]).read_bytes())
        request = review_runner.payload(
            review_runner.SEMANTIC_SYSTEM,
            {
                "raw_reasoning": packet["raw_reasoning"],
                "accepted_compact_anchor": packet["accepted_compact_anchor"],
                "requested_action": packet["requested_action"],
                "candidate_state": packet["candidate_state"],
            },
            SEMANTIC_CHECKS,
            953000 + index,
        )
        journal_path = (
            run_dir
            / "review-journals"
            / f"semantic-canary-{task_id}"
            / f"{entry['packet_id']}.json"
        )
        journal = strict_json_loads(journal_path.read_bytes())
        if (
            journal.get("status") != "complete"
            or journal.get("request") != request
            or journal.get("request_sha256") != canonical_hash(request)
        ):
            raise RuntimeError("development semantic review request does not replay")
        review = review_runner.parse_response(
            copy.deepcopy(journal), SEMANTIC_CHECKS
        )
        if journal.get("review") != review:
            raise RuntimeError("development semantic decision differs from raw response")
        checks = {key: review[key] for key in SEMANTIC_CHECKS}
        semantic_decisions.append(
            {
                "packet_id": entry["packet_id"],
                "passed": all(checks.values()),
                "checks": checks,
                "notes": review["notes"],
            }
        )
    try:
        semantic_locked_at = datetime.fromisoformat(semantic["semantic_locked_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("development semantic canary lacks a lock time") from exc
    if (
        semantic_locked_at.tzinfo is None
        or semantic.get("timing")
        != "after_one_task_generation_before_remaining_development_generation"
        or semantic.get("task_id") != task_id
        or semantic.get("entries") != entries
        or semantic.get("passed") is not all(
            decision["passed"] for decision in semantic_decisions
        )
        or semantic.get("passed") is not True
        or semantic.get("decisions") != semantic_decisions
        or semantic.get("reviewer") != review_runner.REVIEWER
        or semantic.get("reviewer_condition_labels_withheld") is not True
        or semantic.get("continuation_outcomes_inspected") is not False
        or semantic.get("pending_tool_results_inspected") is not False
        or semantic.get("review_workers") != workers
        or not isinstance(semantic.get("command"), list)
    ):
        raise RuntimeError("development semantic canary does not recompute")
    _verify_frozen_file_manifest(
        run_dir / "review-journals" / f"semantic-canary-{task_id}",
        semantic["review_journal_manifest_sha256"],
    )
    style_decisions = []
    semantic_by_id = {
        decision["packet_id"]: decision for decision in semantic_decisions
    }
    for index, entry in enumerate(entries):
        packet = strict_json_loads((run_dir / entry["packet_file"]).read_bytes())
        variant = strict_json_loads((run_dir / entry["variant_file"]).read_bytes())
        generated = next(
            item
            for item in variant["generated_turns"]
            if item["turn"] == entry["turn"]
        )
        request = review_runner.payload(
            review_runner.STYLE_SYSTEM,
            {
                "tier": entry["tier"],
                "tier_instruction": TIER_INSTRUCTIONS[entry["tier"]],
                "raw_reasoning": packet["raw_reasoning"],
                "accepted_compact_anchor": packet["accepted_compact_anchor"],
                "candidate_state": packet["candidate_state"],
                "word_interval_inclusive": generated["word_interval"],
                "candidate_words": generated["automatic_audit"]["variant_words"],
            },
            {"tier_compliance", "no_semantic_padding"},
            954000 + index,
        )
        journal_path = (
            run_dir
            / "review-journals"
            / f"style-canary-{task_id}"
            / f"{entry['packet_id']}.json"
        )
        journal = strict_json_loads(journal_path.read_bytes())
        if (
            journal.get("status") != "complete"
            or journal.get("request") != request
            or journal.get("request_sha256") != canonical_hash(request)
        ):
            raise RuntimeError("development style review request does not replay")
        review = review_runner.parse_response(
            copy.deepcopy(journal), {"tier_compliance", "no_semantic_padding"}
        )
        if journal.get("review") != review:
            raise RuntimeError("development style decision differs from raw response")
        checks = {
            "tier_compliance": review["tier_compliance"],
            "dose_compliance": bool(generated["automatic_audit"]["passed"]),
            "no_semantic_padding": review["no_semantic_padding"],
        }
        passed = all(checks.values())
        style_decisions.append(
            {
                "packet_id": entry["packet_id"],
                "tier": entry["tier"],
                "passed": passed,
                "accepted": semantic_by_id[entry["packet_id"]]["passed"] and passed,
                "checks": checks,
                "notes": review["notes"],
            }
        )
    try:
        style_reviewed_at = datetime.fromisoformat(style["style_reviewed_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("development style canary lacks a review time") from exc
    if (
        style_reviewed_at.tzinfo is None
        or style_reviewed_at < semantic_locked_at
        or style.get("timing")
        != "after_semantic_canary_lock_before_remaining_development_generation"
        or style.get("task_id") != task_id
        or style.get("semantic_canary_sha256") != file_sha256(semantic_path)
        or style.get("entries") != entries
        or style.get("passed") is not all(
            decision["accepted"] for decision in style_decisions
        )
        or style.get("passed") is not True
        or style.get("decisions") != style_decisions
        or style.get("reviewer") != review_runner.REVIEWER
        or style.get("continuation_outcomes_inspected") is not False
        or style.get("pending_tool_results_inspected") is not False
        or style.get("review_workers") != workers
        or not isinstance(style.get("command"), list)
    ):
        raise RuntimeError("development style canary does not recompute")
    _verify_frozen_file_manifest(
        run_dir / "review-journals" / f"style-canary-{task_id}",
        style["review_journal_manifest_sha256"],
    )
    return {
        "development_semantic_canary_sha256": file_sha256(semantic_path),
        "development_style_canary_sha256": file_sha256(style_path),
        "development_semantic_canary_review_manifest_sha256": semantic[
            "review_journal_manifest_sha256"
        ],
        "development_style_canary_review_manifest_sha256": style[
            "review_journal_manifest_sha256"
        ],
    }


def run_generate(
    run_dir: Path,
    parent_run: Path,
    rewriter: dict[str, Any],
    generation_tasks: tuple[str, ...] | None = None,
) -> None:
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text())
    if run.get("outcomes_launched") or run.get("variants_frozen"):
        raise RuntimeError("cannot generate variants after the pre-outcome freeze")
    if rewriter != run["snapshot"]["rewriter"]:
        raise RuntimeError("generation rewriter differs from the frozen profile")
    protocol_name = run["snapshot"].get("protocol", {}).get("name")
    if (
        rewriter.get("reference_mode") == "raw_only"
        and protocol_name != "deepseek_self_expansion"
    ):
        canary_path = run_dir / "rewriter-canary-journal.json"
        if (
            run.get("rewriter_canary_passed") is not True
            or not canary_path.exists()
            or file_sha256(canary_path) != run.get("rewriter_canary_journal_sha256")
        ):
            raise RuntimeError("pass the frozen DeepSeek structured-output canary first")
    selected_sources = load_parent_sources(
        parent_run,
        tuple(run["snapshot"]["selected_models"]),
        tuple(run["snapshot"]["selected_tasks"]),
    )
    generated_tiers = tuple(run["snapshot"]["generated_tiers"])
    tier_mode = run["snapshot"]["tier_mode"]
    if generated_tiers != generated_tiers_for(tier_mode):
        raise RuntimeError("frozen tier mode and generated-tier set disagree")
    selected_task_set = set(run["snapshot"]["selected_tasks"])
    requested_task_set = set(generation_tasks or selected_task_set)
    if not requested_task_set or not requested_task_set <= selected_task_set:
        raise ValueError("generation task subset is empty or outside the frozen cohort")
    canary_bindings: dict[str, str] | None = None
    if (
        tier_mode == "four_fresh"
        and selected_task_set == SELF_REWRITE_DEVELOPMENT_TASKS
        and requested_task_set != {"hospital-decision-v2"}
    ):
        try:
            canary_bindings = _validate_development_canary_audits(
                run_dir, run, "hospital-decision-v2"
            )
        except (FileNotFoundError, KeyError) as exc:
            raise RuntimeError(
                "pass the hospital blinded rewrite canary before mass generation"
            ) from exc
    record_invocation(run_path, run, "generate")
    jobs = [
        (source_path, source, tier)
        for source_path, source in selected_sources
        if source["task_id"] in requested_task_set
        for tier in generated_tiers
    ]
    for source_path, source, tier in jobs:
        prepare_generation_cell(
            run_dir,
            source_path,
            source,
            tier,
            rewriter,
            run["snapshot"]["executable_manifest_sha256"],
        )

    def execute(job: tuple[Path, dict[str, Any], str]) -> None:
        source_path, source, tier = job
        generate_cell(
            run_dir,
            source_path,
            source,
            tier,
            rewriter,
            run["snapshot"]["audit_packet_salt"],
            run["snapshot"]["executable_manifest_sha256"],
        )

    workers = run["snapshot"]["execution"]["workers"]
    run_bounded_jobs(jobs, workers, execute)
    expected_paths = {
        f"{tier}/{source['file']}"
        for source in run["snapshot"]["intended_sources"]
        for tier in generated_tiers
    }
    journals = sorted((run_dir / "generation-journals").glob("*/*.json"))
    actual_paths = {
        str(path.relative_to(run_dir / "generation-journals")) for path in journals
    }
    if not actual_paths <= expected_paths:
        raise RuntimeError("generation journals contain cells outside the frozen matrix")
    journal_values = {path: strict_json_loads(path.read_bytes()) for path in journals}
    requested_paths = {
        f"{tier}/{source_path.name}"
        for source_path, source in selected_sources
        if source["task_id"] in requested_task_set
        for tier in generated_tiers
    }
    if not requested_paths <= actual_paths or any(
        journal_values[run_dir / "generation-journals" / relative].get("status")
        not in TERMINAL_JOURNAL_STATUSES
        for relative in requested_paths
    ):
        raise RuntimeError("requested generation subset did not reach terminal cell states")
    achieved_doses = _completed_source_doses(
        run_dir, selected_sources, generated_tiers, tier_mode
    )
    terminal = sum(
        journal.get("status") in TERMINAL_JOURNAL_STATUSES
        for journal in journal_values.values()
    )
    expected_cells = len(expected_paths)
    if actual_paths != expected_paths or terminal != expected_cells:
        run.update(
            status="variants_generation_in_progress",
            generation_terminal_cells=terminal,
            generated_variant_files=len(
                list((run_dir / "variants").glob("*/*.json"))
            ),
            achieved_source_doses=achieved_doses,
            generation_workers=workers,
            generation_last_subset=sorted(requested_task_set),
            generation_progress_at=now(),
        )
        atomic_json(run_path, run)
        return
    variants = sorted((run_dir / "variants").glob("*/*.json"))
    variant_values = [(path, json.loads(path.read_text())) for path in variants]
    _packet_map_from_variants(
        run_dir, variant_values, run["snapshot"]["audit_packet_salt"]
    )
    packets = sorted((run_dir / "audit-packets").glob("*.json"))
    journal_manifest = _write_manifest(run_dir / "generation-journals", journals)
    candidate_manifest = _write_manifest(run_dir / "variants", variants)
    packet_manifest = _write_manifest(run_dir / "audit-packets", packets)
    generation_values = {
        "status": "variants_and_masked_packets_frozen_pending_semantic_audit",
        "generation_terminal_cells": terminal,
        "generated_variant_files": len(variants),
        "achieved_source_doses": achieved_doses,
        "generation_journal_manifest_sha256": file_sha256(journal_manifest),
        "candidate_variant_manifest_sha256": file_sha256(candidate_manifest),
        "audit_packet_manifest_sha256": file_sha256(packet_manifest),
        "generation_workers": workers,
    }
    if protocol_name == "deepseek_self_development":
        if canary_bindings is None:
            canary_bindings = _validate_development_canary_audits(
                run_dir, run, "hospital-decision-v2"
            )
        generation_values.update(canary_bindings)
    if run.get("status") == generation_values["status"]:
        if any(run.get(key) != value for key, value in generation_values.items()):
            raise RuntimeError("existing generation freeze metadata does not replay")
        return
    run.update(**generation_values, generation_completed_at=now())
    atomic_json(run_path, run)


def _validate_semantic_audit(
    audit: dict[str, Any],
    packet_ids: set[str],
    run: dict[str, Any],
    run_dir: Path,
) -> dict[str, bool]:
    if (
        audit.get("timing") != "before_condition_mapping_and_tier_outcomes"
        or audit.get("audit_packet_manifest_sha256")
        != run["audit_packet_manifest_sha256"]
        or audit.get("condition_mapping_inspected") is not False
        or audit.get("continuation_outcomes_inspected") is not False
        or audit.get("pending_tool_results_inspected") is not False
        or audit.get("source_history_files_inspected") is not False
    ):
        raise RuntimeError("semantic audit lacks the isolated label-blinded attestation")
    active_self_protocol = run.get("snapshot", {}).get("protocol", {}).get("name") in {
        "deepseek_self_development",
        "deepseek_self_expansion",
    }
    review_manifest_sha256 = audit.get("review_journal_manifest_sha256")
    if review_manifest_sha256 is not None:
        _verify_frozen_file_manifest(
            run_dir / "review-journals" / "semantic", review_manifest_sha256
        )
    if active_self_protocol and (
        review_manifest_sha256 is None
        or audit.get("review_workers")
        != run["snapshot"]["execution"]["workers"]
        or not isinstance(audit.get("command"), list)
    ):
        raise RuntimeError("self-rewrite semantic audit lacks frozen reviewer provenance")
    try:
        semantic_locked_at = datetime.fromisoformat(audit["semantic_locked_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("semantic audit lacks a valid lock timestamp") from exc
    if semantic_locked_at.tzinfo is None:
        raise RuntimeError("semantic audit lock timestamp must be timezone-aware")
    decisions = audit.get("decisions", [])
    by_packet = {decision.get("packet_id"): decision for decision in decisions}
    if len(decisions) != len(by_packet) or set(by_packet) != packet_ids:
        raise RuntimeError("semantic audit must decide every opaque packet exactly once")
    passed = {}
    for packet_id, decision in by_packet.items():
        if set(decision) != {"packet_id", "passed", "checks", "notes"}:
            raise RuntimeError("semantic decision contains condition metadata or missing fields")
        checks = decision["checks"]
        recomputed = (
            set(checks) == SEMANTIC_CHECKS
            and all(value is True for value in checks.values())
        )
        if (
            decision["passed"] is not recomputed
            or not isinstance(decision["notes"], str)
        ):
            raise RuntimeError("semantic decision disagrees with its complete rubric")
        passed[packet_id] = recomputed
    return passed


def unblind_packets(run_dir: Path) -> None:
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text())
    if run.get("outcomes_launched") or run.get("variants_frozen"):
        raise RuntimeError("cannot unblind packet labels after tier outcomes/freeze")
    if run["snapshot"].get("protocol", {}).get("name") == "deepseek_self_development":
        canary_bindings = _validate_development_canary_audits(
            run_dir, run, "hospital-decision-v2"
        )
        if any(run.get(key) != value for key, value in canary_bindings.items()):
            raise RuntimeError("development canary evidence changed before unblinding")
    _verify_frozen_file_manifest(
        run_dir / "generation-journals", run["generation_journal_manifest_sha256"]
    )
    _verify_frozen_file_manifest(
        run_dir / "variants", run["candidate_variant_manifest_sha256"]
    )
    _verify_frozen_file_manifest(
        run_dir / "audit-packets", run["audit_packet_manifest_sha256"]
    )
    variant_values = [
        (path, json.loads(path.read_text()))
        for path in sorted((run_dir / "variants").glob("*/*.json"))
    ]
    packet_map = _packet_map_from_variants(
        run_dir, variant_values, run["snapshot"]["audit_packet_salt"]
    )
    packet_ids = {entry["packet_id"] for entry in packet_map["entries"]}
    semantic_path = run_dir / "semantic-audit.json"
    if not semantic_path.exists():
        raise RuntimeError("write label-blinded semantic-audit.json before unblinding")
    semantic_audit = json.loads(semantic_path.read_text())
    _validate_semantic_audit(semantic_audit, packet_ids, run, run_dir)
    map_path = run_dir / "audit-packet-map.json"
    semantic_digest = file_sha256(semantic_path)
    status = run.get("status")
    if status == "semantic_audit_frozen_pending_style_audit":
        if (
            semantic_digest != run.get("semantic_audit_sha256")
            or not map_path.exists()
            or file_sha256(map_path) != run.get("audit_packet_map_sha256")
            or json.loads(map_path.read_text()) != packet_map
        ):
            raise RuntimeError("existing semantic lock or condition map changed")
        return
    if status == "variants_and_masked_packets_frozen_pending_semantic_audit":
        if map_path.exists():
            raise RuntimeError("condition mapping existed before the semantic lock")
        run.update(
            status="semantic_audit_frozen_map_pending",
            semantic_audit_sha256=semantic_digest,
            semantic_locked_at=semantic_audit["semantic_locked_at"],
        )
        atomic_json(run_path, run)
    elif status == "semantic_audit_frozen_map_pending":
        if (
            semantic_digest != run.get("semantic_audit_sha256")
            or semantic_audit["semantic_locked_at"] != run.get("semantic_locked_at")
        ):
            raise RuntimeError("semantic audit changed after its pre-map lock")
    else:
        raise RuntimeError(f"invalid unblinding stage: {status}")
    reconcile_immutable_json(map_path, packet_map)
    run.update(
        status="semantic_audit_frozen_pending_style_audit",
        audit_packet_map_sha256=file_sha256(map_path),
        condition_mapping_created_at=now(),
    )
    atomic_json(run_path, run)


def _write_manifest(directory: Path, paths: list[Path]) -> Path:
    manifest = directory / "SHA256SUMS"
    content = "".join(
        f"{file_sha256(path)}  {path.relative_to(directory)}\n" for path in sorted(paths)
    )
    temporary = manifest.with_name(".SHA256SUMS.tmp")
    if temporary.exists():
        temporary.unlink()
    actual = _manifestable_files(directory, manifest)
    intended = {str(path.relative_to(directory)): path for path in paths}
    if set(actual) != set(intended) or any(actual[name] != intended[name] for name in actual):
        raise RuntimeError(f"manifest inputs are not the exact regular-file set in {directory}")
    if manifest.exists():
        if manifest.read_text() != content:
            raise RuntimeError(f"refusing to overwrite a differing immutable manifest: {manifest}")
        return manifest
    _durable_replace_text(manifest, temporary, content)
    return manifest


def _balanced_randomization(
    blocks: list[dict[str, Any]], tasks: dict[str, dict[str, Any]], replicates: int
) -> dict[str, Any]:
    entries = []
    for model_index, model_name in enumerate(sorted({block["model_name"] for block in blocks})):
        model_blocks = [block for block in blocks if block["model_name"] == model_name]
        units = [
            (block, replicate)
            for block in sorted(model_blocks, key=lambda value: value["task_id"])
            for replicate in range(1, replicates + 1)
        ]
        bases = []
        slot_count = None
        for slot in range(3):
            base = list(ARMS)
            random.Random(910000 + model_index * 100 + slot).shuffle(base)
            bases.append(base)
        for unit_index, (block, replicate) in enumerate(units):
            task = tasks[block["task_id"]]
            slots = [phase["name"] for phase in task["phases"][task["fork_after"] :]] + ["final"]
            if slot_count is None:
                slot_count = len(slots)
            if len(slots) != len(bases):
                raise RuntimeError("tier randomization expects exactly three post-fork slots")
            orders = {}
            for slot_index, slot_name in enumerate(slots):
                base = bases[slot_index]
                rotation = (unit_index + slot_index) % len(base)
                orders[slot_name] = base[rotation:] + base[:rotation]
            entries.append(
                {
                    "model_name": model_name,
                    "task_id": block["task_id"],
                    "parent_source_sha256": block["parent_source_sha256"],
                    "replicate": replicate,
                    "continuation_seed": 930000 + model_index * 100000 + unit_index * 100,
                    "orders": orders,
                }
            )
    return {
        "created_at": now(),
        "timing": "before_tier_outcomes",
        "method": "phase-specific shuffled bases with cyclic rotations balanced within model",
        "arms": list(ARMS),
        "entries": entries,
    }


def block_histories(
    source: dict[str, Any],
    generated: dict[str, dict[str, Any]],
    tier_mode: str,
) -> dict[str, list[dict[str, Any]]]:
    generated_tiers = generated_tiers_for(tier_mode)
    if set(generated) != set(generated_tiers):
        raise RuntimeError("block variants do not match the tier mode")
    return {
        "clean": copy.deepcopy(source["histories"]["clean"]),
        "ultra_telegraphic": copy.deepcopy(
            generated["ultra_telegraphic"]["histories"]["variant"]
        ),
        "telegraphic_compact": copy.deepcopy(
            generated["telegraphic_compact"]["histories"]["variant"]
            if tier_mode == "four_fresh"
            else source["histories"]["rewritten"]
        ),
        "full_sentence_compact": copy.deepcopy(
            generated["full_sentence_compact"]["histories"]["variant"]
        ),
        "relaxed_full_sentence_compact": copy.deepcopy(
            generated["relaxed_full_sentence_compact"]["histories"]["variant"]
        ),
    }


def _history_reasoning_words(history: list[dict[str, Any]]) -> int:
    return sum(
        len(message["reasoning"].split())
        for message in history
        if message.get("role") == "assistant"
        and isinstance(message.get("reasoning"), str)
        and message["reasoning"]
    )


def finalize_variants(run_dir: Path, parent_run: Path, replicates: int) -> None:
    if replicates != 3:
        raise ValueError("the tier protocol requires exactly three replicates")
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text())
    generated_tiers = tuple(run["snapshot"]["generated_tiers"])
    tier_mode = run["snapshot"]["tier_mode"]
    if generated_tiers != generated_tiers_for(tier_mode):
        raise RuntimeError("frozen tier mode and generated-tier set disagree")
    if run["snapshot"].get("protocol", {}).get("name") == "deepseek_self_development":
        canary_bindings = _validate_development_canary_audits(
            run_dir, run, "hospital-decision-v2"
        )
        if any(run.get(key) != value for key, value in canary_bindings.items()):
            raise RuntimeError("development canary evidence changed after generation freeze")
    already_frozen = bool(run.get("variants_frozen"))
    if run.get("outcomes_launched") and not already_frozen:
        raise RuntimeError("outcomes launched before a valid tier freeze")
    if not already_frozen and run.get("status") != "semantic_audit_frozen_pending_style_audit":
        raise RuntimeError("label-blinded semantic audit must freeze before style/finalization")
    if canonical_hash(json.loads((run_dir / "tasks.snapshot.json").read_text())) != run["snapshot"]["task_snapshot_sha256"]:
        raise RuntimeError("tier task snapshot changed before freeze")
    _verify_frozen_file_manifest(
        run_dir / "generation-journals", run["generation_journal_manifest_sha256"]
    )
    _verify_frozen_file_manifest(
        run_dir / "audit-packets", run["audit_packet_manifest_sha256"]
    )
    _verify_frozen_file_manifest(
        run_dir / "variants", run["candidate_variant_manifest_sha256"]
    )

    expected_paths = {
        f"{tier}/{source['file']}"
        for tier in generated_tiers
        for source in run["snapshot"]["intended_sources"]
    }
    journal_paths = sorted((run_dir / "generation-journals").glob("*/*.json"))
    actual_paths = {
        str(path.relative_to(run_dir / "generation-journals")) for path in journal_paths
    }
    journals_by_cell = {
        str(path.relative_to(run_dir / "generation-journals")): json.loads(path.read_text())
        for path in journal_paths
    }
    loaded_parent_sources = load_parent_sources(
        parent_run,
        tuple(run["snapshot"]["selected_models"]),
        tuple(run["snapshot"]["selected_tasks"]),
    )
    parent_by_file = {path.name: source for path, source in loaded_parent_sources}
    if actual_paths != expected_paths or any(
        journal.get("status") not in TERMINAL_JOURNAL_STATUSES
        for journal in journals_by_cell.values()
    ):
        raise RuntimeError("the selected generation matrix lacks terminal journals")
    for journal in journals_by_cell.values():
        source = parent_by_file.get(journal.get("parent_file"))
        if (
            source is None
            or journal.get("code_manifest_sha256")
            != run["snapshot"]["executable_manifest_sha256"]
        ):
            raise RuntimeError("generation journal names an unknown source/code manifest")
        _validate_generation_journal(
            journal, source, journal["tier"], run["snapshot"]["rewriter"]
        )

    expected_variant_paths = {
        cell for cell, journal in journals_by_cell.items() if journal["status"] == "complete"
    }
    variant_paths = sorted((run_dir / "variants").glob("*/*.json"))
    actual_variant_paths = {
        str(path.relative_to(run_dir / "variants")) for path in variant_paths
    }
    if actual_variant_paths != expected_variant_paths:
        raise RuntimeError("completed journals and variant paths are not bijective")
    variants: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in variant_paths:
        cell = str(path.relative_to(run_dir / "variants"))
        journal = journals_by_cell[cell]
        variant = json.loads(path.read_text())
        verify_tier_variant(variant, parent_by_file[journal["parent_file"]])
        if (
            run["snapshot"].get("protocol", {}).get("name")
            in {"deepseek_self_development", "deepseek_self_expansion"}
            and variant.get("require_exact_action") is not True
        ):
            raise RuntimeError("self-rewrite variant lacks exact-action gating")
        if (
            variant["tier"] != journal["tier"]
            or variant["parent_source_sha256"] != journal["parent_source_sha256"]
            or variant["parent_file"] != journal["parent_file"]
            or variant["parent_file_sha256"] != journal["parent_file_sha256"]
            or variant["variant_sha256"] != journal["variant_sha256"]
            or variant["variant_sha256"] in variants
        ):
            raise RuntimeError(f"journal/variant provenance mismatch: {cell}")
        variants[variant["variant_sha256"]] = (path, variant)
    expected_packet_map = _packet_map_from_variants(
        run_dir, list(variants.values()), run["snapshot"]["audit_packet_salt"]
    )
    packet_map_path = run_dir / "audit-packet-map.json"
    if (
        file_sha256(packet_map_path) != run["audit_packet_map_sha256"]
        or json.loads(packet_map_path.read_text()) != expected_packet_map
    ):
        raise RuntimeError("masked audit packet map changed after semantic lock")
    packet_ids = {entry["packet_id"] for entry in expected_packet_map["entries"]}
    entry_by_packet = {
        entry["packet_id"]: entry for entry in expected_packet_map["entries"]
    }
    semantic_path = run_dir / "semantic-audit.json"
    if file_sha256(semantic_path) != run["semantic_audit_sha256"]:
        raise RuntimeError("label-blinded semantic decisions changed after unblinding")
    semantic_audit = json.loads(semantic_path.read_text())
    semantic_pass = _validate_semantic_audit(
        semantic_audit, packet_ids, run, run_dir
    )

    audit_path = run_dir / "variant-audit.json"
    if not audit_path.exists():
        raise RuntimeError("write condition-aware variant-audit.json before finalize")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("timing") != "after_semantic_lock_before_tier_outcomes"
        or audit.get("semantic_audit_sha256") != run["semantic_audit_sha256"]
        or audit.get("audit_packet_map_sha256") != run["audit_packet_map_sha256"]
        or audit.get("continuation_outcomes_inspected") is not False
        or audit.get("pending_tool_results_inspected") is not False
        or audit.get("source_history_files_inspected") is not False
    ):
        raise RuntimeError("tier style audit lacks the required pre-outcome attestation")
    active_self_protocol = run.get("snapshot", {}).get("protocol", {}).get("name") in {
        "deepseek_self_development",
        "deepseek_self_expansion",
    }
    style_review_manifest = audit.get("review_journal_manifest_sha256")
    if style_review_manifest is not None:
        _verify_frozen_file_manifest(
            run_dir / "review-journals" / "style", style_review_manifest
        )
    if active_self_protocol and (
        style_review_manifest is None
        or audit.get("review_workers")
        != run["snapshot"]["execution"]["workers"]
        or not isinstance(audit.get("command"), list)
    ):
        raise RuntimeError("self-rewrite style audit lacks frozen reviewer provenance")
    try:
        semantic_locked_at = datetime.fromisoformat(run["semantic_locked_at"])
        style_reviewed_at = datetime.fromisoformat(audit["style_reviewed_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("tier audit lacks valid semantic/style timestamps") from exc
    if semantic_locked_at > style_reviewed_at:
        raise RuntimeError("condition-aware style review predates semantic decision lock")
    decisions = audit.get("decisions", [])
    decision_by_packet = {decision.get("packet_id"): decision for decision in decisions}
    if len(decisions) != len(decision_by_packet) or set(decision_by_packet) != packet_ids:
        raise RuntimeError("style audit must decide every mapped packet exactly once")
    accepted_packets = set()
    for packet_id, decision in decision_by_packet.items():
        checks = decision.get("checks", {})
        style_passed = (
            decision.get("tier") == entry_by_packet[packet_id]["tier"]
            and set(checks) == STYLE_CHECKS
            and all(value is True for value in checks.values())
        )
        accepted_value = semantic_pass[packet_id] and style_passed
        if (
            set(decision)
            != {"packet_id", "tier", "passed", "accepted", "checks", "notes"}
            or not isinstance(decision.get("notes"), str)
            or decision.get("passed") is not style_passed
            or decision.get("accepted") is not accepted_value
        ):
            raise RuntimeError("style audit decision disagrees with its complete rubric")
        if accepted_value:
            accepted_packets.add(packet_id)
    accepted = {tier: set() for tier in generated_tiers}
    rejected = set()
    for digest, (_path, variant) in variants.items():
        ids = {
            _audit_packet_id(run["snapshot"]["audit_packet_salt"], variant, generated)
            for generated in variant["generated_turns"]
        }
        if ids <= accepted_packets:
            accepted[variant["tier"]].add(digest)
        else:
            rejected.add(digest)

    parent_sources = {
        source["source_sha256"]: (path, source)
        for path, source in loaded_parent_sources
    }
    accepted_by_parent: dict[str, dict[str, dict[str, Any]]] = {}
    for tier, digests in accepted.items():
        for digest in digests:
            variant = variants[digest][1]
            parent_hash = variant["parent_source_sha256"]
            if tier in accepted_by_parent.setdefault(parent_hash, {}):
                raise RuntimeError("multiple accepted variants share a parent/tier cell")
            accepted_by_parent[parent_hash][tier] = variant
    included = [
        parent_hash
        for parent_hash, values in accepted_by_parent.items()
        if set(values) == set(generated_tiers)
    ]
    tasks = {
        task["id"]: task
        for task in json.loads((run_dir / "tasks.snapshot.json").read_text())
    }
    by_model: dict[str, int] = {}
    blocks = []
    for parent_hash in sorted(included):
        source_path, source = parent_sources[parent_hash]
        expected_parent_file_hash = next(
            item["file_sha256"]
            for item in run["snapshot"]["intended_sources"]
            if item["source_sha256"] == parent_hash
        )
        if file_sha256(source_path) != expected_parent_file_hash:
            raise RuntimeError("parent source file changed before block construction")
        generated = accepted_by_parent[parent_hash]
        histories = block_histories(source, generated, tier_mode)
        history_reasoning_words = {
            arm: _history_reasoning_words(history) for arm, history in histories.items()
        }
        dose_order = (
            "clean",
            "relaxed_full_sentence_compact",
            "full_sentence_compact",
            "telegraphic_compact",
            "ultra_telegraphic",
        )
        if any(
            history_reasoning_words[left] <= history_reasoning_words[right]
            for left, right in zip(dose_order, dose_order[1:])
        ):
            raise RuntimeError(
                f"achieved source history dose is not strictly ordered: "
                f"{source['task_id']}: {history_reasoning_words}"
            )
        fork_audits = {
            arm: reasoning_only_fork_audit(histories["clean"], history)
            for arm, history in histories.items()
            if arm != "clean"
        }
        if not all(value["passed"] for value in fork_audits.values()):
            raise RuntimeError("a five-arm block changes fields outside historical reasoning")
        block = {
            "task_id": source["task_id"],
            "task_sha256": canonical_hash(tasks[source["task_id"]]),
            "task_snapshot_sha256": run["snapshot"]["task_snapshot_sha256"],
            "code_manifest_sha256": run["snapshot"]["executable_manifest_sha256"],
            "model_name": source["model_name"],
            "model": source["model"],
            "parent_file": source_path.name,
            "parent_file_sha256": file_sha256(source_path),
            "parent_source_sha256": parent_hash,
            "variant_sha256": {
                tier: generated[tier]["variant_sha256"] for tier in generated_tiers
            },
            "histories": histories,
            "fork_audits": fork_audits,
        }
        block["block_sha256"] = canonical_hash(block)
        blocks.append(block)
        by_model[source["model_name"]] = by_model.get(source["model_name"], 0) + 1
    launch_gate = run["snapshot"]["launch_gate"]
    if len(blocks) < launch_gate["minimum_total_sources"] or any(
        by_model.get(model, 0) < minimum
        for model, minimum in launch_gate["minimum_sources_by_model"].items()
    ):
        raise RuntimeError(
            f"insufficient complete five-arm cohort: total={len(blocks)}, "
            f"by_model={by_model}, gate={launch_gate}"
        )

    block_paths = []
    for block in blocks:
        path = run_dir / "blocks" / f"{block['model_name']}__{block['task_id']}.json"
        reconcile_immutable_json(path, block)
        block_paths.append(path)
    existing_block_paths = set((run_dir / "blocks").glob("*.json"))
    if existing_block_paths != set(block_paths):
        raise RuntimeError("partial pre-freeze block directory contains stale extras")
    block_manifest = _write_manifest(run_dir / "blocks", block_paths)
    randomization_path = run_dir / "randomization.json"
    expected_randomization = _balanced_randomization(blocks, tasks, replicates)
    if randomization_path.exists():
        randomization = json.loads(randomization_path.read_text())
        if (
            not isinstance(randomization.get("created_at"), str)
            or {key: value for key, value in randomization.items() if key != "created_at"}
            != {
                key: value
                for key, value in expected_randomization.items()
                if key != "created_at"
            }
        ):
            raise RuntimeError("partial pre-freeze randomization does not replay")
    else:
        randomization = expected_randomization
    expected_random_keys = {
        (block["model_name"], block["task_id"], replicate)
        for block in blocks
        for replicate in range(1, 4)
    }
    actual_random_keys = {
        (entry["model_name"], entry["task_id"], entry["replicate"])
        for entry in randomization["entries"]
    }
    block_by_key = {(block["model_name"], block["task_id"]): block for block in blocks}
    if (
        actual_random_keys != expected_random_keys
        or len(randomization.get("entries", [])) != len(expected_random_keys)
        or randomization.get("timing") != "before_tier_outcomes"
        or randomization.get("arms") != list(ARMS)
        or any(
            entry.get("parent_source_sha256")
            != block_by_key[(entry["model_name"], entry["task_id"])]["parent_source_sha256"]
            or any(set(order) != set(ARMS) or len(order) != len(ARMS) for order in entry["orders"].values())
            for entry in randomization["entries"]
        )
    ):
        raise RuntimeError("frozen randomization does not cover the exact valid three-replicate matrix")
    reconcile_immutable_json(randomization_path, randomization)
    freeze_values = {
        "status": "five_arm_blocks_frozen_before_outcomes",
        "variants_frozen": True,
        "outcomes_launched": False,
        "included_sources": len(blocks),
        "included_sources_by_model": by_model,
        "style_audit_sha256": file_sha256(audit_path),
        "variant_manifest_sha256": run["candidate_variant_manifest_sha256"],
        "block_manifest_sha256": file_sha256(block_manifest),
        "randomization_sha256": file_sha256(randomization_path),
    }
    if already_frozen:
        if any(run.get(key) != value for key, value in freeze_values.items()):
            raise RuntimeError("existing pre-outcome freeze metadata does not replay")
        return
    run.update(**freeze_values, frozen_at=now())
    atomic_json(run_path, run)


def _verify_frozen_file_manifest(directory: Path, expected_hash: str) -> None:
    manifest = directory / "SHA256SUMS"
    if file_sha256(manifest) != expected_hash:
        raise RuntimeError(f"manifest changed after freeze: {manifest}")
    verify_manifest_exact(directory, manifest)


def _slot_attempt_key(slot_index: int, arm: str) -> str:
    return f"{slot_index}:{arm}"


def _target_response_valid(
    payload: dict[str, Any], model_name: str, response: dict[str, Any]
) -> tuple[bool, bool]:
    choices = response.get("choices")
    message = (
        choices[0].get("message")
        if isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict)
        else None
    )
    reasoning_details = message.get("reasoning_details") if isinstance(message, dict) else None
    shape_valid = (
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and (message.get("content") is None or isinstance(message.get("content"), str))
        and (message.get("reasoning") is None or isinstance(message.get("reasoning"), str))
        and (
            message.get("reasoning_content") is None
            or isinstance(message.get("reasoning_content"), str)
        )
        and (
            reasoning_details is None
            or (
                isinstance(reasoning_details, list)
                and all(isinstance(block, dict) for block in reasoning_details)
            )
        )
    )
    route_valid = (
        response.get("provider") == EXPECTED_PROVIDERS[model_name]
        and response.get("model") == payload["model"]
    )
    return shape_valid, route_valid


def _target_once(
    journal_path: Path,
    journal: dict[str, Any],
    arm: str,
    slot_index: int,
    model_name: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    key = _slot_attempt_key(slot_index, arm)
    attempts = journal["slot_attempts"].setdefault(key, [])
    expected_body, expected_body_sha256 = request_body_record(payload)
    while True:
        if attempts and attempts[-1]["status"] == "in_flight":
            attempts[-1].update(
                status="ambiguous_after_restart",
                ambiguity=(
                    "Pre-call record exists without a returned response and will not be resent automatically."
                ),
            )
            atomic_json(journal_path, journal)
            raise RuntimeError(f"ambiguous target request requires adjudication: {key}")
        if attempts and attempts[-1]["status"] in {
            "ambiguous_after_restart",
            "ambiguous_transport_error",
        }:
            raise RuntimeError(f"ambiguous target request requires adjudication: {key}")
        if (
            attempts
            and attempts[-1]["status"] == "returned_http_error"
            and attempts[-1].get("retryable") is False
        ):
            journal.update(
                status="terminal_transport_failure",
                failed_slot=key,
                completed_at=now(),
            )
            atomic_json(journal_path, journal)
            raise RuntimeError(
                f"target slot received a non-retryable HTTP error: {key}"
            )
        if attempts and attempts[-1]["status"] == "response_received":
            pending = attempts[-1]
            if (
                pending.get("request") != payload
                or pending.get("request_body") != expected_body
                or pending.get("request_body_sha256") != expected_body_sha256
            ):
                raise RuntimeError("journaled target response belongs to a different request")
            try:
                response = parse_persisted_response(pending)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                pending.update(
                    status="returned_error",
                    route_valid=None,
                    usable_response=False,
                    parse_error=f"{type(exc).__name__}: {exc}",
                )
            else:
                pending["response"] = response
                shape_valid, route_valid = _target_response_valid(
                    payload, model_name, response
                )
                explicit_route_mismatch = (
                    response.get("provider") is not None
                    and response.get("provider") != EXPECTED_PROVIDERS[model_name]
                ) or (
                    response.get("model") is not None
                    and response.get("model") != payload["model"]
                )
                pending.update(
                    status=(
                        "usable_response"
                        if shape_valid and route_valid
                        else "returned_error"
                    ),
                    route_valid=route_valid,
                    usable_response=shape_valid and route_valid,
                )
                if explicit_route_mismatch:
                    journal.update(
                        status="terminal_route_failure",
                        failed_slot=key,
                        completed_at=now(),
                    )
                    atomic_json(journal_path, journal)
                    raise RuntimeError("journaled target response has the wrong route/model")
            atomic_json(journal_path, journal)
        recovered = next(
            (attempt for attempt in reversed(attempts) if attempt.get("usable_response")),
            None,
        )
        if recovered is not None:
            if (
                recovered.get("request") != payload
                or recovered.get("request_body") != expected_body
                or recovered.get("request_body_sha256") != expected_body_sha256
            ):
                raise RuntimeError("journaled target response belongs to a different request")
            response = parse_persisted_response(recovered)
            if recovered.get("response") != response:
                raise RuntimeError("parsed target response differs from its raw body")
            shape_valid, route_valid = _target_response_valid(payload, model_name, response)
            if not shape_valid or not route_valid:
                raise RuntimeError("journaled usable target response fails recomputation")
            return response, recovered["wall_seconds"]
        counted = [attempt for attempt in attempts if attempt["status"] != "budget_pause"]
        if len(counted) >= 20:
            journal.update(
                status="terminal_transport_failure", failed_slot=key, completed_at=now()
            )
            atomic_json(journal_path, journal)
            raise RuntimeError(f"target slot exhausted 20 journaled attempts: {key}")
        attempt = {
            "status": "in_flight",
            "created_at": now(),
            "request": copy.deepcopy(payload),
            "request_body": expected_body,
            "request_body_sha256": expected_body_sha256,
        }
        attempts.append(attempt)
        atomic_json(journal_path, journal)
        try:
            raw_body, wall = post_chat_once_raw(payload)
        except BudgetFloorReached as exc:
            attempt.update(status="budget_pause", completed_at=now(), error=str(exc))
            atomic_json(journal_path, journal)
            raise
        except ChatAttemptError as exc:
            if exc.http_status is not None and exc.response_body_bytes is not None:
                persist_returned_http_error(attempt, exc)
                atomic_json(journal_path, journal)
                if not attempt["retryable"]:
                    journal.update(
                        status="terminal_transport_failure",
                        failed_slot=key,
                        completed_at=now(),
                    )
                    atomic_json(journal_path, journal)
                    raise RuntimeError(
                        f"target slot received a non-retryable HTTP error: {key}"
                    ) from exc
                counted = [
                    item for item in attempts if item["status"] != "budget_pause"
                ]
                if len(counted) >= 20:
                    continue
                delay_bucket = int(
                    hashlib.sha256(key.encode()).hexdigest()[:4], 16
                ) % 30
                time.sleep(min(15 * len(counted), 60) + delay_bucket)
                continue
            attempt.update(
                status="ambiguous_transport_error",
                completed_at=now(),
                ambiguity="No response body returned; the paid request will not be resent automatically.",
                **exc.record(),
            )
            atomic_json(journal_path, journal)
            raise RuntimeError(f"ambiguous target transport requires adjudication: {key}") from exc
        persist_raw_response(attempt, raw_body, wall)
        attempt["response_received_at"] = now()
        atomic_json(journal_path, journal)


def _final_response_score(message: Any, expected: str) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {
            "final_correct": False,
            "unexpected_tool_calls": None,
            "format_error": "assistant message must be an object",
        }
    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        return {
            "final_correct": False,
            "unexpected_tool_calls": None,
            "format_error": "final tool_calls must be a list",
        }
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        return {
            "final_correct": False,
            "unexpected_tool_calls": len(calls),
            "format_error": "final content must be a string or null",
        }
    return {
        "final_correct": final_is_correct(content, expected),
        "unexpected_tool_calls": len(calls),
        "format_error": None,
    }


def _initialize_outcome_journal(
    block: dict[str, Any], randomization: dict[str, Any], replicate: int
) -> dict[str, Any]:
    return {
        "created_at": now(),
        "status": "pending",
        "task_id": block["task_id"],
        "model_name": block["model_name"],
        "parent_source_sha256": block["parent_source_sha256"],
        "block_sha256": block["block_sha256"],
        "code_manifest_sha256": block["code_manifest_sha256"],
        "replicate": replicate,
        "continuation_seed": randomization["continuation_seed"],
        "orders": randomization["orders"],
        "slot_attempts": {},
        "completed_slots": [],
        "branches": {
            arm: {
                "messages": copy.deepcopy(block["histories"][arm]),
                "turns": [],
                "actions_correct": True,
                "status": "active",
            }
            for arm in ARMS
        },
    }


def _validate_outcome_journal(
    journal: dict[str, Any],
    block: dict[str, Any],
    task: dict[str, Any],
    randomization: dict[str, Any],
    replicate: int,
) -> None:
    if (
        journal.get("task_id") != block["task_id"]
        or journal.get("model_name") != block["model_name"]
        or journal.get("parent_source_sha256") != block["parent_source_sha256"]
        or journal.get("block_sha256") != block["block_sha256"]
        or journal.get("code_manifest_sha256") != block["code_manifest_sha256"]
        or journal.get("replicate") != replicate
        or journal.get("continuation_seed") != randomization["continuation_seed"]
        or journal.get("orders") != randomization["orders"]
        or set(journal.get("branches", {})) != set(ARMS)
    ):
        raise RuntimeError("outcome journal top-level provenance changed")
    completed = journal.get("completed_slots", [])
    if len(completed) != len(set(completed)):
        raise RuntimeError("outcome journal contains duplicate completed slots")
    action_phases = task["phases"][task["fork_after"] :]
    slot_names = [phase["name"] for phase in action_phases] + ["final"]
    valid_keys = {
        _slot_attempt_key(index, arm)
        for index in range(len(slot_names))
        for arm in ARMS
    }
    if not set(completed) <= valid_keys or not set(journal.get("slot_attempts", {})) <= valid_keys:
        raise RuntimeError("outcome journal references an unknown slot")
    frozen_sequence = [
        _slot_attempt_key(slot_index, arm)
        for slot_index, slot_name in enumerate(slot_names)
        for arm in journal["orders"][slot_name]
    ]
    if completed != frozen_sequence[: len(completed)]:
        raise RuntimeError("completed outcome slots do not follow the frozen global order")
    tools = build_tools(task)
    for arm in ARMS:
        branch = journal["branches"][arm]
        messages = copy.deepcopy(block["histories"][arm])
        turns = branch.get("turns", [])
        actions_correct = True
        expected_status = "active"
        final_correct = False
        expected_requests: dict[int, dict[str, Any]] = {}
        for turn_index, turn in enumerate(turns):
            if turn_index >= len(slot_names):
                raise RuntimeError("outcome journal contains too many turns")
            slot_name = slot_names[turn_index]
            expected_turn_number = (
                len(task["phases"]) + 1
                if slot_name == "final"
                else task["fork_after"] + turn_index + 1
            )
            if turn.get("phase") != slot_name or turn.get("turn") != expected_turn_number:
                raise RuntimeError("outcome journal turn sequence changed")
            phase_offset = task["fork_after"] + turn_index
            request_seed = journal["continuation_seed"] + (
                len(task["phases"]) + 1 if slot_name == "final" else phase_offset
            )
            expected_request = target_payload(block["model"], messages, tools, request_seed)
            expected_requests[turn_index] = expected_request
            if turn.get("request") != expected_request:
                raise RuntimeError("outcome journal request does not reconstruct")
            shape_valid, route_valid = _target_response_valid(
                expected_request, block["model_name"], turn["response"]
            )
            if not shape_valid or not route_valid:
                raise RuntimeError("outcome journal response route/shape changed")
            if turn.get("metrics") != response_metrics(turn["response"], turn["wall_seconds"]):
                raise RuntimeError("outcome journal response metrics do not recompute")
            message = turn["response"]["choices"][0]["message"]
            if slot_name == "final":
                score = _final_response_score(message, task["final_answer"])
                if any(turn.get(key) != value for key, value in score.items()):
                    raise RuntimeError("outcome journal final score does not recompute")
                if branch.get("final_content") != message.get("content"):
                    raise RuntimeError("outcome branch final content does not recompute")
                final_correct = (
                    score["final_correct"]
                    and score["unexpected_tool_calls"] == 0
                    and score["format_error"] is None
                )
                expected_status = "complete"
            else:
                phase = action_phases[turn_index]
                action = evaluate_tool_response(message, phase)
                if turn.get("action") != action:
                    raise RuntimeError("outcome journal action score does not recompute")
                actions_correct = actions_correct and action["correct"]
                if action["correct"]:
                    append_tool_cycle(messages, message, reasoning_text(message), phase)
                else:
                    expected_status = "action_failure"
                    if turn_index + 1 != len(turns):
                        raise RuntimeError("outcome journal continued after an action failure")
                    break
        if len(turns) < len(slot_names) and expected_status == "complete":
            raise RuntimeError("complete outcome branch omits turns")
        if len(turns) == len(slot_names) and expected_status == "active":
            raise RuntimeError("full-length outcome branch was not marked complete")
        if (
            branch.get("status") != expected_status
            or branch.get("actions_correct") != actions_correct
            or (
                "messages" in branch
                and branch["messages"] != messages
            )
        ):
            raise RuntimeError("outcome branch state does not replay from its frozen history")
        if expected_status == "complete" and branch.get("final_correct") != final_correct:
            raise RuntimeError("outcome branch final state changed")
        completed_indices = sorted(
            int(key.split(":", 1)[0])
            for key in completed
            if key.endswith(f":{arm}")
        )
        if completed_indices and completed_indices != list(range(completed_indices[-1] + 1)):
            raise RuntimeError("completed outcome slots are not a chronological prefix")
        if expected_status == "active" and len(completed_indices) != len(turns):
            raise RuntimeError("active branch has skipped or fabricated completed slots")
        if expected_status == "complete" and completed_indices != list(range(len(slot_names))):
            raise RuntimeError("complete branch lacks completed slots")
        if expected_status == "action_failure" and len(completed_indices) < len(turns):
            raise RuntimeError("failed branch lacks its scored action slot")
        if expected_status == "active" and len(turns) < len(slot_names):
            next_index = len(turns)
            phase_offset = task["fork_after"] + next_index
            request_seed = journal["continuation_seed"] + (
                len(task["phases"]) + 1
                if slot_names[next_index] == "final"
                else phase_offset
            )
            expected_requests[next_index] = target_payload(
                block["model"], messages, tools, request_seed
            )
        for slot_index in range(len(slot_names)):
            key = _slot_attempt_key(slot_index, arm)
            attempts = journal["slot_attempts"].get(key, [])
            counted_attempts = [
                attempt for attempt in attempts if attempt.get("status") != "budget_pause"
            ]
            if len(counted_attempts) > 20 or (attempts and slot_index not in expected_requests):
                raise RuntimeError("outcome attempt exists outside reconstructable branch work")
            usable = []
            for attempt_index, attempt in enumerate(attempts):
                expected_request = expected_requests[slot_index]
                body, digest = request_body_record(expected_request)
                if (
                    attempt.get("request") != expected_request
                    or attempt.get("request_body") != body
                    or attempt.get("request_body_sha256") != digest
                ):
                    raise RuntimeError("outcome attempt request does not reconstruct")
                status = attempt.get("status")
                if status in {
                    "in_flight",
                    "response_received",
                    "usable_response",
                    "ambiguous_after_restart",
                    "ambiguous_transport_error",
                } and attempt_index + 1 != len(attempts):
                    raise RuntimeError("outcome attempts continue after unresolved/usable response")
                if status == "returned_http_error":
                    if (
                        not isinstance(attempt.get("http_status"), int)
                        or attempt["http_status"] < 400
                    ):
                        raise RuntimeError("returned target HTTP error lacks an error status")
                    retryable = returned_http_error_is_retryable(
                        attempt["http_status"], persisted_raw_response(attempt)
                    )
                    if attempt.get("retryable") is not retryable:
                        raise RuntimeError("returned target HTTP retry decision changed")
                    if not retryable and attempt_index + 1 != len(attempts):
                        raise RuntimeError("outcome attempts continue after terminal HTTP error")
                    continue
                if status in {
                    "response_received",
                    "usable_response",
                    "returned_error",
                }:
                    try:
                        response = parse_persisted_response(attempt)
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                        if status != "returned_error":
                            raise RuntimeError("outcome attempt status disagrees with raw body")
                        continue
                    if attempt.get("response") is not None and attempt["response"] != response:
                        raise RuntimeError("parsed outcome response differs from raw body")
                    shape_valid, route_valid = _target_response_valid(
                        expected_request, block["model_name"], response
                    )
                    if status == "usable_response":
                        if not shape_valid or not route_valid or not attempt.get("usable_response"):
                            raise RuntimeError("usable outcome response fails recomputation")
                        usable.append(attempt)
                    elif status == "returned_error" and shape_valid and route_valid:
                        raise RuntimeError("usable outcome response was mislabeled returned_error")
                elif status not in {
                    "in_flight",
                    "ambiguous_after_restart",
                    "budget_pause",
                    "transport_error",
                    "ambiguous_transport_error",
                }:
                    raise RuntimeError(f"unknown outcome attempt status: {status}")
            if slot_index < len(turns):
                turn = turns[slot_index]
                if (
                    len(usable) != 1
                    or usable[0]["request"] != turn["request"]
                    or usable[0]["response"] != turn["response"]
                    or usable[0]["wall_seconds"] != turn["wall_seconds"]
                ):
                    raise RuntimeError("completed outcome turn lacks its exact journaled response")
    all_slots_complete = len(completed) == len(frozen_sequence)
    status = journal.get("status")
    if status == "terminal_transport_failure":
        failed_slot = journal.get("failed_slot")
        failed_attempts = (
            journal["slot_attempts"].get(failed_slot)
            if isinstance(failed_slot, str)
            else None
        )
        if (
            not isinstance(failed_attempts, list)
            or failed_slot in completed
        ):
            raise RuntimeError("terminal transport failure has an invalid failed slot")
        counted = [
            attempt
            for attempt in failed_attempts
            if attempt.get("status") != "budget_pause"
        ]
        exhausted = len(counted) >= 20 and not any(
            attempt.get("status") == "usable_response" for attempt in counted
        )
        nonretryable_http_error = bool(counted) and (
            counted[-1].get("status") == "returned_http_error"
            and counted[-1].get("retryable") is False
        )
        if not exhausted and not nonretryable_http_error:
            raise RuntimeError("terminal transport failure lacks a terminal failed slot")
    elif status == "terminal_route_failure":
        if not any(
            attempt.get("response") is not None
            and (
                attempt["response"].get("provider")
                != EXPECTED_PROVIDERS[block["model_name"]]
                or attempt["response"].get("model") != block["model"]["model"]
            )
            for attempts in journal["slot_attempts"].values()
            for attempt in attempts
        ):
            raise RuntimeError("terminal route failure lacks a wrong-route response")
    elif all_slots_complete:
        if status not in {"pending", "capture_ready", "complete"}:
            raise RuntimeError("full outcome journal has an invalid capture state")
    elif status != "pending":
        raise RuntimeError("partial outcome journal has a forged terminal/capture state")
    if status in {"capture_ready", "complete"} and (
        not all_slots_complete
        or not isinstance(journal.get("capture_created_at"), str)
        or not isinstance(journal.get("capture_core_sha256"), str)
    ):
        raise RuntimeError("outcome capture state lacks its deterministic reconstruction fields")


def _capture_from_journal(
    journal: dict[str, Any], block: dict[str, Any], task: dict[str, Any]
) -> dict[str, Any]:
    captured_branches = {}
    for arm, branch in journal["branches"].items():
        captured = {
            key: copy.deepcopy(value) for key, value in branch.items() if key != "messages"
        }
        captured["aggregate"] = aggregate_metrics(
            [turn["metrics"] for turn in captured["turns"]]
        )
        captured.setdefault("final_correct", False)
        captured["success"] = captured["actions_correct"] and captured["final_correct"]
        captured_branches[arm] = captured
    return {
        "status": "complete",
        "task_id": block["task_id"],
        "title": task["title"],
        "model_name": block["model_name"],
        "model": block["model"],
        "parent_source_sha256": block["parent_source_sha256"],
        "block_sha256": block["block_sha256"],
        "code_manifest_sha256": block["code_manifest_sha256"],
        "replicate": journal["replicate"],
        "continuation_seed": journal["continuation_seed"],
        "orders": journal["orders"],
        "fork_after": task["fork_after"],
        "expected_final": task["final_answer"],
        "branches": captured_branches,
    }


def first_replicate_development_gate(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_cases = (
        len(cases) == 3
        and {case.get("task_id") for case in cases} == SELF_REWRITE_DEVELOPMENT_TASKS
        and all(case.get("replicate") == 1 for case in cases)
    )
    strict_successes = sum(
        branch.get("success") is True
        for case in cases
        for branch in case.get("branches", {}).values()
    )
    changes: dict[str, list[float]] = {arm: [] for arm in NONCLEAN_ARMS}
    shorter = 0
    ultra_shorter_tasks = 0
    for case in cases:
        branches = case.get("branches", {})
        clean = branches.get("clean", {})
        clean_tokens = clean.get("aggregate", {}).get("reasoning_tokens")
        if not isinstance(clean_tokens, (int, float)) or clean_tokens <= 0:
            continue
        for arm in NONCLEAN_ARMS:
            branch = branches.get(arm, {})
            treatment_tokens = branch.get("aggregate", {}).get("reasoning_tokens")
            if (
                clean.get("status") != "complete"
                or branch.get("status") != "complete"
                or not isinstance(treatment_tokens, (int, float))
            ):
                continue
            change = (treatment_tokens - clean_tokens) / clean_tokens * 100
            changes[arm].append(change)
            shorter += change < 0
            if arm == "ultra_telegraphic":
                ultra_shorter_tasks += change < 0
    medians = {
        arm: statistics.median(values) if values else None
        for arm, values in changes.items()
    }
    comparisons = sum(len(values) for values in changes.values())
    gate_checks = {
        "all_15_branches_strict_success": exact_cases and strict_successes == 15,
        "at_least_8_of_12_nonclean_shorter": comparisons == 12 and shorter >= 8,
        "ultra_shorter_on_at_least_2_of_3_tasks": (
            len(changes["ultra_telegraphic"]) == 3 and ultra_shorter_tasks >= 2
        ),
        "clean_to_ultra_median_at_most_minus_10_percent": (
            medians["ultra_telegraphic"] is not None
            and medians["ultra_telegraphic"] <= -10
        ),
        "no_nonclean_tier_positive_pooled_median": (
            comparisons == 12
            and all(value is not None and value <= 0 for value in medians.values())
        ),
    }
    return {
        "passed": all(gate_checks.values()),
        "checks": gate_checks,
        "strict_successes": strict_successes,
        "branches": len(cases) * len(ARMS),
        "nonclean_shorter": shorter,
        "nonclean_comparisons": comparisons,
        "median_percent_change_by_arm": medians,
    }


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for position in ordered[cursor:end]:
            ranks[position] = rank
        cursor = end
    return ranks


def _spearman(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    x = _average_ranks(values_x)
    y = _average_ranks(values_y)
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = (
        sum(value * value for value in dx) * sum(value * value for value in dy)
    ) ** 0.5
    return (
        None
        if denominator == 0
        else sum(left * right for left, right in zip(dx, dy)) / denominator
    )


def full_development_expansion_gate(
    cases: list[dict[str, Any]], blocks: list[dict[str, Any]]
) -> dict[str, Any]:
    exact_cases = (
        len(cases) == 9
        and {case.get("task_id") for case in cases} == SELF_REWRITE_DEVELOPMENT_TASKS
        and {case.get("replicate") for case in cases} == {1, 2, 3}
        and len(
            {(case.get("task_id"), case.get("replicate")) for case in cases}
        )
        == 9
    )
    strict_successes = sum(
        case.get("branches", {}).get(arm, {}).get("success") is True
        for case in cases
        for arm in DOSE_ORDER
    )
    changes: dict[str, list[float]] = {arm: [] for arm in NONCLEAN_ARMS}
    pooled_words: list[float] = []
    pooled_reasoning: list[float] = []
    block_by_task = {block["task_id"]: block for block in blocks}
    for case in cases:
        clean = case.get("branches", {}).get("clean", {})
        clean_tokens = clean.get("aggregate", {}).get("reasoning_tokens")
        block = block_by_task.get(case.get("task_id"))
        if block is None:
            continue
        for arm in DOSE_ORDER:
            branch = case.get("branches", {}).get(arm, {})
            tokens = branch.get("aggregate", {}).get("reasoning_tokens")
            if isinstance(tokens, (int, float)):
                pooled_words.append(float(_history_reasoning_words(block["histories"][arm])))
                pooled_reasoning.append(float(tokens))
        if not isinstance(clean_tokens, (int, float)) or clean_tokens <= 0:
            continue
        for arm in NONCLEAN_ARMS:
            branch = case.get("branches", {}).get(arm, {})
            tokens = branch.get("aggregate", {}).get("reasoning_tokens")
            if (
                clean.get("status") == "complete"
                and branch.get("status") == "complete"
                and isinstance(tokens, (int, float))
            ):
                changes[arm].append((tokens - clean_tokens) / clean_tokens * 100)
    comparisons = sum(len(values) for values in changes.values())
    shorter = sum(value < 0 for values in changes.values() for value in values)
    ultra_median = (
        statistics.median(changes["ultra_telegraphic"])
        if changes["ultra_telegraphic"]
        else None
    )
    pooled_rho = _spearman(pooled_words, pooled_reasoning)
    input_order_exact = (
        len(block_by_task) == 3
        and all(
            all(
                _history_reasoning_words(block["histories"][DOSE_ORDER[index]])
                > _history_reasoning_words(
                    block["histories"][DOSE_ORDER[index + 1]]
                )
                for index in range(4)
            )
            for block in blocks
        )
    )
    no_failures = all(
        case.get("branches", {}).get(arm, {}).get("success") is True
        and case["branches"][arm].get("status") == "complete"
        for case in cases
        for arm in DOSE_ORDER
    )
    checks = {
        "all_45_branches_strict_success": exact_cases and strict_successes == 45,
        "at_least_27_of_36_nonclean_shorter": comparisons == 36 and shorter >= 27,
        "clean_to_ultra_median_at_most_minus_10_percent": (
            ultra_median is not None and ultra_median <= -10
        ),
        "positive_pooled_rank_association": pooled_rho is not None and pooled_rho > 0,
        "exact_input_order_all_three_sources": input_order_exact,
        "no_quality_or_technical_failure": no_failures,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "strict_successes": strict_successes,
        "branches": len(cases) * len(DOSE_ORDER),
        "nonclean_shorter": shorter,
        "nonclean_comparisons": comparisons,
    }


def _verify_analysis_manifest(
    run_dir: Path, analysis_name: str, report_name: str, manifest_name: str
) -> None:
    analysis_path = run_dir / analysis_name
    report_path = run_dir / report_name
    manifest_path = run_dir / manifest_name
    expected = (
        f"{file_sha256(analysis_path)}  {analysis_name}\n"
        f"{file_sha256(report_path)}  {report_name}\n"
    )
    if manifest_path.read_text() != expected:
        raise RuntimeError("analysis manifest does not bind its exact outputs")


def verify_first_replicate_development_analysis(
    run_dir: Path,
    blocks: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    random_index: dict[tuple[str, str, int], dict[str, Any]],
    require_pass: bool = True,
) -> dict[str, Any]:
    analysis_name = "analysis-rep-01.json"
    report_name = "report-rep-01.md"
    manifest_name = "ANALYSIS-REP-01_SHA256SUMS"
    try:
        _verify_analysis_manifest(run_dir, analysis_name, report_name, manifest_name)
    except (FileNotFoundError, KeyError) as exc:
        raise RuntimeError("run and freeze replicate-1 analysis before later replicates") from exc
    cases = []
    for block in blocks:
        stem = f"{block['model_name']}__{block['task_id']}__rep-01.json"
        capture_path = run_dir / "continuations" / stem
        journal_path = run_dir / "outcome-journals" / stem
        capture = strict_json_loads(capture_path.read_bytes())
        journal = strict_json_loads(journal_path.read_bytes())
        randomization = random_index[(block["model_name"], block["task_id"], 1)]
        task = tasks[block["task_id"]]
        _validate_outcome_journal(journal, block, task, randomization, 1)
        expected = _capture_from_journal(journal, block, task)
        if {key: value for key, value in capture.items() if key != "created_at"} != expected:
            raise RuntimeError("replicate-1 analysis input capture does not reconstruct")
        cases.append(capture)
    gate = first_replicate_development_gate(cases)
    capture_names = {
        f"{case['model_name']}__{case['task_id']}__rep-{case['replicate']:02d}.json"
        for case in cases
    }
    analysis_input_sha256 = canonical_hash(
        {
            "block_manifest_sha256": file_sha256(run_dir / "blocks" / "SHA256SUMS"),
            "randomization_sha256": file_sha256(run_dir / "randomization.json"),
            "continuations": {
                name: file_sha256(run_dir / "continuations" / name)
                for name in sorted(capture_names)
            },
            "outcome_journals": {
                name: file_sha256(run_dir / "outcome-journals" / name)
                for name in sorted(capture_names)
            },
        }
    )
    analysis = strict_json_loads((run_dir / analysis_name).read_bytes())
    if (
        analysis.get("analysis_input_sha256") != analysis_input_sha256
        or analysis.get("partial_replicate_limit") != 1
        or analysis.get("first_replicate_development_gate") != gate
        or analysis.get("tasks") != sorted(SELF_REWRITE_DEVELOPMENT_TASKS)
        or analysis.get("cases") != 3
    ):
        raise RuntimeError("replicate-1 analysis gate does not recompute")
    if require_pass and gate["passed"] is not True:
        raise RuntimeError("replicate-1 development gate failed; preserve baseline and use fallback")
    return gate


def _development_components(
    run_dir: Path, run: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
]:
    tasks = {
        task["id"]: task
        for task in strict_json_loads((run_dir / "tasks.snapshot.json").read_bytes())
    }
    blocks = [
        strict_json_loads(path.read_bytes())
        for path in sorted((run_dir / "blocks").glob("*.json"))
    ]
    randomization = strict_json_loads((run_dir / "randomization.json").read_bytes())
    random_index = {
        (entry["model_name"], entry["task_id"], entry["replicate"]): entry
        for entry in randomization["entries"]
    }
    return blocks, tasks, random_index


def _reconstruct_full_development_cases(
    run_dir: Path, run: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks, tasks, random_index = _development_components(run_dir, run)
    cases = []
    for block in blocks:
        for replicate in range(1, 4):
            name = f"{block['model_name']}__{block['task_id']}__rep-{replicate:02d}.json"
            capture = strict_json_loads((run_dir / "continuations" / name).read_bytes())
            journal = strict_json_loads((run_dir / "outcome-journals" / name).read_bytes())
            randomization = random_index[
                (block["model_name"], block["task_id"], replicate)
            ]
            task = tasks[block["task_id"]]
            _validate_outcome_journal(
                journal, block, task, randomization, replicate
            )
            expected = _capture_from_journal(journal, block, task)
            if {
                key: value for key, value in capture.items() if key != "created_at"
            } != expected:
                raise RuntimeError("development capture does not reconstruct")
            cases.append(capture)
    return blocks, cases


def verify_completed_generation_scope(
    run_dir: Path,
    run: dict[str, Any],
    task_ids: tuple[str, ...],
) -> None:
    snapshot = run["snapshot"]
    sources = load_parent_sources(
        Path(snapshot["parent_run"]),
        tuple(snapshot["selected_models"]),
        task_ids,
    )
    source_by_name = {path.name: (path, source) for path, source in sources}
    tiers = tuple(snapshot["generated_tiers"])
    expected = {
        (tier, source_path.name)
        for tier in tiers
        for source_path, _source in sources
    }
    actual = {
        (path.parent.name, path.name)
        for path in (run_dir / "generation-journals").glob("*/*.json")
    }
    if actual != expected:
        raise RuntimeError("failed baseline generation scope is incomplete or has extras")
    for tier, parent_file in sorted(expected):
        journal_path = run_dir / "generation-journals" / tier / parent_file
        journal = strict_json_loads(journal_path.read_bytes())
        source_path, source = source_by_name[parent_file]
        if (
            journal.get("status") != "complete"
            or journal.get("parent_file") != parent_file
            or journal.get("parent_file_sha256") != file_sha256(source_path)
            or journal.get("code_manifest_sha256")
            != snapshot["executable_manifest_sha256"]
        ):
            raise RuntimeError("failed baseline generation journal provenance changed")
        _validate_generation_journal(
            journal,
            source,
            tier,
            snapshot["rewriter"],
        )
        variant_path = run_dir / "variants" / tier / parent_file
        expected_variant = _variant_from_generation_journal(
            source_path, source, tier, journal
        )
        if (
            not variant_path.exists()
            or strict_json_loads(variant_path.read_bytes()) != expected_variant
        ):
            raise RuntimeError("failed baseline variant does not reconstruct")


def verify_failed_baseline_run(run_dir: Path) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    run = strict_json_loads(run_path.read_bytes())
    protocol = run.get("snapshot", {}).get("protocol", {})
    if (
        protocol.get("name") != "deepseek_self_development"
        or protocol.get("prompt_selection") != "baseline_raw_only"
        or run.get("snapshot", {}).get("rewriter", {}).get("profile_name")
        != "deepseek_self"
        or set(run.get("snapshot", {}).get("selected_tasks", []))
        != SELF_REWRITE_DEVELOPMENT_TASKS
    ):
        raise RuntimeError("example retry baseline is not the exact baseline development protocol")
    evidence: dict[str, Any] | None = None
    analysis_path = run_dir / "analysis-rep-01.json"
    if evidence is None and analysis_path.exists():
        blocks, tasks, random_index = _development_components(run_dir, run)
        gate = verify_first_replicate_development_analysis(
            run_dir, blocks, tasks, random_index, require_pass=False
        )
        if gate["passed"] is False:
            evidence = {
                "kind": "failed_first_replicate_behavioral_gate",
                "analysis_sha256": file_sha256(analysis_path),
                "analysis_manifest_sha256": file_sha256(
                    run_dir / "ANALYSIS-REP-01_SHA256SUMS"
                ),
            }
    full_analysis_path = run_dir / "analysis.json"
    if evidence is None and full_analysis_path.exists():
        _verify_analysis_manifest(
            run_dir, "analysis.json", "report.md", "ANALYSIS_SHA256SUMS"
        )
        blocks, cases = _reconstruct_full_development_cases(run_dir, run)
        recomputed = full_development_expansion_gate(cases, blocks)
        full_analysis = strict_json_loads(full_analysis_path.read_bytes())
        if (
            recomputed["passed"] is False
            and full_analysis.get("full_development_expansion_gate") == recomputed
        ):
            evidence = {
                "kind": "failed_full_development_gate",
                "analysis_sha256": file_sha256(full_analysis_path),
                "analysis_manifest_sha256": file_sha256(
                    run_dir / "ANALYSIS_SHA256SUMS"
                ),
            }
    if evidence is None:
        for name, kind in (
            (
                "semantic-canary-audit-hospital-decision-v2.json",
                "failed_hospital_semantic_canary",
            ),
            (
                "style-canary-audit-hospital-decision-v2.json",
                "failed_hospital_style_canary",
            ),
            ("semantic-audit.json", "failed_complete_semantic_gate"),
            ("variant-audit.json", "failed_complete_style_gate"),
        ):
            path = run_dir / name
            if not path.exists():
                continue
            value = strict_json_loads(path.read_bytes())
            failed = value.get("passed") is False or any(
                decision.get("passed") is False
                for decision in value.get("decisions", [])
            )
            if failed:
                evidence = {"kind": kind, "file": name, "file_sha256": file_sha256(path)}
                break
    if evidence is None:
        failed_journals = []
        snapshot = run["snapshot"]
        parent_sources = Path(snapshot["parent_run"]) / "sources"
        selected_sources = load_parent_sources(
            Path(snapshot["parent_run"]),
            tuple(snapshot["selected_models"]),
            tuple(snapshot["selected_tasks"]),
        )
        allowed_cells = {
            (tier, source_path.name)
            for tier in snapshot["generated_tiers"]
            for source_path, _source in selected_sources
        }
        for path in sorted((run_dir / "generation-journals").glob("*/*.json")):
            journal = strict_json_loads(path.read_bytes())
            if (path.parent.name, path.name) not in allowed_cells:
                raise RuntimeError("baseline contains an out-of-scope generation journal")
            if journal.get("status") != "automatic_gate_failure":
                continue
            parent_file = journal.get("parent_file")
            if (
                not isinstance(parent_file, str)
                or Path(parent_file).name != parent_file
                or path.name != parent_file
                or path.parent.name != journal.get("tier")
            ):
                raise RuntimeError("failed generation journal path binding changed")
            source_path = parent_sources / parent_file
            source = strict_json_loads(source_path.read_bytes())
            if (
                journal.get("parent_file_sha256") != file_sha256(source_path)
                or journal.get("parent_source_sha256") != source.get("source_sha256")
                or journal.get("model_name") != source.get("model_name")
                or journal.get("task_id") != source.get("task_id")
                or journal.get("code_manifest_sha256")
                != snapshot["executable_manifest_sha256"]
            ):
                raise RuntimeError("failed generation journal provenance changed")
            _validate_generation_journal(
                journal,
                source,
                journal["tier"],
                snapshot["rewriter"],
            )
            failed_journals.append(
                {
                    "file": str(path.relative_to(run_dir)),
                    "file_sha256": file_sha256(path),
                    "status": journal["status"],
                }
            )
        if failed_journals:
            evidence = {
                "kind": "failed_generation_cell",
                "journals": failed_journals,
            }
    if evidence is None:
        raise RuntimeError("baseline has no frozen prespecified failure evidence")
    if evidence["kind"] != "failed_generation_cell":
        scope = (
            ("hospital-decision-v2",)
            if evidence["kind"] in {
                "failed_hospital_semantic_canary",
                "failed_hospital_style_canary",
            }
            else tuple(sorted(SELF_REWRITE_DEVELOPMENT_TASKS))
        )
        verify_completed_generation_scope(run_dir, run, scope)
    canary_sha256 = verify_completed_rewriter_canary(run_dir, run)
    return {
        "run_dir": str(run_dir),
        "run_json_sha256": file_sha256(run_path),
        "rewriter_canary_journal_sha256": canary_sha256,
        "evidence": evidence,
    }


def verify_passed_development_run(
    run_dir: Path, rewriter: dict[str, Any]
) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    run = strict_json_loads(run_path.read_bytes())
    protocol = run.get("snapshot", {}).get("protocol", {})
    if (
        run.get("status") != "five_arm_outcomes_complete_pending_analysis"
        or protocol.get("name") != "deepseek_self_development"
        or run.get("snapshot", {}).get("rewriter") != rewriter
        or set(run.get("snapshot", {}).get("selected_tasks", []))
        != SELF_REWRITE_DEVELOPMENT_TASKS
    ):
        raise RuntimeError("expansion selection is not one completed matching development run")
    _verify_frozen_file_manifest(
        run_dir / "blocks", run["block_manifest_sha256"]
    )
    _verify_frozen_file_manifest(
        run_dir / "continuations", run["continuation_manifest_sha256"]
    )
    _verify_frozen_file_manifest(
        run_dir / "outcome-journals", run["outcome_journal_manifest_sha256"]
    )
    _verify_analysis_manifest(
        run_dir, "analysis.json", "report.md", "ANALYSIS_SHA256SUMS"
    )
    analysis_path = run_dir / "analysis.json"
    analysis = strict_json_loads(analysis_path.read_bytes())
    blocks, cases = _reconstruct_full_development_cases(run_dir, run)
    recomputed_gate = full_development_expansion_gate(cases, blocks)
    gate = analysis.get("full_development_expansion_gate")
    capture_names = {
        f"{case['model_name']}__{case['task_id']}__rep-{case['replicate']:02d}.json"
        for case in cases
    }
    analysis_input_sha256 = canonical_hash(
        {
            "block_manifest_sha256": run["block_manifest_sha256"],
            "randomization_sha256": file_sha256(run_dir / "randomization.json"),
            "continuations": {
                name: file_sha256(run_dir / "continuations" / name)
                for name in sorted(capture_names)
            },
            "outcome_journals": {
                name: file_sha256(run_dir / "outcome-journals" / name)
                for name in sorted(capture_names)
            },
        }
    )
    if (
        gate != recomputed_gate
        or recomputed_gate.get("passed") is not True
        or analysis.get("analysis_input_sha256") != analysis_input_sha256
        or analysis.get("tasks") != sorted(SELF_REWRITE_DEVELOPMENT_TASKS)
        or analysis.get("cases") != 9
        or analysis.get("partial_replicate_limit") is not None
    ):
        raise RuntimeError("development analysis did not pass the frozen expansion gate")
    outcome_freeze_path = run_dir / "outcome-freeze.json"
    if file_sha256(outcome_freeze_path) != run.get("outcome_freeze_sha256"):
        raise RuntimeError("selected development outcome freeze changed")
    canary_path = run_dir / "rewriter-canary-journal.json"
    if (
        run.get("rewriter_canary_passed") is not True
        or file_sha256(canary_path) != run.get("rewriter_canary_journal_sha256")
    ):
        raise RuntimeError("selected development run lacks its frozen self-rewriter canary")
    return {
        "run_dir": str(run_dir),
        "run_json_sha256": file_sha256(run_path),
        "analysis_sha256": file_sha256(analysis_path),
        "analysis_manifest_sha256": file_sha256(run_dir / "ANALYSIS_SHA256SUMS"),
        "outcome_freeze_sha256": file_sha256(outcome_freeze_path),
        "rewriter_canary_journal_sha256": file_sha256(canary_path),
    }


def continue_block(
    run_dir: Path,
    block: dict[str, Any],
    task: dict[str, Any],
    randomization: dict[str, Any],
    replicate: int,
) -> None:
    stem = f"{block['model_name']}__{block['task_id']}__rep-{replicate:02d}"
    output_path = run_dir / "continuations" / f"{stem}.json"
    journal_path = run_dir / "outcome-journals" / f"{stem}.json"
    if output_path.exists() and not journal_path.exists():
        raise RuntimeError(f"completed capture lacks its outcome journal: {output_path}")
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
    else:
        journal = _initialize_outcome_journal(block, randomization, replicate)
        atomic_json(journal_path, journal)

    if canonical_hash(task) != block["task_sha256"]:
        raise RuntimeError("executable task differs from the task bound into the block")
    if output_path.exists():
        capture = json.loads(output_path.read_text())
        _validate_outcome_journal(journal, block, task, randomization, replicate)
        expected = _capture_from_journal(journal, block, task)
        if (
            {key: value for key, value in capture.items() if key != "created_at"} != expected
            or journal.get("capture_core_sha256") not in {None, canonical_hash(expected)}
            or not isinstance(capture.get("created_at"), str)
            or (
                journal.get("capture_created_at") is not None
                and journal["capture_created_at"] != capture["created_at"]
            )
        ):
            raise RuntimeError("interrupted capture does not reconstruct from its journal")
        if journal.get("status") != "complete":
            journal.update(
                status="complete",
                completed_at=now(),
                capture_created_at=capture["created_at"],
                capture_core_sha256=canonical_hash(expected),
                capture_file=str(output_path.relative_to(run_dir)),
                capture_sha256=file_sha256(output_path),
            )
            for branch in journal["branches"].values():
                branch.pop("messages", None)
            atomic_json(journal_path, journal)
        if (
            journal.get("status") != "complete"
            or journal.get("block_sha256") != block["block_sha256"]
            or journal.get("replicate") != replicate
            or journal.get("continuation_seed") != randomization["continuation_seed"]
            or journal.get("orders") != randomization["orders"]
            or journal.get("capture_sha256") != file_sha256(output_path)
            or journal.get("capture_created_at") != capture.get("created_at")
            or capture.get("status") != "complete"
            or capture.get("block_sha256") != block["block_sha256"]
            or capture.get("replicate") != replicate
            or capture.get("continuation_seed") != randomization["continuation_seed"]
            or capture.get("orders") != randomization["orders"]
            or set(capture.get("branches", {})) != set(ARMS)
        ):
            raise RuntimeError(f"existing tier outcome has wrong provenance: {output_path}")
        return

    _validate_outcome_journal(journal, block, task, randomization, replicate)
    if journal.get("status") in {"terminal_transport_failure", "terminal_route_failure"}:
        raise RuntimeError(f"outcome journal is terminal: {journal.get('status')}")
    tools = build_tools(task)
    action_phases = task["phases"][task["fork_after"] :]
    slot_names = [phase["name"] for phase in action_phases] + ["final"]
    for slot_index, slot_name in enumerate(slot_names):
        for arm in journal["orders"][slot_name]:
            slot_key = _slot_attempt_key(slot_index, arm)
            if slot_key in journal["completed_slots"]:
                continue
            branch = journal["branches"][arm]
            if branch["status"] == "action_failure":
                journal["completed_slots"].append(slot_key)
                atomic_json(journal_path, journal)
                continue
            phase_offset = task["fork_after"] + slot_index
            request_seed = journal["continuation_seed"] + (
                len(task["phases"]) + 1 if slot_name == "final" else phase_offset
            )
            payload = target_payload(block["model"], branch["messages"], tools, request_seed)
            print(
                f"continue {block['model_name']} {block['task_id']} rep={replicate} "
                f"slot={slot_name} arm={arm}",
                flush=True,
            )
            response, wall = _target_once(
                journal_path,
                journal,
                arm,
                slot_index,
                block["model_name"],
                payload,
            )
            turn = {
                "request": copy.deepcopy(payload),
                "response": response,
                "wall_seconds": wall,
                "metrics": response_metrics(response, wall),
                "turn": len(task["phases"]) + 1 if slot_name == "final" else phase_offset + 1,
                "phase": slot_name,
            }
            message = response["choices"][0]["message"]
            if slot_name == "final":
                score = _final_response_score(message, task["final_answer"])
                turn.update(score)
                branch.update(
                    final_correct=(
                        score["final_correct"]
                        and score["unexpected_tool_calls"] == 0
                        and score["format_error"] is None
                    ),
                    final_content=message.get("content") if isinstance(message, dict) else None,
                    status="complete",
                )
            else:
                phase = action_phases[slot_index]
                action = evaluate_tool_response(message, phase)
                turn["action"] = action
                branch["actions_correct"] = branch["actions_correct"] and action["correct"]
                if action["correct"]:
                    append_tool_cycle(
                        branch["messages"], message, reasoning_text(message), phase
                    )
                else:
                    branch["status"] = "action_failure"
            branch["turns"].append(turn)
            journal["completed_slots"].append(slot_key)
            atomic_json(journal_path, journal)

    capture_core = _capture_from_journal(journal, block, task)
    capture_core_sha256 = canonical_hash(capture_core)
    if journal.get("capture_core_sha256") not in {None, capture_core_sha256}:
        raise RuntimeError("journaled capture core no longer reconstructs")
    if not journal.get("capture_created_at"):
        journal.update(
            status="capture_ready",
            capture_created_at=now(),
            capture_core_sha256=capture_core_sha256,
        )
        atomic_json(journal_path, journal)
    capture = {"created_at": journal["capture_created_at"], **capture_core}
    atomic_json(output_path, capture)
    journal.update(
        status="complete",
        completed_at=now(),
        capture_file=str(output_path.relative_to(run_dir)),
        capture_sha256=file_sha256(output_path),
    )
    for branch in journal["branches"].values():
        branch.pop("messages", None)
    atomic_json(journal_path, journal)


def _delivery_record(
    model_name: str,
    condition: str,
    trial: int,
    payload: dict[str, Any],
    expected_tool: str,
    case_id: str,
    response: dict[str, Any],
    wall: float,
    response_received_at: str,
) -> dict[str, Any]:
    message = response["choices"][0]["message"]
    expected_arguments = {"case_id": case_id}
    score = evaluate_tool_response(
        message,
        {"name": expected_tool, "expected_arguments": expected_arguments},
    )
    return {
        "model_name": model_name,
        "model": payload["model"],
        "condition": condition,
        "trial": trial,
        "expected_tool": expected_tool,
        "expected_arguments": expected_arguments,
        "actual_tool": score.get("actual_tool"),
        "actual_arguments": score.get("actual_arguments"),
        "call_count": score["call_count"],
        "correct": score["correct"],
        "tool_score": score,
        "response_received_at": response_received_at,
        "request": copy.deepcopy(payload),
        "response": copy.deepcopy(response),
        "metrics": response_metrics(response, wall),
    }


def run_delivery_audit(run_dir: Path) -> None:
    run = json.loads((run_dir / "run.json").read_text())
    validation_only = bool(run.get("outcomes_launched"))
    if not run.get("variants_frozen"):
        raise RuntimeError("delivery audit requires the frozen five-arm blocks")
    if validation_only and not (run_dir / "delivery-audit.json").exists():
        raise RuntimeError("delivery audit cannot be created after outcomes launch")
    _verify_frozen_file_manifest(run_dir / "blocks", run["block_manifest_sha256"])
    blocks = [
        json.loads(path.read_text())
        for path in sorted((run_dir / "blocks").glob("*.json"))
    ]
    models = {}
    for block in blocks:
        models.setdefault(block["model_name"], block["model"])
        if models[block["model_name"]] != block["model"]:
            raise RuntimeError("blocks disagree on delivery-audit model configuration")
    cells = [
        (model_name, condition, trial)
        for model_name in sorted(models)
        for condition, count in (("reasoning", 5), ("visible", 2), ("absent", 4))
        for trial in range(1, count + 1)
    ]
    journal_path = run_dir / "delivery-audit-journal.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal.get("status") in {
            "terminal_route_failure",
            "terminal_transport_failure",
        }:
            raise RuntimeError(f"delivery audit journal is terminal: {journal['status']}")
        if (
            journal.get("code_manifest_sha256")
            != run["snapshot"]["executable_manifest_sha256"]
            or journal.get("block_manifest_sha256") != run["block_manifest_sha256"]
            or journal.get("cells") != [list(cell) for cell in cells]
        ):
            raise RuntimeError("delivery audit journal provenance changed")
    else:
        journal = {
            "created_at": now(),
            "status": "pending",
            "code_manifest_sha256": run["snapshot"]["executable_manifest_sha256"],
            "block_manifest_sha256": run["block_manifest_sha256"],
            "cells": [list(cell) for cell in cells],
            "slot_attempts": {},
            "records": {},
        }
        atomic_json(journal_path, journal)
    expected_record_keys = {
        f"{model_name}:{condition}:{trial}"
        for model_name, condition, trial in cells
    }
    if journal.get("status") == "complete" and set(journal.get("records", {})) != expected_record_keys:
        raise RuntimeError("completed delivery journal lacks the exact canary matrix")
    for cell_index, (model_name, condition, trial) in enumerate(cells):
        payload, expected_tool, case_id = delivery_payload_for(
            model_name, models[model_name], condition, trial
        )
        arm = f"delivery-{cell_index:02d}"
        response, wall = _target_once(
            journal_path, journal, arm, 0, model_name, payload
        )
        slot_key = _slot_attempt_key(0, arm)
        completed_attempt = journal["slot_attempts"][slot_key][-1]
        if completed_attempt.get("status") != "usable_response":
            raise RuntimeError("delivery response did not finish on a usable attempt")
        record = _delivery_record(
            model_name,
            condition,
            trial,
            payload,
            expected_tool,
            case_id,
            response,
            wall,
            completed_attempt["response_received_at"],
        )
        key = f"{model_name}:{condition}:{trial}"
        if key in journal["records"] and journal["records"][key] != record:
            raise RuntimeError("delivery audit record no longer reconstructs")
        if key not in journal["records"]:
            journal["records"][key] = record
            atomic_json(journal_path, journal)
    records = sorted(
        journal["records"].values(),
        key=lambda record: (
            record["model_name"], record["condition"], record["trial"]
        ),
    )
    summary = {}
    for model_name in models:
        by_condition = {
            condition: [
                record
                for record in records
                if record["model_name"] == model_name
                and record["condition"] == condition
            ]
            for condition in ("reasoning", "visible", "absent")
        }
        hits = {
            condition: sum(record["correct"] for record in condition_records)
            for condition, condition_records in by_condition.items()
        }
        summary[model_name] = {
            **{f"{condition}_hits": hits[condition] for condition in hits},
            **{
                f"{condition}_trials": len(by_condition[condition])
                for condition in by_condition
            },
            "errors": 0,
            "functional_history_retention": (
                hits["reasoning"] == 5
                and hits["visible"] == 2
                and hits["absent"] <= 1
            ),
        }
    if journal.get("status") != "complete":
        journal.update(status="complete", completed_at=now(), audit_created_at=now())
        atomic_json(journal_path, journal)
    output = {
        "created_at": journal["audit_created_at"],
        "method": "recover an exact tool action stored only in historical assistant reasoning",
        "interpretation_limit": (
            "A positive result demonstrates functional use through the hosted route, not an exact "
            "provider-side serialization or checkpoint identity."
        ),
        "summary": summary,
        "errors": [],
        "records": records,
        "journal_sha256": file_sha256(journal_path),
    }
    reconcile_immutable_json(run_dir / "delivery-audit.json", output)


def _validate_delivery_audit(
    audit: dict[str, Any], models: dict[str, dict[str, Any]], frozen_at: str
) -> None:
    try:
        created = datetime.fromisoformat(audit["created_at"])
        frozen = datetime.fromisoformat(frozen_at)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("delivery audit lacks a valid timestamp") from exc
    if created.tzinfo is None or frozen.tzinfo is None:
        raise RuntimeError("delivery audit timestamps must be timezone-aware")
    age_seconds = (datetime.now(timezone.utc) - created).total_seconds()
    if created < frozen or age_seconds < 0 or age_seconds > 24 * 60 * 60:
        raise RuntimeError("delivery audit is not contemporaneous with the frozen five-arm run")
    if audit.get("errors") or set(audit.get("summary", {})) != set(models):
        raise RuntimeError("delivery audit has errors or the wrong exact route set")
    records = audit.get("records", [])
    expected_cells = {
        (model_name, condition, trial)
        for model_name in models
        for condition, count in (("reasoning", 5), ("visible", 2), ("absent", 4))
        for trial in range(1, count + 1)
    }
    actual_cells = {
        (record.get("model_name"), record.get("condition"), record.get("trial"))
        for record in records
    }
    if len(records) != len(expected_cells) or actual_cells != expected_cells:
        raise RuntimeError("delivery audit does not cover the exact 5/2/4 canary matrix")
    for model_name, model in models.items():
        summary = audit["summary"][model_name]
        if (
            summary.get("reasoning_hits") != 5
            or summary.get("reasoning_trials") != 5
            or summary.get("visible_hits") != 2
            or summary.get("visible_trials") != 2
            or summary.get("absent_trials") != 4
            or summary.get("absent_hits", 5) > 1
            or summary.get("errors") != 0
            or summary.get("functional_history_retention") is not True
        ):
            raise RuntimeError(f"route failed contemporaneous history delivery: {model_name}")
        for record in (item for item in records if item["model_name"] == model_name):
            request = record["request"]
            response = record["response"]
            if (
                request.get("model") != model["model"]
                or request.get("provider") != model["provider"]
                or response.get("model") != model["model"]
                or response.get("provider") != EXPECTED_PROVIDERS[model_name]
            ):
                raise RuntimeError("delivery canary route/model does not match the frozen target")
            messages = request.get("messages")
            if not isinstance(messages, list) or len(messages) != 4:
                raise RuntimeError("delivery canary request must contain four messages")
            historical = messages[2]
            if not isinstance(historical, dict):
                raise RuntimeError("delivery canary historical message must be an object")
            try:
                response_received_at = datetime.fromisoformat(
                    record["response_received_at"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("invalid delivery response timestamp") from exc
            if response_received_at.tzinfo is None:
                raise RuntimeError("delivery response timestamp must be timezone-aware")
            response_age = (
                datetime.now(timezone.utc) - response_received_at
            ).total_seconds()
            if (
                response_received_at < frozen
                or response_age < 0
                or response_age > 24 * 60 * 60
            ):
                raise RuntimeError(
                    "every delivery response must postdate freeze and be within 24 hours"
                )
            serialized_visible = str(historical.get("content", ""))
            serialized_reasoning = str(historical.get("reasoning", ""))
            expected_tool = record["expected_tool"]
            if (
                record["condition"] == "reasoning"
                and not (expected_tool in serialized_reasoning and expected_tool not in serialized_visible)
            ) or (
                record["condition"] == "visible"
                and not (expected_tool in serialized_visible and expected_tool not in serialized_reasoning)
            ) or (
                record["condition"] == "absent"
                and expected_tool in serialized_visible + serialized_reasoning
            ):
                raise RuntimeError("delivery canary condition does not isolate historical reasoning")
            message = response.get("choices", [{}])[0].get("message", {})
            score = evaluate_tool_response(
                message,
                {
                    "name": record["expected_tool"],
                    "expected_arguments": record["expected_arguments"],
                },
            )
            if (
                record.get("tool_score") != score
                or record.get("actual_tool") != score.get("actual_tool")
                or record.get("actual_arguments") != score.get("actual_arguments")
                or record.get("call_count") != score["call_count"]
                or record.get("correct") is not score["correct"]
            ):
                raise RuntimeError("delivery canary score does not recompute")


def outcome_jobs(
    blocks: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    random_index: dict[tuple[str, str, int], dict[str, Any]],
    replicate_limit: int,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]]:
    if replicate_limit not in {1, 2, 3}:
        raise ValueError("replicate limit must be one, two, or three")
    return [
        (
            block,
            tasks[block["task_id"]],
            random_index[(block["model_name"], block["task_id"], replicate)],
            replicate,
        )
        for block in blocks
        for replicate in range(1, replicate_limit + 1)
    ]


def run_continue(
    run_dir: Path,
    parent_run: Path,
    replicates: int,
    replicate_limit: int = 3,
) -> None:
    del parent_run  # All executable task state comes from the run-local frozen snapshot.
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text())
    if not run.get("variants_frozen"):
        raise RuntimeError("five-arm variants must freeze before outcomes")
    already_complete = run.get("status") == "five_arm_outcomes_complete_pending_analysis"
    if already_complete:
        _verify_frozen_file_manifest(
            run_dir / "continuations", run["continuation_manifest_sha256"]
        )
        _verify_frozen_file_manifest(
            run_dir / "outcome-journals", run["outcome_journal_manifest_sha256"]
        )
    if replicates != 3 or run["snapshot"]["replicates"] != 3:
        raise ValueError("the frozen tier protocol requires exactly three replicates")
    if replicate_limit not in {1, 2, 3}:
        raise ValueError("replicate limit must be one, two, or three")
    if already_complete:
        replicate_limit = 3
    protocol_name = run["snapshot"].get("protocol", {}).get("name")
    existing_limit = int(run.get("completed_replicate_limit", 0))
    if not already_complete and replicate_limit < existing_limit:
        raise RuntimeError("replicate limit cannot regress an existing partial capture")
    if (
        protocol_name == "deepseek_self_development"
        and not run.get("outcomes_launched")
        and replicate_limit != 1
    ):
        raise RuntimeError("self-rewrite development must run replicate 1 alone first")
    if protocol_name == "deepseek_self_development":
        canary_bindings = _validate_development_canary_audits(
            run_dir, run, "hospital-decision-v2"
        )
        if any(run.get(key) != value for key, value in canary_bindings.items()):
            raise RuntimeError("development canary evidence changed before target outcomes")
    _verify_frozen_file_manifest(run_dir / "variants", run["variant_manifest_sha256"])
    _verify_frozen_file_manifest(run_dir / "blocks", run["block_manifest_sha256"])
    _verify_frozen_file_manifest(
        run_dir / "generation-journals", run["generation_journal_manifest_sha256"]
    )
    _verify_frozen_file_manifest(
        run_dir / "audit-packets", run["audit_packet_manifest_sha256"]
    )
    if file_sha256(run_dir / "audit-packet-map.json") != run["audit_packet_map_sha256"]:
        raise RuntimeError("audit packet map changed after freeze")
    if file_sha256(run_dir / "semantic-audit.json") != run["semantic_audit_sha256"]:
        raise RuntimeError("semantic audit changed after freeze")
    if file_sha256(run_dir / "variant-audit.json") != run["style_audit_sha256"]:
        raise RuntimeError("style audit changed after freeze")
    if file_sha256(run_dir / "randomization.json") != run["randomization_sha256"]:
        raise RuntimeError("randomization changed after freeze")
    tasks_data = json.loads((run_dir / "tasks.snapshot.json").read_text())
    if canonical_hash(tasks_data) != run["snapshot"]["task_snapshot_sha256"]:
        raise RuntimeError("run-local task snapshot changed after freeze")
    tasks = {task["id"]: task for task in tasks_data}
    block_paths = sorted((run_dir / "blocks").glob("*.json"))
    blocks = [json.loads(path.read_text()) for path in block_paths]
    for path, block in zip(block_paths, blocks):
        if canonical_hash({k: v for k, v in block.items() if k != "block_sha256"}) != block["block_sha256"]:
            raise RuntimeError(f"block hash does not recompute: {path}")
        if canonical_hash(tasks[block["task_id"]]) != block["task_sha256"]:
            raise RuntimeError(f"block/task hash mismatch: {path}")
    randomization = json.loads((run_dir / "randomization.json").read_text())
    random_keys = [
        (entry["model_name"], entry["task_id"], entry["replicate"])
        for entry in randomization["entries"]
    ]
    expected_keys = {
        (block["model_name"], block["task_id"], replicate)
        for block in blocks
        for replicate in range(1, 4)
    }
    if len(random_keys) != len(set(random_keys)) or set(random_keys) != expected_keys:
        raise RuntimeError("randomization does not bind the exact block-by-replicate matrix")
    random_index = {
        (entry["model_name"], entry["task_id"], entry["replicate"]): entry
        for entry in randomization["entries"]
    }
    if (
        protocol_name == "deepseek_self_development"
        and replicate_limit > 1
        and not already_complete
    ):
        if existing_limit < 1:
            raise RuntimeError("replicate 1 must complete before later development replicates")
        verify_first_replicate_development_analysis(
            run_dir, blocks, tasks, random_index
        )
    if not already_complete:
        record_invocation(run_path, run, "continue")
    delivery_path = run_dir / "delivery-audit.json"
    delivery_journal_path = run_dir / "delivery-audit-journal.json"
    if not delivery_path.exists() or not delivery_journal_path.exists():
        raise RuntimeError("run a contemporaneous exact-route delivery audit before outcomes")
    run_delivery_audit(run_dir)
    delivery_audit = json.loads(delivery_path.read_text())
    if delivery_audit.get("journal_sha256") != file_sha256(delivery_journal_path):
        raise RuntimeError("delivery audit does not bind its raw response journal")
    if run.get("delivery_audit_sha256"):
        if (
            file_sha256(delivery_path) != run["delivery_audit_sha256"]
            or file_sha256(delivery_journal_path)
            != run["delivery_audit_journal_sha256"]
        ):
            raise RuntimeError("delivery audit changed after outcome launch")
    else:
        models = {}
        for block in blocks:
            if block["model_name"] in models and models[block["model_name"]] != block["model"]:
                raise RuntimeError("blocks disagree on an exact target model configuration")
            models[block["model_name"]] = block["model"]
        _validate_delivery_audit(delivery_audit, models, run["frozen_at"])
    if not run.get("outcomes_launched"):
        if (run_dir / "continuations").exists() or (run_dir / "outcome-journals").exists():
            raise RuntimeError("outcome artifacts exist before the launch marker")
        run.update(
            outcomes_launched=True,
            outcomes_started_at=now(),
            status="five_arm_outcomes_in_progress",
            delivery_audit_sha256=file_sha256(delivery_path),
            delivery_audit_journal_sha256=file_sha256(delivery_journal_path),
        )
        atomic_json(run_path, run)
    jobs = outcome_jobs(blocks, tasks, random_index, replicate_limit)
    for block, _task, entry, replicate in jobs:
        stem = f"{block['model_name']}__{block['task_id']}__rep-{replicate:02d}"
        output_path = run_dir / "continuations" / f"{stem}.json"
        journal_path = run_dir / "outcome-journals" / f"{stem}.json"
        if output_path.exists() and not journal_path.exists():
            raise RuntimeError(f"completed capture lacks its outcome journal: {output_path}")
        if not journal_path.exists():
            atomic_json(
                journal_path,
                _initialize_outcome_journal(block, entry, replicate),
            )

    def execute(
        job: tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]
    ) -> None:
        block, task, entry, replicate = job
        continue_block(run_dir, block, task, entry, replicate)

    workers = run["snapshot"]["execution"]["workers"]
    run_bounded_jobs(jobs, workers, execute)
    outputs = sorted((run_dir / "continuations").glob("*.json"))
    journals = sorted((run_dir / "outcome-journals").glob("*.json"))
    expected = run["included_sources"] * 3
    expected_names = {
        f"{block['model_name']}__{block['task_id']}__rep-{replicate:02d}.json"
        for block in blocks
        for replicate in range(1, 4)
    }
    expected_through_limit = {
        f"{block['model_name']}__{block['task_id']}__rep-{replicate:02d}.json"
        for block in blocks
        for replicate in range(1, replicate_limit + 1)
    }
    output_names = {path.name for path in outputs}
    journal_names = {path.name for path in journals}
    if (
        not expected_through_limit <= output_names
        or not expected_through_limit <= journal_names
        or not output_names <= expected_names
        or not journal_names <= expected_names
        or output_names != journal_names
    ):
        raise RuntimeError(
            f"tier outcome subset incomplete or contains extras: "
            f"{len(outputs)}/{run['included_sources'] * replicate_limit}"
        )
    if replicate_limit < 3:
        if already_complete:
            raise RuntimeError("completed outcome run cannot regress to a partial status")
        run.update(
            status="five_arm_outcomes_in_progress",
            continuation_blocks=len(outputs),
            completed_replicate_limit=replicate_limit,
            continuation_workers=workers,
            partial_capture_updated_at=now(),
        )
        atomic_json(run_path, run)
        return
    if len(outputs) != expected or output_names != expected_names:
        raise RuntimeError(f"tier outcome matrix incomplete or contains extras: {len(outputs)}/{expected}")
    if already_complete:
        freeze_path = run_dir / "outcome-freeze.json"
        expected_freeze = {
            "version": 1,
            "status": "outcome_capture_frozen",
            "code_manifest_sha256": run["snapshot"]["executable_manifest_sha256"],
            "variant_manifest_sha256": run["variant_manifest_sha256"],
            "block_manifest_sha256": run["block_manifest_sha256"],
            "randomization_sha256": run["randomization_sha256"],
            "semantic_audit_sha256": run["semantic_audit_sha256"],
            "style_audit_sha256": run["style_audit_sha256"],
            "audit_packet_map_sha256": run["audit_packet_map_sha256"],
            "delivery_audit_sha256": run["delivery_audit_sha256"],
            "delivery_audit_journal_sha256": run["delivery_audit_journal_sha256"],
            "continuation_manifest_sha256": run["continuation_manifest_sha256"],
            "outcome_journal_manifest_sha256": run["outcome_journal_manifest_sha256"],
            "continuation_blocks": len(outputs),
        }
        if run["snapshot"].get("tier_mode") == "four_fresh":
            expected_freeze["continuation_workers"] = workers
        if (
            file_sha256(freeze_path) != run["outcome_freeze_sha256"]
            or json.loads(freeze_path.read_text()) != expected_freeze
        ):
            raise RuntimeError("completed outcome freeze marker changed")
        return
    manifest = _write_manifest(run_dir / "continuations", outputs)
    journal_manifest = _write_manifest(run_dir / "outcome-journals", journals)
    freeze = {
        "version": 1,
        "status": "outcome_capture_frozen",
        "code_manifest_sha256": run["snapshot"]["executable_manifest_sha256"],
        "variant_manifest_sha256": run["variant_manifest_sha256"],
        "block_manifest_sha256": run["block_manifest_sha256"],
        "randomization_sha256": run["randomization_sha256"],
        "semantic_audit_sha256": run["semantic_audit_sha256"],
        "style_audit_sha256": run["style_audit_sha256"],
        "audit_packet_map_sha256": run["audit_packet_map_sha256"],
        "delivery_audit_sha256": run["delivery_audit_sha256"],
        "delivery_audit_journal_sha256": run["delivery_audit_journal_sha256"],
        "continuation_manifest_sha256": file_sha256(manifest),
        "outcome_journal_manifest_sha256": file_sha256(journal_manifest),
        "continuation_blocks": len(outputs),
    }
    if run["snapshot"].get("tier_mode") == "four_fresh":
        freeze["continuation_workers"] = workers
    freeze_path = run_dir / "outcome-freeze.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:
        raise RuntimeError("refusing to replace a differing outcome freeze marker")
    if not freeze_path.exists():
        atomic_json(freeze_path, freeze)
    run.update(
        status="five_arm_outcomes_complete_pending_analysis",
        outcomes_completed_at=now(),
        continuation_blocks=len(outputs),
        continuation_workers=workers,
        completed_replicate_limit=3,
        continuation_manifest_sha256=file_sha256(manifest),
        outcome_journal_manifest_sha256=file_sha256(journal_manifest),
        outcome_freeze_sha256=file_sha256(freeze_path),
    )
    atomic_json(run_path, run)


@contextmanager
def exclusive_run_lock(run_dir: Path):
    digest = hashlib.sha256(str(run_dir).encode()).hexdigest()[:24]
    lock_path = Path("/tmp") / f"short-reasoning-history-tier-{digest}.lock"
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns the tier run lock: {run_dir}") from exc
        yield


def _run_main(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    parent_run = args.parent_run.resolve()
    rewriter = load_rewriter_profile(args.rewriter_profile)
    all_task_ids = {
        task["id"] for task in json.loads((parent_run / "tasks.snapshot.json").read_text())
    }
    model_names = selected_names(args.models, set(EXPECTED_PROVIDERS), "model")
    task_ids = selected_names(args.tasks, all_task_ids, "task")
    protocol_context = build_protocol_context(
        args.protocol_profile,
        rewriter,
        model_names,
        task_ids,
        args.tier_mode,
        args.workers,
        not args.no_budget_guard,
        args.baseline_run,
        args.selection_run,
    )
    initialize(
        run_dir,
        parent_run,
        rewriter,
        args.replicates,
        None if args.models == "all" else model_names,
        None if args.tasks == "all" else task_ids,
        tier_mode=args.tier_mode,
        workers=args.workers,
        budget_guard=not args.no_budget_guard,
        protocol_context=protocol_context,
    )
    if args.stage == "init":
        print(run_dir)
        return
    if args.stage == "unblind":
        unblind_packets(run_dir)
        print(run_dir)
        return
    if args.stage == "finalize":
        finalize_variants(run_dir, parent_run, args.replicates)
        print(run_dir)
        return
    load_dotenv(ROOT / ".env")
    if args.no_budget_guard:
        os.environ.pop("OPENROUTER_MIN_BALANCE_USD", None)
        os.environ.pop("OPENROUTER_MAX_CALL_COST_USD", None)
    else:
        os.environ["OPENROUTER_MIN_BALANCE_USD"] = str(args.min_balance)
        os.environ["OPENROUTER_MAX_CALL_COST_USD"] = str(args.call_reserve)
    if args.stage == "canary":
        run_rewriter_canary(run_dir, parent_run, rewriter, args.canary_task)
    elif args.stage == "generate":
        generation_tasks = (
            selected_names(args.generation_tasks, set(task_ids), "generation task")
            if args.generation_tasks
            else None
        )
        run_generate(run_dir, parent_run, rewriter, generation_tasks)
    elif args.stage == "delivery":
        run_delivery_audit(run_dir)
    else:
        run_continue(
            run_dir,
            parent_run,
            args.replicates,
            replicate_limit=args.replicate_limit,
        )
    print(run_dir)


def main() -> None:
    args = parse_args()
    with exclusive_run_lock(args.run_dir.resolve()):
        _run_main(args)


if __name__ == "__main__":
    main()
