#!/usr/bin/env python3
"""Model-assisted, outcome-blind semantic/style review for frozen tier packets."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_history_tiers as runner  # noqa: E402
from short_reasoning import (  # noqa: E402
    BudgetFloorReached,
    ChatAttemptError,
    load_dotenv,
    post_chat_once_raw,
    strict_json_loads,
)

REVIEWER = {
    "model": "openai/gpt-5.6-sol",
    "provider": {
        "only": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": True,
    },
    "reasoning": {"effort": "none", "exclude": True},
    "include_reasoning": False,
}
SEMANTIC_SYSTEM = """You are a strict blinded scientific reviewer of one historical-reasoning rewrite.
You may inspect only the supplied raw reasoning, accepted compact reference, requested action, and candidate.
Judge semantic preservation, not literal wording. A shorter candidate may omit rhetoric and repeated arithmetic,
but it must preserve every decision-relevant fact, number/unit, comparator, uncertainty, evidence-status distinction,
correction, rejected branch that can matter later, dependency, conclusion, and intended action. The raw reasoning is
authoritative if the compact reference adds process-control metadata about the requested action's unexecuted result;
the candidate should omit that unsupported metadata. Exact action function names and arguments can themselves
encode the requested object and case; do not demand redundant prose that merely expands those exact fields. It must
not add facts or imply a pending/future result. Return the exact schema. Set a check false when there is a concrete
omission, distortion, unsupported addition, or leakage;
explain every false check briefly in notes."""
STYLE_SYSTEM = """You are a strict condition-aware style reviewer for a historical-reasoning dose study.
Semantic equivalence was reviewed separately. Judge whether the candidate actually follows its named style tier
and whether it uses no semantic padding merely to hit length. Dose compliance is computed independently by code.
Return the exact schema and explain every false check briefly."""


def schema_for(checks: set[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {key: {"type": "boolean"} for key in sorted(checks)}
    properties["notes"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


def payload(system: str, value: dict[str, Any], checks: set[str], seed: int) -> dict[str, Any]:
    return {
        **copy.deepcopy(REVIEWER),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(value, ensure_ascii=False)},
        ],
        "seed": seed,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "history_tier_review",
                "strict": True,
                "schema": schema_for(checks),
            },
        },
    }


def persist_raw(journal: dict[str, Any], raw: bytes, wall: float) -> None:
    journal.update(
        status="response_received",
        response_received_at=runner.now(),
        response_body_base64=base64.b64encode(raw).decode("ascii"),
        response_body_sha256=hashlib.sha256(raw).hexdigest(),
        wall_seconds=wall,
    )


def parse_response(journal: dict[str, Any], checks: set[str]) -> dict[str, Any]:
    raw = base64.b64decode(journal["response_body_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != journal["response_body_sha256"]:
        raise RuntimeError("persisted review response hash changed")
    response = strict_json_loads(raw)
    choices = response.get("choices") or []
    if (
        response.get("model") != REVIEWER["model"]
        or response.get("provider") != "OpenAI"
        or len(choices) != 1
        or choices[0].get("finish_reason") != "stop"
    ):
        raise RuntimeError("review response route/finish changed")
    content = choices[0].get("message", {}).get("content")
    value = strict_json_loads(content)
    if set(value) != {*checks, "notes"} or not isinstance(value["notes"], str):
        raise RuntimeError("review response has the wrong schema")
    if any(not isinstance(value[key], bool) for key in checks):
        raise RuntimeError("review response check is not boolean")
    journal.update(
        status="complete",
        completed_at=runner.now(),
        response=response,
        review=value,
        usage=copy.deepcopy(response.get("usage")),
    )
    return value


def prepare_review(path: Path, request: dict[str, Any]) -> None:
    request_sha256 = runner.canonical_hash(request)
    if path.exists():
        journal = strict_json_loads(path.read_bytes())
        if journal.get("request_sha256") != request_sha256 or journal.get("request") != request:
            raise RuntimeError(f"review request changed: {path}")
        return
    journal = {
        "status": "prepared",
        "prepared_at": runner.now(),
        "request": request,
        "request_sha256": request_sha256,
        "attempt_policy": "one request; no resend",
    }
    runner.atomic_json(path, journal)


def one_review(path: Path, request: dict[str, Any], checks: set[str]) -> dict[str, Any]:
    prepare_review(path, request)
    journal = strict_json_loads(path.read_bytes())
    if journal.get("status") == "complete":
        expected = parse_response(copy.deepcopy(journal), checks)
        if journal.get("review") != expected:
            raise RuntimeError("stored review differs from raw response")
        return expected
    if journal.get("status") == "response_received":
        value = parse_response(journal, checks)
        runner.atomic_json(path, journal)
        return value
    if journal.get("status") == "in_flight":
        journal.update(status="ambiguous_after_restart", completed_at=runner.now())
        runner.atomic_json(path, journal)
        raise RuntimeError("ambiguous review request will not be resent")
    if journal.get("status") in {
        "ambiguous_after_restart",
        "ambiguous_transport_error",
        "returned_http_error",
    }:
        raise RuntimeError(f"terminal review failure: {path}: {journal.get('status')}")
    if journal.get("status") not in {"prepared", "budget_pause"}:
        raise RuntimeError(f"unknown review status: {path}: {journal.get('status')}")
    journal.update(status="in_flight", sent_at=runner.now())
    runner.atomic_json(path, journal)
    try:
        raw, wall = post_chat_once_raw(request)
    except BudgetFloorReached:
        journal.update(status="budget_pause", completed_at=runner.now())
        runner.atomic_json(path, journal)
        raise
    except ChatAttemptError as exc:
        if exc.http_status is not None and exc.response_body_bytes is not None:
            runner.persist_returned_http_error(journal, exc)
        else:
            journal.update(
                status="ambiguous_transport_error",
                completed_at=runner.now(),
                ambiguity="No response body returned; the review request will not be resent.",
                **exc.record(),
            )
        runner.atomic_json(path, journal)
        raise
    persist_raw(journal, raw, wall)
    runner.atomic_json(path, journal)
    value = parse_response(journal, checks)
    runner.atomic_json(path, journal)
    return value


def write_review_manifest(directory: Path) -> str:
    paths = sorted(directory.glob("*.json"))
    manifest = runner._write_manifest(directory, paths)
    return runner.file_sha256(manifest)


def _frozen_workers(run: dict[str, Any], requested: int | None) -> int:
    workers = run.get("snapshot", {}).get("execution", {}).get("workers", 1)
    if not isinstance(workers, int) or workers < 1:
        raise RuntimeError("run has no valid frozen worker count")
    if requested is not None and requested != workers:
        raise RuntimeError("review worker count differs from the frozen run")
    return workers


def run_semantic(run_dir: Path, requested_workers: int | None = None) -> None:
    run = strict_json_loads((run_dir / "run.json").read_bytes())
    workers = _frozen_workers(run, requested_workers)
    if run.get("status") != "variants_and_masked_packets_frozen_pending_semantic_audit":
        if (run_dir / "semantic-audit.json").exists():
            existing = strict_json_loads((run_dir / "semantic-audit.json").read_bytes())
            if (
                existing.get("review_workers") != workers
                and run.get("snapshot", {}).get("tier_mode") == "four_fresh"
            ):
                raise RuntimeError("existing semantic review worker provenance changed")
            return
        raise RuntimeError("semantic review requires frozen masked packets")
    runner._verify_frozen_file_manifest(
        run_dir / "audit-packets", run["audit_packet_manifest_sha256"]
    )
    packet_paths = sorted((run_dir / "audit-packets").glob("*.json"))
    review_dir = run_dir / "review-journals" / "semantic"
    jobs = []
    for index, packet_path in enumerate(packet_paths):
        packet = strict_json_loads(packet_path.read_bytes())
        request = payload(
            SEMANTIC_SYSTEM,
            {
                "raw_reasoning": packet["raw_reasoning"],
                "accepted_compact_anchor": packet["accepted_compact_anchor"],
                "requested_action": packet["requested_action"],
                "candidate_state": packet["candidate_state"],
            },
            runner.SEMANTIC_CHECKS,
            951000 + index,
        )
        journal_path = review_dir / f"{packet['packet_id']}.json"
        prepare_review(journal_path, request)
        jobs.append((index, packet, request, journal_path))

    def execute(job: tuple[int, dict[str, Any], dict[str, Any], Path]) -> dict[str, Any]:
        index, packet, request, journal_path = job
        print(
            f"semantic review {index + 1}/{len(packet_paths)} {packet['packet_id']}",
            flush=True,
        )
        review = one_review(journal_path, request, runner.SEMANTIC_CHECKS)
        checks = {key: review[key] for key in runner.SEMANTIC_CHECKS}
        return {
            "packet_id": packet["packet_id"],
            "passed": all(checks.values()),
            "checks": checks,
            "notes": review["notes"],
        }

    decisions = runner.run_bounded_jobs(jobs, workers, execute)
    audit = {
        "timing": "before_condition_mapping_and_tier_outcomes",
        "audit_packet_manifest_sha256": run["audit_packet_manifest_sha256"],
        "condition_mapping_inspected": False,
        "continuation_outcomes_inspected": False,
        "pending_tool_results_inspected": False,
        "source_history_files_inspected": False,
        "semantic_locked_at": runner.now(),
        "reviewer": copy.deepcopy(REVIEWER),
        "review_journal_manifest_sha256": write_review_manifest(review_dir),
        "decisions": decisions,
    }
    if run.get("snapshot", {}).get("tier_mode") == "four_fresh":
        audit.update(
            review_workers=workers,
            command=[sys.executable, *sys.argv],
        )
    runner.reconcile_immutable_json(run_dir / "semantic-audit.json", audit)


def tier_instruction(tier: str) -> str:
    from short_reasoning.history_tiers import TIER_INSTRUCTIONS

    return TIER_INSTRUCTIONS[tier]


def _canary_entries(
    run_dir: Path, run: dict[str, Any], task_id: str
) -> list[dict[str, Any]]:
    if run.get("outcomes_launched"):
        raise RuntimeError("rewrite canary review must precede target outcomes")
    generated_tiers = tuple(run["snapshot"]["generated_tiers"])
    variants = []
    for path in sorted((run_dir / "variants").glob("*/*.json")):
        variant = strict_json_loads(path.read_bytes())
        if variant.get("task_id") == task_id:
            runner.verify_tier_variant(variant)
            if variant.get("require_exact_action") is not True:
                raise RuntimeError("rewrite canary variant lacks exact-action gating")
            variants.append((path, variant))
    if (
        len(variants) != len(generated_tiers)
        or {variant["tier"] for _path, variant in variants} != set(generated_tiers)
        or task_id not in run.get("achieved_source_doses", {})
    ):
        raise RuntimeError("rewrite canary task lacks four complete ordered variants")
    entries = []
    for variant_path, variant in variants:
        for generated in variant["generated_turns"]:
            packet_id = runner._audit_packet_id(
                run["snapshot"]["audit_packet_salt"], variant, generated
            )
            packet_path = run_dir / "audit-packets" / f"{packet_id}.json"
            packet = strict_json_loads(packet_path.read_bytes())
            if packet != runner._audit_packet(packet_id, generated):
                raise RuntimeError("rewrite canary packet does not reconstruct")
            entries.append(
                {
                    "packet_id": packet_id,
                    "packet_file": str(packet_path.relative_to(run_dir)),
                    "packet_sha256": runner.file_sha256(packet_path),
                    "variant_file": str(variant_path.relative_to(run_dir)),
                    "variant_sha256": variant["variant_sha256"],
                    "tier": variant["tier"],
                    "turn": generated["turn"],
                }
            )
    expected = len(generated_tiers) * 3
    if len(entries) != expected or len({entry["packet_id"] for entry in entries}) != expected:
        raise RuntimeError("rewrite canary packet set is not exact")
    return sorted(entries, key=lambda entry: entry["packet_id"])


def run_semantic_canary(
    run_dir: Path, task_id: str, requested_workers: int | None = None
) -> None:
    run = strict_json_loads((run_dir / "run.json").read_bytes())
    entries = _canary_entries(run_dir, run, task_id)
    workers = _frozen_workers(run, requested_workers)
    review_dir = run_dir / "review-journals" / f"semantic-canary-{task_id}"
    jobs = []
    for index, entry in enumerate(entries):
        packet = strict_json_loads((run_dir / entry["packet_file"]).read_bytes())
        request = payload(
            SEMANTIC_SYSTEM,
            {
                "raw_reasoning": packet["raw_reasoning"],
                "accepted_compact_anchor": packet["accepted_compact_anchor"],
                "requested_action": packet["requested_action"],
                "candidate_state": packet["candidate_state"],
            },
            runner.SEMANTIC_CHECKS,
            953000 + index,
        )
        journal_path = review_dir / f"{entry['packet_id']}.json"
        prepare_review(journal_path, request)
        jobs.append((index, entry, request, journal_path))

    def execute(job: tuple[int, dict[str, Any], dict[str, Any], Path]) -> dict[str, Any]:
        index, entry, request, journal_path = job
        print(
            f"semantic canary {index + 1}/{len(entries)} {entry['packet_id']}",
            flush=True,
        )
        review = one_review(journal_path, request, runner.SEMANTIC_CHECKS)
        checks = {key: review[key] for key in runner.SEMANTIC_CHECKS}
        return {
            "packet_id": entry["packet_id"],
            "passed": all(checks.values()),
            "checks": checks,
            "notes": review["notes"],
        }

    decisions = runner.run_bounded_jobs(jobs, workers, execute)
    output_path = run_dir / f"semantic-canary-audit-{task_id}.json"
    existing = (
        strict_json_loads(output_path.read_bytes()) if output_path.exists() else None
    )
    audit = {
        "timing": "after_one_task_generation_before_remaining_development_generation",
        "task_id": task_id,
        "reviewer_condition_labels_withheld": True,
        "continuation_outcomes_inspected": False,
        "pending_tool_results_inspected": False,
        "semantic_locked_at": (
            existing["semantic_locked_at"] if existing else runner.now()
        ),
        "reviewer": copy.deepcopy(REVIEWER),
        "review_workers": workers,
        "command": existing["command"] if existing else [sys.executable, *sys.argv],
        "entries": entries,
        "review_journal_manifest_sha256": write_review_manifest(review_dir),
        "passed": all(decision["passed"] for decision in decisions),
        "decisions": decisions,
    }
    runner.reconcile_immutable_json(output_path, audit)


def run_style_canary(
    run_dir: Path, task_id: str, requested_workers: int | None = None
) -> None:
    run = strict_json_loads((run_dir / "run.json").read_bytes())
    entries = _canary_entries(run_dir, run, task_id)
    workers = _frozen_workers(run, requested_workers)
    semantic_path = run_dir / f"semantic-canary-audit-{task_id}.json"
    semantic = strict_json_loads(semantic_path.read_bytes())
    if (
        semantic.get("passed") is not True
        or semantic.get("entries") != entries
        or semantic.get("continuation_outcomes_inspected") is not False
    ):
        raise RuntimeError("style canary requires a passing frozen semantic canary")
    semantic_by_id = {item["packet_id"]: item for item in semantic["decisions"]}
    runner._verify_frozen_file_manifest(
        run_dir / "review-journals" / f"semantic-canary-{task_id}",
        semantic["review_journal_manifest_sha256"],
    )
    review_dir = run_dir / "review-journals" / f"style-canary-{task_id}"
    jobs = []
    for index, entry in enumerate(entries):
        packet = strict_json_loads((run_dir / entry["packet_file"]).read_bytes())
        variant = strict_json_loads((run_dir / entry["variant_file"]).read_bytes())
        generated = next(
            item for item in variant["generated_turns"] if item["turn"] == entry["turn"]
        )
        request = payload(
            STYLE_SYSTEM,
            {
                "tier": entry["tier"],
                "tier_instruction": tier_instruction(entry["tier"]),
                "raw_reasoning": packet["raw_reasoning"],
                "accepted_compact_anchor": packet["accepted_compact_anchor"],
                "candidate_state": packet["candidate_state"],
                "word_interval_inclusive": generated["word_interval"],
                "candidate_words": generated["automatic_audit"]["variant_words"],
            },
            {"tier_compliance", "no_semantic_padding"},
            954000 + index,
        )
        journal_path = review_dir / f"{entry['packet_id']}.json"
        prepare_review(journal_path, request)
        jobs.append((index, entry, generated, request, journal_path))

    def execute(
        job: tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], Path]
    ) -> dict[str, Any]:
        index, entry, generated, request, journal_path = job
        print(
            f"style canary {index + 1}/{len(entries)} {entry['packet_id']}",
            flush=True,
        )
        review = one_review(
            journal_path, request, {"tier_compliance", "no_semantic_padding"}
        )
        checks = {
            "tier_compliance": review["tier_compliance"],
            "dose_compliance": bool(generated["automatic_audit"]["passed"]),
            "no_semantic_padding": review["no_semantic_padding"],
        }
        passed = all(checks.values())
        return {
            "packet_id": entry["packet_id"],
            "tier": entry["tier"],
            "passed": passed,
            "accepted": semantic_by_id[entry["packet_id"]]["passed"] and passed,
            "checks": checks,
            "notes": review["notes"],
        }

    decisions = runner.run_bounded_jobs(jobs, workers, execute)
    output_path = run_dir / f"style-canary-audit-{task_id}.json"
    existing = (
        strict_json_loads(output_path.read_bytes()) if output_path.exists() else None
    )
    audit = {
        "timing": "after_semantic_canary_lock_before_remaining_development_generation",
        "task_id": task_id,
        "semantic_canary_sha256": runner.file_sha256(semantic_path),
        "continuation_outcomes_inspected": False,
        "pending_tool_results_inspected": False,
        "style_reviewed_at": existing["style_reviewed_at"] if existing else runner.now(),
        "reviewer": copy.deepcopy(REVIEWER),
        "review_workers": workers,
        "command": existing["command"] if existing else [sys.executable, *sys.argv],
        "entries": entries,
        "review_journal_manifest_sha256": write_review_manifest(review_dir),
        "passed": all(decision["accepted"] for decision in decisions),
        "decisions": decisions,
    }
    runner.reconcile_immutable_json(output_path, audit)


def run_style(run_dir: Path, requested_workers: int | None = None) -> None:
    run = strict_json_loads((run_dir / "run.json").read_bytes())
    workers = _frozen_workers(run, requested_workers)
    if run.get("status") != "semantic_audit_frozen_pending_style_audit":
        if (run_dir / "variant-audit.json").exists():
            existing = strict_json_loads((run_dir / "variant-audit.json").read_bytes())
            if (
                existing.get("review_workers") != workers
                and run.get("snapshot", {}).get("tier_mode") == "four_fresh"
            ):
                raise RuntimeError("existing style review worker provenance changed")
            return
        raise RuntimeError("style review requires the frozen semantic lock and condition map")
    mapping = strict_json_loads((run_dir / "audit-packet-map.json").read_bytes())
    semantic = strict_json_loads((run_dir / "semantic-audit.json").read_bytes())
    if semantic.get("review_journal_manifest_sha256") is not None:
        runner._verify_frozen_file_manifest(
            run_dir / "review-journals" / "semantic",
            semantic["review_journal_manifest_sha256"],
        )
    semantic_by_id = {item["packet_id"]: item for item in semantic["decisions"]}
    review_dir = run_dir / "review-journals" / "style"
    entries = sorted(mapping["entries"], key=lambda item: item["packet_id"])
    jobs = []
    for index, entry in enumerate(entries):
        packet = strict_json_loads((run_dir / entry["packet_file"]).read_bytes())
        variant = strict_json_loads((run_dir / entry["variant_file"]).read_bytes())
        generated = next(
            item for item in variant["generated_turns"] if item["turn"] == entry["turn"]
        )
        request = payload(
            STYLE_SYSTEM,
            {
                "tier": entry["tier"],
                "tier_instruction": tier_instruction(entry["tier"]),
                "raw_reasoning": packet["raw_reasoning"],
                "accepted_compact_anchor": packet["accepted_compact_anchor"],
                "candidate_state": packet["candidate_state"],
                "word_interval_inclusive": generated["word_interval"],
                "candidate_words": generated["automatic_audit"]["variant_words"],
            },
            {"tier_compliance", "no_semantic_padding"},
            952000 + index,
        )
        journal_path = review_dir / f"{entry['packet_id']}.json"
        prepare_review(journal_path, request)
        jobs.append((index, entry, generated, request, journal_path))

    def execute(
        job: tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], Path]
    ) -> dict[str, Any]:
        index, entry, generated, request, journal_path = job
        print(
            f"style review {index + 1}/{len(entries)} {entry['packet_id']}",
            flush=True,
        )
        review = one_review(
            journal_path, request, {"tier_compliance", "no_semantic_padding"}
        )
        checks = {
            "tier_compliance": review["tier_compliance"],
            "dose_compliance": bool(generated["automatic_audit"]["passed"]),
            "no_semantic_padding": review["no_semantic_padding"],
        }
        style_pass = all(checks.values())
        accepted = semantic_by_id[entry["packet_id"]]["passed"] and style_pass
        return {
            "packet_id": entry["packet_id"],
            "tier": entry["tier"],
            "passed": style_pass,
            "accepted": accepted,
            "checks": checks,
            "notes": review["notes"],
        }

    decisions = runner.run_bounded_jobs(jobs, workers, execute)
    audit = {
        "timing": "after_semantic_lock_before_tier_outcomes",
        "semantic_audit_sha256": run["semantic_audit_sha256"],
        "audit_packet_map_sha256": run["audit_packet_map_sha256"],
        "continuation_outcomes_inspected": False,
        "pending_tool_results_inspected": False,
        "source_history_files_inspected": False,
        "style_reviewed_at": runner.now(),
        "reviewer": copy.deepcopy(REVIEWER),
        "review_journal_manifest_sha256": write_review_manifest(review_dir),
        "decisions": decisions,
    }
    if run.get("snapshot", {}).get("tier_mode") == "four_fresh":
        audit.update(
            review_workers=workers,
            command=[sys.executable, *sys.argv],
        )
    runner.reconcile_immutable_json(run_dir / "variant-audit.json", audit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("semantic-canary", "style-canary", "semantic", "style")
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--no-budget-guard", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.no_budget_guard:
        os.environ.pop("OPENROUTER_MIN_BALANCE_USD", None)
        os.environ.pop("OPENROUTER_MAX_CALL_COST_USD", None)
    run_dir = args.run_dir.resolve()
    if args.stage in {"semantic-canary", "style-canary"} and not args.task:
        parser.error("--task is required for canary review")
    with runner.exclusive_run_lock(run_dir):
        if args.stage == "semantic-canary":
            run_semantic_canary(run_dir, args.task, args.workers)
        elif args.stage == "style-canary":
            run_style_canary(run_dir, args.task, args.workers)
        elif args.stage == "semantic":
            run_semantic(run_dir, args.workers)
        else:
            run_style(run_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
