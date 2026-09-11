#!/usr/bin/env python3
"""Generate immutable exploratory history-control variants before control outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from short_reasoning import load_dotenv  # noqa: E402
from short_reasoning.frozen import continue_frozen_source, frozen_source_hash  # noqa: E402
from short_reasoning.history_controls import (  # noqa: E402
    build_control_variant,
    generate_control_turn,
    verify_control_variant,
)

CONTROLS = ("verbose_paraphrase", "compact_full_sentence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("init", "generate", "finalize", "continue"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--parent-run",
        type=Path,
        default=ROOT / "results" / "20260724-v2-12-task-two-route-screen",
    )
    parser.add_argument("--min-balance", type=float, default=2.0)
    parser.add_argument("--call-reserve", type=float, default=0.10)
    parser.add_argument("--replicates", type=int, default=3)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_parent_source(source: dict[str, Any]) -> None:
    expected = frozen_source_hash(
        source["task_id"],
        source["model_name"],
        source["model"],
        source["histories"]["clean"],
        source["histories"]["rewritten"],
    )
    if expected != source["source_sha256"]:
        raise RuntimeError(f"parent source hash mismatch: {source['task_id']} / {source['model_name']}")
    clean_reasoning = [
        message["reasoning"]
        for message in source["histories"]["clean"]
        if message.get("role") == "assistant" and message.get("reasoning")
    ]
    rewritten_reasoning = [
        message["reasoning"]
        for message in source["histories"]["rewritten"]
        if message.get("role") == "assistant" and message.get("reasoning")
    ]
    if len(source["shared"]) != len(clean_reasoning) or len(source["shared"]) != len(rewritten_reasoning):
        raise RuntimeError("parent shared records do not match history turns")
    clean_assistants = [
        message
        for message in source["histories"]["clean"]
        if message.get("role") == "assistant" and message.get("reasoning")
    ]
    for index, shared in enumerate(source["shared"]):
        rewrite = shared["rewrite"]
        if rewrite["raw_reasoning"] != clean_reasoning[index] or rewrite["rewritten"] != rewritten_reasoning[index]:
            raise RuntimeError("parent shared rewrite text does not match frozen histories")
        calls = clean_assistants[index].get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"] != rewrite["requested_tool"]:
            raise RuntimeError("parent requested tool does not match frozen history")


def source_manifest(parent_run: Path) -> dict[str, str]:
    manifest = {}
    for line in (parent_run / "sources" / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        manifest[name] = digest
    return manifest


def snapshot(parent_run: Path, rewriter: dict[str, Any]) -> dict[str, Any]:
    sources = []
    manifest = source_manifest(parent_run)
    for path in sorted((parent_run / "sources").glob("*.json")):
        source = json.loads(path.read_text())
        verify_parent_source(source)
        if manifest.get(path.name) != file_sha256(path):
            raise RuntimeError(f"parent source file does not match SHA256SUMS: {path.name}")
        sources.append(
            {
                "file": path.name,
                "file_sha256": file_sha256(path),
                "source_sha256": source["source_sha256"],
                "task_id": source["task_id"],
                "model_name": source["model_name"],
            }
        )
    if len(sources) != 24 or set(manifest) != {source["file"] for source in sources}:
        raise RuntimeError("parent source set does not exactly match its manifest")
    return {
        "parent_run": str(parent_run.resolve()),
        "parent_source_manifest_sha256": file_sha256(parent_run / "sources" / "SHA256SUMS"),
        "controls": list(CONTROLS),
        "rewriter": rewriter,
        "sources": sources,
        "outcome_blinding": (
            "Variant requests are constructed only from frozen historical reasoning and requested-action "
            "metadata. Primary outcomes exist but are never loaded by this script."
        ),
    }


def initialize(run_dir: Path, parent_run: Path, rewriter: dict[str, Any]) -> dict[str, Any]:
    current = snapshot(parent_run, rewriter)
    run_file = run_dir / "run.json"
    if run_file.exists():
        recorded = json.loads(run_file.read_text())
        if recorded["snapshot"] != current:
            raise RuntimeError("control run snapshot differs from current parent/configuration")
        return recorded
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "variants_not_generated",
        "control_outcomes_launched": False,
        "snapshot": current,
    }
    atomic_json(run_file, record)
    return record


def variant_path(run_dir: Path, control: str, source_file: Path) -> Path:
    return run_dir / "variants" / control / source_file.name


def attempt_path(run_dir: Path, control: str, source_file: Path) -> Path:
    return run_dir / "attempts" / control / source_file.name


def run_generate(
    run_dir: Path,
    parent_run: Path,
    rewriter: dict[str, Any],
) -> None:
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("control_outcomes_launched") or run.get("variants_frozen"):
        raise RuntimeError("cannot generate or repair variants after freeze/outcomes")
    for source_file in sorted((parent_run / "sources").glob("*.json")):
        source = json.loads(source_file.read_text())
        verify_parent_source(source)
        for control in CONTROLS:
            output = variant_path(run_dir, control, source_file)
            journal_path = attempt_path(run_dir, control, source_file)
            if output.exists():
                existing = json.loads(output.read_text())
                verify_control_variant(existing)
                if (
                    existing.get("parent_source_sha256") != source["source_sha256"]
                    or existing.get("parent_file_sha256") != file_sha256(source_file)
                    or existing.get("control") != control
                ):
                    raise RuntimeError(f"existing control variant has wrong provenance: {output}")
                continue
            if journal_path.exists():
                journal = json.loads(journal_path.read_text())
                raise RuntimeError(
                    f"refusing to repeat or reconstruct interrupted/failed paid attempts; inspect {journal_path} "
                    f"(status={journal.get('status')})"
                )
            print(f"generate {control} {source['model_name']} {source['task_id']}", flush=True)
            journal = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "in_progress",
                "control": control,
                "task_id": source["task_id"],
                "model_name": source["model_name"],
                "parent_source_sha256": source["source_sha256"],
                "parent_file": source_file.name,
                "parent_file_sha256": file_sha256(source_file),
                "turns": [
                    {"turn": index + 1, "status": "pending", "attempts": []}
                    for index in range(len(source["shared"]))
                ],
            }
            atomic_json(journal_path, journal)
            generated_turns = []
            failed = False
            for turn_index, shared in enumerate(source["shared"]):
                rewrite = shared["rewrite"]

                def persist_attempts(attempts: list[dict[str, Any]]) -> None:
                    journal["turns"][turn_index]["status"] = "attempting"
                    journal["turns"][turn_index]["attempts"] = attempts
                    atomic_json(journal_path, journal)

                try:
                    generated = generate_control_turn(
                        rewriter,
                        control,
                        rewrite["raw_reasoning"],
                        rewrite["rewritten"],
                        rewrite["requested_tool"],
                        on_attempt=persist_attempts,
                    )
                except Exception as exc:
                    journal["status"] = "automatic_gate_failure"
                    journal["failed_turn"] = turn_index + 1
                    journal["error"] = f"{type(exc).__name__}: {exc}"
                    journal["failed_at"] = datetime.now(timezone.utc).isoformat()
                    atomic_json(journal_path, journal)
                    failed = True
                    break
                generated_turns.append(generated)
                journal["turns"][turn_index]["status"] = "complete"
                journal["turns"][turn_index]["result"] = generated
                atomic_json(journal_path, journal)
            if failed:
                continue
            variant = build_control_variant(source, control, generated_turns)
            variant["parent_file"] = source_file.name
            variant["parent_file_sha256"] = file_sha256(source_file)
            atomic_json(output, variant)
            journal["status"] = "complete"
            journal["variant_sha256"] = variant["variant_sha256"]
            journal["completed_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(journal_path, journal)
    variants = list((run_dir / "variants").glob("*/*.json"))
    failures = [
        path
        for path in (run_dir / "attempts").glob("*/*.json")
        if json.loads(path.read_text()).get("status") != "complete"
    ]
    run["status"] = "variants_pending_semantic_audit"
    run["variant_files"] = len(variants)
    run["failed_or_interrupted_variants"] = len(failures)
    run["variants_completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(run_dir / "run.json", run)


def finalize_variants(run_dir: Path) -> None:
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("variants_frozen"):
        raise RuntimeError("variants are already frozen")
    audit_path = run_dir / "variant-audit.json"
    if not audit_path.exists():
        raise RuntimeError("independent variant-audit.json is required before freezing")
    audit = json.loads(audit_path.read_text())
    if audit.get("timing") != "before_control_outcomes":
        raise RuntimeError("variant semantic audit does not declare pre-outcome timing")
    accepted = audit.get("accepted_variant_sha256")
    if not isinstance(accepted, dict) or set(accepted) != set(CONTROLS):
        raise RuntimeError("variant audit must provide accepted hashes for both controls")
    expected_journal_paths = {
        f"{control}/{source['file']}"
        for control in CONTROLS
        for source in run["snapshot"]["sources"]
    }
    journal_paths = sorted((run_dir / "attempts").glob("*/*.json"))
    actual_journal_paths = {
        str(path.relative_to(run_dir / "attempts")) for path in journal_paths
    }
    journals = [json.loads(path.read_text()) for path in journal_paths]
    terminal_statuses = {"complete", "automatic_gate_failure"}
    if actual_journal_paths != expected_journal_paths or any(
        journal.get("status") not in terminal_statuses for journal in journals
    ):
        raise RuntimeError("the complete intended control matrix does not have terminal attempt journals")
    rows = []
    available = {control: set() for control in CONTROLS}
    for path in sorted((run_dir / "variants").glob("*/*.json")):
        variant = json.loads(path.read_text())
        verify_control_variant(variant)
        control = variant["control"]
        available[control].add(variant["variant_sha256"])
        rows.append((file_sha256(path), str(path.relative_to(run_dir / "variants"))))
    completed_journals = sum(journal.get("status") == "complete" for journal in journals)
    if len(rows) != completed_journals:
        raise RuntimeError("completed journals and immutable variant files disagree")
    for control in CONTROLS:
        if not isinstance(accepted[control], list) or not accepted[control]:
            raise RuntimeError(f"semantic audit accepts no {control} variants")
        unknown = set(accepted[control]) - available[control]
        if unknown:
            raise RuntimeError(f"semantic audit accepts unknown {control} hashes: {sorted(unknown)}")
    manifest_path = run_dir / "variants" / "SHA256SUMS"
    if manifest_path.exists():
        raise RuntimeError("refusing to overwrite variant manifest")
    manifest_path.write_text("".join(f"{digest}  {name}\n" for digest, name in rows))
    run["status"] = "variants_frozen_before_control_outcomes"
    run["variants_frozen"] = True
    run["variant_audit_sha256"] = file_sha256(audit_path)
    run["variant_manifest_sha256"] = file_sha256(manifest_path)
    run["accepted_variant_counts"] = {
        control: len(set(accepted[control])) for control in CONTROLS
    }
    run["variants_frozen_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(run_dir / "run.json", run)


def run_control_continuations(
    run_dir: Path,
    parent_run: Path,
    replicates: int,
) -> None:
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text())
    if not run.get("variants_frozen"):
        raise RuntimeError("control variants must be semantically audited and frozen before outcomes")
    manifest_path = run_dir / "variants" / "SHA256SUMS"
    audit_path = run_dir / "variant-audit.json"
    if (
        file_sha256(manifest_path) != run.get("variant_manifest_sha256")
        or file_sha256(audit_path) != run.get("variant_audit_sha256")
    ):
        raise RuntimeError("frozen variant manifest/audit provenance changed")
    for line in manifest_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if file_sha256(run_dir / "variants" / name) != digest:
            raise RuntimeError(f"frozen variant file changed after manifest: {name}")
    if not 1 <= replicates <= 3:
        raise ValueError("control continuation replicates must be between one and three")
    audit = json.loads(audit_path.read_text())
    accepted = {
        control: set(audit["accepted_variant_sha256"][control]) for control in CONTROLS
    }
    tasks = {
        task["id"]: task
        for task in json.loads((parent_run / "tasks.snapshot.json").read_text())
    }
    seed_map = {}
    for path in sorted((parent_run / "continuations").glob("*.json")):
        case = json.loads(path.read_text())
        seed_map[(case["model_name"], case["task_id"], case["replicate"])] = case[
            "continuation_seed"
        ]
    if not run.get("control_outcomes_launched"):
        run["control_outcomes_launched"] = True
        run["control_outcomes_started_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(run_path, run)
    condition_labels = {
        "verbose_paraphrase": {"clean": "clean", "rewritten": "verbose_paraphrase"},
        "compact_full_sentence": {
            "clean": "rewritten",
            "rewritten": "compact_full_sentence",
        },
    }
    for path in sorted((run_dir / "variants").glob("*/*.json")):
        variant = json.loads(path.read_text())
        verify_control_variant(variant)
        control = variant["control"]
        if variant["variant_sha256"] not in accepted[control]:
            continue
        task = tasks[variant["task_id"]]
        adapted_source = {
            "task_id": variant["task_id"],
            "title": task["title"],
            "model_name": variant["model_name"],
            "model": variant["model"],
            "source_seed": None,
            "source_sha256": variant["variant_sha256"],
            "status": "selected",
            "fork_after": task["fork_after"],
            "histories": {
                "clean": variant["histories"]["baseline"],
                "rewritten": variant["histories"]["variant"],
            },
        }
        for replicate in range(1, replicates + 1):
            output = (
                run_dir
                / "continuations"
                / control
                / f"{path.stem}__rep-{replicate:02d}.json"
            )
            if output.exists():
                existing = json.loads(output.read_text())
                if (
                    existing.get("control_variant_sha256") != variant["variant_sha256"]
                    or existing.get("control") != control
                    or existing.get("replicate") != replicate
                ):
                    raise RuntimeError(f"existing continuation has wrong provenance: {output}")
                continue
            key = (variant["model_name"], variant["task_id"], replicate)
            if key not in seed_map:
                raise RuntimeError(f"missing parent continuation seed for {key}")
            print(
                f"continue {control} {variant['model_name']} {variant['task_id']} rep={replicate}",
                flush=True,
            )
            case = continue_frozen_source(task, adapted_source, seed_map[key], replicate)
            case["control"] = control
            case["condition_labels"] = condition_labels[control]
            case["control_variant_sha256"] = variant["variant_sha256"]
            case["parent_source_sha256"] = variant["parent_source_sha256"]
            case["variant_audit_sha256"] = run["variant_audit_sha256"]
            atomic_json(output, case)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    parent_run = args.parent_run.resolve()
    rewriter = json.loads((parent_run / "models.snapshot.json").read_text())["rewriter"]
    initialize(run_dir, parent_run, rewriter)
    if args.stage == "init":
        print(run_dir)
        return
    if args.stage == "finalize":
        finalize_variants(run_dir)
        print(run_dir)
        return
    os.environ["OPENROUTER_MIN_BALANCE_USD"] = str(args.min_balance)
    os.environ["OPENROUTER_MAX_CALL_COST_USD"] = str(args.call_reserve)
    load_dotenv(ROOT / ".env")
    if args.stage == "generate":
        run_generate(run_dir, parent_run, rewriter)
    else:
        run_control_continuations(run_dir, parent_run, args.replicates)
    print(run_dir)


if __name__ == "__main__":
    main()
