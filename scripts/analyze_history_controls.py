#!/usr/bin/env python3
"""Validate and summarize frozen exploratory history-control outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_frozen_run import model_summary, summarize_case  # noqa: E402
from short_reasoning import (  # noqa: E402
    aggregate_metrics,
    append_tool_cycle,
    build_tools,
    evaluate_tool_response,
    final_is_correct,
    percent_change,
    reasoning_text,
    response_metrics,
    target_payload,
)
from short_reasoning.frozen import frozen_source_hash  # noqa: E402
from short_reasoning.history_controls import verify_control_variant  # noqa: E402

CONTROLS = ("verbose_paraphrase", "compact_full_sentence")
CONDITION_LABELS = {
    "verbose_paraphrase": {"clean": "clean", "rewritten": "verbose_paraphrase"},
    "compact_full_sentence": {
        "clean": "rewritten",
        "rewritten": "compact_full_sentence",
    },
}
EXPECTED_PROVIDERS = {
    "deepseek-v4-flash": "DeepInfra",
    "laguna-s-2.1": "Poolside",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def group_summary(control: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["model_name"]].append(case)
    return {
        "branch_semantics": CONDITION_LABELS[control],
        "overall": model_summary(cases),
        "by_model": {
            name: model_summary(values) for name, values in sorted(grouped.items())
        },
    }


def verify_captured_case(
    raw: dict[str, Any],
    variant: dict[str, Any],
    task: dict[str, Any],
) -> None:
    tools = build_tools(task)
    branch_aggregates = {}
    branch_statuses = {}
    for condition, history_key in (("clean", "baseline"), ("rewritten", "variant")):
        branch = raw["branches"][condition]
        messages = copy.deepcopy(variant["histories"][history_key])
        actions_correct = True
        turn_index = 0
        status = None
        for phase_index in range(task["fork_after"], len(task["phases"])):
            if turn_index >= len(branch["turns"]):
                raise SystemExit("capture ended before a required action or final failure")
            phase = task["phases"][phase_index]
            turn = branch["turns"][turn_index]
            expected_request = target_payload(
                variant["model"], messages, tools, raw["continuation_seed"] + phase_index
            )
            if turn["request"] != expected_request:
                raise SystemExit("captured action request does not reconstruct exactly")
            if turn["metrics"] != response_metrics(turn["response"], turn["wall_seconds"]):
                raise SystemExit("captured response metrics do not recompute")
            message = turn["response"]["choices"][0]["message"]
            action = evaluate_tool_response(message, phase)
            if turn.get("action") != action or turn.get("phase") != phase["name"]:
                raise SystemExit("captured action score does not recompute")
            actions_correct = actions_correct and action["correct"]
            turn_index += 1
            if not action["correct"]:
                status = "action_failure"
                break
            append_tool_cycle(messages, message, reasoning_text(message), phase)
        if status is None:
            if turn_index >= len(branch["turns"]):
                raise SystemExit("capture omitted the required final turn")
            final_turn = len(task["phases"]) + 1
            turn = branch["turns"][turn_index]
            expected_request = target_payload(
                variant["model"], messages, tools, raw["continuation_seed"] + final_turn
            )
            if turn["request"] != expected_request:
                raise SystemExit("captured final request does not reconstruct exactly")
            if turn["metrics"] != response_metrics(turn["response"], turn["wall_seconds"]):
                raise SystemExit("captured final metrics do not recompute")
            message = turn["response"]["choices"][0]["message"]
            final_correct = final_is_correct(message.get("content"), task["final_answer"])
            unexpected_calls = len(message.get("tool_calls") or [])
            if (
                turn.get("phase") != "final"
                or turn.get("final_correct") != final_correct
                or turn.get("unexpected_tool_calls") != unexpected_calls
            ):
                raise SystemExit("captured final score does not recompute")
            turn_index += 1
            status = "complete"
        if turn_index != len(branch["turns"]):
            raise SystemExit("capture contains turns after its terminal outcome")
        aggregate = aggregate_metrics([turn["metrics"] for turn in branch["turns"]])
        final_success = status == "complete" and final_correct and unexpected_calls == 0
        expected_fields = {
            "actions_correct": actions_correct,
            "status": status,
            "aggregate": aggregate,
            "final_correct": final_success,
            "success": actions_correct and final_success,
        }
        if any(branch.get(key) != value for key, value in expected_fields.items()):
            raise SystemExit("captured branch aggregate or quality outcome does not recompute")
        branch_aggregates[condition] = aggregate
        branch_statuses[condition] = status
    clean = branch_aggregates["clean"]
    rewritten = branch_aggregates["rewritten"]
    comparable = (
        branch_statuses["clean"] == branch_statuses["rewritten"] == "complete"
        and clean["turn_count"] == rewritten["turn_count"]
    )
    expected_pair = {"comparable_horizon": comparable}
    for output, metric in (
        ("reasoning_token_change_percent", "reasoning_tokens"),
        ("completion_token_change_percent", "completion_tokens"),
        ("reasoning_word_change_percent", "reasoning_words"),
        ("prompt_token_change_percent", "prompt_tokens"),
    ):
        expected_pair[output] = percent_change(clean.get(metric), rewritten.get(metric)) if comparable else None
    if raw.get("pair") != expected_pair:
        raise SystemExit("captured paired effects do not recompute")


def change(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def summary_line(label: str, summary: dict[str, Any]) -> str:
    ci = summary["clustered_bootstrap"]["median_effect_percent_ci95"]
    return (
        f"- **{label}:** {summary['reasoning_reduced_cases']}/"
        f"{summary['comparable_horizon_cases']} shorter; median paired change "
        f"{change(summary['median_reasoning_token_change_percent'])}; clustered 95% CI "
        f"[{change(ci[0])}, {change(ci[1])}]; comparable pooled "
        f"{summary['comparable_pooled_clean_reasoning_tokens']}→"
        f"{summary['comparable_pooled_rewritten_reasoning_tokens']} "
        f"({change(summary['comparable_pooled_reasoning_token_change_percent'])}); "
        f"strict success {summary['clean_successes']}/{summary['cases']}→"
        f"{summary['rewritten_successes']}/{summary['cases']}."
    )


def render_report(analysis: dict[str, Any]) -> str:
    labels = {
        "verbose_paraphrase": "Clean → length-matched verbose paraphrase",
        "compact_full_sentence": "Telegraphic compact → full-sentence compact",
    }
    lines = [
        "# Frozen history-control outcomes",
        "",
        "Each control uses fresh contemporaneous baseline and treatment branches from a pre-outcome frozen history variant. Controls are exploratory and analyzed separately.",
    ]
    for control in CONTROLS:
        result = analysis["controls"][control]
        lines.extend(["", f"## {labels[control]}", "", summary_line("Overall", result["overall"])])
        for model, summary in result["by_model"].items():
            lines.append(summary_line(model, summary))
    lines.extend([
        "",
        "## Common-source sensitivity",
        "",
        f"Both controls have complete outcomes for {analysis['common_parent_sources']} common parent sources ({analysis['common_source_cases_per_control']} cases per control).",
        summary_line(
            "Clean → verbose paraphrase",
            analysis["common_source_sensitivity"]["verbose_paraphrase"]["overall"],
        ),
        summary_line(
            "Telegraphic → full-sentence compact",
            analysis["common_source_sensitivity"]["compact_full_sentence"]["overall"],
        ),
        "",
        "## Validity and limits",
        "",
        f"- First-request treatment invariants: {analysis['validity']['first_request_invariants_passed']}/{analysis['validity']['cases']}.",
        f"- Exact accepted-variant/replicate matrix: {analysis['validity']['cases']}/{analysis['validity']['expected_cases']}.",
        f"- Provider-reported continuation cost in completed captures: ${analysis['continuation_cost_usd']:.4f}.",
        "- Laguna does not support deterministic endpoint seeds; its three replicates represent endpoint variability.",
        "- Quality counts and clustered intervals must be reported; equal or similar counts do not establish non-inferiority.",
        "- The controls do not isolate a single linguistic property: alternate wording can change salience, and full-sentence conversion changes syntax and lexical realization.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run = json.loads((run_dir / "run.json").read_text())
    audit_path = run_dir / "variant-audit.json"
    audit = json.loads(audit_path.read_text())
    if not run.get("variants_frozen") or not run.get("control_outcomes_launched"):
        raise SystemExit("variants were not frozen before control outcomes")
    if hashlib.sha256(audit_path.read_bytes()).hexdigest() != run["variant_audit_sha256"]:
        raise SystemExit("variant audit changed after outcome launch")

    manifest_path = run_dir / "variants" / "SHA256SUMS"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != run["variant_manifest_sha256"]:
        raise SystemExit("variant manifest changed after outcome launch")
    for line in manifest_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((run_dir / "variants" / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"variant file changed after manifest: {name}")
    parent_run = Path(run["snapshot"]["parent_run"])
    parent_manifest_path = parent_run / "sources" / "SHA256SUMS"
    if hashlib.sha256(parent_manifest_path.read_bytes()).hexdigest() != run["snapshot"]["parent_source_manifest_sha256"]:
        raise SystemExit("parent source manifest changed after control initialization")
    tasks = {
        task["id"]: task
        for task in json.loads((parent_run / "tasks.snapshot.json").read_text())
    }
    variant_index = {}
    for path in sorted((run_dir / "variants").glob("*/*.json")):
        variant = json.loads(path.read_text())
        verify_control_variant(variant)
        parent_path = parent_run / "sources" / variant["parent_file"]
        if hashlib.sha256(parent_path.read_bytes()).hexdigest() != variant["parent_file_sha256"]:
            raise SystemExit(f"parent source file changed: {parent_path}")
        parent = json.loads(parent_path.read_text())
        recomputed_parent_hash = frozen_source_hash(
            parent["task_id"],
            parent["model_name"],
            parent["model"],
            parent["histories"]["clean"],
            parent["histories"]["rewritten"],
        )
        if (
            parent["source_sha256"] != variant["parent_source_sha256"]
            or recomputed_parent_hash != parent["source_sha256"]
            or variant["task_id"] != parent["task_id"]
            or variant["model_name"] != parent["model_name"]
            or variant["model"] != parent["model"]
            or variant["histories"]["baseline"]
            != parent["histories"][variant["baseline_condition"]]
        ):
            raise SystemExit(f"variant/parent source mismatch: {path}")
        if variant["variant_sha256"] in variant_index:
            raise SystemExit(f"duplicate variant hash: {variant['variant_sha256']}")
        variant_index[variant["variant_sha256"]] = variant
    accepted = {
        control: set(audit["accepted_variant_sha256"][control]) for control in CONTROLS
    }
    if any(
        digest not in variant_index or variant_index[digest]["control"] != control
        for control, digests in accepted.items()
        for digest in digests
    ):
        raise SystemExit("semantic audit accepts an absent or wrong-control variant")
    for control, digests in accepted.items():
        parents = [variant_index[digest]["parent_source_sha256"] for digest in digests]
        if len(parents) != len(set(parents)):
            raise SystemExit(f"multiple accepted {control} variants share one parent")
    expected = {
        (control, digest, replicate)
        for control in CONTROLS
        for digest in accepted[control]
        for replicate in range(1, args.replicates + 1)
    }
    parent_seed_map = {}
    for path in sorted((parent_run / "continuations").glob("*.json")):
        parent_case = json.loads(path.read_text())
        parent_seed_map[(parent_case["model_name"], parent_case["task_id"], parent_case["replicate"])] = parent_case["continuation_seed"]
    observed = set()
    cases_by_control: dict[str, list[dict[str, Any]]] = {control: [] for control in CONTROLS}
    paths = sorted((run_dir / "continuations").glob("*/*.json"))
    for path in paths:
        raw = json.loads(path.read_text())
        identity = (raw["control"], raw["control_variant_sha256"], raw["replicate"])
        if identity not in expected or identity in observed:
            raise SystemExit(f"unexpected or duplicate continuation identity: {identity}")
        variant = variant_index[raw["control_variant_sha256"]]
        if raw.get("status") != "complete":
            raise SystemExit(f"continuation capture is not terminal: {path}")
        if (
            raw["source_sha256"] != variant["variant_sha256"]
            or raw["task_id"] != variant["task_id"]
            or raw["model_name"] != variant["model_name"]
            or raw["model"] != variant["model"]
            or raw["parent_source_sha256"] != variant["parent_source_sha256"]
            or raw["condition_labels"] != CONDITION_LABELS[raw["control"]]
            or raw["continuation_seed"]
            != parent_seed_map[(raw["model_name"], raw["task_id"], raw["replicate"])]
        ):
            raise SystemExit(f"continuation/variant provenance mismatch: {path}")
        if raw["variant_audit_sha256"] != run["variant_audit_sha256"]:
            raise SystemExit(f"continuation has wrong audit provenance: {path}")
        for condition, expected_history in (
            ("clean", variant["histories"]["baseline"]),
            ("rewritten", variant["histories"]["variant"]),
        ):
            turns = raw["branches"][condition]["turns"]
            if not turns or turns[0]["request"]["messages"] != expected_history:
                raise SystemExit(f"first request does not replay the frozen variant: {path}")
            for turn in turns:
                request = turn["request"]
                if (
                    request.get("model") != variant["model"]["model"]
                    or request.get("provider") != variant["model"]["provider"]
                    or turn["response"].get("provider") != EXPECTED_PROVIDERS[variant["model_name"]]
                ):
                    raise SystemExit(f"target route mismatch: {path}")
        observed.add(identity)
        verify_captured_case(raw, variant, tasks[raw["task_id"]])
        case = summarize_case(raw)
        if not case["first_request_audit"]["passed_with_request_fields"]:
            raise SystemExit(f"first-request treatment invariant failed: {path}")
        case.update(
            control=raw["control"],
            control_variant_sha256=raw["control_variant_sha256"],
            parent_source_sha256=raw["parent_source_sha256"],
            condition_labels=raw["condition_labels"],
        )
        cases_by_control[raw["control"]].append(case)
    missing = sorted(expected - observed)
    if missing:
        raise SystemExit(f"incomplete control matrix: {len(missing)} cases missing")

    parent_sets = {
        control: {case["parent_source_sha256"] for case in cases}
        for control, cases in cases_by_control.items()
    }
    common = set.intersection(*parent_sets.values())
    common_cases = {
        control: [case for case in cases if case["parent_source_sha256"] in common]
        for control, cases in cases_by_control.items()
    }
    if any(len(cases) != len(common) * args.replicates for cases in common_cases.values()):
        raise SystemExit("common-source controls do not have equal complete replicate cardinality")
    controls = {
        control: group_summary(control, cases) for control, cases in cases_by_control.items()
    }
    common_results = {
        control: group_summary(control, cases) for control, cases in common_cases.items()
    }
    all_cases = [case for cases in cases_by_control.values() for case in cases]
    analysis = {
        "design": {
            "controls": list(CONTROLS),
            "replicates": args.replicates,
            "accepted_variants": {control: len(values) for control, values in accepted.items()},
            "fresh_contemporaneous_baselines": True,
        },
        "controls": controls,
        "common_parent_sources": len(common),
        "common_source_cases_per_control": len(next(iter(common_cases.values()))),
        "common_source_sensitivity": common_results,
        "validity": {
            "cases": len(all_cases),
            "expected_cases": len(expected),
            "first_request_invariants_passed": sum(
                case["first_request_audit"]["passed_with_request_fields"] for case in all_cases
            ),
        },
        "continuation_cost_usd": sum(case["continuation_cost_usd"] for case in all_cases),
        "cases": all_cases,
    }
    for path, content in (
        (run_dir / "analysis.json", json.dumps(analysis, indent=2) + "\n"),
        (run_dir / "report.md", render_report(analysis)),
    ):
        if path.exists() and not args.replace:
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_text(content)
    checksum_path = run_dir / "continuations" / "SHA256SUMS"
    if checksum_path.exists() and not args.replace:
        raise SystemExit(f"refusing to overwrite {checksum_path}")
    checksum_path.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(run_dir / 'continuations')}\n"
            for path in paths
        )
    )
    print(json.dumps({control: result["overall"] for control, result in controls.items()}, indent=2))


if __name__ == "__main__":
    main()
