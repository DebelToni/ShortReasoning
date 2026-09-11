#!/usr/bin/env python3
"""Offline camera-ready robustness audit for the six-route controlled experiment.

The script reads immutable per-continuation captures and reports both the
original-first-capture and four-branch retrospective retry-augmented analyses.
It uses only the Python standard library and never contacts a model or network
service.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROUTES = (
    "deepseek-v4-flash",
    "laguna-s-2.1",
    "minimax-m3",
    "glm-4.7-flash",
    "glm-5.1",
    "deepseek-v4-pro",
)
RUN_SELECTIONS = (
    ("results/20260724-v2-12-task-two-route-screen", {"deepseek-v4-flash", "laguna-s-2.1"}),
    ("results/20260730-four-model-paired-replication-v1", {"minimax-m3"}),
    ("results/20260730-glm-four-model-replication-v2", {"glm-4.7-flash"}),
    ("results/20260730-latest-model-paired-replication-v1", {"glm-5.1", "deepseek-v4-pro"}),
)
AUDIT_SELECTIONS = tuple((run, models) for run, models in RUN_SELECTIONS)
PROVENANCE_PATHS = (
    "scripts/analyze_camera_ready_controlled.py",
    "scripts/run_four_model_replication.py",
    "results/20260724-v2-12-task-two-route-screen/models.snapshot.json",
    "results/20260724-v2-12-task-two-route-screen/source-exclusions.json",
    "results/20260730-four-model-paired-replication-v1/config.snapshot.json",
    "results/20260730-glm-four-model-replication-v2/config.snapshot.json",
    "results/20260730-latest-model-paired-replication-v1/config.snapshot.json",
    "results/20260803-table1-infrastructure-reruns-v1/protocol.json",
    "results/20260803-table1-infrastructure-reruns-v1/summary.json",
    "results/20260803-table1-infrastructure-reruns-v1/SHA256SUMS",
)
FIDELITY_RUBRIC = (
    "Audit one compact reasoning-state rewrite before its pending tool result exists. "
    "Pass only if it preserves every decision-relevant fact, number, uncertainty, "
    "conclusion, rejected alternative, evidence-status distinction, and intended next "
    "action from the raw reasoning; adds no unsupported fact; and neither states nor "
    "implies that the requested tool result has occurred. The tool call is separately "
    "preserved and is supplied only to check action consistency. Return only one JSON "
    "object with exactly these keys: verdict ('pass' or 'fail'), "
    "critical_or_material_issue (boolean), future_result_leakage (boolean), "
    "requested_action_preserved (boolean), explanation (short string)."
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(clean: int | float, compact: int | float) -> float | None:
    return None if clean == 0 else 100.0 * (compact - clean) / clean


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "q05": quantile(values, 0.05),
        "q25": quantile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "q75": quantile(values, 0.75),
        "q90": quantile(values, 0.90),
        "q95": quantile(values, 0.95),
        "q99": quantile(values, 0.99),
        "max": max(values),
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "positive": sum(value > 0 for value in values),
    }


def last_tool_observation(request: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(request.get("messages", [])):
        if message.get("role") == "tool":
            return {"name": message.get("name"), "content": message.get("content")}
    return None


def reasoning_only_request_match(clean: dict[str, Any], compact: dict[str, Any]) -> bool:
    """Compare requests after removing only assistant historical reasoning fields."""
    clean = copy.deepcopy(clean)
    compact = copy.deepcopy(compact)
    for request in (clean, compact):
        for message in request.get("messages", []):
            if message.get("role") == "assistant":
                message.pop("reasoning", None)
                message.pop("reasoning_details", None)
    return clean == compact


def branch_record(branch: dict[str, Any]) -> dict[str, Any]:
    aggregate = branch["aggregate"]
    turns = branch.get("turns", [])
    last_finish = turns[-1].get("metrics", {}).get("finish_reason") if turns else None
    return {
        "success": bool(branch.get("success")),
        "status": branch.get("status"),
        "actions_correct": bool(branch.get("actions_correct")),
        "final_correct": bool(branch.get("final_correct")),
        "turn_count": int(aggregate["turn_count"]),
        "input_tokens": int(aggregate["prompt_tokens"]),
        "output_tokens": int(aggregate["completion_tokens"]),
        "reasoning_tokens": int(aggregate["reasoning_tokens"]),
        "input_tokens_by_turn": list(aggregate["prompt_tokens_by_turn"]),
        "output_tokens_by_turn": list(aggregate["completion_tokens_by_turn"]),
        "reasoning_tokens_by_turn": list(aggregate["reasoning_tokens_by_turn"]),
        "last_finish_reason": last_finish,
    }


def case_record(raw: dict[str, Any], capture: str, replacements: list[str]) -> dict[str, Any]:
    clean_branch = raw["branches"]["clean"]
    compact_branch = raw["branches"]["rewritten"]
    clean = branch_record(clean_branch)
    compact = branch_record(compact_branch)
    clean_turns = clean_branch.get("turns", [])
    compact_turns = compact_branch.get("turns", [])
    shared_before = []
    for index in range(min(len(clean_turns), len(compact_turns))):
        shared_before.append(
            last_tool_observation(clean_turns[index]["request"])
            == last_tool_observation(compact_turns[index]["request"])
        )
    first_invariant = bool(clean_turns and compact_turns) and reasoning_only_request_match(
        clean_turns[0]["request"], compact_turns[0]["request"]
    )
    clean_reason = clean["reasoning_tokens"]
    compact_reason = compact["reasoning_tokens"]
    return {
        "route": raw["model_name"],
        "task_id": raw["task_id"],
        "replicate": int(raw["replicate"]),
        "source_id": f"{raw['model_name']}|{raw['task_id']}",
        "source_sha256": raw.get("source_sha256"),
        "continuation_seed": raw.get("continuation_seed"),
        "capture": capture,
        "replacement_captures": replacements,
        "comparable_horizon": (
            clean["status"] == "complete"
            and compact["status"] == "complete"
            and clean["turn_count"] == compact["turn_count"]
        ),
        "clean": clean,
        "compact": compact,
        "reasoning_delta_tokens": compact_reason - clean_reason,
        "reasoning_change_percent": percent(clean_reason, compact_reason),
        "first_reasoning_delta_tokens": (
            compact["reasoning_tokens_by_turn"][0] - clean["reasoning_tokens_by_turn"][0]
        ),
        "first_reasoning_change_percent": percent(
            clean["reasoning_tokens_by_turn"][0], compact["reasoning_tokens_by_turn"][0]
        ),
        "subsequent_reasoning_delta_tokens": (
            sum(compact["reasoning_tokens_by_turn"][1:])
            - sum(clean["reasoning_tokens_by_turn"][1:])
        ),
        "subsequent_reasoning_change_percent": percent(
            sum(clean["reasoning_tokens_by_turn"][1:]),
            sum(compact["reasoning_tokens_by_turn"][1:]),
        ),
        "input_delta_tokens": compact["input_tokens"] - clean["input_tokens"],
        "output_delta_tokens": compact["output_tokens"] - clean["output_tokens"],
        "turn_delta": compact["turn_count"] - clean["turn_count"],
        "first_request_differs_only_in_assistant_reasoning": first_invariant,
        "shared_preceding_tool_result_by_turn": shared_before,
    }


def load_cases(
    root: Path, corrected_input: Path, apply_replacements: bool = True
) -> tuple[list[dict[str, Any]], list[Path]]:
    corrected = read_json(corrected_input / "analysis.json")
    replacement_map: dict[tuple[str, str], str] = {
        (item["parent"], item["condition"]): item["correction_result"]
        for item in corrected["replacements"]
    }
    cases: list[dict[str, Any]] = []
    inputs: list[Path] = [corrected_input / "analysis.json"]
    for run_rel, models in RUN_SELECTIONS:
        run = root / run_rel
        for path in sorted((run / "continuations").glob("*.json")):
            raw = read_json(path)
            if raw["model_name"] not in models:
                continue
            relative = path.relative_to(root).as_posix()
            replacement_paths = []
            for condition in ("clean", "rewritten"):
                replacement_rel = replacement_map.get((relative, condition))
                if replacement_rel and apply_replacements:
                    replacement_path = root / replacement_rel
                    raw["branches"][condition] = read_json(replacement_path)["branch"]
                    replacement_paths.append(replacement_rel)
                    inputs.append(replacement_path)
            cases.append(case_record(raw, relative, replacement_paths))
            inputs.append(path)
    cases.sort(key=lambda item: (ROUTES.index(item["route"]), item["task_id"], item["replicate"]))
    if len(cases) != 174:
        raise ValueError(f"expected 174 selected pairs, found {len(cases)}")
    return cases, inputs


def exclusion_explanations(record: dict[str, Any]) -> list[str]:
    return [
        review["review"]["explanation"]
        for review in record.get("reviews", [])
        if not review.get("review", {}).get("passed", False)
    ]


def load_sources(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    sources: list[dict[str, Any]] = []
    inputs: list[Path] = []
    for run_rel, models in AUDIT_SELECTIONS:
        path = root / run_rel / "source-audit.json"
        audit = read_json(path)
        inputs.append(path)
        records = audit.get("sources", audit.get("records", []))
        for source in records:
            if source["model_name"] not in models:
                continue
            if run_rel.endswith("two-route-screen"):
                accepted = source.get("status") == "selected"
                category = None if accepted else "fidelity_exclusion"
            else:
                accepted = bool(source.get("eligible"))
                category = None
                if not accepted:
                    category = (
                        "source_action_failure"
                        if source.get("status") == "source_not_selected"
                        else "fidelity_exclusion"
                    )
            sources.append({
                "route": source["model_name"],
                "task_id": source["task_id"],
                "source_id": f"{source['model_name']}|{source['task_id']}",
                "accepted": accepted,
                "exclusion_category": category,
                "gate_status": source.get("status"),
                "source_sha256": source.get("source_sha256"),
                "raw_words": source.get("raw_words"),
                "compact_words": source.get("rewritten_words"),
                "model_review_explanations": exclusion_explanations(source),
            })
    sources.sort(key=lambda item: (ROUTES.index(item["route"]), item["task_id"]))
    if len(sources) != 72:
        raise ValueError(f"expected 72 attempted route-task sources, found {len(sources)}")
    return sources, inputs


def paired_values(cases: list[dict[str, Any]], scope: str) -> tuple[list[float], list[float]]:
    if scope == "first":
        return (
            [case["clean"]["reasoning_tokens_by_turn"][0] for case in cases],
            [case["compact"]["reasoning_tokens_by_turn"][0] for case in cases],
        )
    if scope == "subsequent":
        return (
            [sum(case["clean"]["reasoning_tokens_by_turn"][1:]) for case in cases],
            [sum(case["compact"]["reasoning_tokens_by_turn"][1:]) for case in cases],
        )
    if scope == "trajectory":
        return (
            [case["clean"]["reasoning_tokens"] for case in cases],
            [case["compact"]["reasoning_tokens"] for case in cases],
        )
    raise ValueError(scope)


def paired_summary(clean: list[float], compact: list[float]) -> dict[str, Any]:
    deltas = [b - a for a, b in zip(clean, compact)]
    ratios = [value for a, b in zip(clean, compact) if (value := percent(a, b)) is not None]
    return {
        "pairs": len(clean),
        "clean_total_tokens": sum(clean),
        "compact_total_tokens": sum(compact),
        "pooled_change_percent": percent(sum(clean), sum(compact)),
        "delta_tokens": distribution(deltas),
        "paired_change_percent": distribution(ratios),
    }


def terminal_summary(cases: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    return {
        "statuses": dict(sorted(Counter(case[arm]["status"] for case in cases).items())),
        "last_finish_reasons": dict(sorted(Counter(case[arm]["last_finish_reason"] for case in cases).items())),
        "actions_correct": sum(case[arm]["actions_correct"] for case in cases),
        "final_correct": sum(case[arm]["final_correct"] for case in cases),
        "exact_success": sum(case[arm]["success"] for case in cases),
        "failures": len(cases) - sum(case[arm]["success"] for case in cases),
    }


def cohort_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [case for case in cases if case["comparable_horizon"]]
    clean_success = sum(case["clean"]["success"] for case in cases)
    compact_success = sum(case["compact"]["success"] for case in cases)
    discordant = {
        "compact_only": sum(case["compact"]["success"] and not case["clean"]["success"] for case in cases),
        "clean_only": sum(case["clean"]["success"] and not case["compact"]["success"] for case in cases),
    }
    by_turn = []
    max_turn = max(max(case["clean"]["turn_count"], case["compact"]["turn_count"]) for case in cases)
    for index in range(max_turn):
        paired = [
            case for case in cases
            if len(case["clean"]["reasoning_tokens_by_turn"]) > index
            and len(case["compact"]["reasoning_tokens_by_turn"]) > index
        ]
        clean = [case["clean"]["reasoning_tokens_by_turn"][index] for case in paired]
        compact = [case["compact"]["reasoning_tokens_by_turn"][index] for case in paired]
        by_turn.append({"turn": index + 1, **paired_summary(clean, compact)})
    return {
        "assigned_pairs": len(cases),
        "source_clusters": len({case["source_id"] for case in cases}),
        "comparable_horizon_pairs": len(comparable),
        "unequal_horizon_pairs": len(cases) - len(comparable),
        "quality_all_assigned": {
            "clean_successes": clean_success,
            "compact_successes": compact_success,
            "difference_percentage_points": 100 * (compact_success - clean_success) / len(cases),
            "discordant_pairs": discordant,
        },
        "comparable_trajectory_reasoning": paired_summary(
            [case["clean"]["reasoning_tokens"] for case in comparable],
            [case["compact"]["reasoning_tokens"] for case in comparable],
        ),
        "first_continuation_reasoning_all_assigned": paired_summary(*paired_values(cases, "first")),
        "subsequent_reasoning_all_assigned": paired_summary(*paired_values(cases, "subsequent")),
        "full_trajectory_reasoning_all_assigned": paired_summary(*paired_values(cases, "trajectory")),
        "reasoning_by_turn_when_both_arms_reach_turn": by_turn,
        "all_assigned_workload": {
            "turns": paired_summary(
                [case["clean"]["turn_count"] for case in cases],
                [case["compact"]["turn_count"] for case in cases],
            ),
            "input_tokens": paired_summary(
                [case["clean"]["input_tokens"] for case in cases],
                [case["compact"]["input_tokens"] for case in cases],
            ),
            "output_tokens": paired_summary(
                [case["clean"]["output_tokens"] for case in cases],
                [case["compact"]["output_tokens"] for case in cases],
            ),
            "clean_termination": terminal_summary(cases, "clean"),
            "compact_termination": terminal_summary(cases, "compact"),
        },
    }


def bootstrap(cases: list[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        clusters[case["source_id"]].append(case)
    names = sorted(clusters)
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        sample = [case for _ in names for case in clusters[rng.choice(names)]]
        comparable = [case for case in sample if case["comparable_horizon"]]
        values["comparable_median_change_percent"].append(statistics.median(
            case["reasoning_change_percent"] for case in comparable
        ))
        values["comparable_median_delta_tokens"].append(statistics.median(
            case["reasoning_delta_tokens"] for case in comparable
        ))
        values["first_median_change_percent"].append(statistics.median(
            case["first_reasoning_change_percent"] for case in sample
        ))
        values["first_median_delta_tokens"].append(statistics.median(
            case["first_reasoning_delta_tokens"] for case in sample
        ))
        values["full_trajectory_median_delta_tokens"].append(statistics.median(
            case["reasoning_delta_tokens"] for case in sample
        ))
        values["success_difference_percentage_points"].append(
            100 * sum(case["compact"]["success"] - case["clean"]["success"] for case in sample)
            / len(sample)
        )
    return {
        "method": "resample model-route/task source clusters with replacement; retain all replicates",
        "cluster_count": len(names),
        "resamples": resamples,
        "seed": seed,
        "ci95": {key: [quantile(samples, 0.025), quantile(samples, 0.975)] for key, samples in values.items()},
    }


def task_block_bootstrap(cases: list[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    """Resample 12 shared task definitions, retaining all observed routes/replicates."""
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        blocks[case["task_id"]].append(case)
    names = sorted(blocks)
    rng = random.Random(seed)
    effects: list[float] = []
    quality: list[float] = []
    absolute_deltas: list[float] = []
    for _ in range(resamples):
        sample = [case for _ in names for case in blocks[rng.choice(names)]]
        comparable = [case for case in sample if case["comparable_horizon"]]
        effects.append(statistics.median(case["reasoning_change_percent"] for case in comparable))
        absolute_deltas.append(statistics.median(case["reasoning_delta_tokens"] for case in comparable))
        quality.append(
            100 * sum(case["compact"]["success"] - case["clean"]["success"] for case in sample)
            / len(sample)
        )
    return {
        "method": (
            "resample the 12 shared task definitions with replacement; retain every observed "
            "accepted route-specific source and continuation replicate in each sampled task block"
        ),
        "scope": (
            "preserves cross-route dependence induced by shared task definitions while retaining "
            "route-specific source-gate missingness; remains conditional on accepted sources and routes"
        ),
        "block_count": len(names),
        "resamples": resamples,
        "seed": seed,
        "ci95": {
            "comparable_median_change_percent": [quantile(effects, 0.025), quantile(effects, 0.975)],
            "comparable_median_delta_tokens": [quantile(absolute_deltas, 0.025), quantile(absolute_deltas, 0.975)],
            "success_difference_percentage_points": [quantile(quality, 0.025), quantile(quality, 0.975)],
        },
    }


def replacement_audit(
    root: Path,
    corrected_input: Path,
    original_cases: list[dict[str, Any]],
    augmented_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    corrected = read_json(corrected_input / "analysis.json")
    recovery = root / "results/20260803-table1-infrastructure-reruns-v1"
    summary = read_json(recovery / "summary.json")
    protocol = read_json(recovery / "protocol.json")
    summary_by_key = {
        (item["parent_path"], item["condition"]): item for item in summary["cases"]
    }
    original_by_key = {
        (case["capture"], case["route"], case["task_id"], case["replicate"]): case
        for case in original_cases
    }
    augmented_by_key = {
        (case["capture"], case["route"], case["task_id"], case["replicate"]): case
        for case in augmented_cases
    }
    records = []
    for replacement in corrected["replacements"]:
        parent = replacement["parent"]
        condition = replacement["condition"]
        raw = read_json(root / parent)
        key = (parent, raw["model_name"], raw["task_id"], int(raw["replicate"]))
        original = original_by_key[key][condition if condition == "clean" else "compact"]
        augmented = augmented_by_key[key][condition if condition == "clean" else "compact"]
        rerun = summary_by_key[(parent, condition)]
        result = read_json(root / replacement["correction_result"])
        trigger = rerun["original_failure"]
        records.append({
            "route": raw["model_name"],
            "task_id": raw["task_id"],
            "replicate": int(raw["replicate"]),
            "arm": "compact" if condition == "rewritten" else condition,
            "classification": "retrospective_retry_augmentation",
            "original_trigger": trigger,
            "trigger_interpretation": (
                "observed 8,192-token length-cap outcome; not established as an infrastructure fault"
                if trigger == "length"
                else "captured provider HTTP 502 error"
            ),
            "original_capture": parent,
            "retry_augmented_capture": replacement["correction_result"],
            "maximum_attempts_per_retried_turn": protocol["retry_policy"]["maximum_attempts_per_turn"],
            "recovery_run_total_request_attempts": result["branch"]["total_infrastructure_attempts"],
            "recovery_attempts_by_turn": [
                turn.get("infrastructure_attempts", 1) for turn in result["branch"]["turns"]
            ],
            "original_first_capture_branch": original,
            "retry_augmented_branch": augmented,
            "branch_change_from_retry": {
                "success": int(augmented["success"]) - int(original["success"]),
                "turns": augmented["turn_count"] - original["turn_count"],
                "input_tokens": augmented["input_tokens"] - original["input_tokens"],
                "output_tokens": augmented["output_tokens"] - original["output_tokens"],
                "reasoning_tokens": augmented["reasoning_tokens"] - original["reasoning_tokens"],
            },
        })
    return records


def timing_audit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    max_turn = max(len(case["shared_preceding_tool_result_by_turn"]) for case in cases)
    shared = []
    for index in range(max_turn):
        reached = [case for case in cases if len(case["shared_preceding_tool_result_by_turn"]) > index]
        shared.append({
            "turn": index + 1,
            "pairs_both_reached": len(reached),
            "same_preceding_tool_result": sum(case["shared_preceding_tool_result_by_turn"][index] for case in reached),
        })
    return {
        "first_request_reasoning_only_invariant": sum(
            case["first_request_differs_only_in_assistant_reasoning"] for case in cases
        ),
        "pairs": len(cases),
        "shared_preceding_results": shared,
        "interpretation": (
            "The first continuation is the only contrast with no post-treatment generated mediator: "
            "rewriting was frozen before the pending result, then both arms received that same result. "
            "A later preceding tool result can still be identical, but later generation follows "
            "treatment-affected reasoning/actions and therefore combines direct and trajectory-mediated effects."
        ),
    }


def source_gate_summary(sources: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [source for source in sources if source["accepted"]]
    counts = Counter(source["exclusion_category"] for source in sources if not source["accepted"])
    by_route = {}
    for route in ROUTES:
        route_sources = [source for source in sources if source["route"] == route]
        by_route[route] = {
            "attempted": len(route_sources),
            "accepted": sum(source["accepted"] for source in route_sources),
            "source_action_failures": sum(source["exclusion_category"] == "source_action_failure" for source in route_sources),
            "fidelity_exclusions": sum(source["exclusion_category"] == "fidelity_exclusion" for source in route_sources),
        }
    return {
        "attempted_route_task_sources": len(sources),
        "accepted_sources": len(accepted),
        "assigned_pairs": 3 * len(accepted),
        "exclusions": dict(sorted(counts.items())),
        "by_route": by_route,
        "reviewer": "openai/gpt-5.6-sol in separate model calls; no blinded human agreement study",
        "exact_extension_model_review_rubric": FIDELITY_RUBRIC,
        "extension_gate_logic": {
            "required": [
                "all three source actions correct",
                "aggregate rewritten history shorter",
                "reasoning-only fork invariant",
                "every nonidentity rewrite passes model semantic review",
            ],
            "semantic_pass": (
                "verdict=pass AND critical_or_material_issue=false AND "
                "future_result_leakage=false AND requested_action_preserved=true"
            ),
        },
        "initial_gate_note": (
            "The initial two-route source audit additionally records candidate-ID and hard-threshold "
            "gates for every accepted rewrite. Its seven superseded artifacts concern fidelity or "
            "strict temporal order during reacquisition; they are not 14 excluded route-task sources."
        ),
    }


def verify_against_corrected_input(cases: list[dict[str, Any]], corrected_input: Path) -> dict[str, Any]:
    expected = read_json(corrected_input / "analysis.json")
    by_route = {route: cohort_summary([case for case in cases if case["route"] == route]) for route in ROUTES}
    checks = []
    for row in expected["table1_rows"]:
        actual = by_route[row["model_name"]]
        comp = actual["comparable_trajectory_reasoning"]
        checks.append({
            "route": row["model_name"],
            "total_cases_match": actual["assigned_pairs"] == row["total_cases"],
            "comparable_match": actual["comparable_horizon_pairs"] == row["comparable"],
            "shorter_match": comp["delta_tokens"]["negative"] == row["shorter"],
            "success_match": (
                actual["quality_all_assigned"]["clean_successes"] == row["clean_successes"]
                and actual["quality_all_assigned"]["compact_successes"] == row["rewritten_successes"]
            ),
            "median_change_match": abs(comp["paired_change_percent"]["median"] - row["median_change_percent"]) < 1e-12,
        })
    return {"all_checks_pass": all(all(value for key, value in check.items() if key != "route") for check in checks), "routes": checks}


def build_analysis(
    cases: list[dict[str, Any]],
    original_cases: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
    resamples: int,
    corrected_input: Path,
) -> dict[str, Any]:
    overall = cohort_summary(cases)
    original_overall = cohort_summary(original_cases)
    by_route = {}
    for route in ROUTES:
        route_cases = [case for case in cases if case["route"] == route]
        by_route[route] = cohort_summary(route_cases)
        by_route[route]["source_cluster_bootstrap"] = bootstrap(
            route_cases, resamples, seed=761 + len(route_cases)
        )
    leave_one_out = {}
    for route in ROUTES:
        remaining = [case for case in cases if case["route"] != route]
        leave_one_out[route] = cohort_summary(remaining)
    original_by_route = {}
    for route in ROUTES:
        route_cases = [case for case in original_cases if case["route"] == route]
        original_by_route[route] = cohort_summary(route_cases)
        original_by_route[route]["source_cluster_bootstrap"] = bootstrap(
            route_cases, resamples, seed=761 + len(route_cases)
        )
    comp = overall["comparable_trajectory_reasoning"]
    quality = overall["quality_all_assigned"]
    original_comp = original_overall["comparable_trajectory_reasoning"]
    return {
        "schema_version": 2,
        "analysis_scope": "existing-data, six-route controlled paired forks; no acquisition",
        "primary_estimand_classification": {
            "paper_result": "retrospective retry-augmented analysis accepted as the paper estimand",
            "sensitivity": "original first-captured attempts with no rescoring or suppression of length-cap outcomes",
            "critical_caveat": (
                "Three clean Laguna branches reached the stated 8,192 completion cap. Their classification "
                "as infrastructure failures is not independently established; retrying them changes the estimand. "
                "The fourth replacement follows a captured compact-arm provider HTTP 502."
            ),
            "retry_rule": "same captured request, up to 20 attempts per retried turn",
        },
        "sign_convention": "delta = compact - clean; negative means fewer tokens/turns under compact history",
        "cohort_accounting": source_gate_summary(sources),
        "headline_verification": {
            "compact_shorter_comparable_pairs": comp["delta_tokens"]["negative"],
            "comparable_pairs": comp["pairs"],
            "clean_exact_success": quality["clean_successes"],
            "compact_exact_success": quality["compact_successes"],
            "all_assigned_pairs": overall["assigned_pairs"],
            "success_difference_percentage_points": quality["difference_percentage_points"],
            "unit_warning": (
                "The 174 pairs are three continuation replicates nested in 58 accepted route-task sources, "
                "not 174 or 172 independent tasks. The 172 denominator applies only to equal-horizon token ratios."
            ),
        },
        "overall": overall,
        "by_route": by_route,
        "source_cluster_bootstrap": bootstrap(cases, resamples, seed=20260910),
        "task_definition_block_bootstrap": task_block_bootstrap(cases, resamples, seed=20260910),
        "retry_augmentation_records": replacements,
        "original_first_capture_sensitivity": {
            "headline": {
                "comparable_pairs": original_comp["pairs"],
                "compact_shorter_comparable_pairs": original_comp["delta_tokens"]["negative"],
                "clean_exact_success": original_overall["quality_all_assigned"]["clean_successes"],
                "compact_exact_success": original_overall["quality_all_assigned"]["compact_successes"],
                "all_assigned_pairs": original_overall["assigned_pairs"],
            },
            "overall": original_overall,
            "by_route": original_by_route,
            "source_cluster_bootstrap": bootstrap(original_cases, resamples, seed=20260910),
            "task_definition_block_bootstrap": task_block_bootstrap(
                original_cases, resamples, seed=20260910
            ),
            "right_tail": {
                "comparable_absolute_unit_token_deltas": original_comp["delta_tokens"],
                "positive_tail_total_extra_tokens": sum(
                    max(0, case["reasoning_delta_tokens"])
                    for case in original_cases if case["comparable_horizon"]
                ),
                "negative_tail_total_tokens_saved": -sum(
                    min(0, case["reasoning_delta_tokens"])
                    for case in original_cases if case["comparable_horizon"]
                ),
                "largest_increases": [
                    {key: case[key] for key in (
                        "route", "task_id", "replicate", "reasoning_delta_tokens", "reasoning_change_percent"
                    )}
                    for case in sorted(
                        [case for case in original_cases if case["comparable_horizon"]],
                        key=lambda item: item["reasoning_delta_tokens"], reverse=True,
                    )[:10]
                ],
            },
            "interpretation": (
                "All 174 originally assigned pairs remain in success, turn, termination, input, output, and "
                "full-trajectory totals. Equal-horizon ratios use only the originally comparable pairs."
            ),
        },
        "leave_one_route_out": leave_one_out,
        "causal_timing_audit": timing_audit(cases),
        "right_tail": {
            "comparable_absolute_unit_token_deltas": comp["delta_tokens"],
            "positive_tail_total_extra_tokens": sum(max(0, case["reasoning_delta_tokens"]) for case in cases if case["comparable_horizon"]),
            "negative_tail_total_tokens_saved": -sum(min(0, case["reasoning_delta_tokens"]) for case in cases if case["comparable_horizon"]),
            "largest_increases": [
                {key: case[key] for key in ("route", "task_id", "replicate", "reasoning_delta_tokens", "reasoning_change_percent")}
                for case in sorted(
                    [case for case in cases if case["comparable_horizon"]],
                    key=lambda item: item["reasoning_delta_tokens"], reverse=True,
                )[:10]
            ],
            "interpretation": (
                "Raw-token deltas avoid ratio-denominator amplification. Positive values are genuine reversal "
                "cases, so a negative median does not impose a budget or rule out large increases."
            ),
        },
        "estimand_interpretation": {
            "comparable_trajectory": (
                "Per-protocol paired reasoning effect conditional on an accepted rewrite and equal post-fork "
                "turn count; excludes two pairs and is not a task-population average."
            ),
            "first_continuation": (
                "Conditional direct first-output contrast after the shared result that was hidden from the "
                "rewriter; all 174 assigned pairs contribute."
            ),
            "subsequent_and_full_trajectory": (
                "Assigned-cohort trajectory outcomes. They include treatment-induced action, observation, and "
                "termination differences; shorter failed traces are not interpreted as efficiency successes."
            ),
            "input_output": (
                "Prompt-token totals sum provider-reported inputs across requests and therefore represent billed "
                "request workload, not unique context. Completion tokens include reasoning plus visible answer/tool output."
            ),
            "uncertainty": (
                "The 58-source bootstrap preserves within-source replicate dependence but treats route-task "
                "sources as exchangeable and does not preserve dependence from shared task definitions across "
                "routes. The 12-task-block sensitivity preserves that dependence. Both remain conditional on "
                "fixed accepted sources, routes, captured hosted runs, and the self-judged one-sided fidelity gate."
            ),
        },
        "corrected_input_fidelity_check": verify_against_corrected_input(cases, corrected_input),
    }


def render_readme(analysis: dict[str, Any]) -> str:
    h = analysis["headline_verification"]
    o = analysis["overall"]
    c = analysis["cohort_accounting"]
    first = o["first_continuation_reasoning_all_assigned"]
    later = o["subsequent_reasoning_all_assigned"]
    full = o["full_trajectory_reasoning_all_assigned"]
    ci = analysis["source_cluster_bootstrap"]["ci95"]
    work = o["all_assigned_workload"]
    original = analysis["original_first_capture_sensitivity"]
    original_overall = original["overall"]
    original_first = original_overall["first_continuation_reasoning_all_assigned"]
    original_full = original_overall["full_trajectory_reasoning_all_assigned"]
    original_work = original_overall["all_assigned_workload"]
    original_route_lines = []
    for route in ROUTES:
        summary = original["by_route"][route]
        token = summary["comparable_trajectory_reasoning"]
        quality = summary["quality_all_assigned"]
        original_route_lines.append(
            f"| {route} | {token['pairs']}/{summary['assigned_pairs']} | "
            f"{token['delta_tokens'].get('negative', 0)} | "
            f"{token['paired_change_percent'].get('median', float('nan')):+.1f}% | "
            f"{quality['clean_successes']}→{quality['compact_successes']} |"
        )
    original_route_table = "\n".join(original_route_lines)
    loro_lines = []
    for omitted in ROUTES:
        summary = analysis["leave_one_route_out"][omitted]
        token = summary["comparable_trajectory_reasoning"]
        quality = summary["quality_all_assigned"]
        loro_lines.append(
            f"| {omitted} | {token['pairs']} | {token['delta_tokens']['negative']} | "
            f"{token['paired_change_percent']['median']:+.1f}% | "
            f"{token['delta_tokens']['median']:+.0f} | "
            f"{quality['clean_successes']}→{quality['compact_successes']} "
            f"({quality['difference_percentage_points']:+.1f} pp) |"
        )
    loro_table = "\n".join(loro_lines)
    return f"""# Camera-ready controlled robustness audit v2

Offline recomputation from immutable continuation captures. The paper result is retained but classified as **retrospective retry-augmented**; original first-captured outcomes are reported separately. No model calls or network access are used.

## Retry-augmented paper estimand

- Cohort accounting is **{c['attempted_route_task_sources']} attempted route-task sources → {c['accepted_sources']} accepted sources → {h['all_assigned_pairs']} assigned continuation pairs** (three replicates per accepted source). These are not 174 independent tasks.
- Compact history is shorter in **{h['compact_shorter_comparable_pairs']}/{h['comparable_pairs']} equal-horizon pairs** ({100*h['compact_shorter_comparable_pairs']/h['comparable_pairs']:.1f}%). Exact success over every assigned pair is **{h['clean_exact_success']}/{h['all_assigned_pairs']} → {h['compact_exact_success']}/{h['all_assigned_pairs']}** ({h['success_difference_percentage_points']:+.1f} pp).
- At the first continuation, where both arms have the same just-revealed result and no post-treatment generated mediator, the paired median is **{first['paired_change_percent']['median']:+.1f}%** and the pooled token change is **{first['pooled_change_percent']:+.1f}%** over {first['pairs']} pairs. Subsequent reasoning has paired median **{later['paired_change_percent']['median']:+.1f}%** and pooled change **{later['pooled_change_percent']:+.1f}%**; it is a trajectory effect, not a direct effect.
- Full-trajectory reasoning over all assigned pairs changes from **{full['clean_total_tokens']:.0f} to {full['compact_total_tokens']:.0f} tokens** ({full['pooled_change_percent']:+.1f}%). The equal-horizon paired median is **{o['comparable_trajectory_reasoning']['paired_change_percent']['median']:+.1f}%**, with source-cluster bootstrap 95% interval **[{ci['comparable_median_change_percent'][0]:+.1f}, {ci['comparable_median_change_percent'][1]:+.1f}]%**.
- In raw token units, the equal-horizon paired median is **{o['comparable_trajectory_reasoning']['delta_tokens']['median']:+.0f} tokens** (bootstrap interval **[{ci['comparable_median_delta_tokens'][0]:+.0f}, {ci['comparable_median_delta_tokens'][1]:+.0f}]**). There are {o['comparable_trajectory_reasoning']['delta_tokens']['positive']} increases and {o['comparable_trajectory_reasoning']['delta_tokens']['zero']} equality; the maximum increase is **{o['comparable_trajectory_reasoning']['delta_tokens']['max']:+.0f} tokens**, so the right tail remains practically important.
- Without equal-horizon filtering, turns are **{work['turns']['clean_total_tokens']:.0f} → {work['turns']['compact_total_tokens']:.0f}**, accumulated input tokens **{work['input_tokens']['clean_total_tokens']:.0f} → {work['input_tokens']['compact_total_tokens']:.0f}**, and completion tokens **{work['output_tokens']['clean_total_tokens']:.0f} → {work['output_tokens']['compact_total_tokens']:.0f}**. There are **{work['clean_termination']['failures']} → {work['compact_termination']['failures']} exact failures**, including one action-failure termination in each arm.

The source-cluster 95% interval treats 58 route-task sources as exchangeable. Resampling the 12 shared task definitions instead gives **[{analysis['task_definition_block_bootstrap']['ci95']['comparable_median_change_percent'][0]:+.1f}, {analysis['task_definition_block_bootstrap']['ci95']['comparable_median_change_percent'][1]:+.1f}]%**; this preserves cross-route task dependence and observed gate missingness.

## Original-first-capture sensitivity

The three clean Laguna length-cap outcomes and the compact provider-502 outcome remain exactly as first captured. Exact success is **{original['headline']['clean_exact_success']}/{original['headline']['all_assigned_pairs']} → {original['headline']['compact_exact_success']}/{original['headline']['all_assigned_pairs']}** ({original_overall['quality_all_assigned']['difference_percentage_points']:+.1f} pp). Full-trajectory reasoning is **{original_full['clean_total_tokens']:.0f} → {original_full['compact_total_tokens']:.0f} tokens** ({original_full['pooled_change_percent']:+.1f}%), and first-turn paired median change is **{original_first['paired_change_percent']['median']:+.2f}%**.

| Route | Comparable/all | Shorter | Median change | Exact success clean→compact |
|---|---:|---:|---:|---:|
{original_route_table}

Across the original **{original['headline']['comparable_pairs']}** comparable pairs, **{original['headline']['compact_shorter_comparable_pairs']}** shorten; raw median delta is **{original_overall['comparable_trajectory_reasoning']['delta_tokens']['median']:+.0f} tokens**, with {original_overall['comparable_trajectory_reasoning']['delta_tokens']['positive']} increases, equality count {original_overall['comparable_trajectory_reasoning']['delta_tokens']['zero']}, and maximum increase **{original_overall['comparable_trajectory_reasoning']['delta_tokens']['max']:+.0f} tokens**. Without horizon filtering, turns are **{original_work['turns']['clean_total_tokens']:.0f} → {original_work['turns']['compact_total_tokens']:.0f}** ({original_work['turns']['pooled_change_percent']:+.1f}%), input tokens **{original_work['input_tokens']['clean_total_tokens']:.0f} → {original_work['input_tokens']['compact_total_tokens']:.0f}** ({original_work['input_tokens']['pooled_change_percent']:+.1f}%), completion tokens **{original_work['output_tokens']['clean_total_tokens']:.0f} → {original_work['output_tokens']['compact_total_tokens']:.0f}** ({original_work['output_tokens']['pooled_change_percent']:+.1f}%), and exact failures **{original_work['clean_termination']['failures']} → {original_work['compact_termination']['failures']}**. These failure-inclusive workload totals retain cap outcomes; shorter failed traces are not efficiency successes.

## Retry classification

Three clean branches—Laguna grid rep 2, manufacturing rep 3, and rescue rep 2—ended at the stated 8,192 completion cap. The available records do not establish that a cap outcome is an infrastructure fault. Laguna rescue rep 2 compact instead captured a provider HTTP 502. All four were retrospectively rerun under a rule permitting up to 20 attempts per retried turn; recovery-run request-attempt totals were 3, 3, 5, and 3; these totals exclude the original first-capture branches. `retry_augmentation_records.json` exports both branch versions, trigger classification, per-turn recovery attempt counts, and metric changes.

## Leave-one-route-out sensitivity of retry-augmented result

| Omitted route | Comparable pairs | Shorter | Median change | Median token delta | Exact success clean→compact |
|---|---:|---:|---:|---:|---:|
{loro_table}

The equal-horizon token effect remains negative after omitting any route. The aggregate quality sign changes when GLM 4.7 Flash is omitted, showing that the +0.6 pp total is not route-robust.

## Selection, timing, and uncertainty

The additional-route gate excluded {c['exclusions']['source_action_failure']} sources with no action-qualified source and {c['exclusions']['fidelity_exclusion']} for model-judged fidelity, leaving 58 accepted sources. GPT-5.6 Sol generated and separately judged rewrites; there was no independent blinded human validation. The controlled estimand is therefore conditional on passing a one-sided, compressor-family gate. `per_source.json` preserves each gate decision and compact rejection explanation available in the audits.

Route retention in displayed order is 12/12, 12/12, 10/12, 5/12, 10/12, and 9/12. The rewrite was completed before the pending tool result was executed or exposed. The first continuation received the same result in 174/174 pairs; the preceding result was also identical in 173/173 pairs reaching turn 2 in both arms and 172/172 reaching turn 3, but later text still follows treatment-affected generation and can contain mediated effects. Bootstrap intervals resample the 58 route-task source clusters and retain their continuation replicates; they do not estimate new routes, rejected sources, provider reruns, or seed variability.

## Provenance

`results/20260804-six-model-common-slice-infrastructure-corrected-v2/analysis.json` defines the four retry augmentations and the accepted paper reference rows; `results/20260803-table1-infrastructure-reruns-v1/protocol.json` defines their retry policy. Pair-level values are recalculated from `results/20260724-v2-12-task-two-route-screen/continuations/` for DeepSeek V4 Flash and Laguna S 2.1; `results/20260730-four-model-paired-replication-v1/continuations/` for MiniMax M3; `results/20260730-glm-four-model-replication-v2/continuations/` for GLM 4.7 Flash; and `results/20260730-latest-model-paired-replication-v1/continuations/` for GLM 5.1 and DeepSeek V4 Pro. Source decisions come from the corresponding source audits. The exact additional-route model-review rubric is copied from `scripts/run_four_model_replication.py` and exported in `per_source.json`; the initial source audit used additional candidate-ID and threshold gates documented in its immutable audit files.

## Reproduction

```bash
python3 scripts/analyze_camera_ready_controlled.py \\
  --root . \\
  --input results/20260804-six-model-common-slice-infrastructure-corrected-v2 \\
  --output results/20260910-camera-ready-controlled-audit-v2
python3 -m unittest tests/test_camera_ready_controlled.py
```

Outputs are deterministic: `analysis.json` contains aggregate estimands and caveats; explicitly named per-pair files preserve both versions; `retry_augmentation_records.json` binds original and replacement branches; `per_source.json` preserves source gates; and `manifest.json` hashes every consumed capture/audit plus generated output. Paths in JSON are repository-relative. The script uses only the Python standard library.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inferred_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=inferred_root)
    parser.add_argument("--input", type=Path, default=Path("results/20260804-six-model-common-slice-infrastructure-corrected-v2"))
    parser.add_argument("--output", type=Path, default=Path("results/20260910-camera-ready-controlled-audit-v2"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def resolve_under_root(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else root / path


def run(root: Path, corrected_input: Path, output: Path, resamples: int = 10000) -> dict[str, Any]:
    root = root.resolve()
    corrected_input = resolve_under_root(root, corrected_input)
    output = resolve_under_root(root, output)
    cases, case_inputs = load_cases(root, corrected_input, apply_replacements=True)
    original_cases, original_case_inputs = load_cases(
        root, corrected_input, apply_replacements=False
    )
    sources, source_inputs = load_sources(root)
    replacements = replacement_audit(
        root, corrected_input, original_cases, cases
    )
    analysis = build_analysis(
        cases, original_cases, sources, replacements, resamples, corrected_input
    )
    if not analysis["corrected_input_fidelity_check"]["all_checks_pass"]:
        raise ValueError("raw recomputation does not match corrected-v2 route rows")
    original = analysis["original_first_capture_sensitivity"]
    original_full = original["overall"]["full_trajectory_reasoning_all_assigned"]
    if (
        original["headline"]["clean_exact_success"],
        original["headline"]["compact_exact_success"],
        original_full["clean_total_tokens"],
        original_full["compact_total_tokens"],
    ) != (162, 165, 152475, 94357):
        raise ValueError("original-first-capture sensitivity failed frozen reference checks")
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "analysis.json", analysis)
    dump_json(output / "per_pair_retry_augmented.json", {
        "schema_version": 1,
        "estimand": "retrospective_retry_augmented",
        "pairs": cases,
    })
    dump_json(output / "per_pair_original_first_capture.json", {
        "schema_version": 1,
        "estimand": "original_first_capture_without_replacement",
        "pairs": original_cases,
    })
    dump_json(output / "retry_augmentation_records.json", {
        "schema_version": 1,
        "records": replacements,
    })
    dump_json(output / "per_source.json", {
        "schema_version": 1,
        "rubric": analysis["cohort_accounting"]["exact_extension_model_review_rubric"],
        "sources": sources,
    })
    (output / "README.md").write_text(render_readme(analysis), encoding="utf-8")
    consumed = sorted(set(
        case_inputs + original_case_inputs + source_inputs
        + [root / path for path in PROVENANCE_PATHS]
    ))
    missing = [path for path in consumed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing provenance inputs: {missing}")
    generated = [output / name for name in (
        "README.md",
        "analysis.json",
        "per_pair_retry_augmented.json",
        "per_pair_original_first_capture.json",
        "retry_augmentation_records.json",
        "per_source.json",
    )]
    manifest = {
        "schema_version": 1,
        "portable_root": ".",
        "inputs": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in consumed
        ],
        "outputs": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in generated
        ],
    }
    dump_json(output / "manifest.json", manifest)
    return analysis


def main() -> None:
    args = parse_args()
    run(args.root, args.input, args.output, args.bootstrap_resamples)


if __name__ == "__main__":
    main()
