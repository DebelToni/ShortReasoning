#!/usr/bin/env python3
"""Audit and summarize an immutable frozen-fork run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from short_reasoning import reasoning_text  # noqa: E402
from short_reasoning.frozen import reasoning_only_fork_audit  # noqa: E402

LOOP_MARKERS = (
    "however",
    "but wait",
    "wait,",
    "reconsider",
    "on the other hand",
    "actually",
    "double-check",
    "let me check",
    "the issue",
    "problem",
    "uncertain",
    "ambigu",
    "conflict",
    "could",
    "maybe",
    "perhaps",
    "should i",
    "we still need",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def percent(clean: int | float, rewritten: int | float) -> float | None:
    return None if not clean else (rewritten - clean) / clean * 100


def loop_signature(turn: dict[str, Any]) -> dict[str, Any]:
    message = turn["response"]["choices"][0]["message"]
    text = reasoning_text(message)
    lower = text.lower()
    hits = {marker: lower.count(marker) for marker in LOOP_MARKERS if marker in lower}
    tokens = turn["metrics"].get("reasoning_tokens") or 0
    return {
        "tokens": tokens,
        "words": len(text.split()),
        "marker_count": sum(hits.values()),
        "marker_hits": hits,
        "qualifies": tokens >= 350 and sum(hits.values()) >= 3,
    }


def provider_names(case: dict[str, Any]) -> list[str]:
    values = set()
    for branch in case["branches"].values():
        for turn in branch["turns"]:
            provider = turn["response"].get("provider")
            if provider:
                values.add(provider)
    return sorted(values)


def usage_cost(response: Any) -> float:
    return float(response.get("usage", {}).get("cost") or 0) if isinstance(response, dict) else 0.0


def source_rewrite_cost(sources: list[dict[str, Any]]) -> float:
    return sum(
        usage_cost(step.get("rewrite", {}).get("response"))
        for source in sources
        for step in source.get("shared", [])
    )


def summarize_case(case: dict[str, Any]) -> dict[str, Any]:
    branches = case["branches"]
    loop_data = {}
    for condition, branch in branches.items():
        signatures = [loop_signature(turn) for turn in branch["turns"]]
        qualifying = [item for item in signatures if item["qualifies"]]
        loop_data[condition] = {
            "entered": bool(qualifying),
            "qualifying_turns": len(qualifying),
            "qualifying_tokens": sum(item["tokens"] for item in qualifying),
            "turns": signatures,
        }

    first_clean = branches["clean"]["turns"][0]["request"]
    first_rewritten = branches["rewritten"]["turns"][0]["request"]
    clean_without_messages = {key: value for key, value in first_clean.items() if key != "messages"}
    rewritten_without_messages = {
        key: value for key, value in first_rewritten.items() if key != "messages"
    }
    first_audit = reasoning_only_fork_audit(
        first_clean["messages"], first_rewritten["messages"]
    )
    first_audit["non_message_request_fields_identical"] = (
        clean_without_messages == rewritten_without_messages
    )
    first_audit["passed_with_request_fields"] = (
        first_audit["passed"] and first_audit["non_message_request_fields_identical"]
    )

    clean_by_turn = branches["clean"]["aggregate"]["reasoning_tokens_by_turn"]
    rewritten_by_turn = branches["rewritten"]["aggregate"]["reasoning_tokens_by_turn"]
    return {
        "model_name": case["model_name"],
        "task_id": case["task_id"],
        "replicate": case["replicate"],
        "source_sha256": case["source_sha256"],
        "continuation_seed": case["continuation_seed"],
        "status": case["status"],
        "providers": provider_names(case),
        "comparable_horizon": case["pair"]["comparable_horizon"],
        "clean_success": branches["clean"]["success"],
        "rewritten_success": branches["rewritten"]["success"],
        "clean_reasoning_tokens": branches["clean"]["aggregate"]["reasoning_tokens"],
        "rewritten_reasoning_tokens": branches["rewritten"]["aggregate"]["reasoning_tokens"],
        "reasoning_token_delta": (
            branches["rewritten"]["aggregate"]["reasoning_tokens"]
            - branches["clean"]["aggregate"]["reasoning_tokens"]
        ),
        "reasoning_token_change_percent": case["pair"]["reasoning_token_change_percent"],
        "clean_completion_tokens": branches["clean"]["aggregate"]["completion_tokens"],
        "rewritten_completion_tokens": branches["rewritten"]["aggregate"]["completion_tokens"],
        "completion_token_change_percent": case["pair"]["completion_token_change_percent"],
        "clean_wall_seconds": branches["clean"]["aggregate"]["wall_seconds"],
        "rewritten_wall_seconds": branches["rewritten"]["aggregate"]["wall_seconds"],
        "wall_change_percent": percent(
            branches["clean"]["aggregate"]["wall_seconds"],
            branches["rewritten"]["aggregate"]["wall_seconds"],
        ),
        "clean_cost_usd": branches["clean"]["aggregate"]["cost"],
        "rewritten_cost_usd": branches["rewritten"]["aggregate"]["cost"],
        "cost_change_percent": percent(
            branches["clean"]["aggregate"]["cost"],
            branches["rewritten"]["aggregate"]["cost"],
        ),
        "clean_prompt_tokens": branches["clean"]["aggregate"]["prompt_tokens"],
        "rewritten_prompt_tokens": branches["rewritten"]["aggregate"]["prompt_tokens"],
        "prompt_token_change_percent": case["pair"]["prompt_token_change_percent"],
        "clean_reasoning_by_turn": clean_by_turn,
        "rewritten_reasoning_by_turn": rewritten_by_turn,
        "first_turn_reasoning_change_percent": (
            percent(clean_by_turn[0], rewritten_by_turn[0])
            if clean_by_turn and rewritten_by_turn else None
        ),
        "later_turn_reasoning_change_percent": (
            percent(sum(clean_by_turn[1:]), sum(rewritten_by_turn[1:]))
            if len(clean_by_turn) == len(rewritten_by_turn) and len(clean_by_turn) > 1 else None
        ),
        "first_request_audit": first_audit,
        "loops": loop_data,
        "continuation_cost_usd": sum(
            float(turn["metrics"].get("cost") or 0)
            for branch in branches.values()
            for turn in branch["turns"]
        ),
    }


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def clustered_bootstrap(cases: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        clusters[f"{case['model_name']}|{case['task_id']}"].append(case)
    names = sorted(clusters)
    rng = random.Random(seed)
    effect_statistics = []
    quality_statistics = []
    for _ in range(10000):
        sample = [case for _ in names for case in clusters[rng.choice(names)]]
        effects = [
            case["reasoning_token_change_percent"]
            for case in sample
            if case["reasoning_token_change_percent"] is not None
        ]
        if effects:
            effect_statistics.append(statistics.median(effects))
        quality_statistics.append(
            (sum(case["rewritten_success"] for case in sample) - sum(case["clean_success"] for case in sample))
            / len(sample)
            * 100
        )
    return {
        "resamples": 10000,
        "cluster_count": len(names),
        "median_effect_percent_ci95": [
            quantile(effect_statistics, 0.025),
            quantile(effect_statistics, 0.975),
        ],
        "success_difference_percentage_points_ci95": [
            quantile(quality_statistics, 0.025),
            quantile(quality_statistics, 0.975),
        ],
    }


def model_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    comparable_cases = [case for case in cases if case["comparable_horizon"]]
    effects = [
        case["reasoning_token_change_percent"]
        for case in comparable_cases
        if case["reasoning_token_change_percent"] is not None
    ]
    task_effects = [
        statistics.median(
            case["reasoning_token_change_percent"]
            for case in comparable_cases
            if case["task_id"] == task_id
        )
        for task_id in sorted({case["task_id"] for case in comparable_cases})
    ]
    clean_reasoning = sum(case["clean_reasoning_tokens"] for case in comparable_cases)
    rewritten_reasoning = sum(case["rewritten_reasoning_tokens"] for case in comparable_cases)
    clean_prompt = sum(case["clean_prompt_tokens"] for case in comparable_cases)
    rewritten_prompt = sum(case["rewritten_prompt_tokens"] for case in comparable_cases)
    clean_loop_cases = [case for case in cases if case["loops"]["clean"]["entered"]]
    rewritten_loop_cases = [case for case in cases if case["loops"]["rewritten"]["entered"]]
    max_turns = max((len(case["clean_reasoning_by_turn"]) for case in comparable_cases), default=0)
    pooled_by_turn = []
    for turn_index in range(max_turns):
        clean_turn = sum(case["clean_reasoning_by_turn"][turn_index] for case in comparable_cases)
        rewritten_turn = sum(case["rewritten_reasoning_by_turn"][turn_index] for case in comparable_cases)
        pooled_by_turn.append({
            "turn": turn_index + 1,
            "clean_reasoning_tokens": clean_turn,
            "rewritten_reasoning_tokens": rewritten_turn,
            "change_percent": percent(clean_turn, rewritten_turn),
        })
    return {
        "cases": len(cases),
        "comparable_horizon_cases": len(comparable_cases),
        "first_request_invariants_passed": sum(
            case["first_request_audit"]["passed_with_request_fields"] for case in cases
        ),
        "clean_successes": sum(case["clean_success"] for case in cases),
        "rewritten_successes": sum(case["rewritten_success"] for case in cases),
        "paired_success_difference_percentage_points": (
            sum(case["rewritten_success"] for case in cases)
            - sum(case["clean_success"] for case in cases)
        ) / len(cases) * 100,
        "reasoning_reduced_cases": sum(effect < 0 for effect in effects),
        "reasoning_equal_cases": sum(effect == 0 for effect in effects),
        "reasoning_increased_cases": sum(effect > 0 for effect in effects),
        "mean_reasoning_token_change_percent": statistics.mean(effects) if effects else None,
        "sample_variance_reasoning_percent_squared": (
            statistics.variance(effects) if len(effects) > 1 else None
        ),
        "median_reasoning_token_change_percent": statistics.median(effects) if effects else None,
        "reasoning_change_percent_iqr": (
            [quantile(effects, 0.25), quantile(effects, 0.75)] if effects else None
        ),
        "median_reasoning_token_delta": statistics.median(
            case["reasoning_token_delta"] for case in comparable_cases
        ) if comparable_cases else None,
        "median_completion_token_change_percent": statistics.median(
            case["completion_token_change_percent"]
            for case in comparable_cases
            if case["completion_token_change_percent"] is not None
        ) if comparable_cases else None,
        "median_wall_change_percent": statistics.median(
            case["wall_change_percent"]
            for case in comparable_cases
            if case["wall_change_percent"] is not None
        ) if comparable_cases else None,
        "median_cost_change_percent": statistics.median(
            case["cost_change_percent"]
            for case in comparable_cases
            if case["cost_change_percent"] is not None
        ) if comparable_cases else None,
        "median_task_level_reasoning_token_change_percent": statistics.median(task_effects) if task_effects else None,
        "median_first_turn_reasoning_change_percent": statistics.median(
            case["first_turn_reasoning_change_percent"] for case in comparable_cases
        ) if comparable_cases else None,
        "median_later_turn_reasoning_change_percent": statistics.median(
            case["later_turn_reasoning_change_percent"] for case in comparable_cases
        ) if comparable_cases else None,
        "comparable_pooled_reasoning_by_turn": pooled_by_turn,
        "comparable_pooled_clean_reasoning_tokens": clean_reasoning,
        "comparable_pooled_rewritten_reasoning_tokens": rewritten_reasoning,
        "comparable_pooled_reasoning_token_change_percent": percent(clean_reasoning, rewritten_reasoning),
        "comparable_pooled_clean_prompt_tokens": clean_prompt,
        "comparable_pooled_rewritten_prompt_tokens": rewritten_prompt,
        "comparable_pooled_prompt_token_change_percent": percent(clean_prompt, rewritten_prompt),
        "all_workload_clean_reasoning_tokens": sum(case["clean_reasoning_tokens"] for case in cases),
        "all_workload_rewritten_reasoning_tokens": sum(case["rewritten_reasoning_tokens"] for case in cases),
        "clustered_bootstrap": clustered_bootstrap(cases, seed=761 + len(cases)),
        "clean_loop_entries": len(clean_loop_cases),
        "rewritten_loop_entries": len(rewritten_loop_cases),
        "clean_median_qualifying_loop_tokens_if_entered": (
            statistics.median(case["loops"]["clean"]["qualifying_tokens"] for case in clean_loop_cases)
            if clean_loop_cases else None
        ),
        "rewritten_median_qualifying_loop_tokens_if_entered": (
            statistics.median(case["loops"]["rewritten"]["qualifying_tokens"] for case in rewritten_loop_cases)
            if rewritten_loop_cases else None
        ),
        "continuation_cost_usd": sum(case["continuation_cost_usd"] for case in cases),
    }


def render_report(analysis: dict[str, Any]) -> str:
    def change_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.1f}%"

    lines = [
        "# Frozen-fork run report",
        "",
        "Only clean and faithfully rewritten historical assistant reasoning differ at each fork.",
        "",
        "| Model | Task | Rep | Success C/R | Reasoning C→R | Change | Prompt change | Post-hoc loops C/R |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in analysis["cases"]:
        lines.append(
            f"| {case['model_name']} | {case['task_id']} | {case['replicate']} | "
            f"{int(case['clean_success'])}/{int(case['rewritten_success'])} | "
            f"{case['clean_reasoning_tokens']}→{case['rewritten_reasoning_tokens']} | "
            f"{change_text(case['reasoning_token_change_percent'])} | "
            f"{change_text(case['prompt_token_change_percent'])} | "
            f"{int(case['loops']['clean']['entered'])}/{int(case['loops']['rewritten']['entered'])} |"
        )
    lines.extend(["", "## Model summaries", ""])
    for model_name, summary in analysis["by_model"].items():
        lines.append(
            f"- **{model_name}:** reasoning fell in {summary['reasoning_reduced_cases']}/{summary['comparable_horizon_cases']}; "
            f"median paired change {summary['median_reasoning_token_change_percent']:+.1f}%; comparable pooled "
            f"{summary['comparable_pooled_clean_reasoning_tokens']}→{summary['comparable_pooled_rewritten_reasoning_tokens']} "
            f"({summary['comparable_pooled_reasoning_token_change_percent']:+.1f}%); task-clustered median CI "
            f"[{summary['clustered_bootstrap']['median_effect_percent_ci95'][0]:+.1f}%, "
            f"{summary['clustered_bootstrap']['median_effect_percent_ci95'][1]:+.1f}%]; success "
            f"{summary['clean_successes']}/{summary['cases']} clean and "
            f"{summary['rewritten_successes']}/{summary['cases']} rewritten; post-hoc loop entry "
            f"{summary['clean_loop_entries']}→{summary['rewritten_loop_entries']}."
        )
    overall = analysis["overall"]
    lines.extend([
        "",
        "## Validity and scope",
        "",
        f"- First-request reasoning-only invariants: {overall['first_request_invariants_passed']}/{overall['cases']}.",
        f"- Comparable horizons: {overall['comparable_horizon_cases']}/{overall['cases']}.",
        f"- Continuation cost reported by providers: ${overall['continuation_cost_usd']:.4f}.",
        "- Loop labels use the prespecified post-hoc lexical/length detector (≥350 reasoning tokens and ≥3 reconsideration markers); they are descriptive, not confirmatory.",
        "- This controlled synthetic-task run is a validity and signal screen, not benchmark-quality evidence.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    paths = sorted((run_dir / "continuations").glob("*.json"))
    if not paths:
        raise SystemExit("no continuation captures")
    captures = [json.loads(path.read_text()) for path in paths]
    complete_captures = [
        capture
        for capture in captures
        if capture.get("status") == "complete" and isinstance(capture.get("branches"), dict)
    ]
    failures = [
        {
            "path": str(path.relative_to(run_dir)),
            "model_name": capture.get("model_name"),
            "task_id": capture.get("task_id"),
            "replicate": capture.get("replicate"),
            "status": capture.get("status"),
            "error": capture.get("error"),
        }
        for path, capture in zip(paths, captures, strict=True)
        if capture not in complete_captures
    ]
    if not complete_captures:
        raise SystemExit("no complete continuation captures")
    cases = [summarize_case(capture) for capture in complete_captures]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["model_name"]].append(case)
    by_model = {name: model_summary(values) for name, values in sorted(grouped.items())}
    overall = model_summary(cases)

    sources = [json.loads(path.read_text()) for path in sorted((run_dir / "sources").glob("*.json"))]
    acquisition_target_cost = sum(
        float(step["metrics"].get("cost") or 0)
        for source in sources
        for step in source.get("shared", [])
    )
    rewrite_cost = source_rewrite_cost(sources)
    source_audit_path = run_dir / "source-audit.json"
    source_audit = json.loads(source_audit_path.read_text()) if source_audit_path.exists() else None
    semantic_review_cost = (
        float(source_audit["model_review_cost_usd"]) if source_audit else 0.0
    )
    analysis = {
        "requested_cases": len(captures),
        "complete_cases": len(complete_captures),
        "capture_failures": failures,
        "source_audit_sha256": (
            hashlib.sha256(source_audit_path.read_bytes()).hexdigest()
            if source_audit_path.exists() else None
        ),
        "source_audit_summary": (
            {
                "requested_sources": source_audit["requested_sources"],
                "eligible_sources": source_audit["eligible_sources"],
                "actual_human_review_performed": source_audit[
                    "actual_human_review_performed"
                ],
                "raw_words": source_audit["raw_words"],
                "rewritten_words": source_audit["rewritten_words"],
                "model_review_cost_usd": source_audit["model_review_cost_usd"],
            }
            if source_audit else None
        ),
        "cases": cases,
        "by_model": by_model,
        "overall": overall,
        "cost": {
            "acquisition_target_usd": acquisition_target_cost,
            "rewrite_usd": rewrite_cost,
            "continuation_usd": overall["continuation_cost_usd"],
            "semantic_review_usd": semantic_review_cost,
            "total_excluding_semantic_review_usd": (
                acquisition_target_cost
                + rewrite_cost
                + overall["continuation_cost_usd"]
            ),
            "total_usd": (
                acquisition_target_cost
                + rewrite_cost
                + overall["continuation_cost_usd"]
                + semantic_review_cost
            ),
        },
    }
    for name, value in (("analysis.json", analysis),):
        path = run_dir / name
        if path.exists() and not args.replace:
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    report = run_dir / "report.md"
    if report.exists() and not args.replace:
        raise SystemExit(f"refusing to overwrite {report}")
    report.write_text(render_report(analysis))
    checksum_path = run_dir / "continuations" / "SHA256SUMS"
    if checksum_path.exists() and not args.replace:
        raise SystemExit(f"refusing to overwrite {checksum_path}")
    checksum_path.write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in paths)
    )
    root_manifest = run_dir / "SHA256SUMS"
    if root_manifest.exists() and not args.replace:
        raise SystemExit(f"refusing to overwrite {root_manifest}")
    manifest_paths = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path != root_manifest and not path.name.endswith(".tmp")
    )
    root_manifest.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(run_dir)}\n"
            for path in manifest_paths
        )
    )
    print(json.dumps({"by_model": by_model, "cost": analysis["cost"]}, indent=2))


if __name__ == "__main__":
    main()
