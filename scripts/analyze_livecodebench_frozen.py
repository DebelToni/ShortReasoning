#!/usr/bin/env python3
"""Analyze a pinned LiveCodeBench frozen-history run."""

from __future__ import annotations

import argparse
import difflib
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

from short_reasoning.frozen import reasoning_only_fork_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def percent(clean: float, rewritten: float) -> float | None:
    return None if not clean else (rewritten - clean) / clean * 100


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap(cases: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        clusters[f"{case['model_name']}|{case['question_id']}"] .append(case)
    names = sorted(clusters)
    rng = random.Random(seed)
    effects = []
    quality = []
    for _ in range(10000):
        sample = [case for __ in names for case in clusters[rng.choice(names)]]
        sample_effects = [
            case["reasoning_change_percent"]
            for case in sample
            if case["reasoning_change_percent"] is not None
        ]
        if sample_effects:
            effects.append(statistics.median(sample_effects))
        quality.append(
            (sum(case["rewritten_success"] for case in sample) - sum(case["clean_success"] for case in sample))
            / len(sample)
            * 100
        )
    return {
        "clusters": len(names),
        "resamples": 10000,
        "median_reasoning_change_percent_ci95": [quantile(effects, 0.025), quantile(effects, 0.975)],
        "success_difference_points_ci95": [quantile(quality, 0.025), quantile(quality, 0.975)],
    }


def case_summary(case: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    branches = case["branches"]
    first_clean = branches["clean"]["turns"][0]["request"]
    first_rewritten = branches["rewritten"]["turns"][0]["request"]
    audit = reasoning_only_fork_audit(first_clean["messages"], first_rewritten["messages"])
    non_message_equal = (
        {key: value for key, value in first_clean.items() if key != "messages"}
        == {key: value for key, value in first_rewritten.items() if key != "messages"}
    )
    code_changes = {}
    for condition, branch in branches.items():
        submitted = branch.get("submitted_code")
        code_changes[condition] = {
            "submitted": submitted is not None,
            "identical_to_recorded": submitted == source["recorded_code"] if submitted is not None else None,
            "edit_distance_ratio": (
                1 - difflib.SequenceMatcher(None, source["recorded_code"], submitted).ratio()
                if submitted is not None else None
            ),
        }
    providers = sorted(
        {
            turn["response"].get("provider")
            for branch in branches.values()
            for turn in branch["turns"]
            if turn["response"].get("provider")
        }
    )
    clean_submit_tokens = branches["clean"]["turns"][-1]["metrics"]["reasoning_tokens"]
    rewritten_submit_tokens = branches["rewritten"]["turns"][-1]["metrics"]["reasoning_tokens"]
    public_pass = case["fixed_observations"]["candidate_evaluation"]["public_tests"]["passed"]
    return {
        "model_name": case["model_name"],
        "question_id": case["question_id"],
        "task_id": case["task_id"],
        "replicate": case["replicate"],
        "source_sha256": case["source_sha256"],
        "providers": providers,
        "comparable_horizon": case["pair"]["comparable_horizon"],
        "reasoning_change_percent": case["pair"]["reasoning_token_change_percent"],
        "prompt_change_percent": case["pair"]["prompt_token_change_percent"],
        "clean_reasoning_tokens": branches["clean"]["aggregate"]["reasoning_tokens"],
        "rewritten_reasoning_tokens": branches["rewritten"]["aggregate"]["reasoning_tokens"],
        "clean_prompt_tokens": branches["clean"]["aggregate"]["prompt_tokens"],
        "rewritten_prompt_tokens": branches["rewritten"]["aggregate"]["prompt_tokens"],
        "clean_status": branches["clean"]["status"],
        "rewritten_status": branches["rewritten"]["status"],
        "clean_success": branches["clean"]["success"],
        "rewritten_success": branches["rewritten"]["success"],
        "clean_official_pass": branches["clean"].get("evaluation", {}).get("passed", False),
        "rewritten_official_pass": branches["rewritten"].get("evaluation", {}).get("passed", False),
        "recorded_candidate_public_pass": public_pass,
        "clean_submit_reasoning_tokens": clean_submit_tokens,
        "rewritten_submit_reasoning_tokens": rewritten_submit_tokens,
        "clean_extended_verification": (
            public_pass and branches["clean"]["status"] == "complete" and clean_submit_tokens >= 350
        ),
        "rewritten_extended_verification": (
            public_pass and branches["rewritten"]["status"] == "complete" and rewritten_submit_tokens >= 350
        ),
        "first_request_audit": audit,
        "non_message_request_fields_identical": non_message_equal,
        "code_changes": code_changes,
        "continuation_cost_usd": sum(
            float(turn["metrics"].get("cost") or 0)
            for branch in branches.values()
            for turn in branch["turns"]
        ),
    }


def group_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [case for case in cases if case["comparable_horizon"]]
    effects = [case["reasoning_change_percent"] for case in comparable]
    clean_reasoning = sum(case["clean_reasoning_tokens"] for case in comparable)
    rewritten_reasoning = sum(case["rewritten_reasoning_tokens"] for case in comparable)
    clean_prompt = sum(case["clean_prompt_tokens"] for case in comparable)
    rewritten_prompt = sum(case["rewritten_prompt_tokens"] for case in comparable)
    return {
        "cases": len(cases),
        "unique_sources": len({case["source_sha256"] for case in cases}),
        "comparable_cases": len(comparable),
        "reasoning_reduced_cases": sum(effect < 0 for effect in effects),
        "median_reasoning_change_percent": statistics.median(effects) if effects else None,
        "comparable_pooled_reasoning_tokens": [clean_reasoning, rewritten_reasoning],
        "comparable_pooled_reasoning_change_percent": percent(clean_reasoning, rewritten_reasoning),
        "comparable_pooled_prompt_tokens": [clean_prompt, rewritten_prompt],
        "comparable_pooled_prompt_change_percent": percent(clean_prompt, rewritten_prompt),
        "clean_successes": sum(case["clean_success"] for case in cases),
        "rewritten_successes": sum(case["rewritten_success"] for case in cases),
        "clean_official_passes": sum(case["clean_official_pass"] for case in cases),
        "rewritten_official_passes": sum(case["rewritten_official_pass"] for case in cases),
        "first_request_invariants": sum(
            case["first_request_audit"]["passed"] and case["non_message_request_fields_identical"]
            for case in cases
        ),
        "recorded_candidate_public_pass_cases": sum(case["recorded_candidate_public_pass"] for case in cases),
        "clean_extended_verification_cases": sum(case["clean_extended_verification"] for case in cases),
        "rewritten_extended_verification_cases": sum(case["rewritten_extended_verification"] for case in cases),
        "clean_median_submit_reasoning_tokens": statistics.median(
            case["clean_submit_reasoning_tokens"] for case in cases
        ),
        "rewritten_median_submit_reasoning_tokens": statistics.median(
            case["rewritten_submit_reasoning_tokens"] for case in cases
        ),
        "clean_submissions_identical_to_recorded": sum(
            case["code_changes"]["clean"]["identical_to_recorded"] is True for case in cases
        ),
        "rewritten_submissions_identical_to_recorded": sum(
            case["code_changes"]["rewritten"]["identical_to_recorded"] is True for case in cases
        ),
        "clustered_bootstrap": bootstrap(cases, seed=20260724 + len(cases)),
        "continuation_cost_usd": sum(case["continuation_cost_usd"] for case in cases),
    }


def render(analysis: dict[str, Any]) -> str:
    def change(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.1f}%"

    lines = [
        "# LiveCodeBench v6 frozen-fork screen",
        "",
        "| Model | Item | Rep | Success C/R | Reasoning C→R | Change | Prompt change |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in analysis["cases"]:
        lines.append(
            f"| {case['model_name']} | {case['question_id']} | {case['replicate']} | "
            f"{int(case['clean_success'])}/{int(case['rewritten_success'])} | "
            f"{case['clean_reasoning_tokens']}→{case['rewritten_reasoning_tokens']} | "
            f"{change(case['reasoning_change_percent'])} | {change(case['prompt_change_percent'])} |"
        )
    lines.extend(["", "## Model summaries", ""])
    for model, summary in analysis["by_model"].items():
        ci = summary["clustered_bootstrap"]["median_reasoning_change_percent_ci95"]
        lines.append(
            f"- **{model}:** {summary['reasoning_reduced_cases']}/{summary['comparable_cases']} comparable pairs shorter; "
            f"median {summary['median_reasoning_change_percent']:+.1f}% (model-specific-source-clustered 95% CI "
            f"[{ci[0]:+.1f}%, {ci[1]:+.1f}%]); strict success "
            f"{summary['clean_successes']}/{summary['cases']} clean and "
            f"{summary['rewritten_successes']}/{summary['cases']} rewritten."
        )
    overall = analysis["overall"]
    acquisition = analysis["acquisition"]
    lines.extend([
        "",
        "## Scope and validity",
        "",
        f"- Source acquisition succeeded for {acquisition['selected']}/{acquisition['total_model_items']} model/items; "
        f"{acquisition['shared_action_failures']} length-truncated failures were retained and not resampled.",
        f"- First-request reasoning-only invariants: {overall['first_request_invariants']}/{overall['cases']}.",
        f"- Comparable horizons: {overall['comparable_cases']}/{overall['cases']}.",
        f"- Official strict success: {overall['clean_successes']}/{overall['cases']} clean and "
        f"{overall['rewritten_successes']}/{overall['cases']} rewritten.",
        f"- Post-public-test extended verification (≥350 submit-turn reasoning tokens): "
        f"{overall['clean_extended_verification_cases']} clean and "
        f"{overall['rewritten_extended_verification_cases']} rewritten cases.",
        f"- Provider-reported continuation cost: ${overall['continuation_cost_usd']:.4f}.",
        f"- Unchanged shared-candidate submissions: {overall['clean_submissions_identical_to_recorded']} clean and "
        f"{overall['rewritten_submissions_identical_to_recorded']} rewritten cases; this screen primarily measures post-solution verification/termination, not repair quality.",
        "- The selected-source effect is conditional on completing shared acquisition and must not be generalized to all benchmark items.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    source_records = [json.loads(path.read_text()) for path in sorted((run_dir / "sources").glob("*.json"))]
    selected = {
        (source["model_name"], source["question_id"]): source
        for source in source_records
        if source.get("status") == "selected"
    }
    raw_cases = [json.loads(path.read_text()) for path in sorted((run_dir / "continuations").glob("*.json"))]
    cases = [case_summary(case, selected[(case["model_name"], case["question_id"])]) for case in raw_cases]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["model_name"]].append(case)
    acquisition = {
        "total_model_items": len(source_records),
        "selected": sum(source.get("status") == "selected" for source in source_records),
        "shared_action_failures": sum(source.get("status") == "shared_action_failure" for source in source_records),
        "by_model": {
            model: {
                "selected": sum(source.get("status") == "selected" and source["model_name"] == model for source in source_records),
                "shared_action_failures": sum(source.get("status") == "shared_action_failure" and source["model_name"] == model for source in source_records),
            }
            for model in sorted({source["model_name"] for source in source_records})
        },
    }
    analysis = {
        "cases": cases,
        "by_model": {model: group_summary(values) for model, values in sorted(grouped.items())},
        "overall": group_summary(cases),
        "acquisition": acquisition,
    }
    for name, content in (("analysis.json", json.dumps(analysis, indent=2) + "\n"), ("report.md", render(analysis))):
        path = run_dir / name
        if path.exists() and not args.replace:
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_text(content)
    checksum = run_dir / "continuations" / "SHA256SUMS"
    if checksum.exists() and not args.replace:
        raise SystemExit(f"refusing to overwrite {checksum}")
    paths = sorted((run_dir / "continuations").glob("*.json"))
    checksum.write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in paths))
    print(json.dumps({"by_model": analysis["by_model"], "acquisition": acquisition}, indent=2))


if __name__ == "__main__":
    main()
