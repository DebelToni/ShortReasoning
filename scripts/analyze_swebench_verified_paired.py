#!/usr/bin/env python3
"""Analyze a frozen paired SWE-bench Verified screen without mutating raw records."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from short_reasoning import strict_json_loads
from short_reasoning.durable import write_json_atomic

ARMS = ("clean_full_history", "external_sol_compact_history")
QUALITY_CLASSES = ("both_pass", "clean_only", "compact_only", "both_fail")
EXECUTION_CLASSES = (
    "officially_evaluated",
    "infrastructure_failure",
    "transport_failure",
    "unequal_or_invalid_horizon",
    "source_ineligible",
    "pending_or_unknown",
)


def read_json(path: Path) -> Any:
    return strict_json_loads(path.read_bytes())


def median(values: list[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None


def paired_normal_interval(differences: list[int]) -> dict[str, Any] | None:
    if not differences:
        return None
    n = len(differences)
    estimate = sum(differences) / n
    if n == 1:
        standard_error = 0.0
    else:
        standard_error = statistics.stdev(differences) / math.sqrt(n)
    return {
        "method": "paired item-level normal approximation",
        "estimate_pp": 100 * estimate,
        "two_sided_95_percent_interval_pp": [
            100 * (estimate - 1.959963984540054 * standard_error),
            100 * (estimate + 1.959963984540054 * standard_error),
        ],
        "one_sided_95_percent_lower_bound_pp": 100
        * (estimate - 1.6448536269514722 * standard_error),
        "n": n,
    }


def binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index) * probability**index * (1 - probability) ** (n - index)
        for index in range(k + 1)
    )


def conservative_exact_paired_lower_bound(
    differences: list[int], alpha: float = 0.05
) -> dict[str, Any] | None:
    if not differences:
        return None
    n = len(differences)
    clean_only = sum(value == -1 for value in differences)
    if clean_only == n:
        upper_loss_probability = 1.0
    else:
        lower = clean_only / n
        upper = 1.0
        for _ in range(100):
            midpoint = (lower + upper) / 2
            if binomial_cdf(clean_only, n, midpoint) > alpha:
                lower = midpoint
            else:
                upper = midpoint
        upper_loss_probability = upper
    return {
        "method": (
            "one-sided Clopper-Pearson upper bound on clean-only loss; "
            "compact-only gain conservatively lower-bounded by zero"
        ),
        "alpha": alpha,
        "clean_only_count": clean_only,
        "n": n,
        "clean_only_probability_upper_bound": upper_loss_probability,
        "compact_minus_clean_one_sided_95_percent_lower_bound_pp": (
            -100 * upper_loss_probability
        ),
    }


def official_outcomes(run_dir: Path, model: str, arm: str) -> dict[str, bool]:
    path = run_dir / "evaluation" / model / arm / "official-outcomes.json"
    if not path.is_file():
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or not all(
        isinstance(item, bool) for item in value.values()
    ):
        raise ValueError(f"malformed official outcome sidecar: {path}")
    return value


def evaluator_errors(run_dir: Path, model: str, arm: str) -> set[str]:
    path = run_dir / "evaluation" / model / arm / "official-report-binding.json"
    if not path.is_file():
        return set()
    value = read_json(path)
    errors = value.get("infrastructure_errors") if isinstance(value, dict) else None
    return set(map(str, errors)) if isinstance(errors, list) else set()


def sum_usage(turns: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = [turn["usage"]["prompt_tokens"] for turn in turns]
    completion = [turn["usage"]["completion_tokens"] for turn in turns]
    reasoning = [
        turn["usage"]["completion_tokens_details"]["reasoning_tokens"] for turn in turns
    ]
    costs = [turn["usage"].get("cost") for turn in turns]
    return {
        "turns": len(turns),
        "prompt_tokens_by_turn": prompt,
        "completion_tokens_by_turn": completion,
        "reasoning_tokens_by_turn": reasoning,
        "prompt_tokens_total": sum(prompt),
        "completion_tokens_total": sum(completion),
        "reasoning_tokens_total": sum(reasoning),
        "wall_seconds_total": sum(
            float(turn.get("wall_seconds") or 0) for turn in turns
        ),
        "reported_cost_usd_total": (
            sum(float(value) for value in costs)
            if all(value is not None for value in costs)
            else None
        ),
        "reported_cost_complete": all(value is not None for value in costs),
        "tool_names": [turn["tool_name"] for turn in turns],
    }


def load_turns(root: Path) -> list[dict[str, Any]]:
    return (
        [read_json(path) for path in sorted(root.glob("*/complete.json"))]
        if root.exists()
        else []
    )


def execution_class(
    run_dir: Path,
    job_root: Path,
    item_id: str,
    clean_known: bool,
    compact_known: bool,
    evaluator_error: bool,
) -> tuple[str, str]:
    if evaluator_error:
        return "infrastructure_failure", "official evaluator recorded an instance error"
    source_path = job_root / "source" / "state.json"
    if not source_path.is_file():
        preflight = run_dir / "preflight" / "items" / f"{item_id}.json"
        return (
            "pending_or_unknown" if preflight.is_file() else "infrastructure_failure",
            "source state absent",
        )
    source = read_json(source_path)
    screen_path = job_root / "screen.json"
    if source.get("status") != "source_complete":
        reason = str(source.get("reason") or source.get("status"))
        error = str(source.get("error") or "")
        combined = f"{reason}: {error}" if error else reason
        if "budget" in combined.lower():
            return "infrastructure_failure", combined
        if any(
            word in combined.lower()
            for word in ("ambiguous", "transport", "http", "timeout")
        ):
            return "transport_failure", combined
        return "source_ineligible", combined
    if not screen_path.is_file():
        return "source_ineligible", "source audit is absent"
    screen = read_json(screen_path)
    if not screen.get("eligible"):
        error = str(screen.get("error") or "")
        if "budget" in error.lower():
            return "infrastructure_failure", error
        if any(
            word in error.lower()
            for word in ("ambiguous", "transport", "http", "timeout")
        ):
            return "transport_failure", error
        return "source_ineligible", "automatic or isolated model review did not pass"
    pair_path = job_root / "pair" / "state.json"
    if not pair_path.is_file():
        return "pending_or_unknown", "fork absent"
    pair = read_json(pair_path)
    stopped = pair.get("stopped") or {}
    if any(
        value.get("reason") == "terminal_branch_error" for value in stopped.values()
    ):
        errors = " ".join(str(value.get("error") or "") for value in stopped.values())
        if "budget" in errors.lower():
            return "infrastructure_failure", errors
        if any(
            word in errors.lower()
            for word in ("ambiguous", "http", "transport", "timeout")
        ):
            return "transport_failure", errors
        return "unequal_or_invalid_horizon", errors
    turns = {
        arm: load_turns(job_root / "pair" / "arms" / arm / "turns") for arm in ARMS
    }
    if len(turns[ARMS[0]]) != len(turns[ARMS[1]]):
        return "unequal_or_invalid_horizon", "arms ended after different turn counts"
    if clean_known and compact_known:
        return "officially_evaluated", "both official arm outcomes available"
    return "pending_or_unknown", "official outcome missing"


def known_cost(usage: Any) -> float | None:
    if not isinstance(usage, dict) or usage.get("cost") is None:
        return None
    return float(usage["cost"])


def source_accounting(
    run_dir: Path, config: dict[str, Any], jobs: list[dict[str, Any]], model: str
) -> dict[str, Any]:
    job_map = {(job["model_name"], job["item_id"]): job for job in jobs}
    target_reasoning = []
    target_prompt = []
    target_completion = []
    target_costs = []
    rewrite_costs = []
    review_costs = []
    target_wall = []
    rewrite_wall = []
    review_wall = []
    route_failures = []
    for item_id in config["selector"]["item_ids"]:
        job = job_map[(model, item_id)]
        root = run_dir / "jobs" / job["job_id"]
        turns = load_turns(root / "source" / "turns")
        if turns:
            usages = [turn["target_usage"] for turn in turns]
            target_prompt.append(sum(value["prompt_tokens"] for value in usages))
            target_completion.append(
                sum(value["completion_tokens"] for value in usages)
            )
            target_reasoning.append(
                sum(
                    value["completion_tokens_details"]["reasoning_tokens"]
                    for value in usages
                )
            )
            costs = [known_cost(value) for value in usages]
            if all(value is not None for value in costs):
                target_costs.append(sum(value for value in costs if value is not None))
            target_wall.append(
                sum(float(turn["target_wall_seconds"]) for turn in turns)
            )
            rewrite = [
                known_cost(turn.get("rewrite", {}).get("usage")) for turn in turns
            ]
            if all(value is not None for value in rewrite):
                rewrite_costs.append(
                    sum(value for value in rewrite if value is not None)
                )
            rewrite_wall.append(
                sum(
                    float(turn.get("rewrite", {}).get("wall_seconds") or 0)
                    for turn in turns
                )
            )
        screen_path = root / "screen.json"
        if screen_path.is_file():
            screen = read_json(screen_path)
            reviews = screen.get("reviews") or screen.get("completed_reviews") or []
            costs = [known_cost(review.get("usage")) for review in reviews]
            if costs and all(value is not None for value in costs):
                review_costs.append(sum(value for value in costs if value is not None))
            if reviews:
                review_wall.append(
                    sum(float(review.get("wall_seconds") or 0) for review in reviews)
                )
        source_state = root / "source" / "state.json"
        if source_state.is_file():
            state = read_json(source_state)
            error = str(state.get("error") or "")
            if "route" in error.lower() or "provider" in error.lower():
                route_failures.append(
                    {"instance_id": item_id, "stage": "source", "error": error}
                )
        pair_state = root / "pair" / "state.json"
        if pair_state.is_file():
            state = read_json(pair_state)
            for arm, stop in (state.get("stopped") or {}).items():
                error = str(stop.get("error") or "")
                if "route" in error.lower() or "provider" in error.lower():
                    route_failures.append(
                        {"instance_id": item_id, "stage": arm, "error": error}
                    )
    freeze_path = run_dir / "source-freeze" / f"{model}.json"
    if freeze_path.is_file():
        freeze = read_json(freeze_path)
        selection = {
            key: freeze[key]
            for key in (
                "requested",
                "acquired_exactly_three_cycles",
                "mechanically_valid",
                "semantically_eligible",
                "excluded",
            )
        }
    else:
        selection = {
            "requested": 30,
            "acquired_exactly_three_cycles": None,
            "mechanically_valid": None,
            "semantically_eligible": None,
            "excluded": [],
        }
    return {
        "selection_denominator": selection,
        "target_source_reasoning_tokens_full_distribution": target_reasoning,
        "target_source_reasoning_tokens_median": median(target_reasoning),
        "target_source_prompt_tokens_full_distribution": target_prompt,
        "target_source_completion_tokens_full_distribution": target_completion,
        "target_source_reported_cost_usd_full_distribution": target_costs,
        "target_source_wall_seconds_full_distribution": target_wall,
        "rewrite_reported_cost_usd_full_distribution": rewrite_costs,
        "rewrite_wall_seconds_full_distribution": rewrite_wall,
        "review_reported_cost_usd_full_distribution": review_costs,
        "review_wall_seconds_full_distribution": review_wall,
        "route_provider_failures": route_failures,
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.snapshot.json")
    jobs = read_json(run_dir / "jobs.json")
    job_map = {(job["model_name"], job["item_id"]): job for job in jobs}
    result: dict[str, Any] = {
        "schema_version": 1,
        "label": "PROSPECTIVE_EXPLORATORY_SWE_SCREEN",
        "requested_items_per_route": 30,
        "screening_noninferiority_margin_pp": config[
            "screening_noninferiority_margin_pp"
        ],
        "actual_human_review_performed": False,
        "models": {},
    }
    for model in config["models"]:
        outcomes = {arm: official_outcomes(run_dir, model, arm) for arm in ARMS}
        evaluation_errors = {arm: evaluator_errors(run_dir, model, arm) for arm in ARMS}
        rows = []
        quality = Counter()
        execution = Counter()
        arm_distributions = {arm: [] for arm in ARMS}
        for item_id in config["selector"]["item_ids"]:
            job = job_map[(model, item_id)]
            root = run_dir / "jobs" / job["job_id"]
            clean = outcomes[ARMS[0]].get(item_id)
            compact = outcomes[ARMS[1]].get(item_id)
            quality_class = None
            if isinstance(clean, bool) and isinstance(compact, bool):
                quality_class = (
                    "both_pass"
                    if clean and compact
                    else "clean_only"
                    if clean
                    else "compact_only"
                    if compact
                    else "both_fail"
                )
                quality[quality_class] += 1
            kind, detail = execution_class(
                run_dir,
                root,
                item_id,
                isinstance(clean, bool),
                isinstance(compact, bool),
                any(item_id in evaluation_errors[arm] for arm in ARMS),
            )
            execution[kind] += 1
            usages = {}
            for arm in ARMS:
                turns = load_turns(root / "pair" / "arms" / arm / "turns")
                usages[arm] = sum_usage(turns)
                if turns:
                    arm_distributions[arm].append(usages[arm])
            rows.append(
                {
                    "instance_id": item_id,
                    "clean_pass": clean,
                    "compact_pass": compact,
                    "quality_class": quality_class,
                    "execution_class": kind,
                    "execution_detail": detail,
                    "arms": usages,
                }
            )
        paired = [
            int(row["compact_pass"]) - int(row["clean_pass"])
            for row in rows
            if isinstance(row["clean_pass"], bool)
            and isinstance(row["compact_pass"], bool)
        ]
        evaluated = len(paired)
        clean_count = sum(row["clean_pass"] is True for row in rows)
        compact_count = sum(row["compact_pass"] is True for row in rows)
        token_summary = {}
        for arm in ARMS:
            blocks = arm_distributions[arm]
            reasoning = [block["reasoning_tokens_total"] for block in blocks]
            completion = [block["completion_tokens_total"] for block in blocks]
            prompt = [block["prompt_tokens_total"] for block in blocks]
            wall = [block["wall_seconds_total"] for block in blocks]
            costs = [
                block["reported_cost_usd_total"]
                for block in blocks
                if block["reported_cost_usd_total"] is not None
            ]
            token_summary[arm] = {
                "reasoning_tokens_full_distribution": reasoning,
                "reasoning_tokens_median": median(reasoning),
                "completion_tokens_full_distribution": completion,
                "completion_tokens_median": median(completion),
                "prompt_tokens_full_distribution": prompt,
                "prompt_tokens_median": median(prompt),
                "wall_seconds_full_distribution": wall,
                "wall_seconds_median": median(wall),
                "reported_cost_usd_full_distribution": costs,
                "reported_cost_usd_median": median(costs),
            }
        paired_reasoning = []
        for row in rows:
            clean_tokens = row["arms"][ARMS[0]]["reasoning_tokens_total"]
            compact_tokens = row["arms"][ARMS[1]]["reasoning_tokens_total"]
            if (
                row["arms"][ARMS[0]]["turns"]
                and row["arms"][ARMS[0]]["turns"] == row["arms"][ARMS[1]]["turns"]
            ):
                paired_reasoning.append(
                    {
                        "instance_id": row["instance_id"],
                        "clean_reasoning_tokens": clean_tokens,
                        "compact_reasoning_tokens": compact_tokens,
                        "difference_tokens": compact_tokens - clean_tokens,
                        "percent_change": (
                            100 * (compact_tokens - clean_tokens) / clean_tokens
                            if clean_tokens
                            else None
                        ),
                    }
                )
        percent_changes = [
            row["percent_change"]
            for row in paired_reasoning
            if row["percent_change"] is not None
        ]
        interval = paired_normal_interval(paired)
        exact_lower_bound = conservative_exact_paired_lower_bound(paired)
        result["models"][model] = {
            "exact_route": {
                key: config["models"][model][key]
                for key in ("model", "provider_slug", "expected_provider")
            },
            "assigned_items": 30,
            "officially_paired_items": evaluated,
            "clean_pass_count": clean_count,
            "compact_pass_count": compact_count,
            "clean_pass_rate": clean_count / evaluated if evaluated else None,
            "compact_pass_rate": compact_count / evaluated if evaluated else None,
            "compact_minus_clean_pass_rate_pp": (
                100 * (compact_count - clean_count) / evaluated if evaluated else None
            ),
            "screening_margin_cleared": (
                exact_lower_bound[
                    "compact_minus_clean_one_sided_95_percent_lower_bound_pp"
                ]
                >= config["screening_noninferiority_margin_pp"]
                if exact_lower_bound and evaluated == 30
                else None
            ),
            "paired_quality_counts": {name: quality[name] for name in QUALITY_CLASSES},
            "execution_counts": {name: execution[name] for name in EXECUTION_CLASSES},
            "source_and_screen": source_accounting(run_dir, config, jobs, model),
            "paired_interval": interval,
            "conservative_exact_paired_lower_bound": exact_lower_bound,
            "paired_reasoning_tokens": {
                "items": paired_reasoning,
                "percent_change_full_distribution": percent_changes,
                "percent_change_median": median(percent_changes),
            },
            "token_latency_cost": token_summary,
            "items": rows,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir.resolve())
    output = args.output or args.run_dir / "analysis.json"
    write_json_atomic(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
