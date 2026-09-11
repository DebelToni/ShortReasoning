#!/usr/bin/env python3
"""Synthesize the V4/V5 SWE-bench screen without mutating raw captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ARMS = ("clean_full_history", "external_sol_compact_history")
MODEL_RUNS = {
    "deepseek-v4-flash": "20260731-swebench-verified-paired-screen-v4",
    "deepseek-v4-pro": "20260731-swebench-verified-paired-screen-v5",
    "glm-5.1": "20260731-swebench-verified-paired-screen-v5",
    "minimax-m3": "20260731-swebench-verified-paired-screen-v5",
}
LABELS = {
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "glm-5.1": "GLM 5.1",
    "minimax-m3": "MiniMax M3",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def median(values: list[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None


def arm_outcomes(run: Path, model: str, arm: str) -> dict[str, bool]:
    standard = run / "evaluation" / model / arm / "official-outcomes.json"
    if standard.is_file():
        return read_json(standard)
    reports = list((run / "evaluation-derived" / arm).glob(f"{model}--{arm}.*.json"))
    if len(reports) != 1:
        return {}
    report = read_json(reports[0])
    resolved = set(report["resolved_ids"])
    return {item: item in resolved for item in report["submitted_ids"]}


def turn_records(root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted(root.glob("*/complete.json"))]


def normal_interval(differences: list[int]) -> dict[str, Any]:
    estimate = statistics.mean(differences)
    se = (
        statistics.stdev(differences) / math.sqrt(len(differences))
        if len(differences) > 1
        else 0.0
    )
    return {
        "method": "descriptive paired item-level normal approximation over officially submitted items",
        "n": len(differences),
        "estimate_pp": 100 * estimate,
        "two_sided_95_percent_interval_pp": [
            100 * (estimate - 1.959963984540054 * se),
            100 * (estimate + 1.959963984540054 * se),
        ],
        "one_sided_95_percent_lower_bound_pp": 100
        * (estimate - 1.6448536269514722 * se),
    }


def clustered_median_interval(
    items: list[dict[str, Any]], *, seed: int, replicates: int = 10_000
) -> dict[str, Any] | None:
    usable = [item for item in items if item.get("percent_change") is not None]
    if not usable:
        return None
    clusters: dict[str, list[float]] = {}
    for item in usable:
        repo = item["instance_id"].split("__", 1)[0]
        clusters.setdefault(repo, []).append(float(item["percent_change"]))
    names = sorted(clusters)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(names) for _ in names]
        values = [value for name in sampled for value in clusters[name]]
        estimates.append(float(statistics.median(values)))
    estimates.sort()
    return {
        "method": "repository-clustered percentile bootstrap of the item median",
        "seed": seed,
        "replicates": replicates,
        "items": len(usable),
        "repositories": len(names),
        "two_sided_95_percent_interval": [
            estimates[int(0.025 * (replicates - 1))],
            estimates[int(0.975 * (replicates - 1))],
        ],
    }


def analyze_model(results_root: Path, model: str) -> dict[str, Any]:
    run = results_root / MODEL_RUNS[model]
    config = read_json(run / "config.snapshot.json")
    item_ids = config["selector"]["item_ids"]
    official = {arm: arm_outcomes(run, model, arm) for arm in ARMS}
    screens: dict[str, dict[str, Any]] = {}
    source_states: dict[str, dict[str, Any]] = {}
    rows = []
    continuation_pairs = []
    source_reductions = []

    for item in item_ids:
        root = run / "jobs" / f"{model}__{item}"
        screen_path = root / "screen.json"
        state_path = root / "source" / "state.json"
        screen = read_json(screen_path) if screen_path.is_file() else {}
        state = read_json(state_path) if state_path.is_file() else {}
        screens[item] = screen
        source_states[item] = state
        clean_observed = official[ARMS[0]].get(item)
        compact_observed = official[ARMS[1]].get(item)
        clean = clean_observed is True
        compact = compact_observed is True
        if clean and compact:
            quality_class = "both_pass"
        elif clean:
            quality_class = "clean_only"
        elif compact:
            quality_class = "compact_only"
        else:
            quality_class = "both_fail"
        rows.append(
            {
                "instance_id": item,
                "officially_submitted": item in official[ARMS[0]]
                and item in official[ARMS[1]],
                "fidelity_eligible": screen.get("eligible") is True,
                "clean_pass": clean,
                "compact_pass": compact,
                "quality_class": quality_class,
                "source_status": state.get("status", "absent"),
                "source_reason": state.get("reason"),
            }
        )

        if screen.get("eligible") is True:
            source_turns = turn_records(root / "source" / "turns")
            if len(source_turns) == 3:
                raw = sum(
                    turn["rewrite"]["automatic_gate"]["raw_tokens"]
                    for turn in source_turns
                )
                compact_source = sum(
                    turn["rewrite"]["automatic_gate"]["state_tokens"]
                    for turn in source_turns
                )
                source_reductions.append(
                    100 * (compact_source - raw) / raw if raw else None
                )

        turns = {
            arm: turn_records(root / "pair" / "arms" / arm / "turns")
            for arm in ARMS
        }
        if turns[ARMS[0]] and len(turns[ARMS[0]]) == len(turns[ARMS[1]]):
            totals = {}
            for arm in ARMS:
                totals[arm] = sum(
                    turn["usage"]["completion_tokens_details"]["reasoning_tokens"]
                    for turn in turns[arm]
                )
            continuation_pairs.append(
                {
                    "instance_id": item,
                    "turns_per_arm": len(turns[ARMS[0]]),
                    "clean_reasoning_tokens": totals[ARMS[0]],
                    "compact_reasoning_tokens": totals[ARMS[1]],
                    "difference_tokens": totals[ARMS[1]] - totals[ARMS[0]],
                    "percent_change": (
                        100 * (totals[ARMS[1]] - totals[ARMS[0]]) / totals[ARMS[0]]
                        if totals[ARMS[0]]
                        else None
                    ),
                }
            )

    differences = [
        int(row["compact_pass"]) - int(row["clean_pass"])
        for row in rows
        if row["officially_submitted"]
    ]
    clean_count = sum(row["clean_pass"] for row in rows)
    compact_count = sum(row["compact_pass"] for row in rows)
    percent_changes = [
        row["percent_change"]
        for row in continuation_pairs
        if row["percent_change"] is not None
    ]
    source_percent_changes = [value for value in source_reductions if value is not None]
    failure_reasons = Counter()
    for state in source_states.values():
        if state.get("status") == "source_complete":
            continue
        error = str(state.get("error") or "")
        if "provider-visible reasoning" in error:
            reason = "missing_provider_visible_reasoning"
        elif "exactly one tool" in error:
            reason = "wrong_tool_call_count"
        elif "HTTP 429" in error:
            reason = "returned_http_429"
        else:
            reason = str(state.get("reason") or state.get("status") or "absent")
        failure_reasons[reason] += 1
    profile = config["models"][model]
    return {
        "model": model,
        "label": LABELS[model],
        "run": MODEL_RUNS[model],
        "route": f"{profile['expected_provider']} / {profile['model']}",
        "assigned_items": 30,
        "source_complete": sum(
            state.get("status") == "source_complete" for state in source_states.values()
        ),
        "source_ineligible_or_absent": sum(
            state.get("status") != "source_complete" for state in source_states.values()
        ),
        "source_failure_reasons": dict(sorted(failure_reasons.items())),
        "fidelity_eligible": sum(
            screen.get("eligible") is True for screen in screens.values()
        ),
        "officially_submitted_per_arm": len(official[ARMS[0]]),
        "clean_pass_count": clean_count,
        "compact_pass_count": compact_count,
        "clean_pass_rate_assigned": clean_count / 30,
        "compact_pass_rate_assigned": compact_count / 30,
        "compact_minus_clean_pp_assigned": 100 * (compact_count - clean_count) / 30,
        "exploratory_point_margin_pp": config["screening_noninferiority_margin_pp"],
        "exploratory_point_margin_cleared": (
            100 * (compact_count - clean_count) / 30
            >= config["screening_noninferiority_margin_pp"]
            if len(official[ARMS[0]]) == 30 and len(official[ARMS[1]]) == 30
            else None
        ),
        "paired_interval_officially_submitted": normal_interval(differences),
        "quality_counts_all_assigned": {
            name: sum(row["quality_class"] == name for row in rows)
            for name in ("both_pass", "clean_only", "compact_only", "both_fail")
        },
        "source_compaction_percent_change": {
            "n": len(source_percent_changes),
            "median": median(source_percent_changes),
            "full_distribution": source_percent_changes,
        },
        "continuation_reasoning_tokens": {
            "metric": "provider-reported reasoning tokens; not hidden compute",
            "equal_nonzero_horizon_pairs": len(percent_changes),
            "median_percent_change": median(percent_changes),
            "repository_clustered_median_interval": clustered_median_interval(
                continuation_pairs,
                seed=int(hashlib.sha256(model.encode()).hexdigest()[:16], 16),
            ),
            "shorter_count": sum(value < 0 for value in percent_changes),
            "full_distribution": percent_changes,
            "items": continuation_pairs,
        },
        "items": rows,
    }


def write_plots(models: list[dict[str, Any]], output: Path) -> None:
    labels = [model["label"] for model in models]
    x = list(range(len(labels)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [value - width / 2 for value in x],
        [100 * model["clean_pass_rate_assigned"] for model in models],
        width,
        label="Clean",
        color="#9c8f83",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [100 * model["compact_pass_rate_assigned"] for model in models],
        width,
        label="Compact",
        color="#4e6f8e",
    )
    ax.set_ylabel("SWE-bench Verified success (% of 30 assigned)")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, 20)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "quality-pass-rates.png", dpi=180)
    plt.close(fig)

    medians = [model["continuation_reasoning_tokens"]["median_percent_change"] for model in models]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4e6f8e" if value is not None and value < 0 else "#b06f5f" for value in medians]
    ax.bar(x, [value or 0 for value in medians], color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Median paired reasoning-token change (%)")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    for index, model in enumerate(models):
        count = model["continuation_reasoning_tokens"]["equal_nonzero_horizon_pairs"]
        value = medians[index] or 0
        ax.text(index, value - 2 if value < 0 else value + 2, f"n={count}", ha="center", va="top" if value < 0 else "bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "reasoning-token-change.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = [analyze_model(args.results_root, model) for model in MODEL_RUNS]
    synthesis = {
        "schema_version": 1,
        "label": "PROSPECTIVE_EXPLORATORY_30_ITEM_SWE_SCREEN",
        "actual_human_review_performed": False,
        "requested_items_per_route": 30,
        "models": {model["model"]: model for model in models},
        "limitations": [
            "This is a deterministic 30-item screen, not full-benchmark non-inferiority evidence.",
            "All acquisition, fidelity, transport, fork, and evaluator failures remain failures in the assigned denominator.",
            "Reasoning tokens are provider-reported visible accounting and do not measure hidden compute.",
            "Equal failures do not establish semantic quality preservation.",
            "DeepSeek V4 Flash official evaluation used a derived 19-item prediction file with empty patches for two preserved pair-fork infrastructure failures; the other 11 assigned items failed before eligibility.",
        ],
    }
    (args.output / "analysis.json").write_text(json.dumps(synthesis, indent=2) + "\n")
    with (args.output / "model-summary.csv").open("w", newline="") as handle:
        fields = [
            "model",
            "route",
            "source_complete",
            "fidelity_eligible",
            "officially_submitted_per_arm",
            "clean_pass_count",
            "compact_pass_count",
            "compact_minus_clean_pp_assigned",
            "source_compaction_median_percent_change",
            "continuation_equal_horizon_pairs",
            "continuation_reasoning_median_percent_change",
            "continuation_reasoning_shorter_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in models:
            writer.writerow(
                {
                    "model": model["model"],
                    "route": model["route"],
                    "source_complete": model["source_complete"],
                    "fidelity_eligible": model["fidelity_eligible"],
                    "officially_submitted_per_arm": model["officially_submitted_per_arm"],
                    "clean_pass_count": model["clean_pass_count"],
                    "compact_pass_count": model["compact_pass_count"],
                    "compact_minus_clean_pp_assigned": model["compact_minus_clean_pp_assigned"],
                    "source_compaction_median_percent_change": model["source_compaction_percent_change"]["median"],
                    "continuation_equal_horizon_pairs": model["continuation_reasoning_tokens"]["equal_nonzero_horizon_pairs"],
                    "continuation_reasoning_median_percent_change": model["continuation_reasoning_tokens"]["median_percent_change"],
                    "continuation_reasoning_shorter_count": model["continuation_reasoning_tokens"]["shorter_count"],
                }
            )
    write_plots(models, args.output)


if __name__ == "__main__":
    main()
