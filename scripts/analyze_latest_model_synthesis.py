#!/usr/bin/env python3
"""Bind six controlled model cohorts and exact cross-model/version intersections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "20260730-six-model-common-slice-v1"
SOURCES = {
    "controlled": (
        ROOT / "results/20260724-v2-12-task-two-route-screen/analysis.json",
        "94bca6674c802dd9824b50d402825e2da7b53fa29a3c691321d07d84b08cfb6c",
    ),
    "extension_v1": (
        ROOT / "results/20260730-four-model-paired-replication-v1/analysis.json",
        "7d2c40fe64961f34d6dbc1080069cd60c3be153d154968d1b4fd656f81dd1060",
    ),
    "glm47_successor": (
        ROOT / "results/20260730-glm-four-model-replication-v2/analysis.json",
        "e6ee3a8404667172a8be16b16c2d0cb4c326b4431ef09b3d604697cc8f3ffc7c",
    ),
    "latest": (
        ROOT / "results/20260730-latest-model-paired-replication-v1/analysis.json",
        "09a59cecc731c3a9ecc1a8deed44b9d7fdba5f37e408ff5934fc5a392f13039a",
    ),
    "latest_audit": (
        ROOT / "results/20260730-latest-model-paired-replication-v1/source-audit.json",
        "4bba170541b07983424ec697bd52b88203383adc8e9070417c86f8776b729ff1",
    ),
    "delivery_v5": (
        ROOT / "results/20260730-multimodel-history-delivery-v5/analysis.json",
        "6a6da7763e73988ecd6fc11b7434c724de0a7bda2e8cd76df45557bda70a26af",
    ),
    "delivery_v6": (
        ROOT / "results/20260730-multimodel-history-delivery-v6/analysis.json",
        "bc1f2bedafbbc26ae1530c0b8dc9b30462fbfc6d60e09e62769de1cbfc3980f0",
    ),
}
MODEL_SOURCES = {
    "deepseek-v4-flash": "controlled",
    "laguna-s-2.1": "controlled",
    "minimax-m3": "extension_v1",
    "glm-4.7-flash": "glm47_successor",
    "glm-5.1": "latest",
    "deepseek-v4-pro": "latest",
}
DISPLAY = {
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "laguna-s-2.1": "Laguna S 2.1",
    "minimax-m3": "MiniMax M3",
    "glm-4.7-flash": "GLM-4.7-Flash",
    "glm-5.1": "GLM 5.1",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}
ROUTES = {
    "deepseek-v4-flash": "DeepInfra",
    "laguna-s-2.1": "Poolside",
    "minimax-m3": "DeepInfra",
    "glm-4.7-flash": "DeepInfra",
    "glm-5.1": "Z.AI",
    "deepseek-v4-pro": "Together",
}
SOURCE_COVERAGE = {
    "deepseek-v4-flash": (12, 12),
    "laguna-s-2.1": (12, 12),
    "minimax-m3": (10, 12),
    "glm-4.7-flash": (5, 12),
    "glm-5.1": (10, 12),
    "deepseek-v4-pro": (9, 12),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources() -> dict[str, Any]:
    loaded = {}
    for name, (path, expected) in SOURCES.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"{name} source changed: {observed}")
        loaded[name] = json.loads(path.read_text())
    return loaded


def model_rows(loaded: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        model: [
            row
            for row in loaded[source]["cases"]
            if row["model_name"] == model
        ]
        for model, source in MODEL_SOURCES.items()
    }


def keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (row["task_id"], row["replicate"])
        for row in rows
        if row["comparable_horizon"]
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [row for row in rows if row["comparable_horizon"]]
    effects = [row["reasoning_token_change_percent"] for row in comparable]
    return {
        "cases": len(rows),
        "comparable_horizon_cases": len(comparable),
        "reasoning_reduced_cases": sum(value < 0 for value in effects),
        "reasoning_equal_cases": sum(value == 0 for value in effects),
        "reasoning_increased_cases": sum(value > 0 for value in effects),
        "median_reasoning_token_change_percent": median(effects),
        "mean_reasoning_token_change_percent": sum(effects) / len(effects),
        "clean_successes": sum(row["clean_success"] for row in rows),
        "rewritten_successes": sum(row["rewritten_success"] for row in rows),
    }


def summarize_slice(rows: list[dict[str, Any]], common: set[tuple[str, int]]) -> dict[str, Any]:
    selected = [
        row for row in rows if (row["task_id"], row["replicate"]) in common
    ]
    effects = [row["reasoning_token_change_percent"] for row in selected]
    clean = [row["clean_reasoning_tokens"] for row in selected]
    compact = [row["rewritten_reasoning_tokens"] for row in selected]
    return {
        "pairs": len(selected),
        "reasoning_reduced_cases": sum(value < 0 for value in effects),
        "clean_median_reasoning_tokens": median(clean),
        "compact_median_reasoning_tokens": median(compact),
        "median_reasoning_token_change_percent": median(effects),
        "clean_successes": sum(row["clean_success"] for row in selected),
        "compact_successes": sum(row["rewritten_success"] for row in selected),
    }


def exact_slice(
    rows_by_model: dict[str, list[dict[str, Any]]], models: tuple[str, ...]
) -> dict[str, Any]:
    common = set.intersection(*(keys(rows_by_model[model]) for model in models))
    return {
        "models": list(models),
        "pairs_per_model": len(common),
        "task_count": len({task for task, _replicate in common}),
        "task_replicate_keys": [
            {"task_id": task, "replicate": replicate}
            for task, replicate in sorted(common)
        ],
        "by_model": {
            model: {
                "display_name": DISPLAY[model],
                "provider": ROUTES[model],
                **summarize_slice(rows_by_model[model], common),
            }
            for model in models
        },
    }


def delivery_exclusion(loaded: dict[str, Any]) -> dict[str, Any]:
    v6 = next(
        row
        for row in loaded["delivery_v6"]["all_confirmed_configurations"]
        if row["model"] == "z-ai/glm-5.2"
    )
    return {
        "model": "z-ai/glm-5.2",
        "behavioral_cases": 0,
        "reason": "no tested native-reasoning route passed the frozen delivery gate",
        "best_confirmed_route": v6["route_id"],
        "reasoning_hits": v6["reasoning_hits"],
        "reasoning_trials": v6["reasoning_trials"],
        "visible_hits": v6["visible_hits"],
        "visible_trials": v6["visible_trials"],
        "absent_false_hits": v6["absent_false_hits"],
        "absent_trials": v6["absent_trials"],
        "promotion_gate_passed": v6["promotion_gate_passed"],
    }


def build_analysis() -> dict[str, Any]:
    loaded = load_sources()
    rows_by_model = model_rows(loaded)
    full = {}
    for model, rows in rows_by_model.items():
        source_summary = loaded[MODEL_SOURCES[model]]["by_model"][model]
        eligible, requested = SOURCE_COVERAGE[model]
        full[model] = {
            "display_name": DISPLAY[model],
            "provider": ROUTES[model],
            "eligible_sources": eligible,
            "requested_sources": requested,
            **summarize_rows(rows),
            "cluster_count": source_summary["clustered_bootstrap"]["cluster_count"],
            "median_effect_percent_ci95": source_summary["clustered_bootstrap"][
                "median_effect_percent_ci95"
            ],
        }
    all_models = tuple(MODEL_SOURCES)
    return {
        "schema_version": 1,
        "analysis_code_sha256": sha256(Path(__file__)),
        "source_bindings": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": expected}
            for name, (path, expected) in SOURCES.items()
        },
        "contract": {
            "full_cohorts_are_not_pooled": True,
            "common_slices_require_comparable_horizon_for_every_listed_model": True,
            "version_comparisons_are_confounded_with_provider": True,
            "fidelity_review_is_model_review_not_human_review": True,
        },
        "full_primary_cohorts": full,
        "excluded_before_behavior": {"glm-5.2": delivery_exclusion(loaded)},
        "exact_six_model_common_slice": exact_slice(rows_by_model, all_models),
        "exact_deepseek_version_slice": exact_slice(
            rows_by_model, ("deepseek-v4-flash", "deepseek-v4-pro")
        ),
        "exact_glm_version_slice": exact_slice(
            rows_by_model, ("glm-4.7-flash", "glm-5.1")
        ),
    }


def report(analysis: dict[str, Any]) -> str:
    lines = [
        "# Six-model controlled synthesis",
        "",
        "| Model / exact route | Eligible sources | Comparable | Shorter | Median [95% CI] | Success C→R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["full_primary_cohorts"].values():
        low, high = row["median_effect_percent_ci95"]
        lines.append(
            f'| {row["display_name"]} / {row["provider"]} | '
            f'{row["eligible_sources"]}/{row["requested_sources"]} | '
            f'{row["comparable_horizon_cases"]}/{row["cases"]} | '
            f'{row["reasoning_reduced_cases"]} | '
            f'{row["median_reasoning_token_change_percent"]:.1f}% '
            f'[{low:.1f}%, {high:.1f}%] | '
            f'{row["clean_successes"]}/{row["cases"]}→'
            f'{row["rewritten_successes"]}/{row["cases"]} |'
        )
    exclusion = analysis["excluded_before_behavior"]["glm-5.2"]
    lines += [
        "",
        "GLM 5.2 has no behavioral row: its best confirmed native-history route recovered "
        f'{exclusion["reasoning_hits"]}/{exclusion["reasoning_trials"]} reasoning-only actions, '
        "below the frozen 3/4 gate.",
    ]
    for title, key in (
        ("Exact six-model common slice", "exact_six_model_common_slice"),
        ("DeepSeek Flash/Pro exact version slice", "exact_deepseek_version_slice"),
        ("GLM 4.7/5.1 exact version slice", "exact_glm_version_slice"),
    ):
        block = analysis[key]
        lines += [
            "",
            f'## {title} ({block["pairs_per_model"]} keys, {block["task_count"]} tasks)',
            "",
            "| Model | Provider | Clean median | Compact median | Paired median | Shorter | Success C→R |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in block["by_model"].values():
            lines.append(
                f'| {row["display_name"]} | {row["provider"]} | '
                f'{row["clean_median_reasoning_tokens"]:.1f} | '
                f'{row["compact_median_reasoning_tokens"]:.1f} | '
                f'{row["median_reasoning_token_change_percent"]:.1f}% | '
                f'{row["reasoning_reduced_cases"]}/{row["pairs"]} | '
                f'{row["clean_successes"]}/{row["pairs"]}→'
                f'{row["compact_successes"]}/{row["pairs"]} |'
            )
    lines += [
        "",
        "Common-slice raw tokens are descriptive, not a quality-adjusted leaderboard.",
        "Version slices also change provider and therefore do not isolate checkpoint version.",
        "Source fidelity was model-reviewed, not human-reviewed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to replace {args.output}")
    analysis = build_analysis()
    args.output.mkdir(parents=True)
    analysis_path = args.output / "analysis.json"
    report_path = args.output / "report.md"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    report_path.write_text(report(analysis))
    manifest_path = args.output / "SHA256SUMS"
    manifest_path.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n" for path in (analysis_path, report_path)
        )
    )
    print(report(analysis), end="")


if __name__ == "__main__":
    main()
