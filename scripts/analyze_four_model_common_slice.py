#!/usr/bin/env python3
"""Bind four controlled model cohorts and their exact common complete-case slice."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "20260730-four-model-common-slice-v1"
SOURCES = {
    "controlled": (
        ROOT / "results/20260724-v2-12-task-two-route-screen/analysis.json",
        "94bca6674c802dd9824b50d402825e2da7b53fa29a3c691321d07d84b08cfb6c",
    ),
    "extension_v1": (
        ROOT / "results/20260730-four-model-paired-replication-v1/analysis.json",
        "7d2c40fe64961f34d6dbc1080069cd60c3be153d154968d1b4fd656f81dd1060",
    ),
    "glm_successor": (
        ROOT / "results/20260730-glm-four-model-replication-v2/analysis.json",
        "e6ee3a8404667172a8be16b16c2d0cb4c326b4431ef09b3d604697cc8f3ffc7c",
    ),
    "extension_v1_audit": (
        ROOT / "results/20260730-four-model-paired-replication-v1/source-audit.json",
        "4877a9dd0519bb26959f99bf90a8785517e143282e808d8557754112ec35e573",
    ),
    "glm_successor_audit": (
        ROOT / "results/20260730-glm-four-model-replication-v2/source-audit.json",
        "05b89d458e61bfe841c201053d972d670955faff1472438cfcb67b58d27a54db",
    ),
}
PRIMARY_MODELS = {
    "deepseek-v4-flash": "controlled",
    "laguna-s-2.1": "controlled",
    "minimax-m3": "extension_v1",
    "glm-4.7-flash": "glm_successor",
}
DISPLAY_NAMES = {
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "laguna-s-2.1": "Laguna S 2.1",
    "minimax-m3": "MiniMax M3",
    "glm-4.7-flash": "GLM-4.7-Flash",
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


def selected_rows(loaded: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        model: [
            row
            for row in loaded[source_name]["cases"]
            if row["model_name"] == model
        ]
        for model, source_name in PRIMARY_MODELS.items()
    }


def effect_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [row for row in rows if row["comparable_horizon"]]
    effects = [row["reasoning_token_change_percent"] for row in comparable]
    return {
        "cases": len(rows),
        "comparable_horizon_cases": len(comparable),
        "reasoning_reduced_cases": sum(value < 0 for value in effects),
        "reasoning_equal_cases": sum(value == 0 for value in effects),
        "reasoning_increased_cases": sum(value > 0 for value in effects),
        "median_reasoning_token_change_percent": median(effects),
        "clean_successes": sum(row["clean_success"] for row in rows),
        "rewritten_successes": sum(row["rewritten_success"] for row in rows),
    }


def common_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effects = [row["reasoning_token_change_percent"] for row in rows]
    clean = [row["clean_reasoning_tokens"] for row in rows]
    rewritten = [row["rewritten_reasoning_tokens"] for row in rows]
    return {
        "pairs": len(rows),
        "clean_median_reasoning_tokens": median(clean),
        "rewritten_median_reasoning_tokens": median(rewritten),
        "median_reasoning_token_change_percent": median(effects),
        "reasoning_reduced_cases": sum(value < 0 for value in effects),
        "clean_successes": sum(row["clean_success"] for row in rows),
        "rewritten_successes": sum(row["rewritten_success"] for row in rows),
        "pooled_clean_reasoning_tokens": sum(clean),
        "pooled_rewritten_reasoning_tokens": sum(rewritten),
    }


def audit_counts(audit: dict[str, Any], model: str) -> dict[str, Any]:
    rows = [row for row in audit["records"] if row["model_name"] == model]
    return {
        "requested_sources": len(rows),
        "selected_sources": sum(row["status"] == "audited" for row in rows),
        "eligible_sources": sum(row["eligible"] for row in rows),
        "actual_human_review_performed": audit["actual_human_review_performed"],
    }


def build_analysis() -> dict[str, Any]:
    loaded = load_sources()
    rows_by_model = selected_rows(loaded)
    common_keys = set.intersection(
        *(
            {
                (row["task_id"], row["replicate"])
                for row in rows
                if row["comparable_horizon"]
            }
            for rows in rows_by_model.values()
        )
    )
    common_rows = {
        model: [
            row
            for row in rows
            if (row["task_id"], row["replicate"]) in common_keys
        ]
        for model, rows in rows_by_model.items()
    }
    full = {}
    for model, source_name in PRIMARY_MODELS.items():
        source_summary = loaded[source_name]["by_model"][model]
        full[model] = {
            **effect_summary(rows_by_model[model]),
            "display_name": DISPLAY_NAMES[model],
            "cluster_count": source_summary["clustered_bootstrap"]["cluster_count"],
            "median_effect_percent_ci95": source_summary["clustered_bootstrap"][
                "median_effect_percent_ci95"
            ],
        }
    full["minimax-m3"]["source_coverage"] = audit_counts(
        loaded["extension_v1_audit"], "minimax-m3"
    )
    full["glm-4.7-flash"]["source_coverage"] = audit_counts(
        loaded["glm_successor_audit"], "glm-4.7-flash"
    )
    sensitivity_source = loaded["extension_v1"]
    sensitivity_rows = [
        row for row in sensitivity_source["cases"] if row["model_name"] == "glm-4.7-flash"
    ]
    sensitivity = {
        **effect_summary(sensitivity_rows),
        "cluster_count": sensitivity_source["by_model"]["glm-4.7-flash"][
            "clustered_bootstrap"
        ]["cluster_count"],
        "median_effect_percent_ci95": sensitivity_source["by_model"]["glm-4.7-flash"][
            "clustered_bootstrap"
        ]["median_effect_percent_ci95"],
        "source_coverage": audit_counts(loaded["extension_v1_audit"], "glm-4.7-flash"),
        "role": "sensitivity_only_not_pooled_with_glm_successor",
    }
    return {
        "schema_version": 1,
        "analysis_code_sha256": sha256(Path(__file__)),
        "source_bindings": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": expected}
            for name, (path, expected) in SOURCES.items()
        },
        "contract": {
            "primary_models": PRIMARY_MODELS,
            "glm_v1_role": "sensitivity_only",
            "pool_glm_v1_with_successor": False,
            "common_slice_requires_comparable_horizon_for_all_four_models": True,
        },
        "full_primary_cohorts": full,
        "exact_four_model_common_slice": {
            "pairs_per_model": len(common_keys),
            "task_count": len({task for task, _ in common_keys}),
            "task_replicate_keys": [
                {"task_id": task, "replicate": replicate}
                for task, replicate in sorted(common_keys)
            ],
            "by_model": {
                model: {**common_summary(rows), "display_name": DISPLAY_NAMES[model]}
                for model, rows in common_rows.items()
            },
            "cases": [
                {
                    "model_name": model,
                    "task_id": row["task_id"],
                    "replicate": row["replicate"],
                    "clean_reasoning_tokens": row["clean_reasoning_tokens"],
                    "rewritten_reasoning_tokens": row["rewritten_reasoning_tokens"],
                    "reasoning_token_change_percent": row[
                        "reasoning_token_change_percent"
                    ],
                    "clean_success": row["clean_success"],
                    "rewritten_success": row["rewritten_success"],
                }
                for model, rows in common_rows.items()
                for row in rows
            ],
        },
        "glm_v1_sensitivity": sensitivity,
    }


def report(analysis: dict[str, Any]) -> str:
    lines = [
        "# Four-model controlled synthesis",
        "",
        "Full cohorts retain their prospective source contracts; GLM V1 is sensitivity-only.",
        "",
        "| Model | Eligible/Requested sources | Comparable | Shorter | Median [95% CI] | Success C→R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in analysis["full_primary_cohorts"].items():
        coverage = row.get("source_coverage")
        source_text = (
            f'{coverage["eligible_sources"]}/{coverage["requested_sources"]}'
            if coverage else "12/12"
        )
        low, high = row["median_effect_percent_ci95"]
        lines.append(
            f'| {row["display_name"]} | {source_text} | '
            f'{row["comparable_horizon_cases"]}/{row["cases"]} | '
            f'{row["reasoning_reduced_cases"]} | '
            f'{row["median_reasoning_token_change_percent"]:.1f}% '
            f'[{low:.1f}%, {high:.1f}%] | '
            f'{row["clean_successes"]}/{row["cases"]}→'
            f'{row["rewritten_successes"]}/{row["cases"]} |'
        )
    common = analysis["exact_four_model_common_slice"]
    lines += [
        "",
        f'Exact common complete-case slice: {common["pairs_per_model"]} task/replicate keys '
        f'across {common["task_count"]} tasks.',
        "",
        "| Model | Clean median | Compact median | Paired median change | Success C→R |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in common["by_model"].values():
        lines.append(
            f'| {row["display_name"]} | {row["clean_median_reasoning_tokens"]:.0f} | '
            f'{row["rewritten_median_reasoning_tokens"]:.0f} | '
            f'{row["median_reasoning_token_change_percent"]:.1f}% | '
            f'{row["clean_successes"]}/{row["pairs"]}→'
            f'{row["rewritten_successes"]}/{row["pairs"]} |'
        )
    sensitivity = analysis["glm_v1_sensitivity"]
    low, high = sensitivity["median_effect_percent_ci95"]
    lines += [
        "",
        "GLM V1 sensitivity (not pooled): "
        f'{sensitivity["reasoning_reduced_cases"]}/'
        f'{sensitivity["comparable_horizon_cases"]} comparable pairs shortened; '
        f'median {sensitivity["median_reasoning_token_change_percent"]:.1f}% '
        f'[{low:.1f}%, {high:.1f}%].',
        "",
        "The common slice is descriptive raw-token accounting, not a quality-adjusted leaderboard.",
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
    manifest = args.output / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n" for path in (analysis_path, report_path)
        )
    )
    print(report(analysis), end="")


if __name__ == "__main__":
    main()
