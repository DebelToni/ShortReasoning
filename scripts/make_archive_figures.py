#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ROUTES = (
    "deepseek-v4-flash", "laguna-s-2.1", "minimax-m3",
    "glm-4.7-flash", "glm-5.1", "deepseek-v4-pro",
)


def load(root: Path, relative: str):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_data(root: Path) -> dict:
    pair_path = root / "results/20260910-camera-ready-controlled-audit-v2/per_pair_retry_augmented.json"
    controlled_path = root / "results/20260910-camera-ready-controlled-audit-v2/analysis.json"
    style_path = root / "results/20260726-deepseek-five-tier-luna-v6-manual-style-adjudication/analysis.json"
    controls_path = root / "results/20260725-frozen-history-controls/analysis.json"
    corrected_path = root / "results/20260804-six-model-common-slice-infrastructure-corrected-v2/analysis.json"
    pairs = json.loads(pair_path.read_text())["pairs"]
    controlled = json.loads(controlled_path.read_text())
    style = json.loads(style_path.read_text())
    controls = json.loads(controls_path.read_text())
    corrected = json.loads(corrected_path.read_text())

    initial = [
        row for row in pairs
        if row["route"] in {"deepseek-v4-flash", "laguna-s-2.1"}
        and row["comparable_horizon"]
    ]
    if len(initial) != 72:
        raise RuntimeError(f"retry-augmented initial cohort changed: {len(initial)} pairs")
    turn_reductions = []
    for turn in range(3):
        reached = [
            row for row in initial
            if len(row["clean"]["reasoning_tokens_by_turn"]) > turn
            and len(row["compact"]["reasoning_tokens_by_turn"]) > turn
        ]
        clean = sum(row["clean"]["reasoning_tokens_by_turn"][turn] for row in reached)
        compact = sum(row["compact"]["reasoning_tokens_by_turn"][turn] for row in reached)
        turn_reductions.append({
            "turn": turn + 1, "pairs": len(reached),
            "clean_tokens": clean, "compact_tokens": compact,
            "reduction_percent": 100 * (clean - compact) / clean,
        })

    route_data = {}
    for route in ROUTES:
        route_pairs = [row for row in pairs if row["route"] == route and row["comparable_horizon"]]
        summary = controlled["by_route"][route]
        route_data[route] = {
            "values": [row["reasoning_change_percent"] for row in route_pairs],
            "median": summary["comparable_trajectory_reasoning"]["paired_change_percent"]["median"],
            "bootstrap_ci95": summary["source_cluster_bootstrap"]["ci95"]["comparable_median_change_percent"],
            "shorter_percent": 100 * sum(row["reasoning_delta_tokens"] < 0 for row in route_pairs) / len(route_pairs),
        }

    style_keys = (
        ("Clean", None),
        ("Casual", "clean__to__relaxed_full_sentence_compact"),
        ("Short", "clean__to__full_sentence_compact"),
        ("Telegraphic", "clean__to__telegraphic_compact"),
        ("Ultra", "clean__to__ultra_telegraphic"),
    )
    style_data = []
    for label, key in style_keys:
        if key is None:
            style_data.append({"label": label, "reduction_percent": 0.0, "ci95": [0.0, 0.0]})
        else:
            row = style["clean_contrasts"][key]
            style_data.append({
                "label": label,
                "reduction_percent": -row["median_percent_change"],
                "ci95": [-row["source_clustered_95_ci"][1], -row["source_clustered_95_ci"][0]],
            })

    primary = corrected["corrected_primary_overall"]
    verbose = controls["controls"]["verbose_paraphrase"]["overall"]
    length_quality = [
        {
            "label": "Compact rewrite",
            "reasoning_change_percent": primary["median_change_percent"],
            "reasoning_ci95": primary["clustered_bootstrap"]["median_effect_percent_ci95"],
            "success_change_pp": 100 * (primary["rewritten_successes"] - primary["clean_successes"]) / primary["cases"],
        },
        {
            "label": "Length matched",
            "reasoning_change_percent": verbose["median_reasoning_token_change_percent"],
            "reasoning_ci95": verbose["clustered_bootstrap"]["median_effect_percent_ci95"],
            "success_change_pp": verbose["paired_success_difference_percentage_points"],
        },
    ]
    inputs = (pair_path, controlled_path, style_path, controls_path, corrected_path)
    return {
        "schema_version": 2,
        "input_sha256": {path.relative_to(root).as_posix(): file_sha256(path) for path in inputs},
        "panel_metadata": {
            "controlled_summary_a": {
                "cohort": "retry-augmented initial 72 assigned pairs",
                "aggregation": "pooled turn-specific reasoning-token totals",
                "displayed_reductions_percent": [8.8, 34.4, 56.1],
                "corrected_primary_source": corrected_path.relative_to(root).as_posix(),
                "pair_records_source": pair_path.relative_to(root).as_posix(),
            },
            "length_quality": {
                "compact_comparator": "corrected primary overall",
                "compact_source": corrected_path.relative_to(root).as_posix(),
                "length_matched_source": controls_path.relative_to(root).as_posix(),
            },
            "paired_fork_protocol": {
                "rewrite_timing": "rewrite and mechanical checks precede shared-result acquisition",
                "semantic_review_timing": "semantic acceptance follows shared source acquisition and precedes continuation-fork sampling",
            },
        },
        "turn_reductions": turn_reductions,
        "route_distributions": route_data,
        "style_dose": style_data,
        "length_quality": length_quality,
    }


def save(fig, output: Path, name: str) -> Path:
    path = output / name
    fig.savefig(
        path, bbox_inches="tight",
        metadata={"Creator": "Anonymous artifact", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return path


def write_source_data(data_dir: Path, data: dict, pairs: list[dict]) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "active-figure-values.json"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = data_dir / "controlled-effect-distributions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model_name", "task_id", "replicate", "comparable_horizon", "reasoning_change_percent"),
        )
        writer.writeheader()
        for row in pairs:
            writer.writerow({
                "model_name": row["route"], "task_id": row["task_id"],
                "replicate": row["replicate"], "comparable_horizon": row["comparable_horizon"],
                "reasoning_change_percent": row["reasoning_change_percent"],
            })
    return json_path, csv_path


def protocol_figure(output: Path) -> Path:
    import fitz

    # Preserve the original vector artwork, including its embedded Times fonts.
    artwork = ROOT / "figures/source-data/paired-fork-protocol-layout.pdf"
    path = output / "paired-fork-protocol.pdf"
    with fitz.open(artwork) as original, fitz.open() as result:
        source = original[0]
        font_xref = next(row[0] for row in source.get_fonts() if row[3].endswith("TimesNewRomanPSMT"))
        font_data = original.extract_font(font_xref)[3]
        font = fitz.Font(fontbuffer=font_data)
        old_label = source.search_for("rewrite + fidelity gate")[0]
        old_label.y0 = 95.2  # Avoid the overlapping text bounds of "Compact history".
        source.add_redact_annot(old_label, fill=(237 / 255, 233 / 255, 254 / 255))
        source.apply_redactions(images=0, graphics=0)
        source.insert_font(fontname="ProtocolTimes", fontbuffer=font_data)
        label = "rewrite + checks"
        source.insert_text(
            (209.1906 - font.text_length(label, fontsize=6.4) / 2, 99.94962),
            label, fontname="ProtocolTimes", fontsize=6.4,
        )
        page = result.new_page(width=source.rect.width, height=120)
        page.show_pdf_page(source.rect, original, 0)
        page.insert_font(fontname="ProtocolNotes", fontbuffer=font_data)
        for center, baseline, text in (
            (209.1906, 116.4, "rewrite before result"),
            (298.0962, 90.5, "fidelity gate after shared result"),
            (298.0962, 96.8, "before continuations"),
        ):
            page.insert_text(
                (center - font.text_length(text, fontsize=5.4) / 2, baseline),
                text, fontname="ProtocolNotes", fontsize=5.4,
                color=(75 / 255, 85 / 255, 99 / 255),
            )
        result.set_metadata({"creator": "Anonymous artifact", "creationDate": "", "modDate": ""})
        result.save(path, garbage=4, deflate=True)
    return path


def controlled_summary_figure(output: Path, data: dict) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.25))
    turns = data["turn_reductions"]
    values = [row["reduction_percent"] for row in turns]
    axes[0].bar(range(3), values, color="#2563eb", width=.68)
    axes[0].set_xticks(range(3), ["Turn 1", "Turn 2", "Turn 3"])
    axes[0].set_ylabel("Future reasoning\nreduction (%)", labelpad=2)
    axes[0].set_title("(a) Retry-augmented initial 72 pairs\nPooled reduction grows across turns", loc="left", fontweight="bold")
    axes[0].set_ylim(0, 66)
    for index, value in enumerate(values): axes[0].text(index, value + 2, f"{value:.1f}", ha="center", fontsize=7)
    style = data["style_dose"]
    y = [row["reduction_percent"] for row in style]
    lower = [row["reduction_percent"] - row["ci95"][0] for row in style]
    upper = [row["ci95"][1] - row["reduction_percent"] for row in style]
    axes[1].errorbar(range(5), y, yerr=[lower, upper], marker="o", color="#2563eb", capsize=2, lw=1.1)
    axes[1].set_xticks(range(5), [row["label"] for row in style], rotation=18)
    axes[1].set_ylabel("Median reduction\nfrom clean (%)", labelpad=2)
    axes[1].set_title("(b) Exploratory rewrite-style comparison", loc="left", fontweight="bold")
    axes[1].set_ylim(-2, 48)
    for ax in axes: ax.grid(axis="y", alpha=.25); ax.spines[["top","right"]].set_visible(False)
    fig.subplots_adjust(left=.22, right=.98, top=.91, bottom=.14, hspace=.82)
    return save(fig, output, "controlled-summary.pdf")


def raincloud_figure(output: Path, data: dict) -> Path:
    import fitz

    artwork = ROOT / "figures/source-data/controlled-effect-raincloud-layout.pdf"
    path = output / "controlled-effect-raincloud.pdf"
    with fitz.open(artwork) as document:
        page = document[0]
        drawings = page.get_drawings()
        # The original artwork's x-axis maps these points to 0% and 25%.
        zero = 151.42984008789062
        scale = (177.34568786621094 - zero) / 25
        points = sorted([
            d["rect"] for d in drawings if d["fill"] is not None
            and 0 < d["rect"].width < 4 and 0 < d["rect"].height < 4
            and d["items"][0][0] == "c"
        ], key=lambda r: r.y0 + r.y1)
        intervals = sorted([
            d["rect"] for d in drawings if d["color"] and d["color"][0] < .1
            and 8 < d["rect"].width < 100 and d["rect"].height < .001
        ], key=lambda r: r.y0)
        if len(points) != sum(len(data["route_distributions"][r]["values"]) for r in ROUTES) or len(intervals) != len(ROUTES):
            raise RuntimeError("original distribution artwork has incompatible cohorts")
        offset = 0
        for route, interval in zip(ROUTES, intervals):
            row = data["route_distributions"][route]
            values = sorted(row["values"])
            plotted = sorted(((r.x0 + r.x1) / 2 - zero) / scale for r in points[offset:offset + len(values)])
            if any(abs(a - b) > .001 for a, b in zip(values, plotted)):
                raise RuntimeError(f"original distribution points no longer match {route}")
            plotted_ci = [(x - zero) / scale for x in (interval.x0, interval.x1)]
            if any(abs(a - b) > .051 for a, b in zip(row["bootstrap_ci95"], plotted_ci)):
                raise RuntimeError(f"original distribution intervals no longer match {route}")
            offset += len(values)
        title = page.search_for("Effects lie below zero on every route")[0]
        title.y1 = min(title.y1, page.search_for("no change")[0].y0 - .1)
        page.add_redact_annot(title, fill=(1, 1, 1))
        page.apply_redactions(images=0, graphics=0)
        font = Path(mpl.get_data_path()) / "fonts/ttf/DejaVuSerif-Bold.ttf"
        page.insert_font(fontname="DistributionTitle", fontfile=str(font))
        page.insert_text((58.132801, 15.871994), "Most paired effects are negative", fontname="DistributionTitle", fontsize=8.2)
        document.subset_fonts()
        document.set_metadata({"creator": "Paper figure", "creationDate": "", "modDate": ""})
        document.save(path, garbage=4, deflate=True)
    return path


def length_quality_figure(output: Path, data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    colors = ("#2563eb", "#7c3aed")
    for row, color in zip(data["length_quality"], colors):
        x = row["reasoning_change_percent"]; low, high = row["reasoning_ci95"]; y = row["success_change_pp"]
        ax.plot([low, high], [y, y], color=color, lw=2)
        ax.scatter([x], [y], color=color, s=20, zorder=3)
        label = f"{row['label']}\n{x:+.1f}% reasoning, {y:+.1f} pp success"
        if x < -10:
            ax.text(x + 4, y - .7, label, color=color, fontsize=6.5, fontweight="bold")
        else:
            ax.text(x - 1, y - .7, label, color=color, fontsize=6.5, fontweight="bold", ha="right")
    ax.axhline(0, color="#6b7280", lw=.7)
    ax.text(10, .25, "clean baseline", ha="right", color="#4b5563", fontsize=6)
    ax.set_xlim(-52, 12); ax.set_ylim(-16, 1.5)
    ax.set_xlabel("Future reasoning change (%) ← shorter")
    ax.set_ylabel("Exact success change\n(percentage points)")
    ax.grid(alpha=.25); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return save(fig, output, "length-quality-control.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    root = args.archive_root.resolve()
    output = args.output_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve() if args.data_dir else output / "source-data"
    output.mkdir(parents=True, exist_ok=True)
    pair_path = root / "results/20260910-camera-ready-controlled-audit-v2/per_pair_retry_augmented.json"
    pairs = json.loads(pair_path.read_text(encoding="utf-8"))["pairs"]
    data = derive_data(root)
    data_path, csv_path = write_source_data(data_dir, data, pairs)
    generated = []
    if not args.data_only:
        mpl.rcParams.update({
            "font.family": "serif", "font.size": 8, "axes.titlesize": 9,
            "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
        })
        generated = [
            protocol_figure(output), controlled_summary_figure(output, data),
            raincloud_figure(output, data), length_quality_figure(output, data),
        ]
    print(json.dumps({
        "status": "passed", "figures": len(generated), "output": str(output),
        "data": {
            data_path.name: file_sha256(data_path),
            csv_path.name: file_sha256(csv_path),
        },
        "figure_sha256": {path.name: file_sha256(path) for path in generated},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
