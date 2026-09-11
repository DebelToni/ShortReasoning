from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("archive_smoke", ROOT / "scripts/run_archive_smoke.py")
assert SPEC and SPEC.loader
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class ArchiveTests(unittest.TestCase):
    def test_required_structure(self) -> None:
        SMOKE.check_structure()

    def test_complete_manifest(self) -> None:
        self.assertGreater(SMOKE.check_manifest(), 100)

    def test_salient_metrics(self) -> None:
        SMOKE.check_metrics()

    def test_figure_source_data(self) -> None:
        self.assertGreaterEqual(SMOKE.check_figure_data(), 1)

    def test_figure_protocol_labels_and_sources(self) -> None:
        source = (ROOT / "scripts/make_archive_figures.py").read_text()
        for label in (
            "rewrite + checks", "rewrite before result",
            "fidelity gate after shared result", "before continuations", "Most paired effects are negative",
            "Exploratory rewrite-style comparison",
        ):
            self.assertIn(label, source)
        metadata = SMOKE.load("figures/source-data/active-figure-values.json")["panel_metadata"]
        self.assertEqual(
            metadata["controlled_summary_a"]["cohort"],
            "retry-augmented initial 72 assigned pairs",
        )
        self.assertEqual(
            metadata["length_quality"]["compact_comparator"],
            "corrected primary overall",
        )

    def test_json_and_python_parse(self) -> None:
        json_files, python_files = SMOKE.check_json_and_python()
        self.assertGreater(json_files, 100)
        self.assertGreater(python_files, 10)


if __name__ == "__main__":
    unittest.main()
