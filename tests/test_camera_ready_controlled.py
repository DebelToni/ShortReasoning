from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "camera_ready_controlled", ROOT / "scripts" / "analyze_camera_ready_controlled.py"
)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def synthetic_case(source: str, comparable: bool, clean_reason: list[int], compact_reason: list[int], clean_ok: bool, compact_ok: bool):
    def arm(reason, success):
        turns = len(reason)
        return {
            "success": success,
            "status": "complete",
            "actions_correct": True,
            "final_correct": success,
            "turn_count": turns,
            "input_tokens": 10 * turns,
            "output_tokens": sum(reason) + turns,
            "reasoning_tokens": sum(reason),
            "input_tokens_by_turn": [10] * turns,
            "output_tokens_by_turn": [value + 1 for value in reason],
            "reasoning_tokens_by_turn": reason,
            "last_finish_reason": "stop",
        }

    clean = arm(clean_reason, clean_ok)
    compact = arm(compact_reason, compact_ok)
    return {
        "route": "r",
        "task_id": source,
        "replicate": 1,
        "source_id": f"r|{source}",
        "comparable_horizon": comparable,
        "clean": clean,
        "compact": compact,
        "reasoning_delta_tokens": sum(compact_reason) - sum(clean_reason),
        "reasoning_change_percent": audit.percent(sum(clean_reason), sum(compact_reason)),
        "first_reasoning_delta_tokens": compact_reason[0] - clean_reason[0],
        "first_reasoning_change_percent": audit.percent(clean_reason[0], compact_reason[0]),
        "subsequent_reasoning_delta_tokens": sum(compact_reason[1:]) - sum(clean_reason[1:]),
        "subsequent_reasoning_change_percent": audit.percent(sum(clean_reason[1:]), sum(compact_reason[1:])),
        "input_delta_tokens": compact["input_tokens"] - clean["input_tokens"],
        "output_delta_tokens": compact["output_tokens"] - clean["output_tokens"],
        "turn_delta": compact["turn_count"] - clean["turn_count"],
        "first_request_differs_only_in_assistant_reasoning": True,
        "shared_preceding_tool_result_by_turn": [True] * min(len(clean_reason), len(compact_reason)),
    }


class UnitTests(unittest.TestCase):
    def test_quantile_and_sign_convention(self):
        self.assertEqual(audit.quantile([0, 10, 20], 0.25), 5)
        self.assertEqual(audit.percent(100, 75), -25)

    def test_all_assigned_and_equal_horizon_estimands_are_separate(self):
        cases = [
            synthetic_case("a", True, [10, 10], [5, 5], True, True),
            synthetic_case("b", False, [10, 10], [5], True, False),
        ]
        result = audit.cohort_summary(cases)
        self.assertEqual(result["assigned_pairs"], 2)
        self.assertEqual(result["comparable_horizon_pairs"], 1)
        self.assertEqual(result["comparable_trajectory_reasoning"]["pairs"], 1)
        self.assertEqual(result["full_trajectory_reasoning_all_assigned"]["pairs"], 2)
        self.assertEqual(result["quality_all_assigned"]["clean_successes"], 2)
        self.assertEqual(result["quality_all_assigned"]["compact_successes"], 1)
        self.assertEqual(result["all_assigned_workload"]["turns"]["clean_total_tokens"], 4)
        self.assertEqual(result["all_assigned_workload"]["turns"]["compact_total_tokens"], 3)

    def test_cluster_bootstrap_keeps_source_replicates(self):
        cases = [
            synthetic_case("a", True, [10], [5], True, True),
            synthetic_case("b", True, [10], [20], True, False),
        ]
        cases[1]["replicate"] = 2
        result = audit.bootstrap(cases, resamples=20, seed=7)
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["resamples"], 20)


class RealEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "out"
        cls.analysis = audit.run(
            ROOT,
            Path("results/20260804-six-model-common-slice-infrastructure-corrected-v2"),
            cls.output,
            resamples=50,
        )
        cls.pairs = json.loads(
            (cls.output / "per_pair_retry_augmented.json").read_text()
        )["pairs"]
        cls.original_pairs = json.loads(
            (cls.output / "per_pair_original_first_capture.json").read_text()
        )["pairs"]
        cls.replacements = json.loads(
            (cls.output / "retry_augmentation_records.json").read_text()
        )["records"]
        cls.sources = json.loads((cls.output / "per_source.json").read_text())["sources"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_headlines_are_recomputed_from_pairs(self):
        comparable = [pair for pair in self.pairs if pair["comparable_horizon"]]
        self.assertEqual(len(self.pairs), 174)
        self.assertEqual(len(comparable), 172)
        self.assertEqual(sum(pair["reasoning_delta_tokens"] < 0 for pair in comparable), 151)
        self.assertEqual(sum(pair["clean"]["success"] for pair in self.pairs), 165)
        self.assertEqual(sum(pair["compact"]["success"] for pair in self.pairs), 166)

    def test_original_first_capture_reference(self):
        original = self.analysis["original_first_capture_sensitivity"]
        headline = original["headline"]
        full = original["overall"]["full_trajectory_reasoning_all_assigned"]
        first = original["overall"]["first_continuation_reasoning_all_assigned"]
        self.assertEqual((headline["comparable_pairs"], headline["compact_shorter_comparable_pairs"]), (169, 149))
        self.assertEqual((headline["clean_exact_success"], headline["compact_exact_success"]), (162, 165))
        self.assertEqual(
            (
                sum(pair["clean"]["success"] for pair in self.original_pairs),
                sum(pair["compact"]["success"] for pair in self.original_pairs),
            ),
            (162, 165),
        )
        self.assertEqual((full["clean_total_tokens"], full["compact_total_tokens"]), (152475, 94357))
        self.assertEqual(
            (
                sum(pair["clean"]["reasoning_tokens"] for pair in self.original_pairs),
                sum(pair["compact"]["reasoning_tokens"] for pair in self.original_pairs),
            ),
            (152475, 94357),
        )
        self.assertAlmostEqual(first["paired_change_percent"]["median"], -20.718578597269225)

    def test_retry_classification_and_attempts(self):
        self.assertEqual(len(self.replacements), 4)
        self.assertEqual([item["original_trigger"] for item in self.replacements].count("length"), 3)
        self.assertEqual([item["original_trigger"] for item in self.replacements].count("provider_502"), 1)
        self.assertEqual([item["recovery_run_total_request_attempts"] for item in self.replacements], [3, 3, 5, 3])
        self.assertTrue(all(item["maximum_attempts_per_retried_turn"] == 20 for item in self.replacements))
        self.assertTrue(all("not established" in item["trigger_interpretation"] for item in self.replacements if item["original_trigger"] == "length"))

    def test_task_definition_block_bootstrap(self):
        result = self.analysis["task_definition_block_bootstrap"]
        self.assertEqual(result["block_count"], 12)
        self.assertIn("cross-route", result["scope"])

    def test_sources_and_gate_exclusions(self): 
        self.assertEqual(len(self.sources), 72)
        self.assertEqual(sum(source["accepted"] for source in self.sources), 58)
        categories = [source["exclusion_category"] for source in self.sources]
        self.assertEqual(categories.count("source_action_failure"), 5)
        self.assertEqual(categories.count("fidelity_exclusion"), 9)

    def test_corrected_primary_and_causal_timing(self):
        target = next(
            pair for pair in self.pairs
            if pair["route"] == "laguna-s-2.1"
            and pair["task_id"] == "grid-decision-v2"
            and pair["replicate"] == 2
        )
        self.assertEqual(target["clean"]["reasoning_tokens"], 2346)
        timing = self.analysis["causal_timing_audit"]
        self.assertEqual(timing["first_request_reasoning_only_invariant"], 174)
        self.assertEqual(timing["shared_preceding_results"][0]["same_preceding_tool_result"], 174)

    def test_all_assigned_workload_is_raw_sum(self):
        workload = self.analysis["overall"]["all_assigned_workload"]
        self.assertEqual(
            workload["input_tokens"]["clean_total_tokens"],
            sum(pair["clean"]["input_tokens"] for pair in self.pairs),
        )
        self.assertEqual(
            workload["output_tokens"]["compact_total_tokens"],
            sum(pair["compact"]["output_tokens"] for pair in self.pairs),
        )
        self.assertEqual(workload["turns"]["clean_total_tokens"], 521)
        self.assertEqual(workload["turns"]["compact_total_tokens"], 520)

    def test_outputs_and_manifest_are_portable(self):
        expected = {
            "README.md",
            "analysis.json",
            "manifest.json",
            "per_pair_retry_augmented.json",
            "per_pair_original_first_capture.json",
            "retry_augmentation_records.json",
            "per_source.json",
        }
        self.assertEqual({path.name for path in self.output.iterdir()}, expected)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertTrue(all(not item["path"].startswith("/") for item in manifest["inputs"]))
        self.assertTrue(self.analysis["corrected_input_fidelity_check"]["all_checks_pass"])


if __name__ == "__main__":
    unittest.main()
