from __future__ import annotations

import base64
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_camera_ready_deployment.py"
RESULT = ROOT / "results/20260910-camera-ready-deployment-audit-v2"
SPEC = importlib.util.spec_from_file_location("camera_ready_deployment", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_journal(root: Path, relative: str, response: dict, wall: float = 2.0) -> None:
    path = root / "journals" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(response, sort_keys=True).encode()
    path.write_text(json.dumps({
        "status": "complete", "wall_seconds": wall,
        "raw_response_base64": base64.b64encode(raw).decode(),
        "raw_response_sha256": MODULE.sha256_bytes(raw), "request_sha256": "a" * 64,
    }))


def response(response_id: str | None, cost: float, choices: bool = True) -> dict:
    row = {
        "provider": "fixture-provider", "model": "fixture-model",
        "usage": {
            "prompt_tokens": 10, "completion_tokens": 4, "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }
    if response_id is not None:
        row["id"] = response_id
    if choices:
        row["choices"] = [{"finish_reason": "length"}]
    else:
        row["error"] = {"message": "fixture"}
    return row


def b300_arm(status: str = "complete", seconds: float = 100.0) -> dict:
    return {
        "status": status, "complete_task_seconds": seconds,
        "summed_request_seconds": seconds - 2, "summed_generation_phase_seconds": seconds - 3,
        "tool_execution_seconds": 1, "future_reasoning_tokens": 10,
        "total_output_tokens": 20, "turns": 2,
        "patch_sha256": "b" * 64 if status == "complete" else None,
        "evaluation": {"resolved": True} if status == "complete" else None,
        "error": None if status == "complete" else "protocol failure",
    }


class DeploymentStdlibTests(unittest.TestCase):
    def test_orphan_success_no_id_error_and_deduplication(self) -> None:
        parent = Path.home() / "tmp"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            root = Path(temporary)
            trajectory = root / "attempts/attempt-001/inference/task-1/task-1.traj.json"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_text(json.dumps({"messages": [{"role": "assistant", "extra": {"response": {"id": "in-trajectory"}}}]}))
            selected = {"task-1": trajectory}
            write_journal(root, "attempt-001/model-a/call-001/attempt-01.json", response("in-trajectory", .25))
            write_journal(root, "attempt-001/model-a/call-002/attempt-01.json", response("journal-only", .75))
            write_journal(root, "attempt-001/model-a/call-003/attempt-01.json", response(None, .50))
            write_journal(root, "attempt-001/model-a/call-004/attempt-01.json", response(None, 0, False), 7.0)
            write_journal(root, "attempt-001/model-a/call-005/attempt-01.json", response("in-trajectory", .25))
            records, reconciliation = MODULE.collect_response_ledger("fixture", "normal", [root], selected)
            successes = [row for row in records if row["kind"] == "model_response"]
            errors = [row for row in records if row["kind"] == "error_body"]
            self.assertEqual(len(successes), 3)
            self.assertAlmostEqual(sum(row["usage"]["reported_cost_usd"] for row in successes), 1.5)
            self.assertEqual(sum(not row["in_any_trajectory"] for row in successes), 2)
            self.assertEqual(sum(row["response_id_sha256"] is None for row in successes), 1)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["wall_seconds"], 7.0)
            self.assertEqual(reconciliation["successful_journal_responses_not_in_trajectory"], 2)
            self.assertEqual(reconciliation["trajectory_only_response_ids"], 0)

    def test_conflicting_duplicate_fails(self) -> None:
        parent = Path.home() / "tmp"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            root = Path(temporary)
            trajectory = root / "inference/task-1/task-1.traj.json"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_text('{"messages":[]}')
            write_journal(root, "attempt-001/model-a/call-001/attempt-01.json", response("duplicate", .25))
            write_journal(root, "attempt-001/model-a/call-002/attempt-01.json", response("duplicate", .75))
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                MODULE.collect_response_ledger("fixture", "normal", [root], {"task-1": trajectory})

    def test_b300_reconstructs_23_and_25(self) -> None:
        original = []
        for block in range(1, 26):
            arms = {arm: b300_arm(seconds=100 - index * 10) for index, arm in enumerate(("normal", "self", "luna"))}
            if block == 2:
                arms["normal"] = b300_arm("failure", 58)
            if block == 17:
                arms["self"] = b300_arm("failure", 39)
            original.append({"block": block, "regime": "overloaded", "source_sha256": f"{block:064x}", "arms": arms})
        retries = [
            {"block": 2, "arm": "normal", "parent_source_sha256": f"{2:064x}", "retry_source_sha256": "c" * 64, "record": b300_arm(seconds=124)},
            {"block": 17, "arm": "self", "parent_source_sha256": f"{17:064x}", "retry_source_sha256": "d" * 64, "record": b300_arm(seconds=91)},
        ]
        selections = MODULE.b300_selections(original, retries)
        self.assertEqual(len(selections["primary_23_contemporaneous"]), 23)
        self.assertEqual(len(selections["retrospective_25_retry_augmented"]), 25)
        augmented = {row["block"]: row for row in selections["retrospective_25_retry_augmented"]}
        self.assertTrue(augmented[2]["arms"]["normal"]["retrospective_replacement"])
        self.assertTrue(augmented[17]["arms"]["self"]["retrospective_replacement"])

    def test_bundled_v2_ledger_reanalysis(self) -> None:
        inputs = MODULE.load(RESULT / "inputs.json")
        ledger_path = RESULT / "response-ledger.jsonl.gz"
        ledger = MODULE.read_ledger(ledger_path)
        self.assertEqual(len(ledger), 77228)
        reproduced = MODULE.analyze(inputs, ledger, ledger_path)
        self.assertEqual(reproduced, MODULE.load(RESULT / "analysis.json"))
        deployment = reproduced["deployment"]
        self.assertEqual(deployment["accuracy_change_range_percentage_points"], [-6.0, 6.0])
        self.assertEqual(reproduced["b300_latency"]["overload_primary_23"]["rows"], 23)
        self.assertEqual(reproduced["b300_latency"]["overload_retrospective_25"]["rows"], 25)
        kimi = deployment["models"]["kimi-k2.6"]["arms"]
        self.assertEqual([kimi[arm]["journal_trajectory_reconciliation"]["target_success_not_in_any_trajectory"] for arm in MODULE.ARMS], [122, 420, 254])
        nemotron = deployment["models"]["nemotron-3-ultra"]["arms"]
        self.assertEqual([nemotron[arm]["journal_trajectory_reconciliation"]["target_success_not_in_any_trajectory"] for arm in MODULE.ARMS], [98, 205, 72])


if __name__ == "__main__":
    unittest.main()
