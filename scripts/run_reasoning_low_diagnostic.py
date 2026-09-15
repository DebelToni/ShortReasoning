#!/usr/bin/env python3
"""Low-effort replay of the same 20 cases; blinded Sol versus historical xhigh."""
import argparse
import copy
import json
import time
from pathlib import Path

from run_reasoning_off_diagnostic import ROOT, JUDGE_PROMPT, call, dump, load_dotenv, openrouter_key, now

REFERENCE = ROOT / "results/20260915-deepseek-0731-reasoning-off-20-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    output = args.output.resolve()
    reference_root = args.reference.resolve()
    output.mkdir(parents=True, exist_ok=False)
    load_dotenv(ROOT / ".env")
    key = openrouter_key()
    previous = json.loads((reference_root / "progress.json").read_text())
    assert previous["status"] == "complete" and previous["successful_judgments"] == 20
    start = time.monotonic()
    progress = {"status": "running", "started_utc": now(), "total": 20, "finished": 0,
                "successful_judgments": 0, "reported_cost_usd": 0.0, "cases": [],
                "comparison": "reasoning low versus historical xhigh; same originals, judge rubric, and A/B positions as the none study",
                "timeout": None, "deepseek_output_cap": None}

    def update():
        progress["elapsed_seconds"] = time.monotonic() - start
        n = progress["finished"]
        progress["eta_remaining_seconds"] = progress["elapsed_seconds"] / n * (20 - n) if n else None
        temporary = output / "progress.tmp"
        dump(temporary, progress)
        temporary.replace(output / "progress.json")

    update()
    for rank in range(1, 21):
        if previous["reported_cost_usd"] + progress["reported_cost_usd"] + 0.50 > 8.0:
            progress["status"] = "stopped_budget"; update(); return
        reference = reference_root / f"case-{rank:02d}"
        directory = output / f"case-{rank:02d}"
        directory.mkdir()
        historical = json.loads((reference / "historical.json").read_text())
        dump(directory / "historical.json", historical)
        original = historical["raw_reasoning"]
        old_text = historical["response"]["choices"][0]["message"].get("content") or ""
        payload = copy.deepcopy(historical["request"])
        payload.update(model="deepseek/deepseek-v4-flash-0731",
                       provider={"only": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True},
                       reasoning={"effort": "low", "exclude": False},
                       stream=True, stream_options={"include_usage": True})
        for field in ("include_reasoning", "max_tokens", "max_completion_tokens"):
            payload.pop(field, None)
        none_request = json.loads((reference / "reasoning-off-request.json").read_text())
        assert payload["messages"] == none_request["messages"]
        progress.update(current_case=rank, phase="reasoning_low"); update()
        rewrite = call(payload, directory, "reasoning-low", key, monitor_reasoning=True, stop_on_reasoning=False)
        progress["reported_cost_usd"] += (rewrite.get("usage") or {}).get("cost", 0) or 0
        row = {"rank": rank, "rewrite_status": rewrite["status"], "rewrite_seconds": rewrite["wall_seconds"],
               "raw_words": len(original.split()), "new_words": len(rewrite["content"].split()),
               "historical_seconds": historical["ledger"]["wall_seconds"]}
        if rewrite["status"] != "complete" or not rewrite["content"]:
            row["judge_status"] = "not_run"
            progress["cases"].append(row); progress["finished"] += 1; update(); continue
        old_mapping = json.loads((reference / "judge-labels.json").read_text())
        mapping = {label: ("reasoning_low" if arm == "reasoning_off" else arm) for label, arm in old_mapping.items()}
        dump(directory / "judge-labels.json", mapping)
        texts = {"historical": old_text, "reasoning_low": rewrite["content"]}
        user = json.dumps({"original": original, "A": texts[mapping["A"]], "B": texts[mapping["B"]]})
        judge_upper = (len((JUDGE_PROMPT + user).encode()) + 2048) * 0.000002 + 4096 * 0.00001
        if judge_upper > 0.50:
            progress["status"] = "stopped_judge_budget"; update(); return
        judge_payload = {"model": "openai/gpt-5.6-sol",
            "provider": {"only": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True},
            "messages": [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
            "reasoning": {"effort": "medium", "exclude": True}, "max_tokens": 4096,
            "response_format": {"type": "json_object"}, "usage": {"include": True}}
        progress["phase"] = "sol_judgment"; update()
        judged = call(judge_payload, directory, "judge", key)
        progress["reported_cost_usd"] += (judged.get("usage") or {}).get("cost", 0) or 0
        row.update(judge_status=judged["status"], judge_seconds=judged["wall_seconds"],
                   label_mapping=mapping, new_is_shorter=row["new_words"] < row["raw_words"])
        if judged["status"] == "complete":
            try:
                verdict = json.loads(judged["content"])
                for label in ("A", "B"):
                    assert verdict[label]["verdict"] in ("Pass", "Problem", "Unsure")
                row["judgments"] = {mapping[label]: verdict[label] for label in ("A", "B")}
                row["preferred"] = mapping.get(verdict.get("preferred"), verdict.get("preferred"))
                dump(directory / "judgment.json", verdict)
                progress["successful_judgments"] += 1
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                row.update(judge_status="invalid_judgment", parse_error=str(exc))
        progress["cases"].append(row); progress["finished"] += 1; update()
        print(json.dumps({k: progress[k] for k in ("finished", "successful_judgments", "elapsed_seconds", "eta_remaining_seconds", "reported_cost_usd")}), flush=True)
        if judged["status"] != "complete":
            progress["status"] = "stopped_judge_error"; update(); return
    progress.update(status="complete", phase="complete"); update()


if __name__ == "__main__":
    main()
