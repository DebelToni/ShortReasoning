#!/usr/bin/env python3
"""Reconstruct B300 primary and retry-augmented selections from portable records."""
import argparse
import copy
import json
from pathlib import Path

ARMS = ("normal", "self", "luna")


def selections(original, retries):
    primary = [row for row in original if all(row["arms"][arm]["status"] == "complete" for arm in ARMS)]
    augmented = copy.deepcopy(original)
    index = {row["block"]: row for row in augmented}
    for retry in retries:
        parent = index[retry["block"]]
        if parent["source_sha256"] != retry["parent_source_sha256"]:
            raise ValueError("retry parent hash mismatch")
        if parent["arms"][retry["arm"]]["status"] == "complete":
            raise ValueError("retry would replace a successful arm")
        parent["arms"][retry["arm"]] = retry["record"] | {
            "retrospective_replacement": True,
            "retry_source_sha256": retry["retry_source_sha256"],
        }
    if len(primary) != 23 or len(augmented) != 25:
        raise ValueError("unexpected selection size")
    if not all(all(row["arms"][arm]["status"] == "complete" for arm in ARMS) for row in augmented):
        raise ValueError("retry-augmented selection remains incomplete")
    return {
        "primary_23_contemporaneous": primary,
        "retrospective_25_retry_augmented": augmented,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())["b300_latency"]
    args.output.write_text(json.dumps(selections(data["overload_original_25"], data["overload_retries"]), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
