#!/usr/bin/env python3
"""Recompute the exploratory compressor-effort results from published captures."""
import collections
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NONE = ROOT / 'results/20260915-deepseek-0731-reasoning-off-20-v1'
LOW = ROOT / 'results/20260915-deepseek-0731-reasoning-low-20-v1'
OUT = ROOT / 'results/20260915-deepseek-compressor-effort-audit-v1'


def load(path):
    return json.loads(path.read_text())


def main():
    times = {a: [] for a in ('xhigh', 'none', 'low')}
    tokens = {a: [] for a in times}
    shorter = collections.Counter()
    labels = {a: collections.Counter() for a in ('none', 'low', 'xhigh_none_round', 'xhigh_low_round')}
    originals = []
    transitions = collections.Counter()
    fingerprints = {}
    for rank in range(1, 21):
        folders = {'none': NONE / f'case-{rank:02d}', 'low': LOW / f'case-{rank:02d}'}
        historical = load(folders['none'] / 'historical.json')
        assert historical == load(folders['low'] / 'historical.json')
        raw = historical['raw_reasoning']; originals.append(raw)
        response = historical['response']
        times['xhigh'].append(historical['ledger']['wall_seconds'])
        tokens['xhigh'].append(response['usage']['completion_tokens_details']['reasoning_tokens'])
        shorter['xhigh'] += len(response['choices'][0]['message']['content'].split()) < len(raw.split())
        current = {}
        for arm, folder in folders.items():
            stem = 'reasoning-off' if arm == 'none' else 'reasoning-low'
            result = load(folder / f'{stem}-result.json')
            request = load(folder / f'{stem}-request.json')
            assert result['status'] == 'complete' and result['finish_reason'] == 'stop'
            assert request['messages'] == historical['request']['messages']
            assert request['reasoning']['effort'] == arm
            times[arm].append(result['wall_seconds'])
            tokens[arm].append(result['usage']['completion_tokens_details']['reasoning_tokens'])
            shorter[arm] += len(result['content'].split()) < len(raw.split())
            mapping = load(folder / 'judge-labels.json')
            judgment = load(folder / 'judgment.json')
            for label, identity in mapping.items():
                key = f'xhigh_{arm}_round' if identity == 'historical' else arm
                verdict = judgment[label]['verdict']
                labels[key][verdict] += 1
                if identity != 'historical': current[arm] = verdict
            for name in ('historical.json', f'{stem}-request.json', f'{stem}-result.json', 'judge-labels.json', 'judgment.json'):
                path = folder / name
                fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        transitions[f"{current['none']} -> {current['low']}"] += 1
    assert len(set(originals)) == 20 and all(x == 0 for x in tokens['none'])
    summary = {'requests': 20, 'median_seconds': {a: statistics.median(v) for a,v in times.items()},
               'median_reasoning_tokens': {a: statistics.median(v) for a,v in tokens.items()},
               'shorter': dict(shorter), 'judge_labels': labels, 'none_to_low': transitions,
               'input_sha256': fingerprints}
    OUT.mkdir(exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
