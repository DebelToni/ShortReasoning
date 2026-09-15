# Exploratory compressor-effort diagnostic

The paper's compact main-text table and Appendix D.6 compare 20 historically slow DeepSeek self-compaction requests. The historical xhigh responses use Baidu and an unversioned model ID. New none/low responses use `deepseek/deepseek-v4-flash-0731` on DeepInfra. Sol judges each new candidate against its historical counterpart, with identities hidden and the same randomized A/B positions across judging rounds.

## Offline reproduction

From the repository root, with Python 3.11 or newer:

```bash
python scripts/analyze_compressor_effort_diagnostic.py
```

This makes no API calls. It checks original-message identity, completion status, recorded token usage, and judge-label mappings, then recreates `summary.json` here. Inputs are the `case-01` through `case-20` directories under:

- `results/20260915-deepseek-0731-reasoning-off-20-v1/`
- `results/20260915-deepseek-0731-reasoning-low-20-v1/`

Each case contains the original request and historical response, new request/output/usage, judge request/output, label mapping, and parsed assessment. API response IDs are hashed; machine-specific source paths and stream reasoning excerpts are omitted. The interrupted initial reasoning-off attempt remains under case 08's `initial-attempt/` directory. Its cost was not reported, so the recorded spending totals are not complete billing totals.

## New inference (paid)

Set `OPENROUTER_API_KEY` in the environment or a local `.env` file. Use new output directories outside this checkout:

```bash
python scripts/run_reasoning_off_diagnostic.py --output /path/to/new-none-run
python scripts/run_reasoning_low_diagnostic.py --output /path/to/new-low-run
```

The low runner defaults to the published source set and label positions. To use a new, fully completed none run as its reference, add `--reference /path/to/new-none-run`.

These are the experiment runners with portable input paths and CLI output paths. The none runner closes the stream and stops if reasoning is detected. The low runner permits reasoning. Both use the same Sol rubric with medium reasoning effort and a 4,096-token judge-output cap. The current runners impose no client timeout or DeepSeek output cap, retain failed attempts, and enforce a conservative spending guard. Invoking `--help` makes no model calls. Provider/model availability can change.

The original none run used a nonbinding 4,096-token cap and a 180-second inactivity timeout; its interrupted eighth request was explicitly retried without those limits. The low run used the uncapped settings. There is no fresh xhigh arm, no downstream agent evaluation, and no estimate of typical deployment savings. The separate model-judging rounds changed one historical label. All results are exploratory.
