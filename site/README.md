# ShortReasoning project site

Static HTML, CSS and JavaScript for https://debeltoni.github.io/ShortReasoning/.
The Pages workflow publishes only this directory, not the research working tree.

## Preview

From the repository root:

```sh
uv run --no-project python -m http.server 8000 --bind 127.0.0.1 --directory site
```

## Scientific boundaries

- `data/results.json` contains public aggregate values from the revised arXiv paper, https://arxiv.org/abs/2611.03852: controlled results from Table 1, retained-run deployment results from Table 3, and absolute/inclusive request timing from Table 10.
- Controlled token comparisons are conditional on accepted histories and equal horizons. Acceptance differs by route. Task success does not establish quality noninferiority.
- All 50 assigned tasks remain in each deployment arm. Spend includes rewriting. Nemotron target costs are common-tariff estimates with captured compressor charges.
- Target-only and rewrite-inclusive request durations are distinct. Neither is end-to-end agent latency. The history-word totals are not billed input-token totals.
- The hero walkthrough is native HTML, not an embedded or linked GIF. The five compactor diagrams are illustrations of design choices. Self/external rewriting and per-block rewriting were tested; other settings shown are proposed experiments.
- The training diagram is a proposal, not a measured result: frozen same-model teacher on compact history, trainable student on original history, matching a common continuation prefix and vocabulary. Forward KL updates the student only. One connected rollout → compact → train scene shows the distributions converging. The animated probability masses are explicitly illustrative, not model outputs or training measurements.
- Source figure exports preserve the paper's original controlled-effects and length/quality figures.

The deployment table offers Spend and Target time. Rewrite-inclusive timing remains in the public data for provenance, but is not a selectable tab. CI graphics use one shared coordinate grid, fixed-size circular median markers and endpoint caps.

No raw reasoning, task patches, response journals, credentials, or private paths belong in this site.

## Deployment

`.github/workflows/pages.yml` checks syntax and public aggregate invariants,
then copies `site/` into a deployment artifact and adds content-hash query versions to CSS/JavaScript URLs before deploying with GitHub Pages Actions. This prevents old cached scripts or styles from being mixed with new HTML. Resource paths are relative so
the site works under the `/ShortReasoning/` project prefix. No server, package
installation, model call, or account credential is required by the site itself.

## Rendering checks

Before deployment, `.github/check_site_ui.py` checks the rendered page in Chromium at 320, 390, 768, 1024 and 1440 pixels. It verifies CI endpoints and marker positions against the source values, circular markers, both metric tabs, distinct model shapes, both HTML walkthroughs, fixed teacher/matching student distributions, automatic progress and reduced motion. Browser screenshots are retained as a workflow artifact.

The November 5 revision was also tested on native Mac WebKit, including an old production page kept open across the deployment and reloaded at the same URL. It upgraded successfully to the cache-versioned scripts and styles.
