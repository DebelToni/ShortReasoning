# Active figures

The four PDFs in this directory are the paper-used snapshots at export time. Rebuild all four from bundled JSON with:

```bash
python3 scripts/make_archive_figures.py --output-dir ~/tmp/short-reasoning-figures
```

`source-data/artificial-analysis-frontier-reasoning.csv` preserves the August 3, 2026 external benchmark snapshot used in the introduction. Its GLM/Fable quantities are reasoning shares of generated output; the DeepSeek cost quantity is a share of total API cost.

`source-data/paired-fork-protocol-layout.pdf` preserves the original protocol diagram's vector artwork and embedded Times fonts. The generator makes only the documented timing-label correction and adds gray timing notes, without redrawing its boxes or arrows.

`source-data/controlled-effect-raincloud-layout.pdf` preserves the six-route distribution artwork. The generator verifies its plotted values against the bundled results and updates its title.

`source-data/active-figure-values.json` is independently derived from the same result inputs and checked by the smoke. Rendering can differ slightly across Matplotlib/font stacks; the generator reproduces data, labels, panels, and dimensions rather than promising byte-identical PDFs.
