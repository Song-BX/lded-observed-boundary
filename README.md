# Reproducibility Package

This processed package accompanies the manuscript and is suitable for journal supplementary data or repository archival.

Included:

- processed geometry descriptors and reduced-state time series;
- fitted superellipsoid parameters and model-selection tables;
- leave-one-condition-out and external CFD holdout validation summaries;
- figure manifest, captions, nomenclature and equation inventory;
- plotting/analysis scripts and LaTeX manuscript sources.

Excluded:

- proprietary FLOW-3D project files;
- raw FLOW-3D molten-region CSV exports, which are available from the corresponding author upon reasonable request subject to project-sharing and software-export constraints.

Reproduction command from the project root:

```bash
python scripts/flow3d_melt_pool_pilot.py
```

The command rebuilds the processed tables, active figures, LaTeX manuscript and compiled PDFs from the available CSV exports.
