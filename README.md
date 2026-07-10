# Reproducibility Materials

These processed materials accompany the manuscript and are suitable for journal supplementary data or repository archival.

Included:

- processed geometry descriptors and reduced-state time series;
- fitted superellipsoid parameters, geometry-selection metrics and model-selection tables;
- leave-one-condition-out and shared-setting numerical holdout summaries for the V-prefixed and A16-A20 cohorts;
- parameter reconciliation, geometry-risk, q_inf, holdout-extrapolation, boundary-extraction and validation-hierarchy assessment tables;
- dynamics fit-asymmetry and numerical-credibility assessment tables that document descriptor limits without adding new simulations;
- input-file index with raw CSV paths, row counts, schemas and SHA256 hashes;
- environment summary; final output checksums are generated in `analysis_outputs/tables/output_checksums.csv`;
- figure index, captions, editable Figure 1 draw.io source, nomenclature and equation inventory;
- a minimal public processed-descriptor example in `example_data/`;
- plotting/analysis scripts and LaTeX manuscript sources.

Excluded:

- proprietary FLOW-3D project files;
- raw FLOW-3D molten-region CSV exports, which are available from the corresponding author upon reasonable request subject to project-sharing and software-export constraints.

Reproduction command from the project root:

```bash
python scripts/flow3d_melt_pool_pilot.py
```

The command rebuilds the processed tables, active figures, LaTeX manuscript, compiled PDFs and these processed reproducibility materials from the available CSV exports. The final archive checksum is written outside the archive in `analysis_outputs/tables/output_checksums.csv` to avoid a self-referential ZIP hash.

Minimal public example command from the root of this reproducibility package:

```bash
python scripts/minimal_example_summary.py --input example_data/minimal_public_modeling_table.csv --output-dir example_outputs
```

This command uses only the processed descriptor-level example and writes a small summary table plus a diagnostic geometry plot. It does not require the raw FLOW-3D exports.
