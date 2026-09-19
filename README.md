# L-DED observed boundary-envelope reproducibility example

This package contains the analysis scripts and a processed example dataset for the manuscript *Observed boundary-envelope identification of evolving L-DED melt pools using superellipsoid manifolds and parsimonious attractor dynamics*.

## Contents

- `scripts/`: analysis, plotting and minimal-example scripts;
- `example_data/`: processed descriptor-level example data;
- `requirements.txt`: Python runtime dependencies.

## Run the example

Python 3.10 or later is required. From the package root, run:

```bash
python -m pip install -r requirements.txt
python scripts/minimal_example_summary.py --input example_data/minimal_public_modeling_table.csv --output-dir example_outputs
```

The example writes a descriptor summary and geometry plots in PNG and PDF formats. The complete analysis scripts are provided for methodological transparency; full-data execution additionally requires the simulation exports used in the study, which are not part of this example package.
