# Minimal Public Example

This folder contains a processed descriptor-level example extracted from the representative baseline condition. It is intended to let readers run a small public example without access to the raw FLOW-3D molten-region CSV exports.

Run from the root of the reproducibility package:

```bash
python scripts/minimal_example_summary.py --input example_data/minimal_public_modeling_table.csv --output-dir example_outputs
```

The command writes `example_outputs/minimal_example_summary.csv` and `example_outputs/minimal_example_geometry.png`. It exercises the descriptor-table and plotting workflow only. It does not reproduce the full manuscript pipeline and does not include proprietary FLOW-3D project files or raw molten-region exports.
