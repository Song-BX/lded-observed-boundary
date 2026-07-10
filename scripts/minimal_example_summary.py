from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lded_minimal_example_mpl_cache"))

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd


SERIES = [
    ("front_length_m", "L_f", 1e3, "mm"),
    ("rear_length_m", "L_r", 1e3, "mm"),
    ("full_width_m", "W", 1e3, "mm"),
    ("height_span_m", "H", 1e3, "mm"),
    ("Tmax_K", "Tmax", 1.0, "K"),
    ("Umax_m_per_s", "Umax", 1.0, "m s^-1"),
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def write_summary(table: pd.DataFrame, output_dir: Path, quasi_start_s: float) -> Path:
    rows = []
    quasi = table[table["time_s"] >= quasi_start_s].copy()
    if quasi.empty:
        quasi = table.copy()
    for column, label, scale, unit in SERIES:
        if column not in table.columns:
            continue
        values = pd.to_numeric(quasi[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "descriptor": label,
                "source_column": column,
                "quasi_start_s": quasi_start_s,
                "mean": float(values.mean() * scale),
                "min": float(values.min() * scale),
                "max": float(values.max() * scale),
                "unit": unit,
            }
        )
    out_path = output_dir / "minimal_example_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def write_figure(table: pd.DataFrame, output_dir: Path, quasi_start_s: float) -> Path:
    configure_matplotlib()
    time = pd.to_numeric(table["time_s"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    for column, label, scale, unit in SERIES[:4]:
        if column in table.columns:
            axes[0].plot(time, pd.to_numeric(table[column], errors="coerce") * scale, "o-", ms=3, lw=1.0, label=label)
    axes[0].axvline(quasi_start_s, color="0.35", lw=0.8, ls=":")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Geometry (mm)")
    axes[0].set_title("Boundary descriptors")
    axes[0].grid(True, axis="y", color="0.90", linewidth=0.5)
    axes[0].legend(loc="best")

    for column, label, scale, unit in SERIES[4:]:
        if column in table.columns:
            axes[1].plot(time, pd.to_numeric(table[column], errors="coerce") * scale, "o-", ms=3, lw=1.0, label=f"{label} ({unit})")
    axes[1].axvline(quasi_start_s, color="0.35", lw=0.8, ls=":")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Thermal and flow descriptors")
    axes[1].grid(True, axis="y", color="0.90", linewidth=0.5)
    axes[1].legend(loc="best")

    out_path = output_dir / "minimal_example_geometry.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "minimal_example_geometry.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal public descriptor example.")
    parser.add_argument("--input", required=True, help="Path to minimal_public_modeling_table.csv")
    parser.add_argument("--output-dir", default="example_outputs", help="Directory for generated example outputs")
    parser.add_argument("--quasi-start-s", type=float, default=0.20, help="Quasi-steady cutoff used for summary statistics")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(input_path)
    if "time_s" not in table.columns:
        raise ValueError("The input CSV must contain a time_s column.")
    table = table.sort_values("time_s")
    summary_path = write_summary(table, output_dir, args.quasi_start_s)
    figure_path = write_figure(table, output_dir, args.quasi_start_s)
    print(f"Wrote {summary_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
