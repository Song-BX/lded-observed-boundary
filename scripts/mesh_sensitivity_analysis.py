"""Mesh-resolution sensitivity analysis for the representative A1/A2/A5 cases."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from flow3d_pipeline.core import parse_case_metadata, summarize_time_step


CASE_RE = re.compile(r"^(A\d+-\d+-\d+-\d+)-mesh(\d+)um$")
METRICS = [
    "front_length_m", "rear_length_m", "full_width_m", "height_span_m",
    "z_min_m", "z_max_m", "deposition_height_m", "melt_depth_m",
    "Tmax_K", "Gmean_K_per_m", "Umax_m_per_s",
    "volume_proxy_m3", "ellipsoid_residual_rmse", "superellipsoid_residual_rmse",
]
GEOMETRY_LABELS = {
    "deposition_height_m": "Deposition\nheight",
    "front_length_m": "Front\nlength",
    "full_width_m": "Full\nwidth",
    "melt_depth_m": "Melt\ndepth",
    "rear_length_m": "Rear\nlength",
}


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    root = Path("additional-data-mesh")
    out = Path("analysis_outputs") / "mesh_sensitivity"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    for mesh_dir in sorted(root.iterdir()):
        if not mesh_dir.is_dir() or mesh_dir.name == "csv_backup_original":
            continue
        match = CASE_RE.fullmatch(mesh_dir.name)
        if not match:
            continue
        case_id, mesh_um = match.group(1), int(match.group(2))
        meta = parse_case_metadata(Path(case_id))
        for path in sorted(mesh_dir.glob("*.csv")):
            try:
                row, _ = summarize_time_step(path, meta)
                row["mesh_um"] = mesh_um
                row["base_case_id"] = case_id
                row["source_path"] = str(path)
                # The substrate top is z=0 in the FLOW-3D setup. These are
                # the experimental-style height/depth proxies used for mesh checks.
                row["deposition_height_m"] = max(float(row["z_max_m"]), 0.0)
                row["melt_depth_m"] = max(-float(row["z_min_m"]), 0.0)
                rows.append(row)
            except Exception as exc:
                errors.append({"source_path": str(path), "error": repr(exc)})

    table = pd.DataFrame(rows).sort_values(["base_case_id", "mesh_um", "time_s"])
    table["quasi_steady"] = table["time_s"] >= 0.5
    table.to_csv(out / "mesh_metrics_by_time.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out / "processing_errors.csv", index=False)

    # Pairwise differences use the 100 um grid as the baseline.
    comparisons = []
    for case_id in sorted(table["base_case_id"].unique()):
        base = table[(table.base_case_id == case_id) & (table.mesh_um == 100)].set_index("time_s")
        for mesh_um in (80, 200):
            other = table[(table.base_case_id == case_id) & (table.mesh_um == mesh_um)].set_index("time_s")
            for t in sorted(set(base.index) & set(other.index)):
                for metric in METRICS:
                    ref, val = float(base.loc[t, metric]), float(other.loc[t, metric])
                    # Relative error is undefined when the 100 um reference
                    # depth/height proxy is effectively zero; retain the
                    # absolute difference but exclude such points from means.
                    denom = abs(ref)
                    rel = abs(val - ref) / denom if denom > 1e-7 else float("nan")
                    comparisons.append({
                        "base_case_id": case_id, "time_s": t, "comparison": f"{mesh_um}_vs_100",
                        "mesh_um": mesh_um, "metric": metric, "reference_100um": ref,
                        "value_mesh": val, "absolute_difference": abs(val - ref),
                        "relative_difference": rel,
                        "quasi_steady": t >= 0.5,
                    })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "mesh_relative_differences.csv", index=False)

    steady = comparison[comparison.quasi_steady].copy()
    summary = (steady.groupby(["base_case_id", "comparison", "mesh_um", "metric"], as_index=False)
               .agg(mean_relative_difference=("relative_difference", "mean"),
                    max_relative_difference=("relative_difference", "max"),
                    n_times=("relative_difference", "size")))
    summary.to_csv(out / "mesh_convergence_summary.csv", index=False)

    # Compact figure for the principal geometric observables.
    geom = summary[summary.metric.isin(["front_length_m", "rear_length_m", "full_width_m", "deposition_height_m", "melt_depth_m"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    for panel, ax, comp in zip(["a", "b"], axes, ["80_vs_100", "200_vs_100"]):
        sub = geom[geom.comparison == comp]
        pivot = sub.pivot_table(index="metric", columns="base_case_id", values="mean_relative_difference")
        pivot = pivot.rename(index=GEOMETRY_LABELS, columns=lambda value: str(value).split("-")[0])
        pivot.plot.bar(ax=ax, rot=0, legend=False)
        mesh_um = comp.split("_")[0]
        ax.set_title(f"({panel}) {mesh_um} vs 100 μm")
        ax.set_ylabel("Mean absolute relative difference")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Condition", loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(out / "mesh_convergence_geometric_metrics.png", dpi=600)
    fig.savefig(out / "mesh_convergence_geometric_metrics.pdf")
    plt.close(fig)

    report = [
        "# Mesh-resolution sensitivity analysis",
        "",
        f"Processed {len(table)} condition-time files across {table.base_case_id.nunique()} cases and meshes {sorted(table.mesh_um.unique())} um.",
        f"Primary convergence window: t >= 0.5 s; early 200 um outputs are retained but not used in the summary statistics.",
        f"Processing errors: {len(errors)}.",
        "",
        "The 100 um grid is the reference. Relative differences are absolute differences divided by the corresponding 100 um value.",
    ]
    (out / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote mesh analysis to {out.resolve()}")
    print(f"Processed files: {len(table)}; errors: {len(errors)}")


if __name__ == "__main__":
    main()
