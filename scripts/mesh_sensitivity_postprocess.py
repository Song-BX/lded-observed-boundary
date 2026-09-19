"""Post-process the cached mesh sensitivity table without rerunning CFD fits."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("analysis_outputs/mesh_sensitivity")
GEOM = ["full_width_m", "deposition_height_m", "melt_depth_m", "front_length_m", "rear_length_m"]
ALL = GEOM + ["Tmax_K", "Gmean_K_per_m", "Umax_m_per_s", "volume_proxy_m3", "ellipsoid_residual_rmse", "superellipsoid_residual_rmse"]

def main():
    table = pd.read_csv(OUT / "mesh_metrics_by_time.csv")
    steady = table[table.quasi_steady].copy()
    stats = (steady.groupby(["base_case_id", "mesh_um"], as_index=False)[ALL]
             .agg(["mean", "std", "min", "max"]))
    stats.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in stats.columns]
    stats.reset_index().to_csv(OUT / "mesh_steady_statistics.csv", index=False)

    agg_rows = []
    for mesh, group in steady.groupby("mesh_um"):
        for metric in ALL:
            vals = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            agg_rows.append({"mesh_um": int(mesh), "metric": metric, "n": int(len(vals)),
                             "mean": float(vals.mean()) if len(vals) else np.nan,
                             "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan})
    pd.DataFrame(agg_rows).to_csv(OUT / "mesh_steady_aggregate_by_mesh.csv", index=False)

    status = (table.groupby(["base_case_id", "mesh_um"], as_index=False)
              .agg(n_times=("time_s", "size"),
                   convex_hull_ok=("convex_hull_status", lambda x: int((x == "ok").sum())),
                   ellipsoid_ok=("fit_status", lambda x: int((x == "ok").sum())),
                   superellipsoid_ok=("superellipsoid_fit_status", lambda x: int((x == "ok").sum())),
                   mean_boundary_points=("boundary_point_count", "mean")))
    status["superellipsoid_success_rate"] = status.superellipsoid_ok / status.n_times
    status.to_csv(OUT / "mesh_fit_status_summary.csv", index=False)

    # Ordering check: whether 80 -> 100 -> 200 is monotonic at each time.
    order_rows = []
    for case, group in table.groupby("base_case_id"):
        piv = group.pivot(index="time_s", columns="mesh_um")
        for metric in GEOM + ["Tmax_K", "Gmean_K_per_m", "Umax_m_per_s"]:
            if not all(m in piv[metric].columns for m in [80, 100, 200]):
                continue
            vals = piv[metric][[80, 100, 200]].to_numpy(float)
            valid = np.isfinite(vals).all(axis=1)
            nondec = np.sum(np.all(np.diff(vals[valid], axis=1) >= -1e-15, axis=1))
            noninc = np.sum(np.all(np.diff(vals[valid], axis=1) <= 1e-15, axis=1))
            order_rows.append({"base_case_id": case, "metric": metric, "n_times": int(valid.sum()),
                               "nondecreasing_fraction": nondec / max(valid.sum(), 1),
                               "nonincreasing_fraction": noninc / max(valid.sum(), 1)})
    pd.DataFrame(order_rows).to_csv(OUT / "mesh_monotonicity_check.csv", index=False)

    # Persistence of the superellipsoid advantage.
    ok = table.dropna(subset=["ellipsoid_residual_rmse", "superellipsoid_residual_rmse"])
    advantage = (ok.groupby(["base_case_id", "mesh_um"], as_index=False)
                 .apply(lambda g: pd.Series({"n_times": len(g),
                    "superellipsoid_lower_residual_count": int((g.superellipsoid_residual_rmse < g.ellipsoid_residual_rmse).sum()),
                    "fraction_lower": float((g.superellipsoid_residual_rmse < g.ellipsoid_residual_rmse).mean())}), include_groups=False)
                 .reset_index(drop=True))
    advantage.to_csv(OUT / "mesh_model_selection_consistency.csv", index=False)

    report = """# Mesh sensitivity backup package\n\n## Scope\nThis package documents the mesh-resolution assessment for A1, A2 and A5 using 80, 100 and 200 um liquid-phase point clouds. The 100 um grid is the reference. Quantitative convergence summaries use 0.5-1.0 s; all early-time records remain available in the full table.\n\n## Recommended evidence placement\n- Main text Methods: state the three tested resolutions, identical liquid-phase extraction and the 100 um reference.\n- Main text Results: report only the principal geometric comparison and the reason for retaining 100 um.\n- Supplementary material: provide the time-resolved curves, steady statistics, fit-status table and monotonicity check.\n- Reviewer response: call this a mesh-resolution sensitivity assessment; acknowledge residual sensitivity in depth, front length and flow quantities.\n\n## Boundary of the claim\nThe data support that 100 um is a reasonable baseline for the principal geometric analysis, especially melt-pool width. They do not support a blanket claim that every transient state variable is mesh independent. The 200 um early-time liquid-phase point clouds are sparse and are excluded from quasi-steady averages, not discarded.\n\n## Draft reviewer-response paragraph\nA mesh-resolution sensitivity assessment was added for three representative process conditions (A1, A2 and A5) using 80, 100 and 200 um grids. The same liquid-phase extraction and descriptor calculations were applied to all resolutions. During 0.5-1.0 s, the mean absolute relative difference in melt-pool width between 80 and 100 um was approximately 5% across the three cases, whereas the 200 um grid showed larger deviations. Height, depth, front length and maximum velocity were more resolution sensitive and are now reported as limitations of the numerical discretization. We therefore retained the 100 um grid as the baseline resolution for the reduced-order analysis and avoid claiming universal mesh independence of all transient variables.\n\n## Draft Methods sentence\nThe mesh assessment used liquid-phase point clouds extracted at 80, 100 and 200 um cell sizes for A1, A2 and A5. Identical coordinate transformation, boundary extraction and descriptor definitions were applied at each resolution. Because the coarsest grid produced sparse liquid-phase point clouds during melt-pool formation, convergence statistics were evaluated over the 0.5-1.0 s quasi-steady interval, with early-time records retained for transparency.\n"""
    (OUT / "mesh_revision_backup_material.md").write_text(report, encoding="utf-8")
    print("Wrote cached mesh post-processing outputs")

if __name__ == "__main__":
    main()
