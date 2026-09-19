"""Single-figure rendering helpers for the Flow3D manuscript pipeline."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config
from .core import (
    compile_latex_package,
    export_paper_figure_set,
    load_property_curves,
    make_active_figure_manifest,
    plot_boundary_fit_comparison,
    plot_convex_alpha_proxy_comparison,
    plot_dimensionless_regime,
    plot_dynamics_model_comparison,
    plot_error_budget_model_selection,
    plot_external_holdout_validation,
    plot_geometry,
    plot_identifiability_overparameterization,
    plot_leave_one_condition_validation,
    plot_modeling_framework,
    plot_moving_frame,
    plot_multi_condition_dynamics_validation,
    plot_multi_condition_geometry_comparison,
    plot_multi_condition_process_matrix,
    plot_multi_condition_response_surfaces,
    plot_simulation_cross_sections,
    plot_supplementary_all_boundary_fits,
    plot_supplementary_dimensionless_grid,
    plot_supplementary_residuals,
    plot_supplementary_superellipsoid_parameters,
    plot_stability_attractor,
    plot_temperature_dependent_properties,
    plot_theory_identifiability_error_bounds,
    plot_thermal_flow,
)


def _read_csv(output_dir: Path, name: str, required_for: str) -> pd.DataFrame:
    path = output_dir / "tables" / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required cache table for {required_for}: {path}. "
            "Run the full pipeline first: python scripts\\flow3d_melt_pool_pilot.py"
        )
    return pd.read_csv(path)


def _copy_formats(source_base: Path, dest_base: Path) -> None:
    dest_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in config.FIGURE_FORMATS:
        src = source_base.with_suffix(f".{suffix}")
        dst = dest_base.with_suffix(f".{suffix}")
        if not src.exists():
            raise FileNotFoundError(f"Expected rendered figure file is missing: {src}")
        shutil.copyfile(src, dst)


def _refresh_manifest_and_compile(output_dir: Path, compile_pdf: bool = True) -> None:
    manifest = make_active_figure_manifest(output_dir)
    manifest.to_csv(output_dir / "tables" / "active_figure_manifest.csv", index=False)
    if compile_pdf:
        compile_latex_package(output_dir)


def _export_single_paper_figure(output_dir: Path, paper_stem: str) -> None:
    source_stem = config.PAPER_FIGURE_MAP[paper_stem]
    _copy_formats(output_dir / "figures" / source_stem, output_dir / "paper_figures" / paper_stem)


def render_paper_figure(paper_stem: str, output_dir: Path = config.OUTPUT_DIR, compile_pdf: bool = True) -> None:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"

    if paper_stem == "paper_fig01_modeling_framework":
        plot_modeling_framework(fig_dir)
    elif paper_stem == "paper_fig02_process_matrix":
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", paper_stem)
        plot_multi_condition_process_matrix(table, fig_dir)
    elif paper_stem == "paper_fig03_data_moving_frame":
        point_cloud = _read_csv(output_dir, "collapsed_point_cloud.csv", paper_stem)
        plot_moving_frame(point_cloud, fig_dir)
    elif paper_stem == "paper_fig04_geometry_quasi_steady":
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", paper_stem)
        plot_geometry(table, fig_dir)
    elif paper_stem == "paper_fig05_free_boundary_model_comparison":
        comparison = _read_csv(output_dir, "geometry_model_comparison.csv", paper_stem)
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", paper_stem)
        point_cloud = _read_csv(output_dir, "collapsed_point_cloud.csv", paper_stem)
        robustness = _read_csv(output_dir, "robustness_summary.csv", paper_stem)
        plot_multi_condition_geometry_comparison(comparison, fig_dir, table, point_cloud, robustness)
    elif paper_stem == "paper_fig06_process_response":
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", paper_stem)
        plot_multi_condition_response_surfaces(table, fig_dir)
    elif paper_stem == "paper_fig07_dimensionless_regime":
        sensitivity = _read_csv(output_dir, "dimensionless_sensitivity_summary.csv", paper_stem)
        plot_dimensionless_regime(sensitivity, fig_dir)
    elif paper_stem == "paper_fig08_dynamics_validation":
        comparison = _read_csv(output_dir, "dynamics_model_comparison.csv", paper_stem)
        plot_multi_condition_dynamics_validation(comparison, fig_dir)
    elif paper_stem == "paper_fig14_dynamics_residuals_by_state":
        predictions = _read_csv(output_dir, "dynamics_predictions.csv", paper_stem)
        coupled_predictions = _read_csv(output_dir, "coupled_dynamics_predictions.csv", paper_stem)
        plot_supplementary_residuals(predictions, coupled_predictions, fig_dir)
    elif paper_stem == "paper_fig09_error_budget_model_selection":
        error_budget = _read_csv(output_dir, "error_budget_summary.csv", paper_stem)
        model_selection = _read_csv(output_dir, "model_selection_summary.csv", paper_stem)
        plot_error_budget_model_selection(error_budget, model_selection, fig_dir)
    elif paper_stem == "paper_fig10_identifiability_overparameterization":
        identifiability = _read_csv(output_dir, "parameter_identifiability.csv", paper_stem)
        coupled_matrix = _read_csv(output_dir, "coupled_A_matrix.csv", paper_stem)
        plot_identifiability_overparameterization(identifiability, coupled_matrix, fig_dir)
    elif paper_stem == "paper_fig11_leave_one_condition_validation":
        loco = _read_csv(output_dir, "leave_one_condition_out_validation.csv", paper_stem)
        plot_leave_one_condition_validation(loco, fig_dir)
    elif paper_stem == "paper_fig12_external_holdout_validation":
        geometry = _read_csv(output_dir, "external_validation_geometry_model_comparison.csv", paper_stem)
        process = _read_csv(output_dir, "external_holdout_process_response_validation.csv", paper_stem)
        dynamics = _read_csv(output_dir, "external_holdout_dynamics_summary.csv", paper_stem)
        plot_external_holdout_validation(geometry, process, dynamics, fig_dir)
    elif paper_stem == "paper_fig13_simulation_cross_sections":
        plot_simulation_cross_sections(fig_dir)
    else:
        raise KeyError(f"Unknown paper figure stem: {paper_stem}")

    _export_single_paper_figure(output_dir, paper_stem)
    _refresh_manifest_and_compile(output_dir, compile_pdf=compile_pdf)


def render_supplementary_figure(
    supp_key: str,
    output_dir: Path = config.OUTPUT_DIR,
    compile_pdf: bool = True,
) -> None:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"

    if supp_key == "supp_figS1":
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", supp_key)
        point_cloud = _read_csv(output_dir, "collapsed_point_cloud.csv", supp_key)
        plot_supplementary_all_boundary_fits(table, point_cloud, fig_dir)
    elif supp_key == "supp_figS2":
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", supp_key)
        plot_supplementary_superellipsoid_parameters(table, fig_dir)
    elif supp_key == "supp_figS3":
        sensitivity = _read_csv(output_dir, "dimensionless_sensitivity_summary.csv", supp_key)
        plot_supplementary_dimensionless_grid(sensitivity, fig_dir)
    elif supp_key == "supp_figS4":
        identifiability = _read_csv(output_dir, "identifiability_diagnostics_v4.csv", supp_key)
        error_bound = _read_csv(output_dir, "error_bound_summary.csv", supp_key)
        sensitivity = _read_csv(output_dir, "dimensionless_sensitivity_summary.csv", supp_key)
        plot_theory_identifiability_error_bounds(identifiability, error_bound, sensitivity, fig_dir)
    elif supp_key == "supp_figS5":
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", supp_key)
        point_cloud = _read_csv(output_dir, "collapsed_point_cloud.csv", supp_key)
        plot_convex_alpha_proxy_comparison(table, point_cloud, fig_dir)
    elif supp_key == "supp_figS6":
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", supp_key)
        dynamics_summary = _read_csv(output_dir, "dynamics_fit_summary.csv", supp_key)
        eigenvalues = _read_csv(output_dir, "stability_eigenvalues.csv", supp_key)
        coupled_eigenvalues = _read_csv(output_dir, "coupled_stability_eigenvalues.csv", supp_key)
        plot_stability_attractor(table, dynamics_summary, eigenvalues, coupled_eigenvalues, fig_dir)
    elif supp_key == "supp_figS7":
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", supp_key)
        point_cloud = _read_csv(output_dir, "collapsed_point_cloud.csv", supp_key)
        plot_boundary_fit_comparison(table, point_cloud, fig_dir)
    elif supp_key == "supp_figS8":
        table = _read_csv(output_dir, "multi_condition_modeling_table.csv", supp_key)
        plot_thermal_flow(table, fig_dir)
    elif supp_key == "supp_figS9":
        predictions = _read_csv(output_dir, "dynamics_predictions.csv", supp_key)
        coupled_predictions = _read_csv(output_dir, "coupled_dynamics_predictions.csv", supp_key)
        comparison = _read_csv(output_dir, "dynamics_model_comparison.csv", supp_key)
        plot_dynamics_model_comparison(predictions, coupled_predictions, comparison, fig_dir)
    elif supp_key == "supp_figS10":
        plot_temperature_dependent_properties(load_property_curves(output_dir.parent), fig_dir)
    else:
        raise KeyError(f"Unknown supplementary figure key: {supp_key}")

    _refresh_manifest_and_compile(output_dir, compile_pdf=compile_pdf)


def render_named_figure(name: str, output_dir: Path = config.OUTPUT_DIR, compile_pdf: bool = True) -> None:
    normalized = name.lower().replace("_", "").replace("-", "")
    if normalized.startswith("fig") and normalized[:5] in {key.replace("_", "") for key in config.MAIN_FIGURE_SCRIPTS}:
        key = f"fig{normalized[3:5]}"
        render_paper_figure(config.MAIN_FIGURE_SCRIPTS[key], output_dir, compile_pdf=compile_pdf)
        return
    if normalized.startswith("suppfigs"):
        key = f"supp_figS{normalized.split('suppfigs', 1)[1]}"
        render_supplementary_figure(key, output_dir, compile_pdf=compile_pdf)
        return
    raise KeyError(f"Unknown figure name: {name}")


def main_cli(render: Callable[..., None], figure_id: str) -> None:
    parser = argparse.ArgumentParser(description=f"Render {figure_id} from cached Flow3D analysis tables.")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--no-compile", action="store_true", help="Render figure files without recompiling LaTeX PDFs.")
    args = parser.parse_args()
    render(args.output_dir, not args.no_compile)
