"""Configuration and figure registry for the Flow3D L-DED pipeline."""

from __future__ import annotations

from pathlib import Path

from .core import (
    BOUNDARY_FIT_TIMES,
    CANONICAL_EXPORT_COLUMNS,
    MATERIAL_CONSTANTS,
    QUASI_STEADY_START_S,
    STATE_COLUMNS,
    STATE_LABELS,
    TRAIN_FRACTION,
)

OUTPUT_DIR = Path("analysis_outputs")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
PAPER_FIGURES_DIR = OUTPUT_DIR / "paper_figures"
LATEX_DIR = OUTPUT_DIR / "latex"
FIGURE_FORMATS = ("svg", "pdf", "tiff", "png")

PAPER_FIGURE_MAP = {
    "paper_fig01_modeling_framework": "fig08_modeling_framework",
    "paper_fig02_process_matrix": "fig13_multicondition_process_matrix",
    "paper_fig03_data_moving_frame": "fig01_moving_frame_point_cloud",
    "paper_fig04_geometry_quasi_steady": "fig02_geometry_evolution",
    "paper_fig05_free_boundary_model_comparison": "fig15_multicondition_geometry_comparison",
    "paper_fig06_process_response": "fig14_multicondition_response_surfaces",
    "paper_fig07_dimensionless_regime": "fig09_dimensionless_regime",
    "paper_fig08_dynamics_validation": "fig16_multicondition_dynamics_validation",
    "paper_fig09_error_budget_model_selection": "fig11_error_budget_model_selection",
    "paper_fig10_identifiability_overparameterization": "fig12_identifiability_overparameterization",
    "paper_fig11_leave_one_condition_validation": "fig17_leave_one_condition_validation",
}

MAIN_FIGURE_SCRIPTS = {
    "fig01": "paper_fig01_modeling_framework",
    "fig02": "paper_fig02_process_matrix",
    "fig03": "paper_fig03_data_moving_frame",
    "fig04": "paper_fig04_geometry_quasi_steady",
    "fig05": "paper_fig05_free_boundary_model_comparison",
    "fig06": "paper_fig06_process_response",
    "fig07": "paper_fig07_dimensionless_regime",
    "fig08": "paper_fig08_dynamics_validation",
    "fig09": "paper_fig09_error_budget_model_selection",
    "fig10": "paper_fig10_identifiability_overparameterization",
    "fig11": "paper_fig11_leave_one_condition_validation",
}

SUPPLEMENTARY_FIGURE_STEMS = {
    "supp_figS1": "supp_figS1_all_boundary_fits",
    "supp_figS2": "supp_figS2_superellipsoid_parameters",
    "supp_figS3": "supp_figS3_dynamics_residuals",
    "supp_figS4": "supp_figS4_dimensionless_sensitivity_grid",
    "supp_figS5": "supp_figS5_theory_identifiability_error_bounds",
    "supp_figS6": "fig03_thermal_flow_evolution",
    "supp_figS7": "fig06_dynamics_model_comparison",
}
