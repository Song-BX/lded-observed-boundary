"""Model fitting and diagnostics interface for the Flow3D pipeline."""

from __future__ import annotations

from .core import (
    compare_dynamics_models,
    fit_attractor_model,
    fit_coupled_attractor_model,
    make_dimensionless_number_table,
    make_dimensionless_sensitivity_summary,
    make_error_budget_summary,
    make_error_summary,
    make_geometry_model_comparison,
    make_model_selection_summary,
    make_parameter_identifiability,
    make_quasi_steady_summary,
    make_superellipsoid_parameters,
    run_robustness_analysis,
)

__all__ = [
    "compare_dynamics_models",
    "fit_attractor_model",
    "fit_coupled_attractor_model",
    "make_dimensionless_number_table",
    "make_dimensionless_sensitivity_summary",
    "make_error_budget_summary",
    "make_error_summary",
    "make_geometry_model_comparison",
    "make_model_selection_summary",
    "make_parameter_identifiability",
    "make_quasi_steady_summary",
    "make_superellipsoid_parameters",
    "run_robustness_analysis",
]
