# Figure Captions

**Figure 1. Modeling framework for CFD-informed observed boundary-envelope identification.** The workflow converts multi-condition FLOW-3D half-domain molten-region point clouds into symmetry-reconstructed moving-frame observed boundary envelopes, fits analytic superellipsoid manifolds, identifies condition-wise attractors and evaluates process response, stability, error budget and parameter identifiability. Source files: `paper_fig01_modeling_framework.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 2. Multi-condition process matrix.** The 15 training FLOW-3D conditions span laser power, scan speed and powder feed, with powder feed converted from particle generation rate. Source files: `paper_fig02_process_matrix.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 3. Moving-frame reconstruction of the molten region.** The representative baseline condition shows raw half-domain export, symmetry reconstruction, moving-frame alignment and the reduced observed boundary-envelope descriptors `Lf`, `Lr`, `W` and `H`. Source files: `paper_fig03_data_moving_frame.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 4. Transient geometry and quasi-steady approach.** Time histories of front length, rear length, full width and height show the evolution from early transient growth toward a quasi-steady regime after approximately 0.20 s. Source files: `paper_fig04_geometry_quasi_steady.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 5. Cross-condition observed boundary-envelope model comparison.** Boundary-envelope data compare the asymmetric ellipsoid baseline with the asymmetric superellipsoid model over the process matrix. Source files: `paper_fig05_free_boundary_model_comparison.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 6. Quasi-steady process-response diagnostics.** Quasi-steady length, width, height and maximum temperature are plotted across the power-speed matrix; marker area encodes powder feed. Source files: `paper_fig06_process_response.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 7. Dimensionless regime and sensitivity.** Baseline values of `Pe`, `Ste`, `E*` and `Ma` are plotted with perturbation ranges under reference-temperature, absorptivity and surface-tension-coefficient changes. Source files: `paper_fig07_dimensionless_regime.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 8. Cross-condition dynamics validation.** Condition-wise and state-wise validation errors compare the diagonal attractor with the coupled ridge attractor. Source files: `paper_fig08_dynamics_validation.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 9. Error budget and model selection.** The diagnostic error budget is shown alongside the model-selection summary. Source files: `paper_fig09_error_budget_model_selection.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 10. Identifiability and overparameterization.** Superellipsoid parameter variation and coupled-matrix diagnostics motivate the selected model and the non-selected coupled comparison. Source files: `paper_fig10_identifiability_overparameterization.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 11. Leave-one-condition-out validation.** A process-response extrapolation test holds out one training condition at a time; the prediction panel uses target-wise normalization so quantities with different units remain visually comparable. Source files: `paper_fig11_leave_one_condition_validation.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 12. External CFD holdout validation.** Five V-prefixed FLOW-3D conditions are withheld from model construction and used to test boundary-model transfer, quasi-steady process-response prediction and process-parameterized diagonal-attractor trajectories. These holdout conditions were not used for boundary-model selection or attractor-baseline selection. Source files: `paper_fig12_external_holdout_validation.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S1. Boundary fits across all time steps.** Top-view superellipsoid overlays are shown for all exported time steps in the representative condition, providing a visual audit of the boundary model beyond the main-text panels. Source files: `supp_figS1_all_boundary_fits.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S2. Superellipsoid parameters versus time.** The fitted semi-axes, center coordinates and shape exponents are plotted over time to show parameter evolution and quasi-steady behavior. Source files: `supp_figS2_superellipsoid_parameters.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S3. Dynamical residuals by state.** Residuals for the diagonal and coupled attractor models are shown for each reduced state variable, separating training behavior from validation behavior. Source files: `supp_figS3_dynamics_residuals.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S4. Dimensionless sensitivity scenario grid.** Relative minimum, baseline and maximum values for `Pe`, `Ste`, `E*` and `Ma` summarize the full perturbation envelope used in the sensitivity analysis. Source files: `supp_figS4_dimensionless_sensitivity_grid.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S5. Theory, identifiability and error-budget diagnostics.** The semi-formal error-budget terms, v4 parameter-identifiability risk levels and nondimensional sensitivity spans are shown together to support the strengthened mathematical modeling argument. Source files: `supp_figS5_theory_identifiability_error_bounds.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S6. Representative-condition stability and attractor evidence.** State-error convergence, fitted diagonal rates and coupled eigenvalues support the stability discussion. Source files: `fig10_stability_attractor.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S7. Representative boundary-envelope time-step overlays.** Top and side views show the ellipsoid and superellipsoid envelopes at selected times for the representative condition. Source files: `fig05_boundary_fit_comparison.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S8. Thermal-flow state evolution.** Time histories of `Tmax`, `Gmean` and `Umax` provide the thermal-flow evidence behind the reduced state variables used in the attractor model. The quasi-steady marker highlights the transition after approximately 0.20 s. Source files: `fig03_thermal_flow_evolution.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S9. Dynamical model trajectory comparison.** State-wise trajectories compare the observed reduced states with the diagonal attractor and coupled ridge attractor. This figure complements the residual plot by showing the prediction curves directly and supports the conclusion that the coupled model does not improve validation accuracy. Source files: `fig06_dynamics_model_comparison.svg`, `.pdf`, `.tiff`, and `.png`.
