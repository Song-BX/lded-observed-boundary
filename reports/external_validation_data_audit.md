# External CFD Holdout Validation Data Audit

- Validation source: `validation data/`
- External cases processed: 5
- External CSV time-step files processed: 60
- Cases ready for full geometry, thermal-flow and dynamics validation: 5/5

## Case Audit

| case_id | power_W | scan_speed_mm_s | particle_rate | powder_feed_g_min | csv_count | time_min_s | time_max_s | geometry_validation_ready | dynamics_validation_ready |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1-450-7-45000 | 450 | 7 | 45000 | 9 | 12 | 0.05 | 1 | True | True |
| V2-650-8.5-55000 | 650 | 8.5 | 55000 | 11 | 12 | 0.05 | 1 | True | True |
| V3-750-9-50000 | 750 | 9 | 50000 | 10 | 12 | 0.05 | 1 | True | True |
| V4-850-7.5-65000 | 850 | 7.5 | 65000 | 13 | 12 | 0.05 | 1 | True | True |
| V5-900-9.5-45000 | 900 | 9.5 | 45000 | 9 | 12 | 0.05 | 1 | True | True |

## Holdout Metrics

| metric | value | unit | interpretation |
| --- | --- | --- | --- |
| external_validation_case_count | 5 | cases | Independent V-prefixed Flow3D holdout conditions. |
| external_validation_time_step_count | 60 | condition-time steps | Total external CFD time-step exports processed by the same descriptor pipeline. |
| external_superellipsoid_boundary_win_rate | 1 | fraction | Superellipsoid boundary residual improves in 5/5 external cases. |
| external_superellipsoid_volume_win_rate | 0.4 | fraction | Superellipsoid volume proxy improves in 2/5 external cases. |
| external_process_response_mean_relative_error | 0.0483877 | relative error | Mean quasi-steady process-response error on V-prefixed holdout cases. |
| external_process_response_max_relative_error | 0.305589 | relative error | Largest quasi-steady process-response relative error over all external case-target pairs. |
| external_process_response_worst_target_mean_relative_error | 0.111266 | relative error | volume_proxy_m3 |
| external_dynamics_mean_relative_rmse | 0.123023 | relative RMSE | Mean process-parameterized diagonal-attractor trajectory error on external cases. |
| external_dynamics_max_relative_rmse | 0.63113 | relative RMSE | Largest state-wise external diagonal-attractor trajectory error. |
| external_dynamics_worst_state_mean_relative_rmse | 0.349661 | relative RMSE | Umax_m_per_s |

## Interpretation

The V-prefixed cases are processed separately from the A-prefixed training process matrix. They are therefore used as an external CFD holdout: the descriptor extraction and boundary fitting are applied to the validation files, while process-response and trajectory-prediction errors are evaluated against relationships learned from the training cases.
