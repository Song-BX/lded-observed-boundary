from _bootstrap import ensure_pipeline_path

ensure_pipeline_path()

from flow3d_pipeline.figures import main_cli, render_paper_figure


def _render(output_dir, compile_pdf: bool) -> None:
    render_paper_figure("paper_fig14_dynamics_residuals_by_state", output_dir, compile_pdf)


if __name__ == "__main__":
    main_cli(_render, "Figure 14 dynamics residuals by state")
