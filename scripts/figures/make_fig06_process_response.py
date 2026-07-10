from _bootstrap import ensure_pipeline_path

ensure_pipeline_path()

from flow3d_pipeline.figures import main_cli, render_paper_figure


def _render(output_dir, compile_pdf):
    render_paper_figure("paper_fig06_process_response", output_dir, compile_pdf)


if __name__ == "__main__":
    main_cli(_render, "Figure 6")
