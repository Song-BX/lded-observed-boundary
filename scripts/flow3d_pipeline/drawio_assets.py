"""Helpers for incorporating externally edited draw.io manuscript figures."""

from __future__ import annotations

import shutil
from pathlib import Path


def _open_flattened_rgb(source_png: Path):
    from PIL import Image

    with Image.open(source_png) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
    return background.convert("RGB")


def save_drawio_figure_assets(
    out_base: Path,
    source_png: Path,
    source_svg: Path,
    *,
    png_dpi: int = 300,
    tiff_dpi: int = 600,
) -> None:
    """Write the four manuscript figure formats from draw.io PNG/SVG exports."""
    missing = [path for path in (source_png, source_svg) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing draw.io Figure 1 export(s): {missing_text}")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_svg, out_base.with_suffix(".svg"))

    image = _open_flattened_rgb(source_png)
    image.save(out_base.with_suffix(".png"), dpi=(png_dpi, png_dpi))
    image.save(out_base.with_suffix(".pdf"), "PDF", resolution=float(png_dpi))
    image.save(
        out_base.with_suffix(".tiff"),
        dpi=(tiff_dpi, tiff_dpi),
        compression="tiff_lzw",
    )
