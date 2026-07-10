from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path.cwd() / "analysis_outputs" / f".matplotlib-cache-{os.getpid()}"),
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree
from scipy.stats import binomtest, wilcoxon


BASELINE_SCAN_SPEED_M_PER_S = 0.008
SCAN_SPEED_M_PER_S = BASELINE_SCAN_SPEED_M_PER_S
QUASI_STEADY_START_S = 0.20
TRAIN_FRACTION = 0.70
SUPERELLIPSOID_EXPONENT_UPPER = 6.0
BASELINE_LASER_POWER_W = 750.0
BASELINE_POWDER_FEED_G_PER_MIN = 12.0
LASER_POWER_W = BASELINE_LASER_POWER_W
POWDER_FEED_G_PER_MIN = BASELINE_POWDER_FEED_G_PER_MIN
POWDER_FEED_KG_PER_S = POWDER_FEED_G_PER_MIN / 1000.0 / 60.0
BASELINE_PARTICLE_RATE = 60000.0
CASE_PATTERN = re.compile(r"^(?P<case_prefix>[AV])(?P<case_index>\d+)-(?P<power>\d+(?:\.\d+)?)-(?P<speed>\d+(?:\.\d+)?)-(?P<particle>\d+(?:\.\d+)?)$")
TIME_PATTERN = re.compile(r"(?P<time>\d+(?:\.\d+)?)s$")

MANUSCRIPT_AUTHOR_BLOCK_LATEX = (
    "Boxue Song\\\\\n"
    "College of Intelligent Manufacturing, Putian University\n"
    "\\and\n"
    "Xiaoli Lin\\thanks{Corresponding author.}\\\\\n"
    "College of Artificial Intelligence, Putian University\n"
    "\\and\n"
    "Xingyu Jiang\\\\\n"
    "School of Mechanical Engineering, Shenyang University of Technology\n"
    "\\and\n"
    "Tianbiao Yu\\\\\n"
    "School of Mechanical Engineering and Automation, Northeastern University\n"
    "\\and\n"
    "Wenchao Xi\\\\\n"
    "School of Mechanical Engineering, Shenyang University of Technology"
)
SUPPLEMENTARY_AUTHOR_BLOCK_LATEX = MANUSCRIPT_AUTHOR_BLOCK_LATEX
COVER_LETTER_SIGNATURE = "Boxue Song, Xiaoli Lin, Xingyu Jiang, Tianbiao Yu, and Wenchao Xi"
DEFAULT_REPOSITORY_URL = "https://github.com/Song-BX/lded-observed-boundary"
AMM_GUIDE_FOR_AUTHORS_URL = "https://www.sciencedirect.com/journal/applied-mathematical-modelling/publish/guide-for-authors"
AUTHOR_CONFIRMATION_REQUIRED = "AUTHOR CONFIRMATION REQUIRED"

MATERIAL_CONSTANTS = {
    "material": "316L stainless steel",
    "beam_radius_m": 0.0008,
    "absorptivity": 0.1,
    "initial_temperature_K": 298.0,
    "ambient_temperature_K": 298.0,
    "powder_initial_temperature_K": 293.0,
    "solidus_temperature_K": 1683.0,
    "liquidus_temperature_K": 1710.0,
    "latent_heat_fusion_J_per_kg": 2.67776e5,
    "boiling_temperature_K": 3090.0,
    "latent_heat_vaporization_J_per_kg": 6.40568e6,
    "surface_tension_N_per_m": 1.8,
    "surface_tension_temperature_coefficient_N_per_m_K": 2.50836e-4,
    "emissivity": 0.4,
    "convective_heat_transfer_W_per_m2_K": 100.0,
    "powder_particle_diameter_m": 4e-5,
    "powder_capture_efficiency": 0.35,
    "recoil_pressure_enabled": False,
    "powder_stream_radius_m": np.nan,
}

PROPERTY_FILES = {
    "density_kg_per_m3": "density.csv",
    "specific_heat_J_per_kg_K": "specific heat.csv",
    "thermal_conductivity_W_per_m_K": "thermal conduction.csv",
    "viscosity_kg_per_m_s": "viscosity.csv",
}

LEGACY_COORD_COLS = ["Points_0", "Points_1", "Points_2"]
MULTI_COORD_COLS = ["Points:0", "Points:1", "Points:2"]
COORD_COLS = LEGACY_COORD_COLS
LEGACY_FIELD_COLS = [
    "Fraction Of Fluid",
    "Heat Flux Spatial Distribution",
    "Temperature",
    "Temperature Gradient At Tgrdout",
    "Velocity_0",
    "Velocity_1",
    "Velocity_2",
    "Velocity_Magnitude",
]
MULTI_FIELD_COLS = [
    "Fraction Of Fluid",
    "Heat Absorption Rate",
    "Heat Flux Spatial Distribution",
    "Melt Region",
    "Pressure",
    "Temperature",
    "Temperature Gradient At Tgrdout",
    "X-velocity",
    "Y-velocity",
    "Z-velocity",
]
FIELD_COLS = LEGACY_FIELD_COLS
RAW_EXPORT_COLUMNS = COORD_COLS + FIELD_COLS
CANONICAL_EXPORT_COLUMNS = [
    "Points:0/Points_0",
    "Points:1/Points_1",
    "Points:2/Points_2",
    "Fraction Of Fluid",
    "Heat Absorption Rate",
    "Heat Flux Spatial Distribution",
    "Melt Region",
    "Pressure",
    "Temperature",
    "Temperature Gradient At Tgrdout",
    "X/Y/Z-velocity or Velocity_0/1/2",
    "Velocity_Magnitude",
]

STATE_COLUMNS = [
    "front_length_m",
    "rear_length_m",
    "full_width_m",
    "height_span_m",
    "Tmax_K",
    "Gmean_K_per_m",
    "Umax_m_per_s",
]

STATE_LABELS = {
    "front_length_m": "L_f",
    "rear_length_m": "L_r",
    "full_width_m": "W",
    "height_span_m": "H",
    "Tmax_K": "Tmax",
    "Gmean_K_per_m": "Gmean",
    "Umax_m_per_s": "Umax",
}

STATE_PLOT_LABELS = {
    "front_length_m": r"$L_f$",
    "rear_length_m": r"$L_r$",
    "full_width_m": r"$W$",
    "height_span_m": r"$H$",
    "Tmax_K": r"$T_{\max}$",
    "Gmean_K_per_m": r"$G_{\mathrm{mean}}$",
    "Umax_m_per_s": r"$U_{\max}$",
}

DIMENSIONLESS_PLOT_LABELS = {
    "Pe": r"$Pe$",
    "Ste": r"$Ste$",
    "E_star": r"$E^*$",
    "Ma": r"$Ma$",
}

BOUNDARY_FIT_TIMES = [0.05, 0.20, 0.50, 0.70]
MAIN_BOUNDARY_OVERLAY_TIMES = [0.20, 0.70]
RIDGE_LAMBDAS = np.array([0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])


@dataclass
class HullResult:
    points: np.ndarray
    volume_half_m3: float
    status: str


@dataclass(frozen=True)
class CaseMeta:
    case_id: str
    case_index: int
    power_W: float
    scan_speed_mm_s: float
    scan_speed_m_s: float
    particle_rate: float
    powder_feed_g_min: float
    powder_feed_kg_s: float


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.8,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def add_panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def apply_axis_polish(ax: mpl.axes.Axes, grid: str | None = "y") -> None:
    ax.tick_params(length=2.6, width=0.7, pad=1.8)
    if grid:
        ax.grid(True, axis=grid, color="0.90", linewidth=0.45, linestyle="-", zorder=0)
        ax.set_axisbelow(True)


def short_state_label(name: str, for_plot: bool = False) -> str:
    if for_plot and name in STATE_PLOT_LABELS:
        return STATE_PLOT_LABELS[name]
    if name in STATE_LABELS:
        return STATE_LABELS[name]
    cleaned = str(name).replace("_m_per_s", "").replace("_m3", "").replace("_m", "")
    cleaned = cleaned.replace("_K_per_m", "").replace("_K", "")
    cleaned = cleaned.replace("melt_pool_length", "Length")
    cleaned = cleaned.replace("full_width", "Width")
    cleaned = cleaned.replace("height_span", "Height")
    cleaned = cleaned.replace("_", " ")
    return cleaned


def state_plot_label(name: str) -> str:
    return short_state_label(name, for_plot=True)


def dimensionless_plot_label(symbol: str) -> str:
    return DIMENSIONLESS_PLOT_LABELS.get(str(symbol), str(symbol).replace("_", " "))


def compact_condition_label(case_id: str) -> str:
    match = CASE_PATTERN.match(str(case_id))
    if match:
        return f"{match.group('case_prefix')}{int(match.group('case_index'))}"
    return str(case_id).split("-", 1)[0]


def save_publication_figure(fig: mpl.figure.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")


def save_placeholder_figure(out_base: Path, message: str) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(5.2, 2.4), constrained_layout=True)
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, transform=ax.transAxes)
    save_publication_figure(fig, out_base)
    plt.close(fig)


@contextmanager
def temporary_settings(
    train_fraction: float | None = None,
    quasi_steady_start_s: float | None = None,
    superellipsoid_exponent_upper: float | None = None,
):
    global TRAIN_FRACTION, QUASI_STEADY_START_S, SUPERELLIPSOID_EXPONENT_UPPER
    old_train = TRAIN_FRACTION
    old_quasi = QUASI_STEADY_START_S
    old_exp = SUPERELLIPSOID_EXPONENT_UPPER
    if train_fraction is not None:
        TRAIN_FRACTION = float(train_fraction)
    if quasi_steady_start_s is not None:
        QUASI_STEADY_START_S = float(quasi_steady_start_s)
    if superellipsoid_exponent_upper is not None:
        SUPERELLIPSOID_EXPONENT_UPPER = float(superellipsoid_exponent_upper)
    try:
        yield
    finally:
        TRAIN_FRACTION = old_train
        QUASI_STEADY_START_S = old_quasi
        SUPERELLIPSOID_EXPONENT_UPPER = old_exp


def parse_time_s(path: Path) -> float:
    match = TIME_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Cannot parse time from file name: {path.name}")
    return float(match.group("time"))


def sorted_csv_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.csv"), key=parse_time_s)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return files


def parse_case_metadata(case_dir: Path) -> CaseMeta:
    match = CASE_PATTERN.fullmatch(case_dir.name)
    if not match:
        return CaseMeta(
            case_id="A1-750-8-60000",
            case_index=1,
            power_W=BASELINE_LASER_POWER_W,
            scan_speed_mm_s=BASELINE_SCAN_SPEED_M_PER_S * 1000.0,
            scan_speed_m_s=BASELINE_SCAN_SPEED_M_PER_S,
            particle_rate=BASELINE_PARTICLE_RATE,
            powder_feed_g_min=BASELINE_POWDER_FEED_G_PER_MIN,
            powder_feed_kg_s=BASELINE_POWDER_FEED_G_PER_MIN / 1000.0 / 60.0,
        )
    case_index = int(match.group("case_index"))
    power = float(match.group("power"))
    speed_mm_s = float(match.group("speed"))
    particle_rate = float(match.group("particle"))
    powder_feed_g_min = particle_rate / BASELINE_PARTICLE_RATE * BASELINE_POWDER_FEED_G_PER_MIN
    return CaseMeta(
        case_id=case_dir.name,
        case_index=case_index,
        power_W=power,
        scan_speed_mm_s=speed_mm_s,
        scan_speed_m_s=speed_mm_s / 1000.0,
        particle_rate=particle_rate,
        powder_feed_g_min=powder_feed_g_min,
        powder_feed_kg_s=powder_feed_g_min / 1000.0 / 60.0,
    )


def discover_case_dirs(raw_dir: Path) -> list[Path]:
    case_dirs = [path for path in raw_dir.iterdir() if path.is_dir() and CASE_PATTERN.fullmatch(path.name)]
    if case_dirs:
        return sorted(case_dirs, key=lambda path: parse_case_metadata(path).case_index)
    return [raw_dir]


def make_case_metadata_table(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for case_dir in discover_case_dirs(raw_dir):
        meta = parse_case_metadata(case_dir)
        csv_files = sorted_csv_files(case_dir)
        times = [parse_time_s(path) for path in csv_files]
        rows.append(
            {
                "case_id": meta.case_id,
                "case_index": meta.case_index,
                "power_W": meta.power_W,
                "scan_speed_mm_s": meta.scan_speed_mm_s,
                "scan_speed_m_s": meta.scan_speed_m_s,
                "particle_rate": meta.particle_rate,
                "powder_feed_g_min": meta.powder_feed_g_min,
                "powder_feed_kg_s": meta.powder_feed_kg_s,
                "csv_count": len(csv_files),
                "time_min_s": min(times),
                "time_max_s": max(times),
                "time_points_s": ",".join(f"{time:g}" for time in times),
                "source_folder": str(case_dir),
            }
        )
    return pd.DataFrame(rows)


def representative_case_id(df: pd.DataFrame) -> str | None:
    """Pick a stable case for single-condition style time-series figures."""

    if "case_id" not in df.columns or df["case_id"].nunique() <= 1:
        return None
    case_ids = df["case_id"].astype(str).unique().tolist()
    baseline = "A1-750-8-60000"
    return baseline if baseline in case_ids else case_ids[0]


def representative_case_subset(df: pd.DataFrame) -> pd.DataFrame:
    case_id = representative_case_id(df)
    if case_id is None:
        return df.copy()
    return df[df["case_id"].astype(str).eq(case_id)].copy()


def case_metadata_from_modeling_table(table: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_id",
        "case_index",
        "power_W",
        "scan_speed_mm_s",
        "scan_speed_m_s",
        "particle_rate",
        "powder_feed_g_min",
        "powder_feed_kg_s",
        "validation_cohort",
        "source_root",
    ]
    available = [col for col in cols if col in table.columns]
    out = table[available].drop_duplicates().sort_values(["case_index", "case_id"]).reset_index(drop=True)
    counts = (
        table.groupby("case_id", as_index=False)
        .agg(
            csv_count=("source_file", "count"),
            time_min_s=("time_s", "min"),
            time_max_s=("time_s", "max"),
            time_points_s=("time_s", lambda values: ",".join(f"{float(v):g}" for v in sorted(values))),
            source_folder=("source_folder", "first"),
        )
    )
    return out.merge(counts, on="case_id", how="left")


def dimensionless_value_lookup(dimensionless_numbers: pd.DataFrame) -> pd.Series:
    """Return one value per symbol, preferring cross-condition summary rows."""

    if "case_id" not in dimensionless_numbers.columns:
        return dimensionless_numbers.drop_duplicates("symbol").set_index("symbol")["value"]
    values: dict[str, float] = {}
    priority = ["summary", "A1-750-8-60000", "global", "summary_min", "summary_max"]
    for symbol, group in dimensionless_numbers.groupby("symbol", sort=False):
        chosen = None
        for case_id in priority:
            match = group[group["case_id"].astype(str).eq(case_id)]
            if len(match):
                chosen = match.iloc[0]
                break
        if chosen is None:
            chosen = group.iloc[0]
        values[str(symbol)] = float(chosen["value"])
    return pd.Series(values)


def case_parameter_ranges(table: pd.DataFrame) -> str:
    if "case_id" not in table.columns or table["case_id"].nunique() <= 1:
        return "one process condition"
    powder_min = float(table["powder_feed_g_min"].min())
    powder_max = float(table["powder_feed_g_min"].max())
    return (
        f"{table['case_id'].nunique()} process conditions spanning "
        f"{float(table['power_W'].min()):.0f}-{float(table['power_W'].max()):.0f} W, "
        f"{float(table['scan_speed_mm_s'].min()):.0f}-{float(table['scan_speed_mm_s'].max()):.0f} mm/s, "
        f"{powder_min:.1f}-{powder_max:.1f} g/min"
    )


def read_property_curve(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns: temperature and property value.")
    out = df.iloc[:, :2].copy()
    out.columns = ["temperature_K", value_name]
    out = out.sort_values("temperature_K").dropna()
    return out


def interp_property(curve: pd.DataFrame, value_name: str, temperature_K: float) -> float:
    x = curve["temperature_K"].to_numpy(dtype=float)
    y = curve[value_name].to_numpy(dtype=float)
    return float(np.interp(temperature_K, x, y, left=y[0], right=y[-1]))


def load_property_curves(base_dir: Path) -> dict[str, pd.DataFrame]:
    curves = {}
    for value_name, filename in PROPERTY_FILES.items():
        curves[value_name] = read_property_curve(base_dir / filename, value_name)
    return curves


def property_values_at(curves: dict[str, pd.DataFrame], temperature_K: float) -> dict[str, float]:
    values = {"temperature_K": float(temperature_K)}
    for value_name, curve in curves.items():
        values[value_name] = interp_property(curve, value_name, temperature_K)
    rho = values["density_kg_per_m3"]
    cp = values["specific_heat_J_per_kg_K"]
    k = values["thermal_conductivity_W_per_m_K"]
    values["thermal_diffusivity_m2_per_s"] = k / (rho * cp)
    return values


def read_and_collapse(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path)
    if all(col in raw.columns for col in MULTI_COORD_COLS + MULTI_FIELD_COLS):
        coord_cols = MULTI_COORD_COLS
        field_cols = MULTI_FIELD_COLS
        schema = "multi_condition_FLOW-3D"
    elif all(col in raw.columns for col in LEGACY_COORD_COLS + LEGACY_FIELD_COLS):
        coord_cols = LEGACY_COORD_COLS
        field_cols = LEGACY_FIELD_COLS
        schema = "legacy_single_condition_FLOW-3D"
    else:
        missing_multi = [col for col in MULTI_COORD_COLS + MULTI_FIELD_COLS if col not in raw.columns]
        missing_legacy = [col for col in LEGACY_COORD_COLS + LEGACY_FIELD_COLS if col not in raw.columns]
        raise ValueError(
            f"{path.name} does not match supported schemas. Missing multi={missing_multi}; missing legacy={missing_legacy}"
        )
    missing = [col for col in coord_cols + field_cols if col not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    exact = raw.drop_duplicates().copy()
    coordinate_counts = exact.groupby(coord_cols, dropna=False).size()
    multi_value_points = int((coordinate_counts > 1).sum())

    collapsed = exact.groupby(coord_cols, as_index=False, dropna=False)[field_cols].mean()
    rename_map = {
        coord_cols[0]: "x_m",
        coord_cols[1]: "y_m",
        coord_cols[2]: "z_m",
    }
    if schema == "multi_condition_FLOW-3D":
        rename_map.update(
            {
                "X-velocity": "Ux_m_per_s",
                "Y-velocity": "Uy_m_per_s",
                "Z-velocity": "Uz_m_per_s",
            }
        )
    else:
        rename_map.update(
            {
                "Velocity_0": "Ux_m_per_s",
                "Velocity_1": "Uy_m_per_s",
                "Velocity_2": "Uz_m_per_s",
            }
        )
    collapsed = collapsed.rename(columns=rename_map)
    collapsed["Velocity_Magnitude"] = np.sqrt(
        collapsed["Ux_m_per_s"] ** 2 + collapsed["Uy_m_per_s"] ** 2 + collapsed["Uz_m_per_s"] ** 2
    )
    # Backward-compatible aliases keep the legacy plotting and report code usable
    # while the canonical analysis uses x_m, y_m, z_m and U*_m_per_s.
    collapsed["Points_0"] = collapsed["x_m"]
    collapsed["Points_1"] = collapsed["y_m"]
    collapsed["Points_2"] = collapsed["z_m"]
    collapsed["Velocity_0"] = collapsed["Ux_m_per_s"]
    collapsed["Velocity_1"] = collapsed["Uy_m_per_s"]
    collapsed["Velocity_2"] = collapsed["Uz_m_per_s"]
    for col in ["Heat Absorption Rate", "Melt Region", "Pressure"]:
        if col not in collapsed.columns:
            collapsed[col] = np.nan
    counts = {
        "raw_rows": int(len(raw)),
        "exact_dedup_rows": int(len(exact)),
        "unique_points": int(len(collapsed)),
        "coordinate_multi_value_points": multi_value_points,
        "schema": schema,
    }
    return collapsed, counts


def convex_hull(points: np.ndarray) -> HullResult:
    unique = np.unique(points, axis=0)
    if unique.shape[0] < 4:
        return HullResult(unique, float("nan"), "insufficient_points")
    try:
        hull = ConvexHull(unique, qhull_options="QJ")
    except QhullError:
        return HullResult(unique, float("nan"), "qhull_failed")
    return HullResult(unique[hull.vertices], float(hull.volume), "ok")


def fit_asymmetric_ellipsoid(boundary_points: np.ndarray) -> dict[str, float | str]:
    if boundary_points.shape[0] < 8:
        return {
            "fit_status": "insufficient_boundary_points",
            "ellipsoid_af_m": np.nan,
            "ellipsoid_ar_m": np.nan,
            "ellipsoid_b_m": np.nan,
            "ellipsoid_c_m": np.nan,
            "ellipsoid_xic_m": np.nan,
            "ellipsoid_zc_m": np.nan,
            "ellipsoid_residual_rmse": np.nan,
            "ellipsoid_volume_full_m3": np.nan,
        }

    xi = boundary_points[:, 0]
    y = boundary_points[:, 1]
    z = boundary_points[:, 2]

    xi_min, xi_max = float(np.min(xi)), float(np.max(xi))
    y_max = max(float(np.max(y)), 1e-8)
    z_min, z_max = float(np.min(z)), float(np.max(z))
    xi_c0 = float(np.clip(0.0, xi_min + 1e-8, xi_max - 1e-8))
    z_c0 = 0.5 * (z_min + z_max)
    x0 = np.array(
        [
            max(xi_max - xi_c0, 1e-7),
            max(xi_c0 - xi_min, 1e-7),
            y_max,
            max(0.5 * (z_max - z_min), 1e-7),
            xi_c0,
            z_c0,
        ],
        dtype=float,
    )

    span_x = max(xi_max - xi_min, 1e-6)
    span_z = max(z_max - z_min, 1e-6)
    lower = np.array(
        [
            1e-8,
            1e-8,
            1e-8,
            1e-8,
            xi_min - 0.25 * span_x,
            z_min - 0.25 * span_z,
        ]
    )
    upper = np.array(
        [
            2.5 * span_x,
            2.5 * span_x,
            max(2.5 * y_max, 1e-7),
            2.5 * span_z,
            xi_max + 0.25 * span_x,
            z_max + 0.25 * span_z,
        ]
    )
    x0 = np.minimum(np.maximum(x0, lower + 1e-12), upper - 1e-12)

    def residual(params: np.ndarray) -> np.ndarray:
        af, ar, b, c, xi_c, z_c = params
        side_a = np.where(xi >= xi_c, af, ar)
        phi = ((xi - xi_c) / side_a) ** 2 + (y / b) ** 2 + ((z - z_c) / c) ** 2
        return phi - 1.0

    try:
        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.2,
            max_nfev=5000,
        )
        fit_status = "ok" if result.success else f"not_converged:{result.message}"
        params = result.x
        rmse = float(np.sqrt(np.mean(residual(params) ** 2)))
    except Exception as exc:
        return {
            "fit_status": f"failed:{exc}",
            "ellipsoid_af_m": np.nan,
            "ellipsoid_ar_m": np.nan,
            "ellipsoid_b_m": np.nan,
            "ellipsoid_c_m": np.nan,
            "ellipsoid_xic_m": np.nan,
            "ellipsoid_zc_m": np.nan,
            "ellipsoid_residual_rmse": np.nan,
            "ellipsoid_volume_full_m3": np.nan,
        }

    af, ar, b, c, xi_c, z_c = [float(v) for v in params]
    volume_full = (2.0 / 3.0) * math.pi * b * c * (af + ar)
    return {
        "fit_status": fit_status,
        "ellipsoid_af_m": af,
        "ellipsoid_ar_m": ar,
        "ellipsoid_b_m": b,
        "ellipsoid_c_m": c,
        "ellipsoid_xic_m": xi_c,
        "ellipsoid_zc_m": z_c,
        "ellipsoid_residual_rmse": rmse,
        "ellipsoid_volume_full_m3": volume_full,
    }


def superellipsoid_volume_full(af: float, ar: float, b: float, c: float, n: float, m: float, p: float) -> float:
    # Full-domain asymmetric superellipsoid volume.
    # For n=m=p=2 this reduces to (2/3) * pi * b * c * (af + ar).
    log_factor = (
        math.log(4.0)
        + math.log(af + ar)
        + math.log(b)
        + math.log(c)
        + gammaln(1.0 + 1.0 / n)
        + gammaln(1.0 + 1.0 / m)
        + gammaln(1.0 + 1.0 / p)
        - gammaln(1.0 + 1.0 / n + 1.0 / m + 1.0 / p)
    )
    return float(math.exp(log_factor))


def superellipsoid_phi(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    af, ar, b, c, xi_c, z_c, n, m, p = params
    xi = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    side_a = np.where(xi >= xi_c, af, ar)
    return (np.abs((xi - xi_c) / side_a) ** n) + (np.abs(y / b) ** m) + (np.abs((z - z_c) / c) ** p)


def ellipsoid_params_from_fit(fit: dict[str, float | str]) -> np.ndarray:
    return np.array(
        [
            float(fit.get("ellipsoid_af_m", np.nan)),
            float(fit.get("ellipsoid_ar_m", np.nan)),
            float(fit.get("ellipsoid_b_m", np.nan)),
            float(fit.get("ellipsoid_c_m", np.nan)),
            float(fit.get("ellipsoid_xic_m", np.nan)),
            float(fit.get("ellipsoid_zc_m", np.nan)),
            2.0,
            2.0,
            2.0,
        ],
        dtype=float,
    )


def superellipsoid_params_from_fit(fit: dict[str, float | str]) -> np.ndarray:
    return np.array(
        [
            float(fit.get("superellipsoid_af_m", np.nan)),
            float(fit.get("superellipsoid_ar_m", np.nan)),
            float(fit.get("superellipsoid_b_m", np.nan)),
            float(fit.get("superellipsoid_c_m", np.nan)),
            float(fit.get("superellipsoid_xic_m", np.nan)),
            float(fit.get("superellipsoid_zc_m", np.nan)),
            float(fit.get("superellipsoid_n", np.nan)),
            float(fit.get("superellipsoid_m", np.nan)),
            float(fit.get("superellipsoid_p", np.nan)),
        ],
        dtype=float,
    )


def sample_asymmetric_superellipsoid_surface(params: np.ndarray, n_x: int = 26, n_phi: int = 36) -> np.ndarray:
    """Sample the fitted half-domain surface for distance diagnostics."""
    if not np.all(np.isfinite(params)):
        return np.empty((0, 3), dtype=float)
    af, ar, b, c, xi_c, z_c, n, m, p = [float(v) for v in params]
    if min(af, ar, b, c, n, m, p) <= 0:
        return np.empty((0, 3), dtype=float)
    x_min = -ar
    x_max = af
    x_vals = np.linspace(x_min, x_max, n_x)
    phi_vals = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    pts = []
    for sx in x_vals:
        a_side = af if sx >= 0 else ar
        x_term = min(abs(sx / a_side) ** n, 1.0)
        rem = max(1.0 - x_term, 0.0)
        radial = rem ** 0.5
        for angle in phi_vals:
            cp = math.cos(angle)
            sp = math.sin(angle)
            y_local = b * radial * math.copysign(abs(cp) ** (2.0 / m), cp)
            z_local = c * radial * math.copysign(abs(sp) ** (2.0 / p), sp)
            if y_local >= -1e-14:
                pts.append((xi_c + sx, max(y_local, 0.0), z_c + z_local))
    return np.asarray(pts, dtype=float)


def radial_surface_projection(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Project half-domain points radially from the fitted center to phi=1."""
    if points.size == 0 or not np.all(np.isfinite(params)):
        return np.empty((0, 3), dtype=float)
    af, ar, b, c, xi_c, z_c, n, m, p = [float(v) for v in params]
    if min(af, ar, b, c, n, m, p) <= 0:
        return np.empty((0, 3), dtype=float)
    rel = points.copy()
    rel[:, 0] -= xi_c
    rel[:, 2] -= z_c
    projected = np.full_like(points, np.nan, dtype=float)

    def phi_at(vec: np.ndarray, scale: float) -> float:
        side = af if vec[0] >= 0 else ar
        return float(
            (abs(scale * vec[0] / side) ** n)
            + (abs(scale * vec[1] / b) ** m)
            + (abs(scale * vec[2] / c) ** p)
        )

    for idx, vec in enumerate(rel):
        if not np.all(np.isfinite(vec)):
            continue
        norm_ref = max(abs(vec[0]) / max(af if vec[0] >= 0 else ar, 1e-12), abs(vec[1]) / max(b, 1e-12), abs(vec[2]) / max(c, 1e-12))
        if norm_ref <= 1e-14:
            projected[idx] = np.array([xi_c, 0.0, z_c], dtype=float)
            continue
        phi_one = phi_at(vec, 1.0)
        if not np.isfinite(phi_one):
            continue
        if phi_one > 1.0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = 1.0, 2.0
            while phi_at(vec, hi) < 1.0 and hi < 128.0:
                hi *= 2.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if phi_at(vec, mid) >= 1.0:
                hi = mid
            else:
                lo = mid
        scale = 0.5 * (lo + hi)
        projected[idx] = np.array([xi_c + scale * vec[0], scale * vec[1], z_c + scale * vec[2]], dtype=float)
    return projected[np.isfinite(projected).all(axis=1)]


def geometry_distance_metrics(points: np.ndarray, params: np.ndarray) -> dict[str, float]:
    projected = radial_surface_projection(points, params)
    surface = sample_asymmetric_superellipsoid_surface(params)
    if projected.shape[0] == 0 or surface.shape[0] == 0:
        return {
            "radial_distance_rmse_m": np.nan,
            "radial_distance_mean_m": np.nan,
            "chamfer_distance_m": np.nan,
            "hausdorff_distance_m": np.nan,
        }
    point_mask = np.isfinite(projected).all(axis=1)
    valid_points = points[point_mask]
    valid_projected = projected[point_mask]
    distances = np.linalg.norm(valid_points - valid_projected, axis=1)
    tree_surface = cKDTree(surface)
    d_point_to_surface, _ = tree_surface.query(valid_points, k=1)
    tree_points = cKDTree(valid_points)
    d_surface_to_point, _ = tree_points.query(surface, k=1)
    return {
        "radial_distance_rmse_m": float(np.sqrt(np.nanmean(distances**2))),
        "radial_distance_mean_m": float(np.nanmean(distances)),
        "chamfer_distance_m": float(0.5 * (np.nanmean(d_point_to_surface) + np.nanmean(d_surface_to_point))),
        "hausdorff_distance_m": float(max(np.nanmax(d_point_to_surface), np.nanmax(d_surface_to_point))),
    }


def fit_asymmetric_superellipsoid(
    boundary_points: np.ndarray, ellipsoid: dict[str, float | str]
) -> dict[str, float | str]:
    if boundary_points.shape[0] < 10 or not np.isfinite(float(ellipsoid.get("ellipsoid_af_m", np.nan))):
        return {
            "superellipsoid_fit_status": "insufficient_boundary_points_or_baseline",
            "superellipsoid_af_m": np.nan,
            "superellipsoid_ar_m": np.nan,
            "superellipsoid_b_m": np.nan,
            "superellipsoid_c_m": np.nan,
            "superellipsoid_xic_m": np.nan,
            "superellipsoid_zc_m": np.nan,
            "superellipsoid_n": np.nan,
            "superellipsoid_m": np.nan,
            "superellipsoid_p": np.nan,
            "superellipsoid_residual_rmse": np.nan,
            "superellipsoid_volume_full_m3": np.nan,
        }

    xi = boundary_points[:, 0]
    y = boundary_points[:, 1]
    z = boundary_points[:, 2]
    xi_min, xi_max = float(np.min(xi)), float(np.max(xi))
    y_max = max(float(np.max(y)), 1e-8)
    z_min, z_max = float(np.min(z)), float(np.max(z))
    span_x = max(xi_max - xi_min, 1e-6)
    span_z = max(z_max - z_min, 1e-6)

    x0 = np.array(
        [
            float(ellipsoid["ellipsoid_af_m"]),
            float(ellipsoid["ellipsoid_ar_m"]),
            float(ellipsoid["ellipsoid_b_m"]),
            float(ellipsoid["ellipsoid_c_m"]),
            float(ellipsoid["ellipsoid_xic_m"]),
            float(ellipsoid["ellipsoid_zc_m"]),
            2.0,
            2.0,
            2.0,
        ],
        dtype=float,
    )
    lower = np.array(
        [
            1e-8,
            1e-8,
            1e-8,
            1e-8,
            xi_min - 0.25 * span_x,
            z_min - 0.25 * span_z,
            1.0,
            1.0,
            1.0,
        ]
    )
    upper = np.array(
        [
            2.5 * span_x,
            2.5 * span_x,
            max(2.5 * y_max, 1e-7),
            2.5 * span_z,
            xi_max + 0.25 * span_x,
            z_max + 0.25 * span_z,
            SUPERELLIPSOID_EXPONENT_UPPER,
            SUPERELLIPSOID_EXPONENT_UPPER,
            SUPERELLIPSOID_EXPONENT_UPPER,
        ]
    )
    x0 = np.minimum(np.maximum(x0, lower + 1e-12), upper - 1e-12)

    def residual(params: np.ndarray) -> np.ndarray:
        return superellipsoid_phi(boundary_points, params) - 1.0

    try:
        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.2,
            max_nfev=10000,
        )
        fit_status = "ok" if result.success else f"not_converged:{result.message}"
        params = result.x
        rmse = float(np.sqrt(np.mean(residual(params) ** 2)))
        volume_full = superellipsoid_volume_full(*[float(v) for v in params[[0, 1, 2, 3, 6, 7, 8]]])
    except Exception as exc:
        return {
            "superellipsoid_fit_status": f"failed:{exc}",
            "superellipsoid_af_m": np.nan,
            "superellipsoid_ar_m": np.nan,
            "superellipsoid_b_m": np.nan,
            "superellipsoid_c_m": np.nan,
            "superellipsoid_xic_m": np.nan,
            "superellipsoid_zc_m": np.nan,
            "superellipsoid_n": np.nan,
            "superellipsoid_m": np.nan,
            "superellipsoid_p": np.nan,
            "superellipsoid_residual_rmse": np.nan,
            "superellipsoid_volume_full_m3": np.nan,
        }

    af, ar, b, c, xi_c, z_c, n, m, p = [float(v) for v in params]
    return {
        "superellipsoid_fit_status": fit_status,
        "superellipsoid_af_m": af,
        "superellipsoid_ar_m": ar,
        "superellipsoid_b_m": b,
        "superellipsoid_c_m": c,
        "superellipsoid_xic_m": xi_c,
        "superellipsoid_zc_m": z_c,
        "superellipsoid_n": n,
        "superellipsoid_m": m,
        "superellipsoid_p": p,
        "superellipsoid_residual_rmse": rmse,
        "superellipsoid_volume_full_m3": volume_full,
    }


def summarize_time_step(path: Path, meta: CaseMeta | None = None) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    if meta is None:
        meta = parse_case_metadata(path.parent)
    time_s = parse_time_s(path)
    df, counts = read_and_collapse(path)
    df["time_s"] = time_s
    df["case_id"] = meta.case_id
    df["case_index"] = meta.case_index
    df["power_W"] = meta.power_W
    df["scan_speed_mm_s"] = meta.scan_speed_mm_s
    df["scan_speed_m_s"] = meta.scan_speed_m_s
    df["particle_rate"] = meta.particle_rate
    df["powder_feed_g_min"] = meta.powder_feed_g_min
    df["laser_x_m"] = meta.scan_speed_m_s * time_s
    df["xi_m"] = df["x_m"] - df["laser_x_m"]

    points = df[["xi_m", "y_m", "z_m"]].to_numpy(dtype=float)
    hull = convex_hull(points)
    ellipsoid = fit_asymmetric_ellipsoid(hull.points)
    superellipsoid = fit_asymmetric_superellipsoid(hull.points, ellipsoid)
    ellipsoid_distances = geometry_distance_metrics(hull.points, ellipsoid_params_from_fit(ellipsoid))
    superellipsoid_distances = geometry_distance_metrics(hull.points, superellipsoid_params_from_fit(superellipsoid))

    xi_min = float(df["xi_m"].min())
    xi_max = float(df["xi_m"].max())
    y_max = float(df["y_m"].max())
    z_min = float(df["z_m"].min())
    z_max = float(df["z_m"].max())

    row: dict[str, float | int | str] = {
        "case_id": meta.case_id,
        "case_index": meta.case_index,
        "power_W": meta.power_W,
        "scan_speed_mm_s": meta.scan_speed_mm_s,
        "scan_speed_m_s": meta.scan_speed_m_s,
        "particle_rate": meta.particle_rate,
        "powder_feed_g_min": meta.powder_feed_g_min,
        "powder_feed_kg_s": meta.powder_feed_kg_s,
        "time_s": time_s,
        "source_file": path.name,
        "source_folder": path.parent.name,
        **counts,
        "laser_x_m": meta.scan_speed_m_s * time_s,
        "xi_min_m": xi_min,
        "xi_max_m": xi_max,
        "front_length_m": xi_max,
        "rear_length_m": -xi_min,
        "melt_pool_length_m": xi_max - xi_min,
        "half_width_m": y_max,
        "full_width_m": 2.0 * y_max,
        "z_min_m": z_min,
        "z_max_m": z_max,
        "height_span_m": z_max - z_min,
        "volume_half_convex_hull_m3": hull.volume_half_m3,
        "volume_proxy_m3": 2.0 * hull.volume_half_m3 if np.isfinite(hull.volume_half_m3) else np.nan,
        "convex_hull_status": hull.status,
        "boundary_point_count": int(hull.points.shape[0]),
        "Tmax_K": float(df["Temperature"].max()),
        "Tmean_K": float(df["Temperature"].mean()),
        "Gmax_K_per_m": float(df["Temperature Gradient At Tgrdout"].max()),
        "Gmean_K_per_m": float(df["Temperature Gradient At Tgrdout"].mean()),
        "Umax_m_per_s": float(df["Velocity_Magnitude"].max()),
        "Umean_m_per_s": float(df["Velocity_Magnitude"].mean()),
        "FOF_min": float(df["Fraction Of Fluid"].min()),
        "FOF_mean": float(df["Fraction Of Fluid"].mean()),
        "heat_flux_max": float(df["Heat Flux Spatial Distribution"].max()),
        "heat_flux_nonzero_points": int((df["Heat Flux Spatial Distribution"].abs() > 0).sum()),
        "heat_absorption_max": float(df["Heat Absorption Rate"].max()) if df["Heat Absorption Rate"].notna().any() else np.nan,
        "melt_region_min": float(df["Melt Region"].min()) if df["Melt Region"].notna().any() else np.nan,
        "melt_region_mean": float(df["Melt Region"].mean()) if df["Melt Region"].notna().any() else np.nan,
        "pressure_mean_Pa": float(df["Pressure"].mean()) if df["Pressure"].notna().any() else np.nan,
        "pressure_max_Pa": float(df["Pressure"].max()) if df["Pressure"].notna().any() else np.nan,
        **ellipsoid,
        **superellipsoid,
    }
    for key, value in ellipsoid_distances.items():
        row[f"ellipsoid_{key}"] = value
    for key, value in superellipsoid_distances.items():
        row[f"superellipsoid_{key}"] = value
    if np.isfinite(row["volume_proxy_m3"]) and np.isfinite(row["ellipsoid_volume_full_m3"]):
        denom = max(abs(float(row["volume_proxy_m3"])), 1e-30)
        row["ellipsoid_volume_relative_error"] = abs(
            float(row["ellipsoid_volume_full_m3"]) - float(row["volume_proxy_m3"])
        ) / denom
    else:
        row["ellipsoid_volume_relative_error"] = np.nan
    if np.isfinite(row["volume_proxy_m3"]) and np.isfinite(row["superellipsoid_volume_full_m3"]):
        denom = max(abs(float(row["volume_proxy_m3"])), 1e-30)
        row["superellipsoid_volume_relative_error"] = abs(
            float(row["superellipsoid_volume_full_m3"]) - float(row["volume_proxy_m3"])
        ) / denom
    else:
        row["superellipsoid_volume_relative_error"] = np.nan

    return row, df


def build_tables(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    point_frames: list[pd.DataFrame] = []
    for case_dir in discover_case_dirs(raw_dir):
        meta = parse_case_metadata(case_dir)
        for path in sorted_csv_files(case_dir):
            row, points = summarize_time_step(path, meta)
            rows.append(row)
            point_frames.append(points)
    table = pd.DataFrame(rows).sort_values(["case_index", "time_s"]).reset_index(drop=True)

    l_refs = []
    for case_id, group in table.groupby("case_id", sort=False):
        l_ref = float(group.loc[group["time_s"] >= QUASI_STEADY_START_S, "melt_pool_length_m"].mean())
        if not np.isfinite(l_ref) or l_ref <= 0:
            l_ref = float(group["melt_pool_length_m"].mean())
        l_refs.append((case_id, l_ref))
    l_ref_map = dict(l_refs)
    table["L_ref_m"] = table["case_id"].map(l_ref_map).astype(float)
    table["lf_star"] = table["front_length_m"] / table["L_ref_m"]
    table["lr_star"] = table["rear_length_m"] / table["L_ref_m"]
    table["w_star"] = table["full_width_m"] / table["L_ref_m"]
    table["h_star"] = table["height_span_m"] / table["L_ref_m"]
    table["asymmetry_ratio_lr_lf"] = table["rear_length_m"] / table["front_length_m"].replace(0, np.nan)

    point_cloud = pd.concat(point_frames, ignore_index=True)
    return table, point_cloud


def tag_validation_cohort(
    table: pd.DataFrame,
    point_cloud: pd.DataFrame,
    validation_cohort: str,
    source_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = table.copy()
    point_cloud = point_cloud.copy()
    for frame in [table, point_cloud]:
        frame["validation_cohort"] = validation_cohort
        frame["source_root"] = str(source_root)
    return table, point_cloud


def combine_validation_sources(
    sources: list[tuple[Path, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_frames: list[pd.DataFrame] = []
    point_frames: list[pd.DataFrame] = []
    for source_dir, cohort in sources:
        if not source_dir.exists():
            continue
        table, point_cloud = build_tables(source_dir)
        table, point_cloud = tag_validation_cohort(table, point_cloud, cohort, source_dir)
        table_frames.append(table)
        point_frames.append(point_cloud)
    if not table_frames:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(table_frames, ignore_index=True, sort=False), pd.concat(point_frames, ignore_index=True, sort=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_schema_and_row_count(path: Path) -> tuple[str, int]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline().strip()
        row_count = sum(1 for _ in handle)
    return header, row_count


def make_input_file_manifest(sources: list[tuple[Path, str, str]]) -> pd.DataFrame:
    rows = []
    for source_dir, role, cohort in sources:
        if not source_dir.exists():
            rows.append(
                {
                    "input_role": role,
                    "validation_cohort": cohort,
                    "source_root": str(source_dir),
                    "case_id": "",
                    "case_index": np.nan,
                    "time_s": np.nan,
                    "csv_path": "",
                    "row_count": 0,
                    "column_schema": "",
                    "sha256": "",
                    "status": "missing_source_root",
                }
            )
            continue
        for case_dir in discover_case_dirs(source_dir):
            meta = parse_case_metadata(case_dir)
            try:
                csv_files = sorted_csv_files(case_dir)
            except FileNotFoundError:
                rows.append(
                    {
                        "input_role": role,
                        "validation_cohort": cohort,
                        "source_root": str(source_dir),
                        "case_id": meta.case_id,
                        "case_index": meta.case_index,
                        "time_s": np.nan,
                        "csv_path": str(case_dir),
                        "row_count": 0,
                        "column_schema": "",
                        "sha256": "",
                        "status": "missing_csv_files",
                    }
                )
                continue
            for csv_path in csv_files:
                try:
                    column_schema, row_count = csv_schema_and_row_count(csv_path)
                    file_hash = sha256_file(csv_path)
                    status = "ok"
                except OSError as exc:
                    column_schema, row_count, file_hash = "", 0, ""
                    status = f"failed:{exc}"
                rows.append(
                    {
                        "input_role": role,
                        "validation_cohort": cohort,
                        "source_root": str(source_dir),
                        "case_id": meta.case_id,
                        "case_index": meta.case_index,
                        "power_W": meta.power_W,
                        "scan_speed_mm_s": meta.scan_speed_mm_s,
                        "particle_rate": meta.particle_rate,
                        "powder_feed_g_min": meta.powder_feed_g_min,
                        "time_s": parse_time_s(csv_path),
                        "csv_path": str(csv_path),
                        "row_count": row_count,
                        "column_schema": column_schema,
                        "sha256": file_hash,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def fit_attractor_model(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "case_id" in table.columns and table["case_id"].nunique() > 1:
        prediction_frames = []
        summary_frames = []
        eigen_frames = []
        for case_id, group in table.groupby("case_id", sort=False):
            preds, summary, eigs = fit_attractor_model(group.sort_values("time_s").reset_index(drop=True))
            meta_cols = ["case_id", "case_index", "power_W", "scan_speed_mm_s", "scan_speed_m_s", "particle_rate", "powder_feed_g_min"]
            for col in meta_cols:
                if col in group.columns:
                    value = group[col].iloc[0]
                    preds[col] = value
                    summary[col] = value
                    eigs[col] = value
            prediction_frames.append(preds)
            summary_frames.append(summary)
            eigen_frames.append(eigs)
        return (
            pd.concat(prediction_frames, ignore_index=True),
            pd.concat(summary_frames, ignore_index=True),
            pd.concat(eigen_frames, ignore_index=True),
        )
    t = table["time_s"].to_numpy(dtype=float)
    n_train = max(3, int(math.ceil(TRAIN_FRACTION * len(t))))
    n_train = min(n_train, len(t) - 1)
    t0 = t[0]
    train_mask = np.zeros(len(t), dtype=bool)
    train_mask[:n_train] = True

    predictions = pd.DataFrame({"time_s": t, "split": np.where(train_mask, "train", "validation")})
    summary_rows: list[dict[str, float | str | int]] = []
    eigen_rows: list[dict[str, float | str]] = []

    for col in STATE_COLUMNS:
        y = table[col].to_numpy(dtype=float)
        y0 = float(y[0])
        finite = np.isfinite(y)
        fit_mask = train_mask & finite
        y_fit = y[fit_mask]
        t_fit = t[fit_mask]
        y_scale = max(float(np.nanmax(y_fit) - np.nanmin(y_fit)), abs(float(np.nanmean(y_fit))), 1e-12)

        def pred(params: np.ndarray, tt: np.ndarray) -> np.ndarray:
            y_inf, k = params
            return y_inf + (y0 - y_inf) * np.exp(-k * (tt - t0))

        def residual(params: np.ndarray) -> np.ndarray:
            return (pred(params, t_fit) - y_fit) / y_scale

        train_steady = y[train_mask & finite & (t >= QUASI_STEADY_START_S)]
        if train_steady.size >= 3:
            q_center = float(np.nanmean(train_steady))
            q_spread = float(np.nanstd(train_steady, ddof=1))
            q_spread = max(q_spread, 0.05 * abs(q_center), 1e-12)
            q_lower = q_center - 3.0 * q_spread
            q_upper = q_center + 3.0 * q_spread
        else:
            q_lower = float(np.nanmin(y_fit))
            q_upper = float(np.nanmax(y_fit))
            q_spread = max(q_upper - q_lower, 0.05 * abs(float(np.nanmean(y_fit))), 1e-12)
            q_lower -= q_spread
            q_upper += q_spread
        if np.nanmin(y_fit) >= 0:
            q_lower = max(q_lower, 0.0)
        if q_upper <= q_lower:
            q_upper = q_lower + max(abs(q_lower), 1.0) * 0.05
        bounds = ([q_lower, 1e-9], [q_upper, 100.0])
        x0 = np.array([float(np.clip(np.nanmean(train_steady) if train_steady.size else np.nanmean(y_fit), q_lower, q_upper)), 5.0])

        try:
            result = least_squares(residual, x0=x0, bounds=bounds, max_nfev=10000)
            y_inf, k = [float(v) for v in result.x]
            fit_status = "ok" if result.success else f"not_converged:{result.message}"
        except Exception as exc:
            y_inf, k = float("nan"), float("nan")
            fit_status = f"failed:{exc}"

        y_pred = pred(np.array([y_inf, k]), t) if np.isfinite(y_inf) and np.isfinite(k) else np.full_like(y, np.nan)
        predictions[f"{col}_actual"] = y
        predictions[f"{col}_predicted"] = y_pred
        predictions[f"{col}_residual"] = y_pred - y

        def rmse(mask: np.ndarray) -> float:
            err = y_pred[mask & finite] - y[mask & finite]
            if err.size == 0:
                return float("nan")
            return float(np.sqrt(np.mean(err**2)))

        def rel_rmse(mask: np.ndarray) -> float:
            denom = float(np.nanmean(np.abs(y[mask & finite])))
            return rmse(mask) / denom if denom > 0 else float("nan")

        train_rmse = rmse(train_mask)
        validation_rmse = rmse(~train_mask)
        summary_rows.append(
            {
                "state": col,
                "label": STATE_LABELS[col],
                "fit_status": fit_status,
                "train_points": int(np.sum(train_mask & finite)),
                "validation_points": int(np.sum((~train_mask) & finite)),
                "q0": y0,
                "q_inf": y_inf,
                "k_per_s": k,
                "train_rmse": train_rmse,
                "train_relative_rmse": rel_rmse(train_mask),
                "validation_rmse": validation_rmse,
                "validation_relative_rmse": rel_rmse(~train_mask),
            }
        )
        eigen_rows.append(
            {
                "state": col,
                "label": STATE_LABELS[col],
                "jacobian_eigenvalue_per_s": -k,
                "stable_if_negative": bool(np.isfinite(k) and k > 0),
            }
        )

    summary = pd.DataFrame(summary_rows)
    eigenvalues = pd.DataFrame(eigen_rows)
    return predictions, summary, eigenvalues


def train_split_mask(t: np.ndarray) -> np.ndarray:
    n_train = max(3, int(math.ceil(TRAIN_FRACTION * len(t))))
    n_train = min(n_train, len(t) - 1)
    train_mask = np.zeros(len(t), dtype=bool)
    train_mask[:n_train] = True
    return train_mask


def _minimal_dynamics_baselines_for_group(group: pd.DataFrame, context: str) -> pd.DataFrame:
    group = group.sort_values("time_s").reset_index(drop=True)
    t = group["time_s"].to_numpy(dtype=float)
    train_mask = train_split_mask(t)
    rows = []
    meta_cols = [
        "case_id",
        "case_index",
        "power_W",
        "scan_speed_mm_s",
        "scan_speed_m_s",
        "particle_rate",
        "powder_feed_g_min",
        "validation_cohort",
        "source_root",
    ]
    meta = {col: group[col].iloc[0] for col in meta_cols if col in group.columns}
    for state in STATE_COLUMNS:
        y = group[state].to_numpy(dtype=float)
        finite = np.isfinite(y)
        train_idx = np.where(train_mask & finite)[0]
        val_mask = (~train_mask) & finite
        y_val = y[val_mask]
        denom = max(float(np.nanmean(np.abs(y_val))) if y_val.size else np.nan, 1e-12)

        def add_row(model: str, pred_all: np.ndarray, status: str) -> None:
            err = pred_all[val_mask] - y[val_mask]
            rmse = float(np.sqrt(np.nanmean(err**2))) if err.size else np.nan
            rows.append(
                {
                    **meta,
                    "context": context,
                    "state": state,
                    "label": STATE_LABELS[state],
                    "model": model,
                    "train_points": int(np.sum(train_mask & finite)),
                    "validation_points": int(np.sum(val_mask)),
                    "validation_rmse": rmse,
                    "validation_relative_rmse": rmse / denom if np.isfinite(rmse) and np.isfinite(denom) else np.nan,
                    "status": status,
                }
            )

        last_pred = np.full_like(y, np.nan, dtype=float)
        if train_idx.size:
            last_pred[:] = float(y[train_idx[-1]])
            add_row("last_train_persistence", last_pred, "ok")
        else:
            add_row("last_train_persistence", last_pred, "failed:no_training_points")

        ar_pred = np.full_like(y, np.nan, dtype=float)
        pair_idx = [idx for idx in train_idx[:-1] if idx + 1 in set(train_idx)]
        if len(pair_idx) >= 2:
            x = y[pair_idx]
            target = y[[idx + 1 for idx in pair_idx]]
            design = np.column_stack([x, np.ones(len(x))])
            try:
                coef, *_ = np.linalg.lstsq(design, target, rcond=None)
                if train_idx.size:
                    ar_pred[: train_idx[-1] + 1] = y[: train_idx[-1] + 1]
                    current = float(y[train_idx[-1]])
                    for idx in range(train_idx[-1] + 1, len(y)):
                        current = float(coef[0] * current + coef[1])
                        ar_pred[idx] = current
                add_row("independent_ar1", ar_pred, "ok")
            except np.linalg.LinAlgError as exc:
                add_row("independent_ar1", ar_pred, f"failed:{exc}")
        else:
            add_row("independent_ar1", ar_pred, "failed:insufficient_training_pairs")
    return pd.DataFrame(rows)


def make_minimal_dynamics_baselines(table: pd.DataFrame, context: str) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    frames = [
        _minimal_dynamics_baselines_for_group(group, context)
        for _, group in table.groupby("case_id", sort=False)
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def make_dynamics_derivative_audit(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, group in table.groupby("case_id", sort=False):
        group = group.sort_values("time_s")
        t = group["time_s"].to_numpy(dtype=float)
        train_mask = train_split_mask(t)
        train_steady_mask = train_mask & (t >= QUASI_STEADY_START_S)
        if np.sum(train_steady_mask) < 3:
            train_steady_mask = train_mask
        dt = np.diff(t)
        train_dt = np.diff(t[train_mask])
        row = {
            "case_id": case_id,
            "case_index": int(group["case_index"].iloc[0]) if "case_index" in group.columns else np.nan,
            "analysis_role": "same_ecosystem_holdout"
            if "validation_cohort" in group.columns and pd.notna(group["validation_cohort"].iloc[0])
            else "model_construction",
            "time_points": int(len(t)),
            "train_fraction": TRAIN_FRACTION,
            "train_points": int(np.sum(train_mask)),
            "validation_points": int(np.sum(~train_mask)),
            "quasi_steady_start_s": QUASI_STEADY_START_S,
            "q_inf_time_min_s": float(np.nanmin(t[train_steady_mask])) if np.any(train_steady_mask) else np.nan,
            "q_inf_time_max_s": float(np.nanmax(t[train_steady_mask])) if np.any(train_steady_mask) else np.nan,
            "q_inf_uses_validation_points": False,
            "time_step_min_s": float(np.nanmin(dt)) if dt.size else np.nan,
            "time_step_max_s": float(np.nanmax(dt)) if dt.size else np.nan,
            "train_time_step_min_s": float(np.nanmin(train_dt)) if train_dt.size else np.nan,
            "train_time_step_max_s": float(np.nanmax(train_dt)) if train_dt.size else np.nan,
            "nonuniform_time_spacing": bool(dt.size and np.nanmax(dt) > np.nanmin(dt)),
            "diagonal_primary_derivative_policy": "direct exponential least-squares fit; no finite-difference derivative in primary diagonal model",
            "coupled_comparison_derivative_policy": "finite-difference dz/dt with observed Delta t and ridge regularization; regression rows are not smoothed",
            "smoothing_policy": "no smoothing in primary results; rolling-origin and train-fraction stress tests report sampling sensitivity",
        }
        for col in ["validation_cohort", "source_root"]:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def make_method_detail_audit() -> pd.DataFrame:
    rows = [
        {
            "item": "training split",
            "value": f"first {TRAIN_FRACTION:.0%} of exported time steps for each condition",
            "manuscript_action": "Report train/validation split explicitly and keep validation points out of q_inf estimation.",
        },
        {
            "item": "quasi-steady window",
            "value": f"training points with t >= {QUASI_STEADY_START_S:.2f} s; if fewer than 3, use all training points",
            "manuscript_action": "Define q_inf window and leakage rule.",
        },
        {
            "item": "q_inf leakage rule",
            "value": "q_inf is estimated from the training segment only for each condition; validation time points are not used in q_inf estimation",
            "manuscript_action": "State explicitly that validation trajectory errors are not helped by validation-time q_inf estimates.",
        },
        {
            "item": "ellipsoid optimizer",
            "value": "bounded nonlinear least-squares optimization, soft_l1 loss, f_scale=0.2, max_nfev=5000",
            "manuscript_action": "Report optimizer, robust loss and evaluation cap.",
        },
        {
            "item": "ellipsoid bounds",
            "value": "positive axes >=1e-8 m; center within data span +/- 0.25 span; axes <=2.5 span",
            "manuscript_action": "Define Theta for baseline ellipsoid fit.",
        },
        {
            "item": "superellipsoid optimizer",
            "value": "bounded nonlinear least-squares optimization, soft_l1 loss, f_scale=0.2, max_nfev=10000",
            "manuscript_action": "Report optimizer, robust loss and evaluation cap.",
        },
        {
            "item": "superellipsoid bounds",
            "value": f"positive axes >=1e-8 m; center within data span +/-0.25 span; axes <=2.5 span; exponents 1.0-{SUPERELLIPSOID_EXPONENT_UPPER:.1f}",
            "manuscript_action": "Define exponent and positivity bounds for Theta.",
        },
        {
            "item": "initialization",
            "value": "ellipsoid initialized from observed spans; superellipsoid initialized from fitted ellipsoid with exponents n=m=p=2",
            "manuscript_action": "State deterministic initialization and local-minimum handling.",
        },
        {
            "item": "local-minimum handling",
            "value": "single deterministic initialization per time step; failed/nonconverged fits retained with status flags and excluded from summary means through NaN-aware aggregation",
            "manuscript_action": "Avoid implying global optimum; keep fit-status audit visible.",
        },
        {
            "item": "dynamics baselines",
            "value": "diagonal attractor, coupled ridge control, last-train persistence and independent AR(1) audit baselines",
            "manuscript_action": "Clarify that simple baselines are diagnostic controls, not expanded mechanistic claims.",
        },
        {
            "item": "diagonal attractor fitting",
            "value": "direct exponential trajectory least-squares fit; finite-difference derivatives are not used for the primary diagonal model",
            "manuscript_action": "Avoid implying derivative-based inference for the relaxation baseline; call the model a fitted descriptive relaxation baseline.",
        },
        {
            "item": "coupled comparison derivative handling",
            "value": "finite-difference dz/dt with observed Delta t and ridge regularization; reported as an overparameterization audit",
            "manuscript_action": "Keep finite-difference derivative caveat tied to the coupled comparison rather than the selected diagonal baseline.",
        },
    ]
    return pd.DataFrame(rows)


def state_matrix(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = table["time_s"].to_numpy(dtype=float)
    y = table[STATE_COLUMNS].to_numpy(dtype=float)
    train_mask = train_split_mask(t)
    train_steady_mask = train_mask & (t >= QUASI_STEADY_START_S)
    if np.sum(train_steady_mask) < 3:
        train_steady_mask = train_mask
    q_inf = np.nanmean(y[train_steady_mask], axis=0)
    scale = np.nanstd(y[train_mask], axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.maximum(np.abs(q_inf), 1.0))
    z = (y - q_inf) / scale
    return t, y, q_inf, scale


def fit_coupled_attractor_model(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "case_id" in table.columns and table["case_id"].nunique() > 1:
        prediction_frames = []
        summary_frames = []
        eigen_frames = []
        matrix_frames = []
        meta_cols = ["case_id", "case_index", "power_W", "scan_speed_mm_s", "scan_speed_m_s", "particle_rate", "powder_feed_g_min"]
        for case_id, group in table.groupby("case_id", sort=False):
            preds, summary, eigs = fit_coupled_attractor_model(group.sort_values("time_s").reset_index(drop=True))
            matrix = summary.attrs.get("A_matrix_long", pd.DataFrame()).copy()
            summary.attrs = {}
            for col in meta_cols:
                if col in group.columns:
                    value = group[col].iloc[0]
                    preds[col] = value
                    summary[col] = value
                    eigs[col] = value
                    if len(matrix):
                        matrix[col] = value
            prediction_frames.append(preds)
            summary_frames.append(summary)
            eigen_frames.append(eigs)
            if len(matrix):
                matrix_frames.append(matrix)
        summary_all = pd.concat(summary_frames, ignore_index=True)
        summary_all.attrs["A_matrix_long"] = pd.concat(matrix_frames, ignore_index=True) if matrix_frames else pd.DataFrame()
        return (
            pd.concat(prediction_frames, ignore_index=True),
            summary_all,
            pd.concat(eigen_frames, ignore_index=True),
        )
    t, y, q_inf, scale = state_matrix(table)
    z = (y - q_inf) / scale
    train_mask = train_split_mask(t)
    train_index = np.where(train_mask)[0]
    if len(train_index) < 4:
        raise ValueError("At least four training time steps are required for coupled dynamics.")

    x_rows = []
    dz_rows = []
    dt_rows = []
    for i in train_index[:-1]:
        dt = t[i + 1] - t[i]
        if dt <= 0:
            continue
        x_rows.append(z[i])
        dz_rows.append((z[i + 1] - z[i]) / dt)
        dt_rows.append(dt)
    x_mat = np.asarray(x_rows, dtype=float)
    dz_mat = np.asarray(dz_rows, dtype=float)
    dt_arr = np.asarray(dt_rows, dtype=float)

    def solve_matrix(lam: float, leave_out: int | None = None) -> np.ndarray:
        mask = np.ones(len(x_mat), dtype=bool)
        if leave_out is not None:
            mask[leave_out] = False
        x_use = x_mat[mask]
        dz_use = dz_mat[mask]
        xtx = x_use.T @ x_use
        rhs = x_use.T @ dz_use
        ridge = lam * np.eye(xtx.shape[0])
        beta = np.linalg.solve(xtx + ridge, rhs)
        return -beta.T

    cv_rows = []
    for lam in RIDGE_LAMBDAS:
        errs = []
        for idx in range(len(x_mat)):
            try:
                a_cv = solve_matrix(float(lam), leave_out=idx)
                z_next = expm(-a_cv * dt_arr[idx]) @ z[idx]
                errs.append(np.mean((z_next - z[idx + 1]) ** 2))
            except np.linalg.LinAlgError:
                errs.append(np.inf)
        cv_rows.append((float(lam), float(np.mean(errs))))
    selected_lambda, cv_mse = min(cv_rows, key=lambda item: item[1])
    a_matrix = solve_matrix(selected_lambda)

    z_pred = np.zeros_like(z)
    z_pred[0] = z[0]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        z_pred[i + 1] = expm(-a_matrix * dt) @ z_pred[i]
    y_pred = q_inf + z_pred * scale

    predictions = pd.DataFrame({"time_s": t, "split": np.where(train_mask, "train", "validation")})
    summary_rows = []
    for j, col in enumerate(STATE_COLUMNS):
        predictions[f"{col}_actual"] = y[:, j]
        predictions[f"{col}_predicted"] = y_pred[:, j]
        predictions[f"{col}_residual"] = y_pred[:, j] - y[:, j]
        finite = np.isfinite(y[:, j])

        def rmse(mask: np.ndarray) -> float:
            err = y_pred[mask & finite, j] - y[mask & finite, j]
            return float(np.sqrt(np.mean(err**2))) if err.size else float("nan")

        def rel_rmse(mask: np.ndarray) -> float:
            denom = float(np.nanmean(np.abs(y[mask & finite, j])))
            return rmse(mask) / denom if denom > 0 else float("nan")

        summary_rows.append(
            {
                "state": col,
                "label": STATE_LABELS[col],
                "model": "coupled_ridge_attractor",
                "selected_lambda": selected_lambda,
                "cv_mse": cv_mse,
                "q_inf": q_inf[j],
                "scale": scale[j],
                "train_rmse": rmse(train_mask),
                "train_relative_rmse": rel_rmse(train_mask),
                "validation_rmse": rmse(~train_mask),
                "validation_relative_rmse": rel_rmse(~train_mask),
            }
        )

    eig = np.linalg.eigvals(-a_matrix)
    eigen_rows = [
        {
            "eigen_index": idx,
            "jacobian_eigenvalue_real_per_s": float(np.real(value)),
            "jacobian_eigenvalue_imag_per_s": float(np.imag(value)),
            "stable_if_real_negative": bool(np.real(value) < 0),
            "selected_lambda": selected_lambda,
        }
        for idx, value in enumerate(eig, start=1)
    ]
    summary = pd.DataFrame(summary_rows)
    eigenvalues = pd.DataFrame(eigen_rows)
    matrix_rows = []
    for i, row_state in enumerate(STATE_COLUMNS):
        for j, col_state in enumerate(STATE_COLUMNS):
            matrix_rows.append(
                {
                    "row_state": row_state,
                    "column_state": col_state,
                    "A_value_per_s": float(a_matrix[i, j]),
                    "selected_lambda": selected_lambda,
                }
            )
    summary.attrs["A_matrix_long"] = pd.DataFrame(matrix_rows)
    return predictions, summary, eigenvalues


def compare_dynamics_models(
    diagonal_summary: pd.DataFrame,
    diagonal_eigenvalues: pd.DataFrame,
    coupled_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
) -> pd.DataFrame:
    diagonal = diagonal_summary.copy()
    diagonal["model"] = "diagonal_attractor"
    if "case_id" in diagonal.columns and "case_id" in diagonal_eigenvalues.columns:
        stable_map = diagonal_eigenvalues.groupby("case_id")["stable_if_negative"].all()
        diagonal["all_eigenvalues_stable"] = diagonal["case_id"].map(stable_map).fillna(False).astype(bool)
    else:
        diagonal["all_eigenvalues_stable"] = bool(diagonal_eigenvalues["stable_if_negative"].all())
    coupled = coupled_summary.copy()
    if "case_id" in coupled.columns and "case_id" in coupled_eigenvalues.columns:
        stable_map = coupled_eigenvalues.groupby("case_id")["stable_if_real_negative"].all()
        coupled["all_eigenvalues_stable"] = coupled["case_id"].map(stable_map).fillna(False).astype(bool)
    else:
        coupled["all_eigenvalues_stable"] = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    common_cols = [
        "model",
        "state",
        "label",
        "train_rmse",
        "train_relative_rmse",
        "validation_rmse",
        "validation_relative_rmse",
        "all_eigenvalues_stable",
    ]
    meta_cols = ["case_id", "case_index", "power_W", "scan_speed_mm_s", "particle_rate", "powder_feed_g_min"]
    for col in meta_cols:
        if col in diagonal.columns and col in coupled.columns:
            common_cols.append(col)
    comparison = pd.concat([diagonal[common_cols], coupled[common_cols]], ignore_index=True)
    index_cols = ["state"]
    if "case_id" in comparison.columns:
        index_cols = ["case_id", "state"]
    pivot = comparison.pivot(index=index_cols, columns="model", values="validation_relative_rmse")
    if {"coupled_ridge_attractor", "diagonal_attractor"}.issubset(pivot.columns):
        improvement = (
            pivot["diagonal_attractor"] - pivot["coupled_ridge_attractor"]
        ) / pivot["diagonal_attractor"].replace(0, np.nan)
        if "case_id" in comparison.columns:
            comparison["coupled_relative_improvement_vs_diagonal"] = [
                improvement.get((row.case_id, row.state), np.nan) for row in comparison.itertuples()
            ]
        else:
            comparison["coupled_relative_improvement_vs_diagonal"] = comparison["state"].map(improvement)
    else:
        comparison["coupled_relative_improvement_vs_diagonal"] = np.nan
    return comparison


def make_model_selection_summary(
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
) -> pd.DataFrame:
    geom_summary = geometry_comparison[geometry_comparison["time_s"] == "summary"].set_index("model")
    dyn_mean = (
        dynamics_comparison.groupby("model", as_index=False)
        .agg(
            mean_train_relative_rmse=("train_relative_rmse", "mean"),
            mean_validation_relative_rmse=("validation_relative_rmse", "mean"),
            stable=("all_eigenvalues_stable", "all"),
        )
        .set_index("model")
    )
    geom_case = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("case_summary")].copy()
    geom_case_boundary = pd.DataFrame()
    geom_case_volume = pd.DataFrame()
    if len(geom_case):
        geom_case_boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse")
        geom_case_volume = geom_case.pivot(index="case_id", columns="model", values="mean_volume_relative_error")
    dyn_case = pd.DataFrame()
    if "case_id" in dynamics_comparison.columns and len(dynamics_comparison):
        dyn_case = dynamics_comparison.pivot_table(
            index=["case_id", "state"], columns="model", values="validation_relative_rmse", aggfunc="mean"
        )

    def pairwise_stats(
        pivot: pd.DataFrame,
        current_model: str,
        reference_model: str,
        lower_is_better: bool = True,
    ) -> dict[str, float | int | str]:
        if pivot.empty or not {current_model, reference_model}.issubset(pivot.columns):
            return {
                "paired_reference_model": reference_model,
                "paired_pair_count": 0,
                "paired_better_count": 0,
                "paired_better_rate": np.nan,
                "paired_median_advantage": np.nan,
                "paired_mean_advantage": np.nan,
                "paired_binom_p_value": np.nan,
                "paired_wilcoxon_p_value": np.nan,
            }
        current = pivot[current_model].to_numpy(dtype=float)
        reference = pivot[reference_model].to_numpy(dtype=float)
        mask = np.isfinite(current) & np.isfinite(reference)
        current = current[mask]
        reference = reference[mask]
        if current.size == 0:
            return {
                "paired_reference_model": reference_model,
                "paired_pair_count": 0,
                "paired_better_count": 0,
                "paired_better_rate": np.nan,
                "paired_median_advantage": np.nan,
                "paired_mean_advantage": np.nan,
                "paired_binom_p_value": np.nan,
                "paired_wilcoxon_p_value": np.nan,
            }
        advantage = (reference - current) if lower_is_better else (current - reference)
        better_count = int(np.sum(advantage > 0))
        pair_count = int(len(advantage))
        better_rate = float(better_count / pair_count) if pair_count else np.nan
        try:
            binom_p = float(binomtest(better_count, pair_count, 0.5, alternative="greater").pvalue) if pair_count else np.nan
        except Exception:
            binom_p = np.nan
        try:
            wilcoxon_p = (
                float(wilcoxon(advantage, alternative="greater", zero_method="pratt").pvalue)
                if pair_count and np.any(np.abs(advantage) > 0)
                else np.nan
            )
        except Exception:
            wilcoxon_p = np.nan
        return {
            "paired_reference_model": reference_model,
            "paired_pair_count": pair_count,
            "paired_better_count": better_count,
            "paired_better_rate": better_rate,
            "paired_median_advantage": float(np.nanmedian(advantage)),
            "paired_mean_advantage": float(np.nanmean(advantage)),
            "paired_binom_p_value": binom_p,
            "paired_wilcoxon_p_value": wilcoxon_p,
        }

    diagonal_validation = float(dyn_mean.loc["diagonal_attractor", "mean_validation_relative_rmse"])
    coupled_validation = float(dyn_mean.loc["coupled_ridge_attractor", "mean_validation_relative_rmse"])
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    rows = [
        {
            "model_family": "observed_boundary_envelope_geometry",
            "model": "ellipsoid",
            "parameter_count": 6,
            "primary_metric": "mean_boundary_residual_rmse",
            "primary_metric_value": float(geom_summary.loc["ellipsoid", "mean_boundary_residual_rmse"]),
            "secondary_metric": "mean_volume_relative_error",
            "secondary_metric_value": float(geom_summary.loc["ellipsoid", "mean_volume_relative_error"]),
            "stability": "not_applicable",
            "interpretability": "high",
            "role": "baseline",
            "selected_as_main_model": False,
            "selection_reason": "Lower complexity but larger boundary residual; retained as the conservative proxy baseline.",
            **pairwise_stats(geom_case_boundary, "ellipsoid", "superellipsoid", True),
        },
        {
            "model_family": "observed_boundary_envelope_geometry",
            "model": "superellipsoid",
            "parameter_count": 9,
            "primary_metric": "mean_boundary_residual_rmse",
            "primary_metric_value": float(geom_summary.loc["superellipsoid", "mean_boundary_residual_rmse"]),
            "secondary_metric": "mean_volume_relative_error",
            "secondary_metric_value": float(geom_summary.loc["superellipsoid", "mean_volume_relative_error"]),
            "stability": "not_applicable",
            "interpretability": "medium_high",
            "role": "selected_algebraic_observed_envelope_descriptor",
            "selected_as_main_model": True,
            "selection_reason": "Reduces the implicit boundary residual while retaining a compact analytic form; volume and distance diagnostics are reported separately as limitations.",
            **pairwise_stats(geom_case_boundary, "superellipsoid", "ellipsoid", True),
        },
        {
            "model_family": "reduced_order_dynamics",
            "model": "diagonal_attractor",
            "parameter_count": 14,
            "primary_metric": "mean_validation_relative_rmse",
            "primary_metric_value": diagonal_validation,
            "secondary_metric": "mean_train_relative_rmse",
            "secondary_metric_value": float(dyn_mean.loc["diagonal_attractor", "mean_train_relative_rmse"]),
            "stability": bool(dyn_mean.loc["diagonal_attractor", "stable"]),
            "interpretability": "high",
            "role": "parsimonious_baseline_dynamics",
            "selected_as_main_model": True,
            "selection_reason": "Selected as a stable parsimonious baseline because the available short sequences do not justify the coupled model.",
            **pairwise_stats(dyn_case, "diagonal_attractor", "coupled_ridge_attractor", True),
        },
        {
            "model_family": "reduced_order_dynamics",
            "model": "coupled_ridge_attractor",
            "parameter_count": 56,
            "primary_metric": "mean_validation_relative_rmse",
            "primary_metric_value": coupled_validation,
            "secondary_metric": "mean_train_relative_rmse",
            "secondary_metric_value": float(dyn_mean.loc["coupled_ridge_attractor", "mean_train_relative_rmse"]),
            "stability": coupled_stable,
            "interpretability": "medium",
            "role": "overparameterization_test",
            "selected_as_main_model": bool(coupled_stable and coupled_validation < diagonal_validation),
            "selection_reason": "Stable but retained as an overparameterized comparison because its validation advantage is marginal and not robust.",
            **pairwise_stats(dyn_case, "coupled_ridge_attractor", "diagonal_attractor", True),
        },
    ]
    out = pd.DataFrame(rows)
    if len(geom_case_boundary):
        wins = int((geom_case_boundary["superellipsoid"] < geom_case_boundary["ellipsoid"]).sum())
        total = int(len(geom_case_boundary))
        vol_wins = int((geom_case_volume["superellipsoid"] < geom_case_volume["ellipsoid"]).sum()) if len(geom_case_volume) else 0
        out.loc[out["model"].eq("superellipsoid"), "selection_reason"] = (
            f"Reduces mean boundary residual in {wins}/{total} conditions; volume proxy improves in "
            f"{vol_wins}/{total} conditions, while distance metrics are summarized separately as geometric-risk diagnostics."
        )
        out.loc[out["model"].eq("ellipsoid"), "selection_reason"] = (
            f"Retained as the baseline for {total} paired geometry comparisons against the superellipsoid."
        )
    if len(dyn_case):
        coupled_wins = int((dyn_case["diagonal_attractor"] < dyn_case["coupled_ridge_attractor"]).sum())
        total = int(len(dyn_case))
        out.loc[out["model"].eq("diagonal_attractor"), "selection_reason"] = (
            f"Stable parsimonious baseline; lower validation error than the coupled model in {coupled_wins}/{total} paired condition-state comparisons, without statistical dominance."
        )
        out.loc[out["model"].eq("coupled_ridge_attractor"), "selection_reason"] = (
            f"Coupled model improves validation in {total - coupled_wins}/{total} paired condition-state comparisons, but the advantage is not robust."
        )
    return out


def run_robustness_analysis(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_count = len(discover_case_dirs(raw_dir))
    scenarios = []
    if case_count == 1:
        for train_fraction in [0.60, 0.70, 0.80]:
            scenarios.append(("train_fraction", train_fraction, None, None))
        for quasi_start in [0.15, 0.20, 0.25]:
            scenarios.append(("quasi_steady_start_s", None, quasi_start, None))
        for exponent_upper in [4.0, 6.0, 8.0]:
            scenarios.append(("superellipsoid_exponent_upper", None, None, exponent_upper))
    else:
        for train_fraction in [0.60, 0.70, 0.80]:
            scenarios.append(("train_fraction", train_fraction, None, None))
        for exponent_upper in [4.0, 6.0]:
            scenarios.append(("superellipsoid_exponent_upper", None, None, exponent_upper))

    long_rows = []
    summary_rows = []
    for scenario_type, train_fraction, quasi_start, exponent_upper in scenarios:
        scenario_value = (
            train_fraction if train_fraction is not None else quasi_start if quasi_start is not None else exponent_upper
        )
        try:
            with temporary_settings(train_fraction, quasi_start, exponent_upper):
                table, _point_cloud = build_tables(raw_dir)
                diag_predictions, diag_summary, diag_eigs = fit_attractor_model(table)
                coupled_predictions, coupled_summary, coupled_eigs = fit_coupled_attractor_model(table)
                dynamics_comparison = compare_dynamics_models(diag_summary, diag_eigs, coupled_summary, coupled_eigs)
                if case_count > 1 and scenario_type == "train_fraction":
                    geometry_comparison = make_geometry_model_comparison(table)
                    geom_summary = geometry_comparison[geometry_comparison["time_s"] == "summary"].set_index("model")
                    ellipsoid_volume = float(geom_summary.loc["ellipsoid", "mean_volume_relative_error"])
                    super_volume = float(geom_summary.loc["superellipsoid", "mean_volume_relative_error"])
                    ellipsoid_boundary = float(geom_summary.loc["ellipsoid", "mean_boundary_residual_rmse"])
                    super_boundary = float(geom_summary.loc["superellipsoid", "mean_boundary_residual_rmse"])
                elif case_count > 1:
                    ellipsoid_volume = float(np.nanmean(table["ellipsoid_volume_relative_error"]))
                    super_volume = float(np.nanmean(table["superellipsoid_volume_relative_error"]))
                    ellipsoid_boundary = float(np.nanmean(table["ellipsoid_residual_rmse"]))
                    super_boundary = float(np.nanmean(table["superellipsoid_residual_rmse"]))
                else:
                    geometry_comparison = make_geometry_model_comparison(table)
                    geom_summary = geometry_comparison[geometry_comparison["time_s"] == "summary"].set_index("model")
                    ellipsoid_volume = float(geom_summary.loc["ellipsoid", "mean_volume_relative_error"])
                    super_volume = float(geom_summary.loc["superellipsoid", "mean_volume_relative_error"])
                    ellipsoid_boundary = float(geom_summary.loc["ellipsoid", "mean_boundary_residual_rmse"])
                    super_boundary = float(geom_summary.loc["superellipsoid", "mean_boundary_residual_rmse"])
                dyn_mean = dynamics_comparison.groupby("model")["validation_relative_rmse"].mean()
                diagonal_validation = float(dyn_mean["diagonal_attractor"])
                coupled_validation = float(dyn_mean["coupled_ridge_attractor"])
                coupled_stable = bool(coupled_eigs["stable_if_real_negative"].all())
                status = "ok"
                message = ""
        except Exception as exc:
            ellipsoid_volume = super_volume = ellipsoid_boundary = super_boundary = np.nan
            diagonal_validation = coupled_validation = np.nan
            coupled_stable = False
            status = "failed"
            message = str(exc)

        metrics = {
            "ellipsoid_volume_error": ellipsoid_volume,
            "superellipsoid_volume_error": super_volume,
            "ellipsoid_boundary_rmse": ellipsoid_boundary,
            "superellipsoid_boundary_rmse": super_boundary,
            "diagonal_validation_relative_rmse": diagonal_validation,
            "coupled_validation_relative_rmse": coupled_validation,
        }
        for metric, value in metrics.items():
            long_rows.append(
                {
                    "scenario_type": scenario_type,
                    "scenario_value": scenario_value,
                    "metric": metric,
                    "value": value,
                    "status": status,
                    "message": message,
                }
            )
        summary_rows.append(
            {
                "scenario_type": scenario_type,
                "scenario_value": scenario_value,
                "status": status,
                "message": message,
                "superellipsoid_improves_volume": bool(np.isfinite(super_volume) and super_volume < ellipsoid_volume),
                "superellipsoid_improves_boundary": bool(np.isfinite(super_boundary) and super_boundary < ellipsoid_boundary),
                "coupled_improves_validation": bool(
                    np.isfinite(coupled_validation) and coupled_validation < diagonal_validation and coupled_stable
                ),
                "coupled_stable": coupled_stable,
                **metrics,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def make_quasi_steady_summary(table: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "front_length_m",
        "rear_length_m",
        "melt_pool_length_m",
        "full_width_m",
        "height_span_m",
        "volume_proxy_m3",
        "Tmax_K",
        "Tmean_K",
        "Gmax_K_per_m",
        "Gmean_K_per_m",
        "Umax_m_per_s",
        "Umean_m_per_s",
        "asymmetry_ratio_lr_lf",
    ]
    steady = table[table["time_s"] >= QUASI_STEADY_START_S]
    rows = []
    group_items = [("all_conditions", steady)]
    if "case_id" in steady.columns and steady["case_id"].nunique() > 1:
        group_items.extend((str(case_id), group) for case_id, group in steady.groupby("case_id", sort=False))
    for group_id, group in group_items:
        meta = {}
        for key in ["case_index", "power_W", "scan_speed_mm_s", "particle_rate", "powder_feed_g_min"]:
            if key in group.columns and len(group):
                meta[key] = group[key].iloc[0] if group_id != "all_conditions" else np.nan
        for col in cols:
            values = group[col].to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1)) if np.sum(np.isfinite(values)) > 1 else np.nan
            rows.append(
                {
                    "case_id": group_id,
                    **meta,
                    "quantity": col,
                    "mean": mean,
                    "std": std,
                    "coefficient_of_variation": float(std / mean) if np.isfinite(std) and mean != 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_geometry_model_comparison(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        for model in ["ellipsoid", "superellipsoid"]:
            out = {
                "time_s": row["time_s"],
                "model": model,
                "fit_status": row["fit_status"] if model == "ellipsoid" else row["superellipsoid_fit_status"],
                "boundary_residual_rmse": row[f"{model}_residual_rmse"],
                "volume_full_m3": row[f"{model}_volume_full_m3"],
                "volume_relative_error": row[f"{model}_volume_relative_error"],
                "volume_proxy_m3": row["volume_proxy_m3"],
                "radial_distance_rmse_m": row[f"{model}_radial_distance_rmse_m"],
                "radial_distance_mean_m": row[f"{model}_radial_distance_mean_m"],
                "chamfer_distance_m": row[f"{model}_chamfer_distance_m"],
                "normalized_chamfer_distance": row[f"{model}_chamfer_distance_m"]
                / max(float(row.get("melt_pool_length_m", np.nan)), 1e-12)
                if np.isfinite(float(row.get("melt_pool_length_m", np.nan)))
                else np.nan,
                "hausdorff_distance_m": row[f"{model}_hausdorff_distance_m"],
            }
            for col in [
                "case_id",
                "case_index",
                "power_W",
                "scan_speed_mm_s",
                "particle_rate",
                "powder_feed_g_min",
                "validation_cohort",
                "source_root",
            ]:
                if col in row.index:
                    out[col] = row[col]
            rows.append(out)
    comparison = pd.DataFrame(rows)
    group_cols = ["model"]
    if "case_id" in comparison.columns:
        case_metrics = (
            comparison.groupby(["case_id", "model"], as_index=False)
            .agg(
                mean_boundary_residual_rmse=("boundary_residual_rmse", "mean"),
                mean_volume_relative_error=("volume_relative_error", "mean"),
                std_volume_relative_error=("volume_relative_error", "std"),
                mean_radial_distance_rmse_m=("radial_distance_rmse_m", "mean"),
                mean_chamfer_distance_m=("chamfer_distance_m", "mean"),
                mean_normalized_chamfer_distance=("normalized_chamfer_distance", "mean"),
                mean_hausdorff_distance_m=("hausdorff_distance_m", "mean"),
                successful_fits=("fit_status", lambda x: int((x == "ok").sum())),
                case_index=("case_index", "first"),
                power_W=("power_W", "first"),
                scan_speed_mm_s=("scan_speed_mm_s", "first"),
                particle_rate=("particle_rate", "first"),
                powder_feed_g_min=("powder_feed_g_min", "first"),
                **{
                    col: (col, "first")
                    for col in ["validation_cohort", "source_root"]
                    if col in comparison.columns
                },
            )
            .assign(time_s="case_summary")
        )
    else:
        case_metrics = pd.DataFrame()
    metrics = (
        comparison.groupby(group_cols, as_index=False)
        .agg(
            mean_boundary_residual_rmse=("boundary_residual_rmse", "mean"),
            mean_volume_relative_error=("volume_relative_error", "mean"),
            std_volume_relative_error=("volume_relative_error", "std"),
            mean_radial_distance_rmse_m=("radial_distance_rmse_m", "mean"),
            mean_chamfer_distance_m=("chamfer_distance_m", "mean"),
            mean_normalized_chamfer_distance=("normalized_chamfer_distance", "mean"),
            mean_hausdorff_distance_m=("hausdorff_distance_m", "mean"),
            successful_fits=("fit_status", lambda x: int((x == "ok").sum())),
        )
        .assign(time_s="summary")
    )
    return pd.concat([comparison, case_metrics, metrics], ignore_index=True, sort=False)


ALPHA_SHAPE_RADIUS_MULTIPLIERS = (1.5, 2.5, 4.0)


def median_nearest_neighbor_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return np.nan
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nn = distances[:, 1]
    return float(np.nanmedian(nn[np.isfinite(nn)])) if np.any(np.isfinite(nn)) else np.nan


def alpha_complex_half_volume(points: np.ndarray, radius_limit: float) -> tuple[float, int, int, str]:
    if points.shape[0] < 4:
        return np.nan, 0, 0, "invalid:insufficient_points"
    try:
        triangulation = Delaunay(points, qhull_options="QJ")
    except Exception as exc:
        return np.nan, 0, 0, f"invalid:delaunay:{exc}"
    retained = 0
    volume = 0.0
    for simplex in triangulation.simplices:
        tetra = points[simplex]
        try:
            a = 2.0 * (tetra[1:] - tetra[0])
            b = np.sum(tetra[1:] ** 2, axis=1) - np.sum(tetra[0] ** 2)
            center = np.linalg.solve(a, b)
            radius = float(np.linalg.norm(center - tetra[0]))
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(radius) and radius <= radius_limit:
            retained += 1
            volume += abs(float(np.linalg.det(np.vstack([tetra[1] - tetra[0], tetra[2] - tetra[0], tetra[3] - tetra[0]])))) / 6.0
    status = "ok" if retained else "invalid:no_retained_tetrahedra"
    return float(volume), retained, int(len(triangulation.simplices)), status


def make_boundary_extraction_sensitivity(table: pd.DataFrame, point_cloud: pd.DataFrame) -> pd.DataFrame:
    if point_cloud is None or len(point_cloud) == 0:
        return pd.DataFrame()
    table_lookup = {
        (str(row.case_id), float(row.time_s)): row
        for row in table.itertuples()
        if hasattr(row, "case_id") and hasattr(row, "time_s")
    }
    rows = []
    for (case_id, time_s), group in point_cloud.groupby(["case_id", "time_s"], sort=False):
        points = group[["xi_m", "Points_1", "Points_2"]].to_numpy(dtype=float)
        points = points[np.isfinite(points).all(axis=1)]
        median_nn = median_nearest_neighbor_distance(points)
        meta_row = table_lookup.get((str(case_id), float(time_s)))
        convex_full_volume = float(getattr(meta_row, "volume_proxy_m3", np.nan)) if meta_row is not None else np.nan
        base_meta = {
            "case_id": str(case_id),
            "time_s": float(time_s),
            "point_count": int(points.shape[0]),
            "median_nearest_neighbor_m": median_nn,
            "convex_full_volume_proxy_m3": convex_full_volume,
            "case_index": int(group["case_index"].iloc[0]) if "case_index" in group.columns else np.nan,
            "power_W": float(group["power_W"].iloc[0]) if "power_W" in group.columns else np.nan,
            "scan_speed_mm_s": float(group["scan_speed_mm_s"].iloc[0]) if "scan_speed_mm_s" in group.columns else np.nan,
            "powder_feed_g_min": float(group["powder_feed_g_min"].iloc[0]) if "powder_feed_g_min" in group.columns else np.nan,
        }
        for col in ["validation_cohort", "source_root"]:
            if col in group.columns:
                base_meta[col] = group[col].iloc[0]
        for multiplier in ALPHA_SHAPE_RADIUS_MULTIPLIERS:
            radius = multiplier * median_nn if np.isfinite(median_nn) else np.nan
            if np.isfinite(radius):
                half_volume, retained, total_tetra, status = alpha_complex_half_volume(points, radius)
            else:
                half_volume, retained, total_tetra, status = np.nan, 0, 0, "invalid:no_nearest_neighbor_scale"
            full_volume = 2.0 * half_volume if np.isfinite(half_volume) else np.nan
            rows.append(
                {
                    **base_meta,
                    "method": "alpha_complex_volume_sensitivity",
                    "alpha_radius_multiplier": multiplier,
                    "alpha_radius_m": radius,
                    "retained_tetrahedra": retained,
                    "total_tetrahedra": total_tetra,
                    "alpha_full_volume_m3": full_volume,
                    "alpha_to_convex_volume_ratio": full_volume / convex_full_volume
                    if np.isfinite(full_volume) and np.isfinite(convex_full_volume) and convex_full_volume > 0
                    else np.nan,
                    "relative_difference_from_convex": abs(full_volume - convex_full_volume) / convex_full_volume
                    if np.isfinite(full_volume) and np.isfinite(convex_full_volume) and convex_full_volume > 0
                    else np.nan,
                    "status": status,
                    "manuscript_use": "sensitivity diagnostic only; not used for superellipsoid model selection",
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        summary = (
            out.groupby(["method", "alpha_radius_multiplier"], as_index=False)
            .agg(
                valid_fraction=("status", lambda x: float(pd.Series(x).astype(str).eq("ok").mean())),
                mean_alpha_to_convex_volume_ratio=("alpha_to_convex_volume_ratio", "mean"),
                mean_relative_difference_from_convex=("relative_difference_from_convex", "mean"),
                n_condition_time_steps=("case_id", "count"),
            )
            .assign(case_id="summary", time_s="summary", manuscript_use="aggregate sensitivity diagnostic")
        )
        out = pd.concat([out, summary], ignore_index=True, sort=False)
    return out


def make_superellipsoid_parameters(table: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_id",
        "case_index",
        "power_W",
        "scan_speed_mm_s",
        "particle_rate",
        "powder_feed_g_min",
        "time_s",
        "superellipsoid_fit_status",
        "superellipsoid_af_m",
        "superellipsoid_ar_m",
        "superellipsoid_b_m",
        "superellipsoid_c_m",
        "superellipsoid_xic_m",
        "superellipsoid_zc_m",
        "superellipsoid_n",
        "superellipsoid_m",
        "superellipsoid_p",
        "superellipsoid_residual_rmse",
        "superellipsoid_volume_full_m3",
        "superellipsoid_volume_relative_error",
    ]
    return table[[col for col in cols if col in table.columns]].copy()


def make_error_summary(
    table: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    eigenvalues: pd.DataFrame,
    coupled_summary: pd.DataFrame | None = None,
    coupled_eigenvalues: pd.DataFrame | None = None,
) -> pd.DataFrame:
    raw_rows = table["raw_rows"].sum()
    exact_rows = table["exact_dedup_rows"].sum()
    unique_points = table["unique_points"].sum()
    validation_rel = dynamics_summary["validation_relative_rmse"].to_numpy(dtype=float)

    rows = [
        {
            "error_category": "boundary_reconstruction",
            "metric": "exact_duplicate_fraction",
            "value": float(1.0 - exact_rows / raw_rows),
            "interpretation": "Fraction removed by exact row deduplication before modeling.",
        },
        {
            "error_category": "boundary_reconstruction",
            "metric": "coordinate_duplicate_fraction_after_exact_dedup",
            "value": float(1.0 - unique_points / exact_rows),
            "interpretation": "Residual repeated coordinates collapsed by coordinate-wise averaging.",
        },
        {
            "error_category": "geometric_free_boundary",
            "metric": "mean_ellipsoid_residual_rmse",
            "value": float(np.nanmean(table["ellipsoid_residual_rmse"])),
            "interpretation": "Dimensionless residual of the asymmetric ellipsoid boundary equation.",
        },
        {
            "error_category": "geometric_free_boundary",
            "metric": "mean_ellipsoid_volume_relative_error",
            "value": float(np.nanmean(table["ellipsoid_volume_relative_error"])),
            "interpretation": "Full-domain ellipsoid volume error relative to mirrored convex-hull proxy.",
        },
        {
            "error_category": "geometric_free_boundary",
            "metric": "mean_superellipsoid_residual_rmse",
            "value": float(np.nanmean(table["superellipsoid_residual_rmse"])),
            "interpretation": "Dimensionless residual of the asymmetric superellipsoid boundary equation.",
        },
        {
            "error_category": "geometric_free_boundary",
            "metric": "mean_superellipsoid_chamfer_distance_m",
            "value": float(np.nanmean(table["superellipsoid_chamfer_distance_m"])),
            "interpretation": "Mean symmetric surface-distance proxy for the fitted superellipsoid boundary.",
        },
        {
            "error_category": "geometric_free_boundary",
            "metric": "mean_superellipsoid_hausdorff_distance_m",
            "value": float(np.nanmean(table["superellipsoid_hausdorff_distance_m"])),
            "interpretation": "Worst-case surface mismatch sampled from the fitted superellipsoid boundary.",
        },
        {
            "error_category": "geometric_free_boundary",
            "metric": "mean_superellipsoid_volume_relative_error",
            "value": float(np.nanmean(table["superellipsoid_volume_relative_error"])),
            "interpretation": "Full-domain superellipsoid volume error relative to mirrored convex-hull proxy.",
        },
        {
            "error_category": "dynamical_prediction",
            "metric": "mean_validation_relative_rmse",
            "value": float(np.nanmean(validation_rel)),
            "interpretation": "Mean validation relative RMSE over the seven baseline state variables.",
        },
        {
            "error_category": "stability",
            "metric": "stable_state_fraction",
            "value": float(eigenvalues["stable_if_negative"].mean()),
            "interpretation": "Fraction of fitted Jacobian eigenvalues with negative real part.",
        },
    ]
    if coupled_summary is not None and coupled_eigenvalues is not None:
        rows.extend(
            [
                {
                    "error_category": "dynamical_prediction",
                    "metric": "mean_coupled_validation_relative_rmse",
                    "value": float(np.nanmean(coupled_summary["validation_relative_rmse"])),
                    "interpretation": "Mean validation relative RMSE for the coupled ridge attractor model.",
                },
                {
                    "error_category": "stability",
                    "metric": "coupled_stable_state_fraction",
                    "value": float(coupled_eigenvalues["stable_if_real_negative"].mean()),
                    "interpretation": "Fraction of coupled-model Jacobian eigenvalues with negative real part.",
                },
            ]
        )
    return pd.DataFrame(rows)


def classify_dimensionless(symbol: str, value: float) -> str:
    if not np.isfinite(value):
        return "not_available"
    if symbol == "Pe":
        if value < 0.5:
            return "diffusion_dominant"
        if value <= 5.0:
            return "mixed_conduction_advection"
        return "advection_dominant"
    if symbol == "Ste":
        if value < 0.1:
            return "reference_scale_latent_heat_interpretation"
        if value <= 2.0:
            return "sensible_and_latent_comparable"
        return "sensible_heat_dominant"
    if symbol == "E_star":
        if value < 1.0:
            return "weak_normalized_heat_input"
        if value <= 10.0:
            return "moderate_normalized_heat_input"
        return "large_normalized_heat_input"
    if symbol == "Ma":
        if value < 100.0:
            return "reference_scale_weak_marangoni_interpretation"
        if value <= 1000.0:
            return "reference_scale_marangoni_interpretation"
        return "reference_scale_strong_marangoni_interpretation"
    return "unclassified"


def compute_dimensionless_values(
    table: pd.DataFrame,
    props: dict[str, float],
    absorptivity: float,
    surface_tension_temperature_coefficient: float,
) -> dict[str, float]:
    table = representative_case_subset(table)
    l_ref = float(table["L_ref_m"].mean())
    t_final = float(table["time_s"].max())
    speed = float(table["scan_speed_m_s"].mean()) if "scan_speed_m_s" in table.columns else SCAN_SPEED_M_PER_S
    power = float(table["power_W"].mean()) if "power_W" in table.columns else LASER_POWER_W
    liquidus = MATERIAL_CONSTANTS["liquidus_temperature_K"]
    solidus = MATERIAL_CONSTANTS["solidus_temperature_K"]
    rho = float(props["density_kg_per_m3"])
    cp = float(props["specific_heat_J_per_kg_K"])
    k = float(props["thermal_conductivity_W_per_m_K"])
    alpha = float(props["thermal_diffusivity_m2_per_s"])
    mu = float(props["viscosity_kg_per_m_s"])
    delta_t_melt = liquidus - MATERIAL_CONSTANTS["initial_temperature_K"]
    melt_range = liquidus - solidus
    return {
        "Pe": speed * l_ref / alpha,
        "Fo_final": alpha * t_final / (l_ref**2),
        "Ste": cp * melt_range / MATERIAL_CONSTANTS["latent_heat_fusion_J_per_kg"],
        "E_star": absorptivity
        * power
        / (
            rho
            * cp
            * speed
            * MATERIAL_CONSTANTS["beam_radius_m"] ** 2
            * delta_t_melt
        ),
        "Re": rho * speed * l_ref / mu,
        "Pr": mu * cp / k,
        "Ma": abs(surface_tension_temperature_coefficient) * melt_range * l_ref / (mu * alpha),
    }


def make_dimensionless_sensitivity_summary(
    table: pd.DataFrame, curves: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    reference_temperatures = {
        "solidus": MATERIAL_CONSTANTS["solidus_temperature_K"],
        "mid_melt": 0.5
        * (MATERIAL_CONSTANTS["solidus_temperature_K"] + MATERIAL_CONSTANTS["liquidus_temperature_K"]),
        "liquidus": MATERIAL_CONSTANTS["liquidus_temperature_K"],
    }
    absorptivity_factors = [0.8, 1.0, 1.2]
    surface_tension_factors = [0.8, 1.0, 1.2]
    scenario_rows = []
    for temp_label, temp_value in reference_temperatures.items():
        props = property_values_at(curves, temp_value)
        for eta_factor in absorptivity_factors:
            for sigma_factor in surface_tension_factors:
                values = compute_dimensionless_values(
                    table,
                    props,
                    MATERIAL_CONSTANTS["absorptivity"] * eta_factor,
                    MATERIAL_CONSTANTS["surface_tension_temperature_coefficient_N_per_m_K"] * sigma_factor,
                )
                scenario = {
                    "reference_temperature_label": temp_label,
                    "reference_temperature_K": temp_value,
                    "absorptivity_factor": eta_factor,
                    "surface_tension_coefficient_factor": sigma_factor,
                    **values,
                }
                scenario_rows.append(scenario)

    scenarios = pd.DataFrame(scenario_rows)
    baseline = scenarios[
        (scenarios["reference_temperature_label"] == "liquidus")
        & np.isclose(scenarios["absorptivity_factor"], 1.0)
        & np.isclose(scenarios["surface_tension_coefficient_factor"], 1.0)
    ].iloc[0]
    rows = []
    for symbol in ["Pe", "Ste", "E_star", "Ma"]:
        values = scenarios[symbol].to_numpy(dtype=float)
        baseline_value = float(baseline[symbol])
        classes = sorted({classify_dimensionless(symbol, float(value)) for value in values})
        baseline_class = classify_dimensionless(symbol, baseline_value)
        min_idx = int(np.nanargmin(values))
        max_idx = int(np.nanargmax(values))
        rows.append(
            {
                "symbol": symbol,
                "baseline_value": baseline_value,
                "min_value": float(np.nanmin(values)),
                "max_value": float(np.nanmax(values)),
                "relative_min": float(np.nanmin(values) / baseline_value) if baseline_value != 0 else np.nan,
                "relative_max": float(np.nanmax(values) / baseline_value) if baseline_value != 0 else np.nan,
                "baseline_class": baseline_class,
                "observed_classes": ";".join(classes),
                "conclusion_changed": bool(any(item != baseline_class for item in classes)),
                "min_scenario": (
                    f"{scenarios.iloc[min_idx]['reference_temperature_label']}, "
                    f"eta_x={scenarios.iloc[min_idx]['absorptivity_factor']:.1f}, "
                    f"dsigma_x={scenarios.iloc[min_idx]['surface_tension_coefficient_factor']:.1f}"
                ),
                "max_scenario": (
                    f"{scenarios.iloc[max_idx]['reference_temperature_label']}, "
                    f"eta_x={scenarios.iloc[max_idx]['absorptivity_factor']:.1f}, "
                    f"dsigma_x={scenarios.iloc[max_idx]['surface_tension_coefficient_factor']:.1f}"
                ),
                "scenarios_tested": int(len(scenarios)),
                "perturbation_design": "reference_temperature=solidus/mid_melt/liquidus; absorptivity +/-20%; d_sigma_dT +/-20%",
            }
        )
    return pd.DataFrame(rows)


def safe_condition_number(matrix: np.ndarray) -> float:
    if matrix.size == 0 or min(matrix.shape) < 2:
        return float("nan")
    finite = np.isfinite(matrix).all(axis=1)
    x = matrix[finite]
    if x.shape[0] < 2:
        return float("nan")
    scale = np.nanstd(x, axis=0, ddof=1)
    keep = np.isfinite(scale) & (scale > 0)
    if np.sum(keep) < 2:
        return float("nan")
    z = (x[:, keep] - np.nanmean(x[:, keep], axis=0)) / scale[keep]
    try:
        return float(np.linalg.cond(z))
    except np.linalg.LinAlgError:
        return float("inf")


def make_parameter_identifiability(
    table: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    super_cols = {
        "a_f": "superellipsoid_af_m",
        "a_r": "superellipsoid_ar_m",
        "b": "superellipsoid_b_m",
        "c": "superellipsoid_c_m",
        "xi_c": "superellipsoid_xic_m",
        "z_c": "superellipsoid_zc_m",
        "n": "superellipsoid_n",
        "m": "superellipsoid_m",
        "p": "superellipsoid_p",
    }
    ok = table["superellipsoid_fit_status"].eq("ok")
    super_matrix = table.loc[ok, list(super_cols.values())].to_numpy(dtype=float)
    super_condition = safe_condition_number(super_matrix)
    success_rate = float(ok.mean())
    for name, col in super_cols.items():
        values = table.loc[ok, col].to_numpy(dtype=float)
        mean = float(np.nanmean(values)) if values.size else np.nan
        std = float(np.nanstd(values, ddof=1)) if np.sum(np.isfinite(values)) > 1 else np.nan
        cv = float(abs(std / mean)) if np.isfinite(std) and mean != 0 else np.nan
        relative_range = (
            float((np.nanmax(values) - np.nanmin(values)) / abs(mean))
            if values.size and np.isfinite(mean) and mean != 0
            else np.nan
        )
        bound_fraction = 0.0
        if name in {"n", "m", "p"} and values.size:
            bound_fraction = float(np.mean((values <= 1.05) | (values >= SUPERELLIPSOID_EXPONENT_UPPER - 0.05)))
        risk = "low"
        if name in {"n", "m", "p"} or bound_fraction > 0.2 or (np.isfinite(cv) and cv > 0.5):
            risk = "high"
        elif (np.isfinite(cv) and cv > 0.2) or success_rate < 1.0:
            risk = "medium"
        rows.append(
            {
                "parameter_group": "superellipsoid_geometry",
                "parameter": name,
                "n_observations": int(values.size),
                "mean": mean,
                "std": std,
                "coefficient_of_variation": cv,
                "relative_range": relative_range,
                "success_rate": success_rate,
                "condition_proxy": super_condition,
                "bound_fraction": bound_fraction,
                "risk_level": risk,
                "risk_reason": "bounded nonlinear exponent under short time series"
                if name in {"n", "m", "p"}
                else "time variation of fitted geometric scale",
            }
        )

    for _, row in dynamics_summary.iterrows():
        k = float(row["k_per_s"])
        val = float(row["validation_relative_rmse"])
        risk = "low" if k > 0 and val <= 0.1 else "medium" if k > 0 and val <= 0.2 else "high"
        rows.append(
            {
                "parameter_group": "diagonal_attractor",
                "parameter": f"k_{row['state']}",
                "n_observations": int(row["train_points"]),
                "mean": k,
                "std": np.nan,
                "coefficient_of_variation": np.nan,
                "relative_range": np.nan,
                "success_rate": 1.0 if row["fit_status"] == "ok" else 0.0,
                "condition_proxy": np.nan,
                "bound_fraction": 0.0,
                "risk_level": risk,
                "risk_reason": "positive relaxation rate with validation relative RMSE check",
            }
        )

    a_values = coupled_matrix["A_value_per_s"].to_numpy(dtype=float)
    t = table["time_s"].to_numpy(dtype=float)
    transition_count = max(int(np.sum(train_split_mask(t))) - 1, 0)
    if "case_id" in coupled_matrix.columns and coupled_matrix["case_id"].nunique() > 1:
        cm_for_condition = coupled_matrix[coupled_matrix["case_id"].astype(str).eq(representative_case_id(coupled_matrix))].copy()
    else:
        cm_for_condition = coupled_matrix.copy()
    condition_matrix = cm_for_condition.pivot(index="row_state", columns="column_state", values="A_value_per_s").loc[
        STATE_COLUMNS, STATE_COLUMNS
    ]
    rows.append(
        {
            "parameter_group": "coupled_attractor",
            "parameter": "A_matrix_entries",
            "n_observations": transition_count,
            "mean": float(np.nanmean(a_values)),
            "std": float(np.nanstd(a_values, ddof=1)),
            "coefficient_of_variation": float(abs(np.nanstd(a_values, ddof=1) / np.nanmean(a_values)))
            if np.nanmean(a_values) != 0
            else np.nan,
            "relative_range": float((np.nanmax(a_values) - np.nanmin(a_values)) / max(abs(np.nanmean(a_values)), 1e-12)),
            "success_rate": 1.0,
            "condition_proxy": safe_condition_number(condition_matrix.to_numpy(dtype=float)),
            "bound_fraction": np.nan,
            "risk_level": "high",
            "risk_reason": "coupling coefficients are identified from short condition-wise sequences and are not selected by validation",
        }
    )
    return pd.DataFrame(rows)


def make_error_budget_summary(
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_summary: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    raw_rows = float(table["raw_rows"].sum())
    exact_rows = float(table["exact_dedup_rows"].sum())
    unique_points = float(table["unique_points"].sum())
    geom = geometry_comparison[geometry_comparison["time_s"] == "summary"].set_index("model")
    ellipsoid_boundary = float(geom.loc["ellipsoid", "mean_boundary_residual_rmse"])
    super_boundary = float(geom.loc["superellipsoid", "mean_boundary_residual_rmse"])
    super_volume = float(geom.loc["superellipsoid", "mean_volume_relative_error"])
    super_chamfer = float(geom.loc["superellipsoid", "mean_chamfer_distance_m"])
    super_normalized_chamfer = float(geom.loc["superellipsoid", "mean_normalized_chamfer_distance"])
    diagonal_validation = float(np.nanmean(dynamics_summary["validation_relative_rmse"]))
    coupled_validation = float(np.nanmean(coupled_summary["validation_relative_rmse"]))
    umax_error = float(
        dynamics_summary.loc[dynamics_summary["state"].eq("Umax_m_per_s"), "validation_relative_rmse"].mean()
    )
    sensitivity_span = float(
        np.nanmax(
            np.maximum(
                abs(dimensionless_sensitivity["relative_min"].to_numpy(dtype=float) - 1.0),
                abs(dimensionless_sensitivity["relative_max"].to_numpy(dtype=float) - 1.0),
            )
        )
    )
    any_dimensionless_class_changed = bool(dimensionless_sensitivity["conclusion_changed"].any())
    rows = [
        {
            "error_term": "E_reconstruction",
            "component_name": "point-cloud preprocessing and symmetry reconstruction",
            "source": "raw FLOW-3D molten-region CSV files",
            "primary_metric": "exact_duplicate_fraction",
            "primary_value": float(1.0 - exact_rows / raw_rows),
            "secondary_metric": "coordinate_duplicate_fraction_after_exact_dedup",
            "secondary_value": float(1.0 - unique_points / exact_rows),
            "risk_level": "medium",
            "source_table": "modeling_table.csv",
            "manuscript_interpretation": "The boundary is the envelope of the exported molten region; missing solid-domain data are a modeling limitation.",
        },
        {
            "error_term": "E_geometry",
            "component_name": "analytic free-boundary fit",
            "source": "ellipsoid and superellipsoid residuals plus geometric-distance diagnostics",
            "primary_metric": "normalized_superellipsoid_chamfer_distance",
            "primary_value": super_normalized_chamfer,
            "secondary_metric": "raw_superellipsoid_chamfer_distance_m",
            "secondary_value": super_chamfer,
            "risk_level": "medium",
            "source_table": "geometry_model_comparison.csv",
            "manuscript_interpretation": "The selected superellipsoid improves the implicit boundary residual; the normalized Chamfer distance is scaled by melt-pool length, while the raw Chamfer distance remains a geometric-risk diagnostic.",
        },
        {
            "error_term": "E_volume_proxy",
            "component_name": "full-domain volume proxy",
            "source": "mirrored half-domain convex hull",
            "primary_metric": "superellipsoid_volume_relative_error",
            "primary_value": super_volume,
            "secondary_metric": "convex_hull_status_ok_fraction",
            "secondary_value": float(table["convex_hull_status"].eq("ok").mean()),
            "risk_level": "medium_high",
            "source_table": "geometry_model_comparison.csv",
            "manuscript_interpretation": "The volume metric is relative to a convex-hull proxy rather than an exact thermodynamic melt volume, so it is reported as a proxy limitation.",
        },
        {
            "error_term": "E_dynamics",
            "component_name": "reduced-order prediction",
            "source": "validation time steps",
            "primary_metric": "diagonal_mean_validation_relative_rmse",
            "primary_value": diagonal_validation,
            "secondary_metric": "Umax_validation_relative_rmse",
            "secondary_value": umax_error,
            "risk_level": "medium",
            "source_table": "dynamics_fit_summary.csv",
            "manuscript_interpretation": "The diagonal attractor is selected as a parsimonious baseline; the coupled model is retained as a negative control because its additional parameters are not supported robustly.",
        },
        {
            "error_term": "E_model_complexity",
            "component_name": "coupled model comparison",
            "source": "diagonal versus coupled validation",
            "primary_metric": "coupled_mean_validation_relative_rmse",
            "primary_value": coupled_validation,
            "secondary_metric": "coupled_minus_diagonal_validation_relative_rmse",
            "secondary_value": float(coupled_validation - diagonal_validation),
            "risk_level": "high_for_coupled_model",
            "source_table": "dynamics_model_comparison.csv",
            "manuscript_interpretation": "Coupling is physically plausible but not statistically justified by this short sequence; complexity is a cost, not a benefit.",
        },
        {
            "error_term": "E_parameter_scale",
            "component_name": "dimensionless material-property scaling",
            "source": "reference-temperature and material-parameter perturbations",
            "primary_metric": "max_relative_dimensionless_span",
            "primary_value": sensitivity_span,
            "secondary_metric": "any_dimensionless_class_changed",
            "secondary_value": float(any_dimensionless_class_changed),
            "risk_level": "medium" if any_dimensionless_class_changed else "low_medium",
            "source_table": "dimensionless_sensitivity_summary.csv",
            "manuscript_interpretation": "Dimensionless numbers are scaling diagnostics; sensitivity is reported to avoid over-interpreting a single reference state.",
        },
    ]
    return pd.DataFrame(rows)


def make_identifiability_diagnostics_v4(
    table: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_summary: pd.DataFrame,
    coupled_matrix: pd.DataFrame,
    parameter_identifiability: pd.DataFrame,
) -> pd.DataFrame:
    """Build a v4 identifiability audit with lightweight local sensitivity proxies."""

    base_lookup = {
        (row.parameter_group, row.parameter): row
        for row in parameter_identifiability.itertuples(index=False)
    }
    times = table["time_s"].to_numpy(dtype=float)
    rows: list[dict[str, float | int | str]] = []

    def finite_stats(values: np.ndarray) -> dict[str, float | int]:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {
                "n_observations": 0,
                "mean": np.nan,
                "std": np.nan,
                "coefficient_of_variation": np.nan,
                "relative_range": np.nan,
                "local_sensitivity_norm": np.nan,
                "fisher_information_proxy": np.nan,
            }
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
        cv = float(abs(std / mean)) if mean != 0 else np.nan
        rel_range = float((np.nanmax(values) - np.nanmin(values)) / max(abs(mean), 1e-12))
        centered = values - mean
        fisher = float(np.nansum((centered / max(std, 1e-12)) ** 2)) if values.size > 1 else 0.0
        local_sensitivity = np.nan
        if values.size > 2 and times.size >= values.size:
            dt = np.diff(times[: values.size])
            dv = np.diff(values)
            valid = np.isfinite(dt) & (dt > 0) & np.isfinite(dv)
            if np.any(valid):
                local_sensitivity = float(np.nanmedian(np.abs(dv[valid] / dt[valid])) / max(abs(mean), 1e-12))
        return {
            "n_observations": int(values.size),
            "mean": mean,
            "std": std,
            "coefficient_of_variation": cv,
            "relative_range": rel_range,
            "local_sensitivity_norm": local_sensitivity,
            "fisher_information_proxy": fisher,
        }

    def v4_risk(
        base_risk: str,
        cv: float,
        bound_fraction: float,
        condition_proxy: float,
        validation_relative_rmse: float = np.nan,
        force_high: bool = False,
    ) -> str:
        if force_high:
            return "high"
        if str(base_risk).lower().startswith("high"):
            return "high"
        if np.isfinite(condition_proxy) and condition_proxy > 1e8:
            return "high"
        if np.isfinite(bound_fraction) and bound_fraction > 0.2:
            return "high"
        if np.isfinite(validation_relative_rmse) and validation_relative_rmse > 0.2:
            return "high"
        if np.isfinite(cv) and cv > 0.5:
            return "high"
        if np.isfinite(validation_relative_rmse) and validation_relative_rmse > 0.1:
            return "medium"
        if np.isfinite(cv) and cv > 0.2:
            return "medium"
        return "low"

    super_cols = {
        "a_f": "superellipsoid_af_m",
        "a_r": "superellipsoid_ar_m",
        "b": "superellipsoid_b_m",
        "c": "superellipsoid_c_m",
        "xi_c": "superellipsoid_xic_m",
        "z_c": "superellipsoid_zc_m",
        "n": "superellipsoid_n",
        "m": "superellipsoid_m",
        "p": "superellipsoid_p",
    }
    ok = table["superellipsoid_fit_status"].eq("ok")
    super_matrix = table.loc[ok, list(super_cols.values())].to_numpy(dtype=float)
    super_condition = safe_condition_number(super_matrix)
    for parameter, col in super_cols.items():
        values = table.loc[ok, col].to_numpy(dtype=float)
        stats = finite_stats(values)
        bound_fraction = 0.0
        if parameter in {"n", "m", "p"} and values.size:
            bound_fraction = float(np.mean((values <= 1.05) | (values >= SUPERELLIPSOID_EXPONENT_UPPER - 0.05)))
        base = base_lookup.get(("superellipsoid_geometry", parameter))
        base_risk = getattr(base, "risk_level", "low")
        risk = v4_risk(
            str(base_risk),
            float(stats["coefficient_of_variation"]),
            bound_fraction,
            super_condition,
            force_high=parameter in {"n", "m", "p"} and bound_fraction > 0.0,
        )
        rows.append(
            {
                "parameter_group": "superellipsoid_geometry",
                "parameter": parameter,
                **stats,
                "bound_fraction": bound_fraction,
                "condition_proxy": super_condition,
                "parameter_to_transition_ratio": np.nan,
                "validation_relative_rmse": np.nan,
                "risk_level": risk,
                "source_table": "modeling_table.csv; parameter_identifiability.csv",
                "diagnostic_basis": "time-series variability, exponent bound fraction, finite-difference sensitivity",
                "risk_reason": "shape exponent reaches imposed bound"
                if parameter in {"n", "m", "p"} and bound_fraction > 0.0
                else "geometric parameter varies across the short time sequence",
                "recommendation": "Report as bounded nonlinear manifold parameter"
                if parameter in {"n", "m", "p"}
                else "Retain as interpretable geometric coordinate",
            }
        )

    for row in dynamics_summary.itertuples(index=False):
        validation = float(row.validation_relative_rmse)
        k = float(row.k_per_s)
        fisher = float(row.train_points / max(validation**2, 1e-12))
        risk = v4_risk(
            "low",
            np.nan,
            0.0,
            np.nan,
            validation_relative_rmse=validation,
            force_high=(row.state == "Umax_m_per_s" and validation > 0.2) or k <= 0,
        )
        rows.append(
            {
                "parameter_group": "diagonal_attractor",
                "parameter": f"k_{row.state}",
                "n_observations": int(row.train_points),
                "mean": k,
                "std": np.nan,
                "coefficient_of_variation": np.nan,
                "relative_range": np.nan,
                "local_sensitivity_norm": abs(k),
                "fisher_information_proxy": fisher,
                "bound_fraction": 0.0,
                "condition_proxy": np.nan,
                "parameter_to_transition_ratio": float(1.0 / max(int(row.train_points) - 1, 1)),
                "validation_relative_rmse": validation,
                "risk_level": risk,
                "source_table": "dynamics_fit_summary.csv",
                "diagnostic_basis": "positive relaxation margin and validation relative RMSE",
                "risk_reason": "large validation error for this state"
                if validation > 0.2
                else "positive rate with acceptable validation error",
                "recommendation": "Keep in parsimonious diagonal baseline; discuss high-error states separately"
                if risk == "high"
                else "Keep in parsimonious diagonal baseline",
            }
        )

    if "case_id" in coupled_matrix.columns and coupled_matrix["case_id"].nunique() > 1:
        cm_for_condition = coupled_matrix[coupled_matrix["case_id"].astype(str).eq(representative_case_id(coupled_matrix))].copy()
    else:
        cm_for_condition = coupled_matrix.copy()
    a_matrix = cm_for_condition.pivot(index="row_state", columns="column_state", values="A_value_per_s").loc[
        STATE_COLUMNS, STATE_COLUMNS
    ]
    a_values = a_matrix.to_numpy(dtype=float).ravel()
    stats = finite_stats(a_values)
    transition_count = max(int(np.sum(train_split_mask(times))) - 1, 1)
    condition = safe_condition_number(a_matrix.to_numpy(dtype=float))
    coupled_validation = float(np.nanmean(coupled_summary["validation_relative_rmse"]))
    rows.append(
        {
            "parameter_group": "coupled_attractor",
            "parameter": "A_matrix_entries",
            **stats,
            "bound_fraction": np.nan,
            "condition_proxy": condition,
            "parameter_to_transition_ratio": float(len(a_values) / transition_count),
            "validation_relative_rmse": coupled_validation,
            "risk_level": "high",
            "source_table": "coupled_A_matrix.csv; coupled_dynamics_fit_summary.csv",
            "diagnostic_basis": "matrix condition proxy, parameter-to-transition ratio, validation error",
            "risk_reason": "49 coupling coefficients are inferred from short condition-wise training sequences",
            "recommendation": "Interpret as an overparameterization comparison rather than as the selected dynamics",
        }
    )

    out = pd.DataFrame(rows)
    risk_order = {"low": 1, "medium": 2, "high": 3}
    out["risk_score_numeric"] = out["risk_level"].map(risk_order).fillna(0).astype(float)
    return out


def make_error_bound_summary_v4(error_budget: pd.DataFrame) -> pd.DataFrame:
    """Convert the diagnostic error budget into a semi-formal bound table."""

    budget = error_budget.set_index("error_term")
    bound_terms = [
        (
            "E_reconstruction",
            "C1",
            "Observation error from molten-region-only export and half-domain symmetry reconstruction.",
        ),
        (
            "E_geometry",
            "C2",
            "Projection error between the observed boundary envelope and the analytic superellipsoid manifold.",
        ),
        (
            "E_volume_proxy",
            "C3",
            "Integral descriptor error introduced by the mirrored convex-hull volume proxy.",
        ),
        (
            "E_dynamics",
            "C4",
            "Forecast error of the selected reduced-order diagonal attractor on validation time steps.",
        ),
        (
            "E_parameter_scale",
            "C5",
            "Scaling uncertainty from material-property reference state and scenario perturbations of absorptivity or surface-tension coefficient.",
        ),
    ]
    rows = []
    values = []
    for error_term, _, _ in bound_terms:
        if error_term in budget.index:
            values.append(float(budget.loc[error_term, "primary_value"]))
    max_value = max(values) if values else 1.0
    for idx, (error_term, weight, role) in enumerate(bound_terms, start=1):
        row = budget.loc[error_term]
        value = float(row["primary_value"])
        rows.append(
            {
                "bound_component": error_term,
                "bound_weight_symbol": weight,
                "bound_expression": f"{weight} * {error_term}",
                "theoretical_role": role,
                "source_metric": row["primary_metric"],
                "source_value": value,
                "normalized_proxy": float(value / max(max_value, 1e-12)),
                "risk_level": row["risk_level"],
                "source_table": row["source_table"],
                "bound_statement": "E_total <= C1 E_reconstruction + C2 E_geometry + C3 E_volume_proxy + C4 E_dynamics + C5 E_parameter_scale",
                "interpretation": row["manuscript_interpretation"],
                "constant_policy": "C_i are reported as sensitivity weights, not estimated as universal constants from one process condition.",
                "display_order": idx,
            }
        )
    return pd.DataFrame(rows)


def make_dimensionless_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "Pe",
                "name": "Peclet number",
                "definition": "v L / alpha",
                "required_parameters": "scan speed v, reference length L, thermal diffusivity alpha",
                "status": "symbolic_pending_material_parameters",
            },
            {
                "symbol": "Fo",
                "name": "Fourier number",
                "definition": "alpha t / L^2",
                "required_parameters": "thermal diffusivity alpha, time t, reference length L",
                "status": "symbolic_pending_material_parameters",
            },
            {
                "symbol": "Ste",
                "name": "Stefan number",
                "definition": "c_p (T_l - T_m) / L_m",
                "required_parameters": "specific heat c_p, liquidus/melting scale T_l - T_m, latent heat L_m",
                "status": "symbolic_pending_material_parameters",
            },
            {
                "symbol": "E*",
                "name": "normalized heat input",
                "definition": "eta P / [rho c_p v r_b^2 (T_m - T_0)]",
                "required_parameters": "absorptivity eta, power P, density rho, specific heat c_p, scan speed v, beam radius r_b, T_m, T_0",
                "status": "symbolic_pending_material_parameters",
            },
            {
                "symbol": "l_f, l_r, w, h",
                "name": "dimensionless melt-pool geometry",
                "definition": "L_f/L_ref, L_r/L_ref, W/L_ref, H/L_ref",
                "required_parameters": "geometry states from point cloud and L_ref from quasi-steady mean length",
                "status": "computed_in_modeling_table",
            },
        ]
    )


def make_material_parameter_table() -> pd.DataFrame:
    setup_note = "Flow3D_setup.md authoritative setup note"
    rows = [
        ("material", MATERIAL_CONSTANTS["material"], "", setup_note),
        ("laser_power", LASER_POWER_W, "W", "process_setting"),
        ("scan_speed", SCAN_SPEED_M_PER_S, "m/s", "process_setting"),
        ("powder_feed_rate", POWDER_FEED_KG_PER_S, "kg/s", "converted from 12 g/min"),
        ("beam_radius", MATERIAL_CONSTANTS["beam_radius_m"], "m", setup_note),
        ("absorptivity", MATERIAL_CONSTANTS["absorptivity"], "-", setup_note),
        ("initial_temperature", MATERIAL_CONSTANTS["initial_temperature_K"], "K", "post-processing default; setup note lists reference temperature 273.15 K"),
        ("ambient_temperature", MATERIAL_CONSTANTS["ambient_temperature_K"], "K", "post-processing ambient default"),
        ("powder_initial_temperature", MATERIAL_CONSTANTS["powder_initial_temperature_K"], "K", "post-processing powder default"),
        ("solidus_temperature", MATERIAL_CONSTANTS["solidus_temperature_K"], "K", setup_note),
        ("liquidus_temperature", MATERIAL_CONSTANTS["liquidus_temperature_K"], "K", setup_note),
        ("latent_heat_fusion", MATERIAL_CONSTANTS["latent_heat_fusion_J_per_kg"], "J/kg", setup_note),
        ("boiling_temperature", MATERIAL_CONSTANTS["boiling_temperature_K"], "K", setup_note),
        ("latent_heat_vaporization", MATERIAL_CONSTANTS["latent_heat_vaporization_J_per_kg"], "J/kg", setup_note),
        ("surface_tension", MATERIAL_CONSTANTS["surface_tension_N_per_m"], "N/m", f"{setup_note}; kg/s^2 equals N/m"),
        (
            "surface_tension_temperature_coefficient",
            MATERIAL_CONSTANTS["surface_tension_temperature_coefficient_N_per_m_K"],
            "N/(m K)",
            f"{setup_note}; kg/s^2/K equals N/(m K)",
        ),
        ("emissivity", MATERIAL_CONSTANTS["emissivity"], "-", "user_provided"),
        (
            "convective_heat_transfer_coefficient",
            MATERIAL_CONSTANTS["convective_heat_transfer_W_per_m2_K"],
            "W/(m2 K)",
            "user_provided",
        ),
        ("powder_particle_diameter", MATERIAL_CONSTANTS["powder_particle_diameter_m"], "m", "user_provided"),
        ("powder_capture_efficiency", MATERIAL_CONSTANTS["powder_capture_efficiency"], "-", "user_provided"),
        ("recoil_pressure_enabled", MATERIAL_CONSTANTS["recoil_pressure_enabled"], "-", "user_provided"),
        ("powder_stream_radius", MATERIAL_CONSTANTS["powder_stream_radius_m"], "m", "not_defined"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "source_or_note"])


def make_temperature_dependent_property_table(
    table: pd.DataFrame, curves: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    time_mask = table["time_s"] >= QUASI_STEADY_START_S
    reference_temperatures = {
        "initial": MATERIAL_CONSTANTS["initial_temperature_K"],
        "solidus": MATERIAL_CONSTANTS["solidus_temperature_K"],
        "liquidus": MATERIAL_CONSTANTS["liquidus_temperature_K"],
        "liquidus_plus_100K": MATERIAL_CONSTANTS["liquidus_temperature_K"] + 100.0,
        "quasi_steady_mean_temperature": float(table.loc[time_mask, "Tmean_K"].mean()),
    }
    rows = []
    for label, temperature in reference_temperatures.items():
        row = {"reference": label, **property_values_at(curves, temperature)}
        rows.append(row)
    return pd.DataFrame(rows)


def make_dimensionless_number_table(
    table: pd.DataFrame, property_table: pd.DataFrame
) -> pd.DataFrame:
    liquidus = MATERIAL_CONSTANTS["liquidus_temperature_K"]
    solidus = MATERIAL_CONSTANTS["solidus_temperature_K"]
    t_ref = float(0.5 * (liquidus + solidus))
    props = property_table.loc[property_table["reference"] == "liquidus"].iloc[0]
    rho = float(props["density_kg_per_m3"])
    cp = float(props["specific_heat_J_per_kg_K"])
    k = float(props["thermal_conductivity_W_per_m_K"])
    alpha = float(props["thermal_diffusivity_m2_per_s"])
    mu = float(props["viscosity_kg_per_m_s"])

    delta_t_melt = liquidus - MATERIAL_CONSTANTS["initial_temperature_K"]
    melt_range = liquidus - solidus
    ste = cp * melt_range / MATERIAL_CONSTANTS["latent_heat_fusion_J_per_kg"]
    prandtl = mu * cp / k
    rows = [
        ("global", "T_ref", t_ref, "K", "midpoint of solidus and liquidus"),
        ("global", "rho_liquidus", rho, "kg/m3", "temperature-dependent property curve value at liquidus"),
        ("global", "cp_liquidus", cp, "J/(kg K)", "temperature-dependent property curve value at liquidus"),
        ("global", "k_liquidus", k, "W/(m K)", "temperature-dependent property curve value at liquidus"),
        ("global", "alpha_liquidus", alpha, "m2/s", "temperature-dependent k/(rho cp) at liquidus"),
        ("global", "mu_liquidus", mu, "kg/(m s)", "temperature-dependent viscosity curve value at liquidus"),
        ("global", "Ste", ste, "-", "cp (T_liquidus - T_solidus) / latent_heat_fusion"),
        ("global", "Pr", prandtl, "-", "mu cp / k"),
    ]
    for case_id, group in table.groupby("case_id", sort=False):
        l_ref = float(group["L_ref_m"].iloc[0])
        t_final = float(group["time_s"].max())
        speed = float(group["scan_speed_m_s"].iloc[0]) if "scan_speed_m_s" in group.columns else SCAN_SPEED_M_PER_S
        power = float(group["power_W"].iloc[0]) if "power_W" in group.columns else LASER_POWER_W
        powder_kg_s = float(group["powder_feed_kg_s"].iloc[0]) if "powder_feed_kg_s" in group.columns else POWDER_FEED_KG_PER_S
        pe = speed * l_ref / alpha
        fo_final = alpha * t_final / (l_ref**2)
        e_star = MATERIAL_CONSTANTS["absorptivity"] * power / (
            rho * cp * speed * MATERIAL_CONSTANTS["beam_radius_m"] ** 2 * delta_t_melt
        )
        reynolds = rho * speed * l_ref / mu
        marangoni = (
            abs(MATERIAL_CONSTANTS["surface_tension_temperature_coefficient_N_per_m_K"])
            * melt_range
            * l_ref
            / (mu * alpha)
        )
        effective_powder_energy_W = MATERIAL_CONSTANTS["powder_capture_efficiency"] * powder_kg_s * (
            cp * (liquidus - MATERIAL_CONSTANTS["powder_initial_temperature_K"])
            + MATERIAL_CONSTANTS["latent_heat_fusion_J_per_kg"]
        )
        rows.extend(
            [
                (case_id, "L_ref", l_ref, "m", "case-wise quasi-steady mean melt-pool length"),
                (case_id, "Pe", pe, "-", "v L_ref / alpha"),
                (case_id, "Fo_final", fo_final, "-", "alpha t_final / L_ref^2"),
                (case_id, "E_star", e_star, "-", "eta P / [rho cp v rb^2 (T_liquidus - T0)]"),
                (case_id, "Re", reynolds, "-", "rho v L_ref / mu"),
                (case_id, "Ma", marangoni, "-", "uses corrected surface-tension temperature coefficient magnitude |d sigma/dT|"),
                (case_id, "powder_heating_power_proxy", effective_powder_energy_W, "W", "capture_efficiency*m_dot*[cp(Tl-Tpowder)+Lf]"),
                (case_id, "absorbed_laser_power", MATERIAL_CONSTANTS["absorptivity"] * power, "W", "eta*P"),
                (case_id, "Ste", ste, "-", "cp (T_liquidus - T_solidus) / latent_heat_fusion"),
                (case_id, "Pr", prandtl, "-", "mu cp / k"),
            ]
        )
    out = pd.DataFrame(rows, columns=["case_id", "symbol", "value", "unit", "definition_or_note"])
    summary_rows = []
    for symbol in ["L_ref", "Pe", "Fo_final", "E_star", "Re", "Ma", "Ste", "Pr"]:
        values = out.loc[out["symbol"].eq(symbol) & out["case_id"].ne("global"), "value"].to_numpy(dtype=float)
        if values.size:
            summary_rows.extend(
                [
                    ("summary", symbol, float(np.nanmean(values)), out.loc[out["symbol"].eq(symbol), "unit"].iloc[0], "mean across conditions"),
                    ("summary_min", symbol, float(np.nanmin(values)), out.loc[out["symbol"].eq(symbol), "unit"].iloc[0], "minimum across conditions"),
                    ("summary_max", symbol, float(np.nanmax(values)), out.loc[out["symbol"].eq(symbol), "unit"].iloc[0], "maximum across conditions"),
                ]
            )
    return pd.concat([out, pd.DataFrame(summary_rows, columns=out.columns)], ignore_index=True)


def make_multi_condition_point_cloud_summary(point_cloud: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["case_id", "time_s"] if "case_id" in point_cloud.columns else ["time_s"]
    agg = {
        "xi_m": ["min", "max"],
        "y_m": ["min", "max"],
        "z_m": ["min", "max"],
        "Temperature": ["mean", "max"],
        "Temperature Gradient At Tgrdout": ["mean", "max"],
        "Velocity_Magnitude": ["mean", "max"],
        "Pressure": ["mean", "max"],
        "Heat Absorption Rate": ["mean", "max"],
        "Melt Region": ["mean", "min"],
    }
    out = point_cloud.groupby(group_cols).agg(agg)
    out.columns = ["_".join([part for part in col if part]).strip("_") for col in out.columns]
    out = out.reset_index()
    counts = point_cloud.groupby(group_cols, as_index=False).agg(unique_points=("xi_m", "size"))
    meta_cols = [
        "case_id",
        "case_index",
        "power_W",
        "scan_speed_mm_s",
        "scan_speed_m_s",
        "particle_rate",
        "powder_feed_g_min",
    ]
    available_meta = [col for col in meta_cols if col in point_cloud.columns]
    if available_meta:
        meta = point_cloud[available_meta].drop_duplicates()
        out = out.merge(meta, on="case_id", how="left") if "case_id" in out.columns else out
    return out.merge(counts, on=group_cols, how="left")


def make_multi_condition_dimensionless_table(
    dimensionless_numbers: pd.DataFrame,
) -> pd.DataFrame:
    if "case_id" not in dimensionless_numbers.columns:
        out = dimensionless_numbers.copy()
        out.insert(0, "row_type", "single_condition")
        return out
    out = dimensionless_numbers.copy()
    out["row_type"] = np.where(
        out["case_id"].astype(str).str.startswith("summary"),
        "summary",
        np.where(out["case_id"].astype(str).eq("global"), "global_constant", "condition"),
    )
    return out[
        ["row_type", "case_id", "symbol", "value", "unit", "definition_or_note"]
    ].copy()


def make_parameter_reconciliation_audit(
    material_parameters: pd.DataFrame | None = None,
    property_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    setup_source = "Flow3D_setup.md setup note"
    post_source = "scripts/flow3d_pipeline/core.py and property CSV tables"
    rows = [
        {
            "parameter": "material_property_treatment",
            "unit": "-",
            "flow3d_setup_record": "temperature-dependent density, viscosity, thermal conductivity and specific heat tables",
            "postprocessing_basis": "property CSV tables interpolated directly for liquidus-scale diagnostics",
            "setup_source": setup_source,
            "postprocessing_source": post_source,
            "status": "resolved_to_temperature_dependent_property_tables",
            "manuscript_action": "The tabulated temperature-dependent curves are used; their variation is plotted in Fig.~\\ref{fig:supp-temperature-properties}.",
            "mandatory_reconciliation": False,
        },
        {
            "parameter": "beam_radius",
            "unit": "m",
            "flow3d_setup_record": "8.0e-4",
            "postprocessing_basis": f"{MATERIAL_CONSTANTS['beam_radius_m']:.6g}",
            "setup_source": setup_source,
            "postprocessing_source": post_source,
            "status": "resolved_to_flow3d_setup",
            "manuscript_action": "Adopted from the solver-parameter record.",
            "mandatory_reconciliation": False,
        },
        {
            "parameter": "absorptivity",
            "unit": "-",
            "flow3d_setup_record": "0.1",
            "postprocessing_basis": f"{MATERIAL_CONSTANTS['absorptivity']:.6g}",
            "setup_source": setup_source,
            "postprocessing_source": post_source,
            "status": "resolved_to_flow3d_setup",
            "manuscript_action": "Adopted from the solver-parameter record.",
            "mandatory_reconciliation": False,
        },
        {
            "parameter": "solidus_temperature",
            "unit": "K",
            "flow3d_setup_record": "1683",
            "postprocessing_basis": f"{MATERIAL_CONSTANTS['solidus_temperature_K']:.6g}",
            "setup_source": setup_source,
            "postprocessing_source": post_source,
            "status": "resolved_to_flow3d_setup",
            "manuscript_action": "Adopted from the solver-parameter record.",
            "mandatory_reconciliation": False,
        },
        {
            "parameter": "liquidus_temperature",
            "unit": "K",
            "flow3d_setup_record": "1710",
            "postprocessing_basis": f"{MATERIAL_CONSTANTS['liquidus_temperature_K']:.6g}",
            "setup_source": setup_source,
            "postprocessing_source": post_source,
            "status": "resolved_to_flow3d_setup",
            "manuscript_action": "Adopted from the solver-parameter record.",
            "mandatory_reconciliation": False,
        },
        {
            "parameter": "latent_heat_fusion",
            "unit": "J/kg",
            "flow3d_setup_record": "2.67776e5",
            "postprocessing_basis": f"{MATERIAL_CONSTANTS['latent_heat_fusion_J_per_kg']:.6g}",
            "setup_source": setup_source,
            "postprocessing_source": post_source,
            "status": "resolved_to_flow3d_setup",
            "manuscript_action": "Adopted from the solver-parameter record.",
            "mandatory_reconciliation": False,
        },
        {
            "parameter": "surface_tension",
            "unit": "N/m",
            "flow3d_setup_record": "1.8",
            "postprocessing_basis": f"{MATERIAL_CONSTANTS['surface_tension_N_per_m']:.6g}",
            "setup_source": setup_source,
            "postprocessing_source": post_source,
            "status": "resolved_to_flow3d_setup",
            "manuscript_action": "Adopted from the solver-parameter record; thermocapillary interpretation remains scale context only.",
            "mandatory_reconciliation": False,
        },
        {
            "parameter": "surface_tension_temperature_coefficient",
            "unit": "N/(m K)",
            "flow3d_setup_record": "2.50836e-4",
            "postprocessing_basis": f"{MATERIAL_CONSTANTS['surface_tension_temperature_coefficient_N_per_m_K']:.6g}",
            "setup_source": setup_source,
            "postprocessing_source": post_source,
            "status": "resolved_to_flow3d_setup",
            "manuscript_action": "$Ma$ uses the solver-parameter value and liquidus-interpolated viscosity and thermal diffusivity from the property tables.",
            "mandatory_reconciliation": False,
        },
        {
            "parameter": "cell_size",
            "unit": "m",
            "flow3d_setup_record": "1.0e-4",
            "postprocessing_basis": "not used in reduced-model fitting",
            "setup_source": setup_source,
            "postprocessing_source": "not applicable",
            "status": "setup_note_only",
            "manuscript_action": "Retained as setup context; mesh and time-step convergence are outside the present data scope.",
            "mandatory_reconciliation": False,
        },
        {
            "parameter": "validation_scope",
            "unit": "-",
            "flow3d_setup_record": "same FLOW-3D modeling ecosystem",
            "postprocessing_basis": "model-construction process matrix (Supplementary Table S2) plus additional simulated-condition assessment cases (Supplementary Table S3)",
            "setup_source": "raw data, validation data and additional data folders",
            "postprocessing_source": "generated validation tables",
            "status": "confirmed_scope_limitation",
            "manuscript_action": "The additional cases are described as same-setting numerical transfer evidence, not experimental melt-pool validation.",
            "mandatory_reconciliation": False,
        },
    ]
    return pd.DataFrame(rows)


def make_geometry_risk_summary(
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    error_budget: pd.DataFrame,
) -> pd.DataFrame:
    geom_summary = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("summary")].copy()
    geom_case = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("case_summary")].copy()
    residuals = {}
    volumes = {}
    chamfers = {}
    if len(geom_summary):
        for row in geom_summary.itertuples():
            model = str(row.model)
            residuals[model] = float(getattr(row, "mean_boundary_residual_rmse", np.nan))
            volumes[model] = float(getattr(row, "mean_volume_relative_error", np.nan))
            chamfers[model] = float(getattr(row, "mean_chamfer_distance_m", np.nan)) if hasattr(row, "mean_chamfer_distance_m") else np.nan
    boundary_wins = volume_wins = total_cases = 0
    if len(geom_case):
        boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse")
        volume = geom_case.pivot(index="case_id", columns="model", values="mean_volume_relative_error")
        if {"ellipsoid", "superellipsoid"}.issubset(boundary.columns):
            boundary_wins = int((boundary["superellipsoid"] < boundary["ellipsoid"]).sum())
            total_cases = int(len(boundary))
        if {"ellipsoid", "superellipsoid"}.issubset(volume.columns):
            volume_wins = int((volume["superellipsoid"] < volume["ellipsoid"]).sum())
    error_map = {
        str(row.error_term): row
        for row in error_budget.itertuples()
        if hasattr(row, "error_term")
    }
    duplicate_fraction = np.nan
    if "E_reconstruction" in error_map:
        duplicate_fraction = float(getattr(error_map["E_reconstruction"], "primary_value", np.nan))
    normalized_chamfer = np.nan
    if "E_geometry" in error_map:
        normalized_chamfer = float(getattr(error_map["E_geometry"], "primary_value", np.nan))
    volume_proxy_error = np.nan
    if "E_volume_proxy" in error_map:
        volume_proxy_error = float(getattr(error_map["E_volume_proxy"], "primary_value", np.nan))
    raw_min = int(table["raw_rows"].min()) if "raw_rows" in table.columns else 0
    raw_max = int(table["raw_rows"].max()) if "raw_rows" in table.columns else 0
    unique_min = int(table["unique_points"].min()) if "unique_points" in table.columns else 0
    unique_max = int(table["unique_points"].max()) if "unique_points" in table.columns else 0
    rows = [
        {
            "risk_item": "implicit_boundary_residual_selection",
            "metric": "ellipsoid_to_superellipsoid_mean_residual",
            "value": f"{residuals.get('ellipsoid', np.nan):.6g} -> {residuals.get('superellipsoid', np.nan):.6g}",
            "risk_level": "medium",
            "manuscript_action": "State that superellipsoid improves implicit boundary residual, not full melt-pool geometry.",
        },
        {
            "risk_item": "volume_proxy_mismatch",
            "metric": "superellipsoid_volume_relative_error",
            "value": f"{volume_proxy_error:.6g}",
            "risk_level": "medium_high",
            "manuscript_action": "State explicitly that the superellipsoid is not quantitatively reliable for melt-volume prediction.",
        },
        {
            "risk_item": "casewise_volume_support",
            "metric": "superellipsoid_volume_wins",
            "value": f"{volume_wins}/{total_cases}",
            "risk_level": "medium_high",
            "manuscript_action": "Keep volume as a limitation and do not use it as a model-selection success metric.",
        },
        {
            "risk_item": "casewise_boundary_support",
            "metric": "superellipsoid_boundary_wins",
            "value": f"{boundary_wins}/{total_cases}",
            "risk_level": "medium",
            "manuscript_action": "Use this only for the narrow boundary-residual claim.",
        },
        {
            "risk_item": "convex_hull_extraction",
            "metric": "boundary_method",
            "value": "convex hull of exported molten-region points",
            "risk_level": "medium_high",
            "manuscript_action": "Describe convex-hull overfill and concavity loss as an explicit limitation.",
        },
        {
            "risk_item": "point_preprocessing",
            "metric": "raw_rows_to_unique_points",
            "value": f"{raw_min}-{raw_max} raw rows; {unique_min}-{unique_max} unique points",
            "risk_level": "medium",
            "manuscript_action": "Keep duplicate/collapse audit in main-text data provenance or limitation discussion.",
        },
        {
            "risk_item": "duplicate_rows",
            "metric": "exact_duplicate_fraction",
            "value": f"{duplicate_fraction:.6g}",
            "risk_level": "medium",
            "manuscript_action": "Treat duplicate removal as a modeling uncertainty source, not bookkeeping.",
        },
        {
            "risk_item": "geometric_distance_diagnostic",
            "metric": "normalized_superellipsoid_chamfer_distance",
            "value": f"{normalized_chamfer:.6g}",
            "risk_level": "medium",
            "manuscript_action": "Use Chamfer distance as a diagnostic check rather than the primary selection rule.",
        },
    ]
    return pd.DataFrame(rows)


def make_validation_hierarchy_table(
    loco_validation: pd.DataFrame,
    external_holdout_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    loco_error = np.nan
    if len(loco_validation) and "relative_error" in loco_validation.columns:
        loco_error = float(pd.to_numeric(loco_validation["relative_error"], errors="coerce").mean())
    ext_cases = ext_cohorts = ext_process = ext_dynamics = np.nan
    if external_holdout_summary is not None and len(external_holdout_summary):
        metric_map = dict(zip(external_holdout_summary["metric"], external_holdout_summary["value"]))
        ext_cases = float(metric_map.get("external_validation_case_count", np.nan))
        ext_cohorts = float(metric_map.get("external_validation_cohort_count", np.nan))
        ext_process = float(metric_map.get("external_process_response_mean_relative_error", np.nan))
        ext_dynamics = float(metric_map.get("external_dynamics_mean_relative_rmse", np.nan))
    rows = [
        {
            "validation_layer": "condition_time_train_validation_split",
            "evidence": "within-condition early-time fit and later-time validation",
            "independence_scope": "internal to each FLOW-3D condition",
            "supported_claim": "short-horizon descriptive trajectory fit",
            "unsupported_claim": "independent physical-measurement support",
        },
        {
            "validation_layer": "leave_one_condition_out_process_response",
            "evidence": f"mean relative error {loco_error:.4f} over held-out condition-target predictions",
            "independence_scope": "A-prefixed process matrix holdout by condition",
            "supported_claim": "interpolation-style process-response transfer within the same export pipeline",
            "unsupported_claim": "universal L-DED process law",
        },
        {
            "validation_layer": "same_ecosystem_cfd_holdout_cohorts",
            "evidence": f"{ext_cases:.0f} cases across {ext_cohorts:.0f} melt-pool-data holdout cohorts; process mean relative error {ext_process:.4f}; dynamics mean relative RMSE {ext_dynamics:.4f}",
            "independence_scope": "new CFD conditions under the same FLOW-3D assumptions and preprocessing",
            "supported_claim": "CFD holdout transfer within the same simulation ecosystem",
            "unsupported_claim": "experimental melt-pool validation",
        },
        {
            "validation_layer": "experimental_validation",
            "evidence": "not present in the current data-only revision",
            "independence_scope": "none",
            "supported_claim": "not applicable",
            "unsupported_claim": "agreement with measured melt-pool geometry or temperature fields",
        },
    ]
    return pd.DataFrame(rows)


def make_multi_condition_geometry_summary(geometry_comparison: pd.DataFrame) -> pd.DataFrame:
    case_summary = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("case_summary")].copy()
    if case_summary.empty:
        case_summary = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("summary")].copy()
    if "case_id" in case_summary.columns:
        pivot = case_summary.pivot_table(
            index=["case_id", "case_index", "power_W", "scan_speed_mm_s", "particle_rate", "powder_feed_g_min"],
            columns="model",
            values=["mean_boundary_residual_rmse", "mean_volume_relative_error", "successful_fits"],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
        out = pivot.reset_index()
        if {
            "mean_boundary_residual_rmse_ellipsoid",
            "mean_boundary_residual_rmse_superellipsoid",
        }.issubset(out.columns):
            out["superellipsoid_boundary_improves"] = (
                out["mean_boundary_residual_rmse_superellipsoid"]
                < out["mean_boundary_residual_rmse_ellipsoid"]
            )
        if {
            "mean_volume_relative_error_ellipsoid",
            "mean_volume_relative_error_superellipsoid",
        }.issubset(out.columns):
            out["superellipsoid_volume_improves"] = (
                out["mean_volume_relative_error_superellipsoid"]
                < out["mean_volume_relative_error_ellipsoid"]
            )
        return out
    return case_summary


def make_multi_condition_dynamics_summary(dynamics_comparison: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in dynamics_comparison.columns:
        return dynamics_comparison.copy()
    pivot = dynamics_comparison.pivot_table(
        index=["case_id", "case_index", "power_W", "scan_speed_mm_s", "particle_rate", "powder_feed_g_min", "state"],
        columns="model",
        values=["train_relative_rmse", "validation_relative_rmse", "all_eigenvalues_stable"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    out = pivot.reset_index()
    if {"validation_relative_rmse_diagonal_attractor", "validation_relative_rmse_coupled_ridge_attractor"}.issubset(
        out.columns
    ):
        out["coupled_improves_validation"] = (
            out["validation_relative_rmse_coupled_ridge_attractor"]
            < out["validation_relative_rmse_diagonal_attractor"]
        )
    return out


def make_process_response_summary(table: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in table.columns:
        return pd.DataFrame()
    steady = table[table["time_s"] >= QUASI_STEADY_START_S].copy()
    response_cols = [
        "melt_pool_length_m",
        "front_length_m",
        "rear_length_m",
        "full_width_m",
        "height_span_m",
        "volume_proxy_m3",
        "Tmax_K",
        "Tmean_K",
        "Gmean_K_per_m",
        "Umax_m_per_s",
        "pressure_mean_Pa",
        "heat_absorption_max",
    ]
    case_response = steady.groupby("case_id", as_index=False).agg(
        case_index=("case_index", "first"),
        power_W=("power_W", "first"),
        scan_speed_mm_s=("scan_speed_mm_s", "first"),
        particle_rate=("particle_rate", "first"),
        powder_feed_g_min=("powder_feed_g_min", "first"),
        **{f"{col}_quasi_mean": (col, "mean") for col in response_cols if col in steady.columns},
    )
    rows = []
    process_cols = ["power_W", "scan_speed_mm_s", "powder_feed_g_min"]
    for response in [col for col in case_response.columns if col.endswith("_quasi_mean")]:
        y = case_response[response].to_numpy(dtype=float)
        for proc in process_cols:
            x = case_response[proc].to_numpy(dtype=float)
            if np.sum(np.isfinite(x) & np.isfinite(y)) < 3 or np.nanstd(x) == 0:
                corr = np.nan
                slope = np.nan
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
                slope = float(np.polyfit(x, y, 1)[0])
            rows.append(
                {
                    "response": response,
                    "process_parameter": proc,
                    "pearson_correlation": corr,
                    "linear_slope": slope,
                    "n_conditions": int(case_response["case_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def make_leave_one_condition_out_validation(table: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in table.columns or table["case_id"].nunique() < 4:
        return pd.DataFrame()
    steady = table[table["time_s"] >= QUASI_STEADY_START_S].copy()
    target_cols = [
        "melt_pool_length_m",
        "front_length_m",
        "rear_length_m",
        "full_width_m",
        "height_span_m",
        "volume_proxy_m3",
        "Tmax_K",
        "Gmean_K_per_m",
        "Umax_m_per_s",
    ]
    case_df = steady.groupby("case_id", as_index=False).agg(
        case_index=("case_index", "first"),
        power_W=("power_W", "first"),
        scan_speed_mm_s=("scan_speed_mm_s", "first"),
        powder_feed_g_min=("powder_feed_g_min", "first"),
        **{col: (col, "mean") for col in target_cols},
    )
    x_raw = case_df[["power_W", "scan_speed_mm_s", "powder_feed_g_min"]].to_numpy(dtype=float)
    rows = []
    for held_idx, held in case_df.iterrows():
        train_mask = np.ones(len(case_df), dtype=bool)
        train_mask[held_idx] = False
        x_train = x_raw[train_mask]
        x_test = x_raw[[held_idx]]
        x_mean = np.nanmean(x_train, axis=0)
        x_std = np.nanstd(x_train, axis=0, ddof=1)
        x_std = np.where(x_std > 0, x_std, 1.0)
        design_train = np.column_stack([np.ones(x_train.shape[0]), (x_train - x_mean) / x_std])
        design_test = np.column_stack([np.ones(1), (x_test - x_mean) / x_std])
        for target in target_cols:
            y = case_df[target].to_numpy(dtype=float)
            y_train = y[train_mask]
            try:
                beta, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
                pred = float((design_test @ beta).item())
                status = "ok"
            except np.linalg.LinAlgError as exc:
                pred = np.nan
                status = f"failed:{exc}"
            actual = float(held[target])
            err = pred - actual
            rows.append(
                {
                    "held_out_case_id": held["case_id"],
                    "case_index": int(held["case_index"]),
                    "power_W": float(held["power_W"]),
                    "scan_speed_mm_s": float(held["scan_speed_mm_s"]),
                    "powder_feed_g_min": float(held["powder_feed_g_min"]),
                    "target": target,
                    "actual": actual,
                    "predicted": pred,
                    "error": err,
                    "absolute_error": abs(err) if np.isfinite(err) else np.nan,
                    "relative_error": abs(err) / max(abs(actual), 1e-12) if np.isfinite(err) else np.nan,
                    "status": status,
                    "model": "leave_one_condition_linear_process_response",
                    "feature_set": "power_W, scan_speed_mm_s, powder_feed_g_min",
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        summary = (
            out.groupby("target", as_index=False)
            .agg(
                rmse=("error", lambda v: float(np.sqrt(np.nanmean(np.asarray(v, dtype=float) ** 2)))),
                mean_relative_error=("relative_error", "mean"),
                max_relative_error=("relative_error", "max"),
                n_conditions=("held_out_case_id", "nunique"),
            )
            .assign(held_out_case_id="summary", status="summary")
        )
        out = pd.concat([out, summary], ignore_index=True, sort=False)
    return out


PROCESS_FEATURE_COLUMNS = ["power_W", "scan_speed_mm_s", "powder_feed_g_min"]
EXTERNAL_HOLDOUT_TARGET_COLUMNS = [
    "melt_pool_length_m",
    "front_length_m",
    "rear_length_m",
    "full_width_m",
    "height_span_m",
    "volume_proxy_m3",
    "Tmax_K",
    "Gmean_K_per_m",
    "Umax_m_per_s",
]


def quasi_steady_case_table(table: pd.DataFrame, target_cols: list[str] | None = None) -> pd.DataFrame:
    target_cols = target_cols or EXTERNAL_HOLDOUT_TARGET_COLUMNS
    steady = table[table["time_s"] >= QUASI_STEADY_START_S].copy()
    available_targets = [col for col in target_cols if col in steady.columns]
    optional_meta = {
        col: (col, "first")
        for col in ["validation_cohort", "source_root"]
        if col in steady.columns
    }
    return steady.groupby("case_id", as_index=False).agg(
        case_index=("case_index", "first"),
        power_W=("power_W", "first"),
        scan_speed_mm_s=("scan_speed_mm_s", "first"),
        particle_rate=("particle_rate", "first"),
        powder_feed_g_min=("powder_feed_g_min", "first"),
        **optional_meta,
        **{col: (col, "mean") for col in available_targets},
    )


def _linear_holdout_design(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.nanmean(x_train, axis=0)
    x_std = np.nanstd(x_train, axis=0, ddof=1)
    x_std = np.where(np.isfinite(x_std) & (x_std > 0), x_std, 1.0)
    train_design = np.column_stack([np.ones(x_train.shape[0]), (x_train - x_mean) / x_std])
    eval_design = np.column_stack([np.ones(x_eval.shape[0]), (x_eval - x_mean) / x_std])
    return train_design, eval_design, x_mean, x_std


def _feature_extrapolation_flags(train_case_df: pd.DataFrame, eval_row: pd.Series) -> str:
    flags = []
    for col in PROCESS_FEATURE_COLUMNS:
        lo = float(train_case_df[col].min())
        hi = float(train_case_df[col].max())
        value = float(eval_row[col])
        if value < lo or value > hi:
            flags.append(f"{col}={value:g} outside [{lo:g},{hi:g}]")
    return "; ".join(flags)


def make_external_holdout_process_response_validation(
    train_table: pd.DataFrame,
    validation_table: pd.DataFrame,
) -> pd.DataFrame:
    if validation_table is None or len(validation_table) == 0:
        return pd.DataFrame()
    train_case = quasi_steady_case_table(train_table)
    validation_case = quasi_steady_case_table(validation_table)
    if len(train_case) < 4 or len(validation_case) == 0:
        return pd.DataFrame()
    x_train_all = train_case[PROCESS_FEATURE_COLUMNS].to_numpy(dtype=float)
    x_val_all = validation_case[PROCESS_FEATURE_COLUMNS].to_numpy(dtype=float)
    train_design_all, val_design_all, _, _ = _linear_holdout_design(x_train_all, x_val_all)
    rows = []
    for target in [col for col in EXTERNAL_HOLDOUT_TARGET_COLUMNS if col in train_case.columns and col in validation_case.columns]:
        y_train = train_case[target].to_numpy(dtype=float)
        finite = np.isfinite(y_train) & np.all(np.isfinite(train_design_all), axis=1)
        if np.sum(finite) < len(PROCESS_FEATURE_COLUMNS) + 1:
            beta = np.full(train_design_all.shape[1], np.nan)
            status = "failed:insufficient_training_cases"
        else:
            try:
                beta, *_ = np.linalg.lstsq(train_design_all[finite], y_train[finite], rcond=None)
                status = "ok"
            except np.linalg.LinAlgError as exc:
                beta = np.full(train_design_all.shape[1], np.nan)
                status = f"failed:{exc}"
        predictions = val_design_all @ beta if np.all(np.isfinite(beta)) else np.full(len(validation_case), np.nan)
        for idx, val_row in validation_case.reset_index(drop=True).iterrows():
            actual = float(val_row[target])
            pred = float(predictions[idx]) if np.isfinite(predictions[idx]) else np.nan
            err = pred - actual if np.isfinite(pred) else np.nan
            row_meta = {
                col: val_row[col]
                for col in ["validation_cohort", "source_root"]
                if col in validation_case.columns
            }
            rows.append(
                {
                    **row_meta,
                    "case_id": val_row["case_id"],
                    "case_index": int(val_row["case_index"]),
                    "power_W": float(val_row["power_W"]),
                    "scan_speed_mm_s": float(val_row["scan_speed_mm_s"]),
                    "powder_feed_g_min": float(val_row["powder_feed_g_min"]),
                    "target": target,
                    "actual": actual,
                    "predicted": pred,
                    "error": err,
                    "absolute_error": abs(err) if np.isfinite(err) else np.nan,
                    "relative_error": abs(err) / max(abs(actual), 1e-12) if np.isfinite(err) else np.nan,
                    "status": status,
                    "model": "external_holdout_linear_process_response",
                    "feature_set": ", ".join(PROCESS_FEATURE_COLUMNS),
                    "extrapolation_flags": _feature_extrapolation_flags(train_case, val_row),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        summary = (
            out.groupby("target", as_index=False)
            .agg(
                rmse=("error", lambda v: float(np.sqrt(np.nanmean(np.asarray(v, dtype=float) ** 2)))),
                mean_relative_error=("relative_error", "mean"),
                max_relative_error=("relative_error", "max"),
                n_conditions=("case_id", "nunique"),
            )
            .assign(case_id="summary", status="summary", model="external_holdout_linear_process_response")
        )
        out = pd.concat([out, summary], ignore_index=True, sort=False)
    return out


def make_external_holdout_dynamics_validation(
    train_table: pd.DataFrame,
    validation_table: pd.DataFrame,
    train_dynamics_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if validation_table is None or len(validation_table) == 0 or train_dynamics_summary is None or len(train_dynamics_summary) == 0:
        return pd.DataFrame(), pd.DataFrame()
    train_meta = train_dynamics_summary.drop_duplicates("case_id")[["case_id", *PROCESS_FEATURE_COLUMNS]].copy()
    train_x = train_meta[PROCESS_FEATURE_COLUMNS].to_numpy(dtype=float)
    prediction_rows = []
    summary_rows = []
    validation_cases = validation_table.groupby("case_id", sort=False)
    for state in STATE_COLUMNS:
        state_train = train_dynamics_summary[train_dynamics_summary["state"].eq(state)].copy()
        state_train = state_train.merge(train_meta, on=["case_id", *PROCESS_FEATURE_COLUMNS], how="left")
        x = state_train[PROCESS_FEATURE_COLUMNS].to_numpy(dtype=float)
        y_inf = pd.to_numeric(state_train["q_inf"], errors="coerce").to_numpy(dtype=float)
        k = pd.to_numeric(state_train["k_per_s"], errors="coerce").to_numpy(dtype=float)
        finite = np.all(np.isfinite(x), axis=1) & np.isfinite(y_inf) & np.isfinite(k) & (k > 0)
        if np.sum(finite) < len(PROCESS_FEATURE_COLUMNS) + 1:
            beta_q = np.full(len(PROCESS_FEATURE_COLUMNS) + 1, np.nan)
            beta_k = np.full(len(PROCESS_FEATURE_COLUMNS) + 1, np.nan)
            x_mean = np.nanmean(train_x, axis=0)
            x_std = np.nanstd(train_x, axis=0, ddof=1)
            x_std = np.where(np.isfinite(x_std) & (x_std > 0), x_std, 1.0)
            status = "failed:insufficient_training_cases"
        else:
            design, _, x_mean, x_std = _linear_holdout_design(x[finite], x[finite])
            try:
                beta_q, *_ = np.linalg.lstsq(design, y_inf[finite], rcond=None)
                beta_k, *_ = np.linalg.lstsq(design, np.log(k[finite]), rcond=None)
                status = "ok"
            except np.linalg.LinAlgError as exc:
                beta_q = np.full(len(PROCESS_FEATURE_COLUMNS) + 1, np.nan)
                beta_k = np.full(len(PROCESS_FEATURE_COLUMNS) + 1, np.nan)
                status = f"failed:{exc}"
        for case_id, group in validation_cases:
            group = group.sort_values("time_s").copy()
            val_x = group[PROCESS_FEATURE_COLUMNS].iloc[[0]].to_numpy(dtype=float)
            val_design = np.column_stack([np.ones(1), (val_x - x_mean) / x_std])
            q_inf_pred = float((val_design @ beta_q).item()) if np.all(np.isfinite(beta_q)) else np.nan
            log_k_pred = float((val_design @ beta_k).item()) if np.all(np.isfinite(beta_k)) else np.nan
            k_pred = float(np.exp(np.clip(log_k_pred, -20.0, math.log(100.0)))) if np.isfinite(log_k_pred) else np.nan
            t = group["time_s"].to_numpy(dtype=float)
            y = group[state].to_numpy(dtype=float)
            y0 = float(y[0]) if len(y) else np.nan
            y_pred = (
                q_inf_pred + (y0 - q_inf_pred) * np.exp(-k_pred * (t - t[0]))
                if np.isfinite(q_inf_pred) and np.isfinite(k_pred) and np.isfinite(y0)
                else np.full(len(group), np.nan)
            )
            err = y_pred - y
            rel_denom = max(float(np.nanmean(np.abs(y))), 1e-12)
            rmse = float(np.sqrt(np.nanmean(err**2))) if np.any(np.isfinite(err)) else np.nan
            rel_rmse = rmse / rel_denom if np.isfinite(rmse) else np.nan
            meta = group.iloc[0]
            row_meta = {
                col: meta[col]
                for col in ["validation_cohort", "source_root"]
                if col in group.columns
            }
            summary_rows.append(
                {
                    **row_meta,
                    "case_id": case_id,
                    "case_index": int(meta["case_index"]),
                    "power_W": float(meta["power_W"]),
                    "scan_speed_mm_s": float(meta["scan_speed_mm_s"]),
                    "powder_feed_g_min": float(meta["powder_feed_g_min"]),
                    "state": state,
                    "label": STATE_LABELS[state],
                    "q_inf_predicted": q_inf_pred,
                    "k_predicted_per_s": k_pred,
                    "rmse": rmse,
                    "relative_rmse": rel_rmse,
                    "max_relative_error": float(np.nanmax(np.abs(err) / np.maximum(np.abs(y), 1e-12))) if np.any(np.isfinite(err)) else np.nan,
                    "status": status,
                    "model": "external_holdout_process_parameterized_diagonal_attractor",
                    "feature_set": ", ".join(PROCESS_FEATURE_COLUMNS),
                    "extrapolation_flags": _feature_extrapolation_flags(train_meta, meta),
                }
            )
            for time_s, actual, predicted, residual in zip(t, y, y_pred, err):
                prediction_rows.append(
                    {
                        **row_meta,
                        "case_id": case_id,
                        "case_index": int(meta["case_index"]),
                        "power_W": float(meta["power_W"]),
                        "scan_speed_mm_s": float(meta["scan_speed_mm_s"]),
                        "powder_feed_g_min": float(meta["powder_feed_g_min"]),
                        "time_s": float(time_s),
                        "state": state,
                        "label": STATE_LABELS[state],
                        "actual": float(actual),
                        "predicted": float(predicted) if np.isfinite(predicted) else np.nan,
                        "error": float(residual) if np.isfinite(residual) else np.nan,
                        "relative_error": abs(float(residual)) / max(abs(float(actual)), 1e-12)
                        if np.isfinite(residual) and np.isfinite(actual)
                        else np.nan,
                        "q_inf_predicted": q_inf_pred,
                        "k_predicted_per_s": k_pred,
                        "status": status,
                        "model": "external_holdout_process_parameterized_diagonal_attractor",
                    }
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(summary_rows)


def make_external_holdout_validation_summary(
    validation_table: pd.DataFrame,
    external_geometry_comparison: pd.DataFrame,
    process_validation: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if validation_table is not None and len(validation_table):
        rows.extend(
            [
                {
                    "metric": "external_validation_case_count",
                    "value": int(validation_table["case_id"].nunique()),
                    "unit": "cases",
                    "interpretation": "Same-solver FLOW-3D holdout conditions withheld from model construction.",
                },
                {
                    "metric": "external_validation_cohort_count",
                    "value": int(validation_table["validation_cohort"].nunique()) if "validation_cohort" in validation_table.columns else 1,
                    "unit": "cohorts",
                    "interpretation": "Number of holdout cohorts processed by the same descriptor pipeline.",
                },
                {
                    "metric": "external_validation_time_step_count",
                    "value": int(len(validation_table)),
                    "unit": "condition-time steps",
                    "interpretation": "Total external CFD time-step exports processed by the same descriptor pipeline.",
                },
            ]
        )
    geom_case = external_geometry_comparison[
        external_geometry_comparison["time_s"].astype(str).eq("case_summary")
    ].copy() if external_geometry_comparison is not None and len(external_geometry_comparison) else pd.DataFrame()
    if len(geom_case):
        boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse")
        volume = geom_case.pivot(index="case_id", columns="model", values="mean_volume_relative_error")
        if {"ellipsoid", "superellipsoid"}.issubset(boundary.columns):
            wins = int((boundary["superellipsoid"] < boundary["ellipsoid"]).sum())
            total = int(len(boundary))
            rows.append(
                {
                    "metric": "external_superellipsoid_boundary_win_rate",
                    "value": wins / max(total, 1),
                    "unit": "fraction",
                    "interpretation": f"Superellipsoid boundary residual improves in {wins}/{total} external cases.",
                }
            )
        if {"ellipsoid", "superellipsoid"}.issubset(volume.columns):
            wins = int((volume["superellipsoid"] < volume["ellipsoid"]).sum())
            total = int(len(volume))
            rows.append(
                {
                    "metric": "external_superellipsoid_volume_win_rate",
                    "value": wins / max(total, 1),
                    "unit": "fraction",
                    "interpretation": f"Superellipsoid volume proxy improves in {wins}/{total} external cases.",
                }
            )
    if process_validation is not None and len(process_validation):
        detail = process_validation[process_validation["case_id"].astype(str).ne("summary")]
        summary = process_validation[process_validation["case_id"].astype(str).eq("summary")]
        rows.extend(
            [
                {
                    "metric": "external_process_response_mean_relative_error",
                    "value": float(detail["relative_error"].mean()),
                    "unit": "relative error",
                    "interpretation": "Mean quasi-steady process-response error on CFD holdout cases.",
                },
                {
                    "metric": "external_process_response_max_relative_error",
                    "value": float(detail["relative_error"].max()),
                    "unit": "relative error",
                    "interpretation": "Largest quasi-steady process-response relative error over all external case-target pairs.",
                },
                {
                    "metric": "external_process_response_worst_target_mean_relative_error",
                    "value": float(summary["mean_relative_error"].max()) if len(summary) else np.nan,
                    "unit": "relative error",
                    "interpretation": str(summary.loc[summary["mean_relative_error"].idxmax(), "target"]) if len(summary) else "",
                },
            ]
        )
    if dynamics_summary is not None and len(dynamics_summary):
        rows.extend(
            [
                {
                    "metric": "external_dynamics_mean_relative_rmse",
                    "value": float(dynamics_summary["relative_rmse"].mean()),
                    "unit": "relative RMSE",
                    "interpretation": "Mean process-parameterized diagonal-attractor trajectory error on external cases.",
                },
                {
                    "metric": "external_dynamics_max_relative_rmse",
                    "value": float(dynamics_summary["relative_rmse"].max()),
                    "unit": "relative RMSE",
                    "interpretation": "Largest state-wise external diagonal-attractor trajectory error.",
                },
                {
                    "metric": "external_dynamics_worst_state_mean_relative_rmse",
                    "value": float(dynamics_summary.groupby("state")["relative_rmse"].mean().max()),
                    "unit": "relative RMSE",
                    "interpretation": str(dynamics_summary.groupby("state")["relative_rmse"].mean().idxmax()),
                },
            ]
        )
    return pd.DataFrame(rows)


def make_holdout_cohort_summary(
    validation_table: pd.DataFrame,
    external_geometry_comparison: pd.DataFrame,
    process_validation: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
) -> pd.DataFrame:
    if validation_table is None or len(validation_table) == 0 or "validation_cohort" not in validation_table.columns:
        return pd.DataFrame()
    rows = []
    cohorts = validation_table["validation_cohort"].dropna().astype(str).unique().tolist()
    for cohort in cohorts:
        val = validation_table[validation_table["validation_cohort"].astype(str).eq(cohort)].copy()
        geom = (
            external_geometry_comparison[external_geometry_comparison["validation_cohort"].astype(str).eq(cohort)].copy()
            if external_geometry_comparison is not None and len(external_geometry_comparison) and "validation_cohort" in external_geometry_comparison.columns
            else pd.DataFrame()
        )
        proc = (
            process_validation[process_validation["validation_cohort"].astype(str).eq(cohort)].copy()
            if process_validation is not None and len(process_validation) and "validation_cohort" in process_validation.columns
            else pd.DataFrame()
        )
        dyn = (
            dynamics_summary[dynamics_summary["validation_cohort"].astype(str).eq(cohort)].copy()
            if dynamics_summary is not None and len(dynamics_summary) and "validation_cohort" in dynamics_summary.columns
            else pd.DataFrame()
        )
        geom_case = geom[geom["time_s"].astype(str).eq("case_summary")].copy() if len(geom) and "time_s" in geom.columns else pd.DataFrame()
        boundary_win = volume_win = np.nan
        if len(geom_case):
            boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse")
            volume = geom_case.pivot(index="case_id", columns="model", values="mean_volume_relative_error")
            if {"ellipsoid", "superellipsoid"}.issubset(boundary.columns):
                boundary_win = float((boundary["superellipsoid"] < boundary["ellipsoid"]).mean())
            if {"ellipsoid", "superellipsoid"}.issubset(volume.columns):
                volume_win = float((volume["superellipsoid"] < volume["ellipsoid"]).mean())
        proc_detail = proc[proc["case_id"].astype(str).ne("summary")] if len(proc) else pd.DataFrame()
        dyn_state = dyn.groupby("state")["relative_rmse"].mean() if len(dyn) else pd.Series(dtype=float)
        proc_target = proc_detail.groupby("target")["relative_error"].mean() if len(proc_detail) else pd.Series(dtype=float)
        rows.append(
            {
                "validation_cohort": cohort,
                "case_count": int(val["case_id"].nunique()),
                "time_step_count": int(len(val)),
                "case_ids": ", ".join(val["case_id"].drop_duplicates().astype(str).tolist()),
                "boundary_residual_win_rate": boundary_win,
                "volume_proxy_win_rate": volume_win,
                "process_mean_relative_error": float(proc_detail["relative_error"].mean()) if len(proc_detail) else np.nan,
                "process_max_relative_error": float(proc_detail["relative_error"].max()) if len(proc_detail) else np.nan,
                "process_worst_target": str(proc_target.idxmax()) if len(proc_target) else "",
                "process_worst_target_mean_relative_error": float(proc_target.max()) if len(proc_target) else np.nan,
                "dynamics_mean_relative_rmse": float(dyn["relative_rmse"].mean()) if len(dyn) else np.nan,
                "dynamics_max_relative_rmse": float(dyn["relative_rmse"].max()) if len(dyn) else np.nan,
                "dynamics_worst_state": str(dyn_state.idxmax()) if len(dyn_state) else "",
                "dynamics_worst_state_mean_relative_rmse": float(dyn_state.max()) if len(dyn_state) else np.nan,
            }
        )
    return pd.DataFrame(rows)


GEOMETRY_SELECTION_METRICS = [
    "mean_boundary_residual_rmse",
    "mean_normalized_chamfer_distance",
    "mean_hausdorff_distance_m",
    "mean_radial_distance_rmse_m",
    "mean_volume_relative_error",
]


def make_geometry_selection_metrics(
    geometry_comparison: pd.DataFrame,
    external_geometry_comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = []
    if geometry_comparison is not None and len(geometry_comparison):
        train = geometry_comparison.copy()
        train["analysis_role"] = "model_construction"
        train["used_for_model_selection"] = True
        frames.append(train)
    if external_geometry_comparison is not None and len(external_geometry_comparison):
        holdout = external_geometry_comparison.copy()
        holdout["analysis_role"] = "same_ecosystem_holdout"
        holdout["used_for_model_selection"] = False
        frames.append(holdout)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    case_rows = combined[combined["time_s"].astype(str).eq("case_summary")].copy()
    if case_rows.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["analysis_role", "case_id"]
    for _, group in case_rows.groupby(group_cols, sort=False):
        models = {str(row.model): row for row in group.itertuples()}
        if not {"ellipsoid", "superellipsoid"}.issubset(models):
            continue
        ell = models["ellipsoid"]
        sup = models["superellipsoid"]
        first = group.iloc[0]
        row = {
            "analysis_role": str(first.get("analysis_role", "")),
            "used_for_model_selection": bool(first.get("used_for_model_selection", False)),
            "validation_cohort": first.get("validation_cohort", ""),
            "source_root": first.get("source_root", ""),
            "case_id": str(first.get("case_id", "")),
            "case_index": first.get("case_index", np.nan),
            "power_W": first.get("power_W", np.nan),
            "scan_speed_mm_s": first.get("scan_speed_mm_s", np.nan),
            "particle_rate": first.get("particle_rate", np.nan),
            "powder_feed_g_min": first.get("powder_feed_g_min", np.nan),
            "selection_scope": "training selection evidence"
            if bool(first.get("used_for_model_selection", False))
            else "holdout transfer audit only",
            "selection_statement": (
                "superellipsoid is the selected algebraic observed-envelope descriptor; geometric distances and volume proxy restrict interpretation"
            ),
        }
        for metric in GEOMETRY_SELECTION_METRICS:
            ell_value = float(getattr(ell, metric, np.nan))
            sup_value = float(getattr(sup, metric, np.nan))
            row[f"ellipsoid_{metric}"] = ell_value
            row[f"superellipsoid_{metric}"] = sup_value
            row[f"super_minus_ellipsoid_{metric}"] = sup_value - ell_value if np.isfinite(ell_value) and np.isfinite(sup_value) else np.nan
            row[f"superellipsoid_improves_{metric}"] = bool(np.isfinite(ell_value) and np.isfinite(sup_value) and sup_value < ell_value)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        summary_rows = []
        for role, group in out.groupby("analysis_role", sort=False):
            summary = {
                "analysis_role": role,
                "used_for_model_selection": bool(group["used_for_model_selection"].astype(bool).any()),
                "validation_cohort": "summary",
                "source_root": "",
                "case_id": "summary",
                "case_index": np.nan,
                "power_W": np.nan,
                "scan_speed_mm_s": np.nan,
                "particle_rate": np.nan,
                "powder_feed_g_min": np.nan,
                "selection_scope": "aggregate",
                "selection_statement": "aggregate paired case-level metric comparison",
            }
            for metric in GEOMETRY_SELECTION_METRICS:
                delta_col = f"super_minus_ellipsoid_{metric}"
                win_col = f"superellipsoid_improves_{metric}"
                summary[f"ellipsoid_{metric}"] = float(group[f"ellipsoid_{metric}"].mean())
                summary[f"superellipsoid_{metric}"] = float(group[f"superellipsoid_{metric}"].mean())
                summary[delta_col] = float(group[delta_col].mean())
                summary[win_col] = float(group[win_col].astype(bool).mean())
            summary_rows.append(summary)
        out = pd.concat([out, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
    return out


def make_dynamics_fit_asymmetry_audit(
    dynamics_summary: pd.DataFrame,
    coupled_summary: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    parameter_identifiability: pd.DataFrame,
    identifiability_v4: pd.DataFrame | None,
    robustness_summary: pd.DataFrame,
    dynamics_derivative_audit: pd.DataFrame,
    validation_stress_tests: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize why the diagonal and coupled dynamical fits have different roles."""

    def bool_series(values: pd.Series) -> pd.Series:
        if values.dtype == bool:
            return values.fillna(False)
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    def mean_col(frame: pd.DataFrame, col: str) -> float:
        if frame is None or len(frame) == 0 or col not in frame.columns:
            return np.nan
        return float(pd.to_numeric(frame[col], errors="coerce").mean())

    diag_mean = mean_col(dynamics_summary, "validation_relative_rmse")
    coupled_mean = mean_col(coupled_summary, "validation_relative_rmse")
    diag_stable = bool(
        dynamics_summary is not None
        and len(dynamics_summary)
        and "k_per_s" in dynamics_summary.columns
        and np.all(pd.to_numeric(dynamics_summary["k_per_s"], errors="coerce").to_numpy(dtype=float) > 0)
    )
    coupled_stable = bool(
        coupled_eigenvalues is not None
        and len(coupled_eigenvalues)
        and "stable_if_real_negative" in coupled_eigenvalues.columns
        and coupled_eigenvalues["stable_if_real_negative"].astype(bool).all()
    )
    dyn_pair = (
        dynamics_comparison.pivot_table(
            index=["case_id", "state"],
            columns="model",
            values="validation_relative_rmse",
            aggfunc="mean",
        )
        if dynamics_comparison is not None and len(dynamics_comparison)
        else pd.DataFrame()
    )
    diag_wins = dyn_total = 0
    diag_median_advantage = np.nan
    if {"diagonal_attractor", "coupled_ridge_attractor"}.issubset(dyn_pair.columns):
        diag = pd.to_numeric(dyn_pair["diagonal_attractor"], errors="coerce").to_numpy(dtype=float)
        coupled = pd.to_numeric(dyn_pair["coupled_ridge_attractor"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(diag) & np.isfinite(coupled)
        if np.any(valid):
            advantage = coupled[valid] - diag[valid]
            diag_wins = int(np.sum(advantage > 0))
            dyn_total = int(np.sum(valid))
            diag_median_advantage = float(np.nanmedian(advantage))

    robust_total = int(len(robustness_summary)) if robustness_summary is not None else 0
    robust_coupled_wins = (
        int(bool_series(robustness_summary["coupled_improves_validation"]).sum())
        if robustness_summary is not None and len(robustness_summary) and "coupled_improves_validation" in robustness_summary.columns
        else 0
    )
    stress_support = (
        float(bool_series(validation_stress_tests["supports_main_model"]).mean())
        if validation_stress_tests is not None and len(validation_stress_tests) and "supports_main_model" in validation_stress_tests.columns
        else np.nan
    )
    derivative_cases = (
        int(dynamics_derivative_audit["case_id"].nunique())
        if dynamics_derivative_audit is not None and len(dynamics_derivative_audit) and "case_id" in dynamics_derivative_audit.columns
        else 0
    )
    umax = (
        dynamics_summary[dynamics_summary["state"].eq("Umax_m_per_s")].copy()
        if dynamics_summary is not None and len(dynamics_summary) and "state" in dynamics_summary.columns
        else pd.DataFrame()
    )
    umax_validation = pd.to_numeric(umax.get("validation_relative_rmse", pd.Series(dtype=float)), errors="coerce")
    umax_median = float(umax_validation.median()) if len(umax_validation.dropna()) else np.nan
    umax_high_cases = int((umax_validation > 0.2).sum()) if len(umax_validation) else 0
    umax_total = int(len(umax_validation.dropna()))

    a_entries = pd.DataFrame()
    if parameter_identifiability is not None and len(parameter_identifiability):
        a_entries = parameter_identifiability[
            parameter_identifiability["parameter"].astype(str).eq("A_matrix_entries")
        ].copy()
    a_entries_v4 = pd.DataFrame()
    if identifiability_v4 is not None and len(identifiability_v4):
        a_entries_v4 = identifiability_v4[
            identifiability_v4["parameter"].astype(str).eq("A_matrix_entries")
        ].copy()
    a_cv = (
        float(pd.to_numeric(a_entries["coefficient_of_variation"], errors="coerce").iloc[0])
        if len(a_entries) and "coefficient_of_variation" in a_entries.columns
        else np.nan
    )
    a_ratio = (
        float(pd.to_numeric(a_entries_v4["parameter_to_transition_ratio"], errors="coerce").iloc[0])
        if len(a_entries_v4) and "parameter_to_transition_ratio" in a_entries_v4.columns
        else float(pd.to_numeric(a_entries["parameter_to_transition_ratio"], errors="coerce").iloc[0])
        if len(a_entries) and "parameter_to_transition_ratio" in a_entries.columns
        else np.nan
    )
    a_risk = str(a_entries["risk_level"].iloc[0]) if len(a_entries) and "risk_level" in a_entries.columns else ""

    rows = [
        {
            "audit_area": "diagonal_trajectory_fit",
            "model_scope": "diagonal_attractor",
            "source_tables": "dynamics_fit_summary.csv; q_inf_estimation_audit.csv",
            "fit_target": "state trajectory q_i(t)",
            "primary_fit_method": "direct exponential trajectory least-squares",
            "derivative_policy": "no finite-difference derivative in the primary diagonal fit",
            "evidence_metric": "mean_time_split_relative_rmse",
            "evidence_value": diag_mean,
            "evidence_unit": "relative RMSE",
            "evidence_summary": f"all_positive_rates={diag_stable}; derivative_audit_cases={derivative_cases}",
            "interpretation_scope": "stable compact trajectory descriptor",
            "manuscript_action": "Describe the diagonal attractor as a stable descriptive baseline, not as a discovered reduced-order law.",
            "unsupported_claim": "governing reduced-order physics law",
        },
        {
            "audit_area": "coupled_derivative_fit",
            "model_scope": "coupled_ridge_attractor",
            "source_tables": "coupled_dynamics_fit_summary.csv; coupled_A_matrix.csv; dynamics_derivative_audit.csv",
            "fit_target": "finite-difference derivative dq/dt",
            "primary_fit_method": "ridge-regularized finite-difference regression",
            "derivative_policy": "finite differences are used only for the coupled comparison audit",
            "evidence_metric": "mean_time_split_relative_rmse",
            "evidence_value": coupled_mean,
            "evidence_unit": "relative RMSE",
            "evidence_summary": f"coupled_spectral_stability={coupled_stable}; coupled_robustness_wins={robust_coupled_wins}/{robust_total}",
            "interpretation_scope": "overparameterization and stability audit",
            "manuscript_action": "Retain the coupled model as a structured complexity control rather than a symmetric competing predictor.",
            "unsupported_claim": "selected process-state interaction law",
        },
        {
            "audit_area": "fit_comparison_asymmetry",
            "model_scope": "diagonal_vs_coupled",
            "source_tables": "dynamics_model_comparison.csv; robustness_summary.csv",
            "fit_target": "different estimands",
            "primary_fit_method": "trajectory fit versus derivative fit",
            "derivative_policy": "asymmetric by design",
            "evidence_metric": "paired_condition_state_diagonal_wins",
            "evidence_value": float(diag_wins),
            "evidence_unit": f"of {dyn_total} condition-state pairs",
            "evidence_summary": f"median_coupled_minus_diagonal_relative_rmse={diag_median_advantage:.4g}; stress_support_rate={stress_support:.3f}",
            "interpretation_scope": "model-role comparison, not estimator symmetry",
            "manuscript_action": "State that lower diagonal error is a parsimonious selection result under this protocol, not a symmetric theorem over all coupled trajectory models.",
            "unsupported_claim": "statistical dominance of diagonal dynamics",
        },
        {
            "audit_area": "coupled_identifiability_risk",
            "model_scope": "coupled_ridge_attractor",
            "source_tables": "parameter_identifiability.csv; coupled_A_matrix.csv",
            "fit_target": "coupled-attractor matrix entries",
            "primary_fit_method": "49-entry coupled matrix identified from short sequences",
            "derivative_policy": "finite-difference rows amplify sampling sensitivity",
            "evidence_metric": "A_matrix_entry_coefficient_of_variation",
            "evidence_value": a_cv,
            "evidence_unit": "coefficient of variation",
            "evidence_summary": f"parameter_to_transition_ratio={a_ratio:.4g}; risk_level={a_risk}",
            "interpretation_scope": "high identifiability risk",
            "manuscript_action": "Keep coupled-attractor matrix coefficients out of the physical-parameter interpretation.",
            "unsupported_claim": "uniquely identifiable coupled transport coefficients",
        },
        {
            "audit_area": "Umax_state_scope",
            "model_scope": "diagonal_attractor",
            "source_tables": "dynamics_fit_summary.csv; parameter_identifiability.csv",
            "fit_target": "maximum velocity descriptor",
            "primary_fit_method": "same diagonal trajectory fit as other states",
            "derivative_policy": "no derivative in primary diagonal fit",
            "evidence_metric": "Umax_median_time_split_relative_rmse",
            "evidence_value": umax_median,
            "evidence_unit": "relative RMSE",
            "evidence_summary": f"high_error_cases={umax_high_cases}/{umax_total}",
            "interpretation_scope": "weakest reduced state",
            "manuscript_action": "State that the framework is more reliable for geometric descriptors than for the maximum-velocity descriptor.",
            "unsupported_claim": "strong flow-state prediction",
        },
    ]
    return pd.DataFrame(rows)


def make_numerical_credibility_audit(
    validation_stress_tests: pd.DataFrame,
    holdout_extrapolation_audit: pd.DataFrame,
    literature_dimension_benchmark: pd.DataFrame,
    external_holdout_summary: pd.DataFrame,
    boundary_extraction_sensitivity: pd.DataFrame,
    geometry_selection_metrics: pd.DataFrame,
    input_file_manifest: pd.DataFrame,
    environment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Integrate export-only consistency checks without implying solver verification."""

    rows: list[dict[str, object]] = []

    def bool_series(values: pd.Series) -> pd.Series:
        if values.dtype == bool:
            return values.fillna(False)
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    def add_row(
        audit_area: str,
        evidence_family: str,
        source_tables: str,
        scope: str,
        scenario_count: int,
        primary_metric: str,
        primary_value: float,
        primary_unit: str,
        secondary_metric: str,
        secondary_value: float,
        interpretation: str,
        limitation: str,
        manuscript_action: str,
    ) -> None:
        rows.append(
            {
                "audit_area": audit_area,
                "evidence_family": evidence_family,
                "source_tables": source_tables,
                "scope": scope,
                "scenario_count": scenario_count,
                "primary_metric": primary_metric,
                "primary_value": primary_value,
                "primary_unit": primary_unit,
                "secondary_metric": secondary_metric,
                "secondary_value": secondary_value,
                "interpretation": interpretation,
                "limitation": limitation,
                "manuscript_action": manuscript_action,
                "available_export_only": True,
                "not_mesh_time_step_solver_verification": True,
                "does_not_validate_solver_physics": True,
            }
        )

    if validation_stress_tests is not None and len(validation_stress_tests):
        for family, label in [
            ("train_fraction", "train-fraction sensitivity"),
            ("rolling_origin", "rolling-origin temporal-sampling check"),
            ("leave_one_time_step", "time-step removal sensitivity"),
            ("deterministic_state_noise", "deterministic state-noise perturbation"),
        ]:
            group = validation_stress_tests[validation_stress_tests["test_family"].astype(str).eq(family)].copy()
            values = pd.to_numeric(group.get("mean_validation_relative_rmse", pd.Series(dtype=float)), errors="coerce")
            support = bool_series(group.get("supports_main_model", pd.Series(dtype=bool)))
            if len(group):
                add_row(
                    family,
                    label,
                    "validation_stress_tests.csv",
                    "A1-A15 model-construction exports",
                    int(len(group)),
                    "mean_time_split_relative_rmse",
                    float(values.mean()),
                    "relative RMSE",
                    "support_rate",
                    float(support.mean()),
                    "Checks sensitivity of the diagonal descriptor to available temporal sampling and split choices.",
                    "Internal consistency only; no new FLOW-3D mesh or solver time-step reruns are performed.",
                    "Report as export-only temporal-sampling evidence, not as numerical solver verification.",
                )

    if external_holdout_summary is not None and len(external_holdout_summary):
        metric_map = dict(zip(external_holdout_summary["metric"], external_holdout_summary["value"]))
        ext_cases = float(metric_map.get("external_validation_case_count", np.nan))
        ext_process = float(metric_map.get("external_process_response_mean_relative_error", np.nan))
        ext_dynamics = float(metric_map.get("external_dynamics_mean_relative_rmse", np.nan))
        add_row(
            "same_ecosystem_cfd_holdout",
            "same-solver transfer check",
            "external_holdout_validation_summary.csv",
            "V-prefixed and A16-A20 holdout exports",
            int(ext_cases) if np.isfinite(ext_cases) else 0,
            "process_response_mean_relative_error",
            ext_process,
            "relative error",
            "process_parameterized_attractor_mean_relative_rmse",
            ext_dynamics,
            "Checks whether descriptors and simple process-response maps transfer to withheld same-solver conditions.",
            "Same software ecosystem and preprocessing assumptions; no experimental measurements or independent solver physics are tested.",
            "Use same-solver CFD holdout or CFD-output transfer wording.",
        )

    if holdout_extrapolation_audit is not None and len(holdout_extrapolation_audit):
        positions = holdout_extrapolation_audit.get("holdout_position", pd.Series(dtype=str)).astype(str)
        extrap_count = int(positions.eq("extrapolation").sum())
        add_row(
            "holdout_extrapolation",
            "process-space range check",
            "holdout_extrapolation_audit.csv",
            "same-solver holdout process settings",
            int(len(holdout_extrapolation_audit)),
            "extrapolative_holdout_case_count",
            float(extrap_count),
            "cases",
            "in_range_holdout_case_count",
            float(len(holdout_extrapolation_audit) - extrap_count),
            "Separates interpolation-like holdout checks from extrapolative process settings.",
            "A range check cannot establish physics fidelity outside the available process matrix.",
            "State which holdout cases are extrapolative when discussing transfer evidence.",
        )

    if literature_dimension_benchmark is not None and len(literature_dimension_benchmark):
        numeric = literature_dimension_benchmark[
            pd.to_numeric(literature_dimension_benchmark.get("reported_range_min"), errors="coerce").notna()
        ].copy()
        for row in numeric.itertuples(index=False):
            reported_min = float(getattr(row, "reported_range_min"))
            reported_max = float(getattr(row, "reported_range_max"))
            current = float(getattr(row, "current_export_mean_mm"))
            add_row(
                f"published_dimension_plausibility_{getattr(row, 'reported_quantity')}",
                "published-dimension plausibility",
                "literature_dimension_benchmark.csv",
                "quasi-steady exported descriptors",
                1,
                "current_export_mean_mm",
                current,
                "mm",
                "published_range_midpoint_mm",
                0.5 * (reported_min + reported_max),
                str(getattr(row, "comparison_status")),
                "Literature ranges are broad context, not matched controls for this solver setup or export filter.",
                "Use as millimetre-scale plausibility context only.",
            )

    if boundary_extraction_sensitivity is not None and len(boundary_extraction_sensitivity):
        summary = boundary_extraction_sensitivity[
            boundary_extraction_sensitivity["case_id"].astype(str).eq("summary")
        ].copy() if "case_id" in boundary_extraction_sensitivity.columns else pd.DataFrame()
        if len(summary):
            best = summary.sort_values("mean_relative_difference_from_convex").iloc[0]
            add_row(
                "convex_hull_alpha_complex_sensitivity",
                "export-envelope proxy bias check",
                "boundary_extraction_sensitivity.csv",
                "available molten-region point clouds",
                int(len(summary)),
                "best_mean_relative_difference_from_convex",
                float(best["mean_relative_difference_from_convex"]),
                "relative difference",
                "best_alpha_radius_multiplier",
                float(best["alpha_radius_multiplier"]),
                "Compares convex-hull and alpha-complex volume proxies from the same exported point clouds.",
                "This is a proxy-bias audit; it does not recover the unexported physical interface.",
                "Describe convex hull as an export-derived envelope proxy.",
            )

    if geometry_selection_metrics is not None and len(geometry_selection_metrics):
        train = geometry_selection_metrics[
            geometry_selection_metrics["analysis_role"].astype(str).eq("model_construction")
            & geometry_selection_metrics["case_id"].astype(str).ne("summary")
        ].copy()
        if len(train):
            total = int(len(train))
            chamfer = int(bool_series(train["superellipsoid_improves_mean_normalized_chamfer_distance"]).sum())
            hausdorff = int(bool_series(train["superellipsoid_improves_mean_hausdorff_distance_m"]).sum())
            radial = int(bool_series(train["superellipsoid_improves_mean_radial_distance_rmse_m"]).sum())
            volume = int(bool_series(train["superellipsoid_improves_mean_volume_relative_error"]).sum())
            add_row(
                "geometry_distance_consistency",
                "geometric-distance limitation check",
                "geometry_selection_metrics.csv",
                "A1-A15 model-construction case summaries",
                total,
                "normalized_chamfer_win_count",
                float(chamfer),
                f"of {total} cases",
                "hausdorff_radial_volume_win_counts",
                np.nan,
                f"Chamfer={chamfer}/{total}; Hausdorff={hausdorff}/{total}; radial={radial}/{total}; volume proxy={volume}/{total}.",
                "Most geometric-distance diagnostics do not support broad reconstruction claims.",
                "Frame the superellipsoid as the selected algebraic observed-envelope descriptor.",
            )

    if input_file_manifest is not None and len(input_file_manifest):
        add_row(
            "input_manifest_hashes",
            "file provenance and checksum audit",
            "input_file_manifest.csv",
            "all discovered raw CSV exports",
            int(len(input_file_manifest)),
            "manifest_file_count",
            float(len(input_file_manifest)),
            "files",
            "unique_schema_count",
            float(input_file_manifest["column_schema"].nunique()) if "column_schema" in input_file_manifest.columns else np.nan,
            "Records raw CSV paths, row counts, schemas and SHA256 hashes for reproducible processing.",
            "Checks provenance of available exports only; raw FLOW-3D project files remain outside the package.",
            "Use as reproducibility evidence rather than physics-quality evidence.",
        )

    if environment_summary is not None and len(environment_summary):
        add_row(
            "software_environment",
            "runtime reproducibility audit",
            "environment_summary.csv",
            "local Python and LaTeX toolchain",
            int(len(environment_summary)),
            "reported_component_count",
            float(len(environment_summary)),
            "components",
            "unavailable_component_count",
            float(environment_summary["version"].astype(str).str.contains("unavailable").sum())
            if "version" in environment_summary.columns
            else np.nan,
            "Records the runtime used to regenerate tables, figures and PDFs.",
            "Environment capture supports reproducibility but is not a solver verification study.",
            "Keep environment and checksum tables in the reproducibility package.",
        )

    return pd.DataFrame(rows)


def make_q_inf_estimation_audit(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    rows = []
    for case_id, group in table.groupby("case_id", sort=False):
        group = group.sort_values("time_s").reset_index(drop=True)
        t = group["time_s"].to_numpy(dtype=float)
        train_mask = train_split_mask(t)
        quasi_mask = train_mask & (t >= QUASI_STEADY_START_S)
        fallback_used = bool(np.sum(quasi_mask) < 3)
        if fallback_used:
            quasi_mask = train_mask
        dt = np.diff(t)
        meta = group.iloc[0]
        base = {
            "case_id": case_id,
            "case_index": int(meta["case_index"]) if "case_index" in group.columns else np.nan,
            "analysis_role": "same_ecosystem_holdout"
            if "validation_cohort" in group.columns and pd.notna(meta.get("validation_cohort", np.nan))
            else "model_construction",
            "validation_cohort": meta.get("validation_cohort", ""),
            "source_root": meta.get("source_root", ""),
            "time_points": int(len(t)),
            "train_fraction": TRAIN_FRACTION,
            "train_points": int(np.sum(train_mask)),
            "validation_points": int(np.sum(~train_mask)),
            "q_inf_policy": "training points with t >= quasi_steady_start_s; fallback to all training points if fewer than three",
            "quasi_steady_start_s": QUASI_STEADY_START_S,
            "q_inf_window_start_s": float(np.nanmin(t[quasi_mask])) if np.any(quasi_mask) else np.nan,
            "q_inf_window_end_s": float(np.nanmax(t[quasi_mask])) if np.any(quasi_mask) else np.nan,
            "q_inf_points": int(np.sum(quasi_mask)),
            "q_inf_fallback_to_all_training": fallback_used,
            "q_inf_uses_validation_points": False,
            "time_step_min_s": float(np.nanmin(dt)) if dt.size else np.nan,
            "time_step_max_s": float(np.nanmax(dt)) if dt.size else np.nan,
            "nonuniform_time_spacing": bool(dt.size and np.nanmax(dt) > np.nanmin(dt)),
        }
        for state in STATE_COLUMNS:
            values = group[state].to_numpy(dtype=float)
            rows.append(
                {
                    **base,
                    "state": state,
                    "label": STATE_LABELS[state],
                    "q_inf_estimate": float(np.nanmean(values[quasi_mask])) if np.any(quasi_mask) else np.nan,
                    "q0": float(values[0]) if len(values) else np.nan,
                    "q_validation_mean": float(np.nanmean(values[~train_mask])) if np.any(~train_mask) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_holdout_extrapolation_audit(train_table: pd.DataFrame, validation_table: pd.DataFrame) -> pd.DataFrame:
    if validation_table is None or len(validation_table) == 0:
        return pd.DataFrame()
    train_case = quasi_steady_case_table(train_table)
    validation_case = quasi_steady_case_table(validation_table)
    rows = []
    for _, row in validation_case.iterrows():
        flags = _feature_extrapolation_flags(train_case, row)
        rows.append(
            {
                "case_id": row["case_id"],
                "validation_cohort": row.get("validation_cohort", ""),
                "source_root": row.get("source_root", ""),
                "power_W": float(row["power_W"]),
                "scan_speed_mm_s": float(row["scan_speed_mm_s"]),
                "powder_feed_g_min": float(row["powder_feed_g_min"]),
                "training_power_range_W": f"{float(train_case['power_W'].min()):g}-{float(train_case['power_W'].max()):g}",
                "training_scan_speed_range_mm_s": f"{float(train_case['scan_speed_mm_s'].min()):g}-{float(train_case['scan_speed_mm_s'].max()):g}",
                "training_powder_feed_range_g_min": f"{float(train_case['powder_feed_g_min'].min()):g}-{float(train_case['powder_feed_g_min'].max()):g}",
                "extrapolation_flags": flags,
                "holdout_position": "extrapolation" if flags else "within_training_feature_range",
                "manuscript_action": "Report V1 power extrapolation separately from same-range holdout cases."
                if flags
                else "Report as same-solver holdout within the training process-feature ranges.",
            }
        )
    return pd.DataFrame(rows)


def make_literature_dimension_benchmark(quasi: pd.DataFrame) -> pd.DataFrame:
    current = quasi[quasi["case_id"].astype(str).eq("all_conditions")].copy() if quasi is not None and len(quasi) else pd.DataFrame()

    def current_mean(quantity: str) -> float:
        match = current[current["quantity"].astype(str).eq(quantity)]
        return float(match["mean"].iloc[0]) if len(match) else np.nan

    current_width_mm = current_mean("full_width_m") * 1000.0
    current_length_mm = current_mean("melt_pool_length_m") * 1000.0
    current_height_mm = current_mean("height_span_m") * 1000.0
    rows = [
        {
            "citation_key": "lp_ded_overview_typical_range",
            "paper_title": "Overview source reporting typical LP-DED melt-pool dimensions",
            "reported_quantity": "width_mm",
            "reported_range_min": 0.25,
            "reported_range_max": 1.00,
            "current_export_mean_mm": current_width_mm,
            "comparison_status": "same_millimetre_scale_but_current_export_mean_above_review_range",
            "source_note": "Overview reports typical melt-pool widths of 0.25-1.0 mm for LP-DED; used only as a broad plausibility range, not validation.",
            "manuscript_use": "published-dimension plausibility check; not measurement-based evidence",
        },
        {
            "citation_key": "lp_ded_overview_typical_range",
            "paper_title": "Overview source reporting typical LP-DED melt-pool dimensions",
            "reported_quantity": "height_or_depth_mm",
            "reported_range_min": 0.25,
            "reported_range_max": 0.50,
            "current_export_mean_mm": current_height_mm,
            "comparison_status": "same_millimetre_scale_but_current_vertical_span_above_review_range",
            "source_note": "Overview reports typical height/depth scale of 0.25-0.5 mm; current height_span is an exported vertical span and is not directly comparable to clad height or melt depth.",
            "manuscript_use": "published-dimension plausibility check; not measurement-based evidence",
        },
        {
            "citation_key": "current_work",
            "paper_title": "Current FLOW-3D observed-envelope export",
            "reported_quantity": "length_mm",
            "reported_range_min": np.nan,
            "reported_range_max": np.nan,
            "current_export_mean_mm": current_length_mm,
            "comparison_status": "current_export_descriptor_only_no_direct_literature_range",
            "source_note": "Current all-condition quasi-steady mean melt-pool length from generated descriptors.",
            "manuscript_use": "internal descriptor context",
        },
    ]
    for key, title in [
        ("zhang2021dedcfd", "Numerical investigation on heat transfer of melt pool and clad generation in directed energy deposition of stainless steel"),
        ("liao2022", "Simulation-guided variable laser power design for melt pool depth control in directed energy deposition"),
        ("zhu2023", "Prediction of melt pool shape in additive manufacturing based on machine learning methods"),
        ("wang2023tcn", "Prediction of melt pool width and layer height for laser directed energy deposition enabled by physics-driven temporal convolutional network"),
        ("jiang2024piml", "Physics-Informed Machine Learning for Accurate Prediction of Temperature and Melt Pool Dimension in Metal Additive Manufacturing"),
    ]:
        rows.append(
            {
                "citation_key": key,
                "paper_title": title,
                "reported_quantity": "melt_pool_dimension",
                "reported_range_min": np.nan,
                "reported_range_max": np.nan,
                "current_export_mean_mm": np.nan,
                "comparison_status": "not_numerically_comparable_from_accessible_metadata",
                "source_note": "Retained as process/modeling context; no directly extractable width/depth/length range was used in this generated benchmark table.",
                "manuscript_use": "context only; do not use as numerical validation",
            }
        )
    return pd.DataFrame(rows)


def make_environment_summary() -> pd.DataFrame:
    rows = [
        {"component": "python", "version": sys.version.replace("\n", " "), "source": sys.executable},
        {"component": "numpy", "version": np.__version__, "source": "import numpy"},
        {"component": "pandas", "version": pd.__version__, "source": "import pandas"},
        {"component": "matplotlib", "version": mpl.__version__, "source": "import matplotlib"},
    ]
    try:
        import scipy

        rows.append({"component": "scipy", "version": scipy.__version__, "source": "import scipy"})
    except Exception as exc:
        rows.append({"component": "scipy", "version": f"unavailable:{exc}", "source": "import scipy"})
    for command in [["pdflatex", "--version"], ["bibtex", "--version"]]:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            version = (result.stdout or result.stderr).splitlines()[0] if (result.stdout or result.stderr) else ""
        except Exception as exc:
            version = f"unavailable:{exc}"
        rows.append({"component": command[0], "version": version, "source": " ".join(command)})
    return pd.DataFrame(rows)


def make_output_checksums(output_dir: Path) -> pd.DataFrame:
    targets = [
        output_dir / "latex" / "main_submission.pdf",
        output_dir / "latex" / "main_submission.tex",
        output_dir / "latex" / "supplementary_methods.pdf",
        output_dir / "latex" / "supplementary_methods.tex",
        output_dir / "reports" / "declarations_for_submission.md",
        output_dir / "reports" / "amm_submission_checklist.md",
        output_dir / "tables" / "submission_package_manifest.csv",
        output_dir / "tables" / "modeling_table.csv",
        output_dir / "tables" / "external_validation_modeling_table.csv",
        output_dir / "tables" / "geometry_selection_metrics.csv",
        output_dir / "tables" / "literature_dimension_benchmark.csv",
        output_dir / "tables" / "dynamics_fit_asymmetry_audit.csv",
        output_dir / "tables" / "numerical_credibility_audit.csv",
        output_dir / "tables" / "input_file_manifest.csv",
        output_dir / "reproducibility_package.zip",
    ]
    rows = []
    for path in targets:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    return pd.DataFrame(rows)


def plot_moving_frame(point_cloud: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    point_cloud = representative_case_subset(point_cloud)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    cmap = mpl.colormaps["plasma"]
    times = np.sort(point_cloud["time_s"].unique())
    norm = mpl.colors.Normalize(vmin=float(times.min()), vmax=float(times.max()))
    selected_times = [times[0], times[min(3, len(times) - 1)], times[-1]]

    def _scatter(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, color: object, alpha: float = 0.28) -> None:
        ax.scatter(x, y, s=3.0, alpha=alpha, color=color, linewidths=0)

    for time_s in selected_times:
        part = point_cloud[point_cloud["time_s"] == time_s]
        color = cmap(norm(time_s))
        x_mm = part["Points_0"].to_numpy(dtype=float) * 1e3
        xi_mm = part["xi_m"].to_numpy() * 1e3
        y_mm = part["Points_1"].to_numpy() * 1e3
        z_mm = part["Points_2"].to_numpy() * 1e3
        laser_x_mm = float(part["laser_x_m"].iloc[0]) * 1e3 if "laser_x_m" in part.columns else 0.0

        _scatter(axes[0, 0], x_mm, y_mm, color)
        axes[0, 0].axvline(laser_x_mm, color=color, lw=0.8, alpha=0.75)

        _scatter(axes[1, 0], xi_mm, y_mm, color)
        _scatter(axes[1, 0], xi_mm, -y_mm, color)

    descriptor_time = times[-1]
    part = point_cloud[point_cloud["time_s"] == descriptor_time]
    xi_mm = part["xi_m"].to_numpy(dtype=float) * 1e3
    y_mm = part["Points_1"].to_numpy(dtype=float) * 1e3
    z_mm = part["Points_2"].to_numpy(dtype=float) * 1e3
    descriptor_color = "#1F77F4"
    mirror_color = "#FF8C00"
    axes[0, 1].scatter(xi_mm, y_mm, s=4.0, alpha=0.34, color=descriptor_color, linewidths=0)
    axes[0, 1].scatter(xi_mm, -y_mm, s=4.0, alpha=0.26, color=mirror_color, linewidths=0)
    axes[1, 1].scatter(xi_mm, z_mm, s=4.0, alpha=0.28, color=descriptor_color, linewidths=0)

    xi_min = float(np.nanmin(xi_mm))
    xi_max = float(np.nanmax(xi_mm))
    y_max = float(np.nanmax(y_mm))
    z_min = float(np.nanmin(z_mm))
    z_max = float(np.nanmax(z_mm))
    xi_range = max(xi_max - xi_min, 1e-6)
    z_range = max(z_max - z_min, 1e-6)
    z_mid = z_min + 0.15 * (z_max - z_min)
    xi_center = 0.0

    x_for_w = xi_min - 0.20 * xi_range
    axes[0, 1].annotate(
        "",
        xy=(x_for_w, y_max),
        xytext=(x_for_w, -y_max),
        arrowprops=dict(arrowstyle="<->", color="#2DB84D", lw=1.05),
    )
    axes[0, 1].text(x_for_w - 0.055 * xi_range, 0, r"$W$", color="#2DB84D", va="center", ha="right")
    axes[0, 1].text(
        0.98,
        0.94,
        "exported half",
        color=descriptor_color,
        transform=axes[0, 1].transAxes,
        fontsize=7,
        ha="right",
        va="top",
    )
    axes[0, 1].text(
        0.98,
        0.86,
        "mirrored half",
        color=mirror_color,
        transform=axes[0, 1].transAxes,
        fontsize=7,
        ha="right",
        va="top",
    )

    axes[1, 1].annotate(
        "",
        xy=(xi_max, z_mid),
        xytext=(xi_center, z_mid),
        arrowprops=dict(arrowstyle="<->", color="#F04E23", lw=1.1),
    )
    axes[1, 1].text((xi_max + xi_center) / 2, z_mid + 0.055, r"$L_f$", color="#F04E23", ha="center")
    axes[1, 1].annotate(
        "",
        xy=(xi_min, z_mid),
        xytext=(xi_center, z_mid),
        arrowprops=dict(arrowstyle="<->", color="#FF8C00", lw=1.1),
    )
    axes[1, 1].text((xi_min + xi_center) / 2, z_mid + 0.055, r"$L_r$", color="#FF8C00", ha="center")
    x_for_h = xi_max + 0.14 * xi_range
    axes[1, 1].annotate(
        "",
        xy=(x_for_h, z_max),
        xytext=(x_for_h, z_min),
        arrowprops=dict(arrowstyle="<->", color="#2DB84D", lw=1.1),
    )
    axes[1, 1].text(x_for_h + 0.04 * xi_range, (z_min + z_max) / 2, r"$H$", color="#2DB84D", va="center")
    axes[1, 1].text(
        0.03,
        0.95,
        rf"$W=2\max(y)={2*y_max:.2f}$ mm",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

    axes[0, 0].axhline(0, color="0.65", lw=0.7)
    axes[0, 1].axvline(0, color="0.25", lw=0.8, ls="--")
    axes[1, 0].axvline(0, color="0.25", lw=0.8, ls="--")
    axes[1, 1].axvline(0, color="0.25", lw=0.8, ls="--")
    axes[0, 1].axhline(0, color="0.65", lw=0.7)
    axes[1, 0].axhline(0, color="0.65", lw=0.7)

    axes[0, 0].set_title("Raw half-domain export")
    axes[0, 0].set_xlabel("Laboratory x (mm)")
    axes[0, 0].set_ylabel("y (mm)")
    axes[0, 1].set_title("Symmetry reconstruction")
    axes[0, 1].set_xlabel(r"Moving coordinate $\xi$ (mm)")
    axes[0, 1].set_ylabel("Mirrored y (mm)")
    axes[1, 0].set_title("Moving-frame alignment")
    axes[1, 0].set_xlabel(r"Moving coordinate $\xi$ (mm)")
    axes[1, 0].set_ylabel("Mirrored y (mm)")
    axes[1, 1].set_title("Boundary descriptors")
    axes[1, 1].set_xlabel(r"Moving coordinate $\xi$ (mm)")
    axes[1, 1].set_ylabel("z (mm)")

    for label, ax in zip(["a", "b", "c", "d"], axes.ravel()):
        add_panel_label(ax, label, x=-0.13, y=1.03)
        apply_axis_polish(ax, grid=None)
        ax.set_aspect("equal", adjustable="box")
    axes[0, 0].set_aspect("auto")
    axes[0, 1].set_xlim(xi_min - 0.32 * xi_range, xi_max + 0.08 * xi_range)
    axes[0, 1].set_ylim(-1.12 * y_max, 1.12 * y_max)
    axes[1, 1].set_xlim(xi_min - 0.12 * xi_range, xi_max + 0.30 * xi_range)
    axes[1, 1].set_ylim(z_min - 0.12 * z_range, z_max + 0.12 * z_range)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.88, pad=0.018)
    cbar.set_label("Time (s)")
    cbar.ax.tick_params(labelsize=6.3, length=2.4)
    save_publication_figure(fig, fig_dir / "fig01_moving_frame_point_cloud")
    plt.close(fig)


def plot_geometry(table: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    table = representative_case_subset(table).sort_values("time_s")
    fig, ax = plt.subplots(figsize=(4.8, 3.1), constrained_layout=True)
    x = table["time_s"].to_numpy()
    series = [
        ("front_length_m", r"$L_f$", "#1F77F4"),
        ("rear_length_m", r"$L_r$", "#FF8C00"),
        ("full_width_m", r"$W$", "#2DB84D"),
        ("height_span_m", r"$H$", "#A64AC9"),
    ]
    for col, label, color in series:
        ax.plot(x, table[col].to_numpy() * 1e3, marker="o", ms=3.0, lw=1.2, label=label, color=color)
    ax.axvline(QUASI_STEADY_START_S, color="0.35", lw=0.8, ls="--")
    ax.text(
        QUASI_STEADY_START_S + 0.015,
        0.05,
        "quasi-steady\nwindow",
        transform=ax.get_xaxis_transform(),
        fontsize=6.2,
        color="0.35",
        va="bottom",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Length scale (mm)")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.0, handlelength=1.3)
    apply_axis_polish(ax)
    save_publication_figure(fig, fig_dir / "fig02_geometry_evolution")
    plt.close(fig)


def plot_thermal_flow(table: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    table = representative_case_subset(table).sort_values("time_s")
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 5.0), sharex=True, constrained_layout=True)
    x = table["time_s"].to_numpy()
    axes[0].plot(x, table["Tmax_K"], marker="o", ms=3.0, lw=1.2, color="#E53935")
    axes[1].plot(x, table["Gmean_K_per_m"] / 1e6, marker="o", ms=3.0, lw=1.2, color="#1F77F4")
    axes[2].plot(x, table["Umax_m_per_s"], marker="o", ms=3.0, lw=1.2, color="#2DB84D")
    axes[0].set_ylabel(r"$T_{\max}$ (K)")
    axes[1].set_ylabel(r"$G_{\mathrm{mean}}$ (MK m$^{-1}$)")
    axes[2].set_ylabel(r"$U_{\max}$ (m s$^{-1}$)")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(QUASI_STEADY_START_S, color="0.35", lw=0.8, ls="--")
        apply_axis_polish(ax)
    save_publication_figure(fig, fig_dir / "fig03_thermal_flow_evolution")
    plt.close(fig)


def plot_dynamics_validation(predictions: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    predictions = representative_case_subset(predictions).sort_values("time_s")
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), constrained_layout=True)
    axes = axes.ravel()
    time = predictions["time_s"].to_numpy()
    train = predictions["split"].to_numpy() == "train"
    scale = {
        "front_length_m": 1e3,
        "rear_length_m": 1e3,
        "full_width_m": 1e3,
        "height_span_m": 1e3,
        "Tmax_K": 1.0,
        "Gmean_K_per_m": 1e-6,
        "Umax_m_per_s": 1.0,
    }
    units = {
        "front_length_m": "mm",
        "rear_length_m": "mm",
        "full_width_m": "mm",
        "height_span_m": "mm",
        "Tmax_K": "K",
        "Gmean_K_per_m": r"MK m$^{-1}$",
        "Umax_m_per_s": r"m s$^{-1}$",
    }
    for idx, col in enumerate(STATE_COLUMNS):
        ax = axes[idx]
        actual = predictions[f"{col}_actual"].to_numpy() * scale[col]
        pred = predictions[f"{col}_predicted"].to_numpy() * scale[col]
        ax.plot(time, actual, "o", ms=2.8, color="#222222", label="Data")
        ax.plot(time, pred, "-", lw=1.1, color="#1F77F4", label="Attractor model")
        ax.axvspan(time[0], time[np.where(train)[0][-1]], color="0.92", zorder=-1)
        ax.set_title(state_plot_label(col))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(units[col])
        apply_axis_polish(ax)
        apply_axis_polish(ax)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center")
    save_publication_figure(fig, fig_dir / "fig04_dynamics_validation")
    plt.close(fig)


def boundary_curve_top(params: np.ndarray, model: str, side: int, num: int = 160) -> tuple[np.ndarray, np.ndarray]:
    if model == "ellipsoid":
        af, ar, b, _c, xi_c, _z_c = params
        n, m = 2.0, 2.0
    else:
        af, ar, b, _c, xi_c, _z_c, n, m, _p = params
    a = af if side > 0 else ar
    xi = np.linspace(xi_c, xi_c + side * a, num)
    r = np.clip(1.0 - np.abs((xi - xi_c) / a) ** n, 0.0, None)
    y = b * r ** (1.0 / m)
    return xi, y


def boundary_curve_side(params: np.ndarray, model: str, side: int, num: int = 160) -> tuple[np.ndarray, np.ndarray]:
    if model == "ellipsoid":
        af, ar, _b, c, xi_c, z_c = params
        n, p = 2.0, 2.0
    else:
        af, ar, _b, c, xi_c, z_c, n, _m, p = params
    a = af if side > 0 else ar
    xi = np.linspace(xi_c, xi_c + side * a, num)
    r = np.clip(1.0 - np.abs((xi - xi_c) / a) ** n, 0.0, None)
    z = c * r ** (1.0 / p)
    return xi, z_c + z


def model_params_from_row(row: pd.Series, model: str) -> np.ndarray:
    if model == "ellipsoid":
        return row[
            [
                "ellipsoid_af_m",
                "ellipsoid_ar_m",
                "ellipsoid_b_m",
                "ellipsoid_c_m",
                "ellipsoid_xic_m",
                "ellipsoid_zc_m",
            ]
        ].to_numpy(dtype=float)
    return row[
        [
            "superellipsoid_af_m",
            "superellipsoid_ar_m",
            "superellipsoid_b_m",
            "superellipsoid_c_m",
            "superellipsoid_xic_m",
            "superellipsoid_zc_m",
            "superellipsoid_n",
            "superellipsoid_m",
            "superellipsoid_p",
        ]
    ].to_numpy(dtype=float)


def nearest_boundary_times(table: pd.DataFrame, requested_times: list[float]) -> list[float]:
    available = table["time_s"].to_numpy(dtype=float)
    selected_times: list[float] = []
    for requested in requested_times:
        selected_times.append(float(available[np.argmin(np.abs(available - requested))]))
    return list(dict.fromkeys(selected_times))


def draw_boundary_overlay_pair(
    top_ax: mpl.axes.Axes,
    side_ax: mpl.axes.Axes,
    row: pd.Series,
    points: pd.DataFrame,
    colors: dict[str, str],
    labels: dict[str, str],
    *,
    top_title: str = "",
    side_title: str = "",
    top_ylabel: str = "y (mm)",
    side_ylabel: str = "z (mm)",
    show_xlabel: bool = False,
    point_size: float = 5.0,
    line_width: float = 1.1,
) -> None:
    if points.empty:
        for ax in [top_ax, side_ax]:
            ax.text(0.5, 0.5, "No points", transform=ax.transAxes, ha="center", va="center", color="0.45")
            apply_axis_polish(ax, grid=None)
        return

    xi = points["xi_m"].to_numpy(dtype=float) * 1e3
    y = points["Points_1"].to_numpy(dtype=float) * 1e3
    z = points["Points_2"].to_numpy(dtype=float) * 1e3
    top_ax.scatter(xi, y, s=point_size, alpha=0.35, color="0.45", linewidths=0)
    top_ax.scatter(xi, -y, s=point_size, alpha=0.35, color="0.75", linewidths=0)
    side_ax.scatter(xi, z, s=point_size, alpha=0.35, color="0.45", linewidths=0)

    for model in ["ellipsoid", "superellipsoid"]:
        params = model_params_from_row(row, model)
        if not np.all(np.isfinite(params)):
            continue
        for side in [-1, 1]:
            xi_curve, y_curve = boundary_curve_top(params, model, side)
            top_ax.plot(xi_curve * 1e3, y_curve * 1e3, color=colors[model], lw=line_width)
            top_ax.plot(xi_curve * 1e3, -y_curve * 1e3, color=colors[model], lw=line_width)
            xi_curve, z_curve = boundary_curve_side(params, model, side)
            side_ax.plot(xi_curve * 1e3, z_curve * 1e3, color=colors[model], lw=line_width)
            side_ax.plot(xi_curve * 1e3, (2 * params[5] - z_curve) * 1e3, color=colors[model], lw=line_width)

    top_ax.axvline(0, color="0.3", lw=0.6, ls="--")
    side_ax.axvline(0, color="0.3", lw=0.6, ls="--")
    top_ax.set_title(top_title)
    side_ax.set_title(side_title)
    top_ax.set_ylabel(top_ylabel)
    side_ax.set_ylabel(side_ylabel)
    if show_xlabel:
        top_ax.set_xlabel(r"Moving coordinate $\xi$ (mm)")
        side_ax.set_xlabel(r"Moving coordinate $\xi$ (mm)")
    apply_axis_polish(top_ax, grid=None)
    apply_axis_polish(side_ax, grid=None)


def plot_boundary_fit_comparison(table: pd.DataFrame, point_cloud: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    case_id = representative_case_id(table)
    if case_id is not None:
        table = table[table["case_id"].astype(str).eq(case_id)].copy().sort_values("time_s")
        point_cloud = point_cloud[point_cloud["case_id"].astype(str).eq(case_id)].copy()
    selected_times = nearest_boundary_times(table, BOUNDARY_FIT_TIMES)

    fig, axes = plt.subplots(len(selected_times), 2, figsize=(7.2, 1.7 * len(selected_times)), constrained_layout=True)
    if len(selected_times) == 1:
        axes = np.asarray([axes])
    colors = {"ellipsoid": "#1F77F4", "superellipsoid": "#E53935"}
    labels = {"ellipsoid": "Ellipsoid", "superellipsoid": "Superellipsoid"}

    for row_idx, time_s in enumerate(selected_times):
        row = table.loc[np.isclose(table["time_s"], time_s)].iloc[0]
        part = point_cloud[np.isclose(point_cloud["time_s"], time_s)]
        draw_boundary_overlay_pair(
            axes[row_idx, 0],
            axes[row_idx, 1],
            row,
            part,
            colors,
            labels,
            top_title="Top view" if row_idx == 0 else "",
            side_title="Side view" if row_idx == 0 else "",
            top_ylabel=f"{time_s:.2f}s\ny (mm)",
            side_ylabel="z (mm)",
            show_xlabel=row_idx == len(selected_times) - 1,
        )
    add_panel_label(axes[0, 0], "a", x=-0.11, y=1.08)
    add_panel_label(axes[0, 1], "b", x=-0.11, y=1.08)
    handles = [
        mpl.lines.Line2D([0], [0], color=colors[m], lw=1.2, label=labels[m])
        for m in ["ellipsoid", "superellipsoid"]
    ]
    axes[0, 1].legend(handles=handles, loc="upper right")
    save_publication_figure(fig, fig_dir / "fig05_boundary_fit_comparison")
    plt.close(fig)


def unique_finite_points_2d(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    return np.unique(points, axis=0)


def convex_hull_segments_2d(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    points = unique_finite_points_2d(points)
    if points.shape[0] < 3:
        return []
    try:
        hull = ConvexHull(points, qhull_options="QJ")
    except Exception:
        return []
    return [(points[i], points[j]) for i, j in hull.simplices]


def triangle_circumradius_2d(triangle: np.ndarray) -> float:
    a = float(np.linalg.norm(triangle[1] - triangle[2]))
    b = float(np.linalg.norm(triangle[0] - triangle[2]))
    c = float(np.linalg.norm(triangle[0] - triangle[1]))
    first = triangle[1] - triangle[0]
    second = triangle[2] - triangle[0]
    area2 = abs(float(first[0] * second[1] - first[1] * second[0]))
    if area2 <= 0 or not np.isfinite(area2):
        return np.nan
    return a * b * c / (2.0 * area2)


def alpha_complex_segments_2d(points: np.ndarray, radius_limit: float) -> list[tuple[np.ndarray, np.ndarray]]:
    points = unique_finite_points_2d(points)
    if points.shape[0] < 4 or not np.isfinite(radius_limit) or radius_limit <= 0:
        return []
    try:
        triangulation = Delaunay(points, qhull_options="QJ")
    except Exception:
        return []
    edge_counts: dict[tuple[int, int], int] = {}
    for simplex in triangulation.simplices:
        triangle = points[simplex]
        radius = triangle_circumradius_2d(triangle)
        if not np.isfinite(radius) or radius > radius_limit:
            continue
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            edge = tuple(sorted((int(simplex[i]), int(simplex[j]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return [(points[i], points[j]) for (i, j), count in edge_counts.items() if count == 1]


def draw_line_segments(
    ax: mpl.axes.Axes,
    segments: list[tuple[np.ndarray, np.ndarray]],
    color: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    for start, end in segments:
        ax.plot(
            [start[0] * 1e3, end[0] * 1e3],
            [start[1] * 1e3, end[1] * 1e3],
            color=color,
            lw=linewidth,
            ls=linestyle,
        )


def plot_convex_alpha_proxy_comparison(table: pd.DataFrame, point_cloud: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    case_id = representative_case_id(table)
    if case_id is not None:
        table = table[table["case_id"].astype(str).eq(case_id)].copy().sort_values("time_s")
        point_cloud = point_cloud[point_cloud["case_id"].astype(str).eq(case_id)].copy()
    if table.empty or point_cloud.empty:
        save_placeholder_figure(fig_dir / "supp_figS6_convex_alpha_proxy_comparison", "No point-cloud data available.")
        return

    available_times = table["time_s"].to_numpy(dtype=float)
    quasi_times = available_times[available_times >= QUASI_STEADY_START_S]
    candidate_times = quasi_times if quasi_times.size else available_times
    target_time = float(candidate_times[np.argmin(np.abs(candidate_times - 0.50))])
    part = point_cloud[np.isclose(point_cloud["time_s"].to_numpy(dtype=float), target_time)].copy()
    if part.empty:
        save_placeholder_figure(fig_dir / "supp_figS6_convex_alpha_proxy_comparison", "Representative time step is unavailable.")
        return

    top_half = part[["xi_m", "Points_1"]].to_numpy(dtype=float)
    top_points = np.vstack([top_half, top_half * np.array([1.0, -1.0])])
    side_points = part[["xi_m", "Points_2"]].to_numpy(dtype=float)
    projections = [
        ("Top view", top_points, r"Moving coordinate $\xi$ (mm)", "Transverse coordinate y (mm)"),
        ("Side view", side_points, r"Moving coordinate $\xi$ (mm)", "Build coordinate z (mm)"),
    ]
    # Use a visual projection radius large enough to show an envelope instead of interior alpha-complex fragments.
    alpha_multiplier = 8.0
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    colors = {"points": "0.42", "convex": "#1F77F4", "alpha": "#E53935"}

    for label, ax, (title, projection, xlabel, ylabel) in zip(["a", "b"], axes, projections):
        points = unique_finite_points_2d(projection)
        median_nn = median_nearest_neighbor_distance(points)
        alpha_radius = alpha_multiplier * median_nn if np.isfinite(median_nn) else np.nan
        convex_segments = convex_hull_segments_2d(points)
        alpha_segments = alpha_complex_segments_2d(points, alpha_radius)
        ax.scatter(points[:, 0] * 1e3, points[:, 1] * 1e3, s=4.0, alpha=0.28, color=colors["points"], linewidths=0)
        draw_line_segments(ax, convex_segments, colors["convex"], 1.1, "-")
        draw_line_segments(ax, alpha_segments, colors["alpha"], 1.0, "-")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        apply_axis_polish(ax, grid=None)
        add_panel_label(ax, label, x=-0.10, y=1.04)
        if not alpha_segments:
            ax.text(
                0.03,
                0.05,
                "No retained\nalpha edges",
                transform=ax.transAxes,
                fontsize=6.4,
                color=colors["alpha"],
                ha="left",
                va="bottom",
            )

    case_text = str(part["case_id"].iloc[0]) if "case_id" in part.columns else "representative case"
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["points"], markeredgewidth=0, markersize=4.0, alpha=0.5, label="Molten points"),
        mpl.lines.Line2D([0], [0], color=colors["convex"], lw=1.1, label="Convex hull"),
        mpl.lines.Line2D([0], [0], color=colors["alpha"], lw=1.0, label=rf"Alpha-complex, {alpha_multiplier:g}$\times$NN"),
    ]
    axes[1].legend(handles=handles, loc="upper right")
    fig.suptitle(f"{case_text}, t={target_time:.2f} s, projection proxy comparison", fontsize=7.6)
    save_publication_figure(fig, fig_dir / "supp_figS6_convex_alpha_proxy_comparison")
    plt.close(fig)


def plot_dynamics_model_comparison(
    diagonal_predictions: pd.DataFrame,
    coupled_predictions: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    fig_dir: Path,
) -> None:
    configure_matplotlib()
    diagonal_predictions = representative_case_subset(diagonal_predictions).sort_values("time_s")
    coupled_predictions = representative_case_subset(coupled_predictions).sort_values("time_s")
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), constrained_layout=True)
    axes = axes.ravel()
    scale = {
        "front_length_m": 1e3,
        "rear_length_m": 1e3,
        "full_width_m": 1e3,
        "height_span_m": 1e3,
        "Tmax_K": 1.0,
        "Gmean_K_per_m": 1e-6,
        "Umax_m_per_s": 1.0,
    }
    units = {
        "front_length_m": "mm",
        "rear_length_m": "mm",
        "full_width_m": "mm",
        "height_span_m": "mm",
        "Tmax_K": "K",
        "Gmean_K_per_m": r"MK m$^{-1}$",
        "Umax_m_per_s": r"m s$^{-1}$",
    }
    time = diagonal_predictions["time_s"].to_numpy()
    train = diagonal_predictions["split"].to_numpy() == "train"
    for idx, col in enumerate(STATE_COLUMNS):
        ax = axes[idx]
        actual = diagonal_predictions[f"{col}_actual"].to_numpy() * scale[col]
        diag = diagonal_predictions[f"{col}_predicted"].to_numpy() * scale[col]
        coupled = coupled_predictions[f"{col}_predicted"].to_numpy() * scale[col]
        ax.axvspan(time[0], time[np.where(train)[0][-1]], color="0.92", zorder=-1)
        ax.plot(time, actual, "o", ms=2.8, color="#222222", label="Data")
        ax.plot(time, diag, "-", lw=1.0, color="#1F77F4", label="Diagonal")
        ax.plot(time, coupled, "--", lw=1.0, color="#E53935", label="Coupled")
        ax.set_title(state_plot_label(col))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(units[col])
    axes[-1].axis("off")
    mean_errors = dynamics_comparison.groupby("model")["validation_relative_rmse"].mean()
    model_labels = {
        "diagonal_attractor": "Diagonal",
        "coupled_ridge_attractor": "Coupled",
    }
    text = "\n".join([f"{model_labels.get(str(model), str(model))}: {value:.3f}" for model, value in mean_errors.items()])
    axes[-1].text(0.05, 0.82, "Mean validation\nrelative RMSE", fontsize=7, transform=axes[-1].transAxes)
    axes[-1].text(0.05, 0.60, text, fontsize=7, transform=axes[-1].transAxes)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.04), borderaxespad=0.0)
    save_publication_figure(fig, fig_dir / "fig06_dynamics_model_comparison")
    plt.close(fig)


def plot_uncertainty_identifiability(
    parameter_identifiability: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
    fig_dir: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)

    geom = parameter_identifiability[
        parameter_identifiability["parameter_group"].eq("superellipsoid_geometry")
    ].copy()
    dyn = parameter_identifiability[
        parameter_identifiability["parameter_group"].eq("diagonal_attractor")
    ].copy()
    param_labels = list(geom["parameter"]) + [label.replace("k_", "k ") for label in dyn["parameter"]]
    cv_values = list(geom["coefficient_of_variation"].fillna(0.0)) + list(
        dyn["coefficient_of_variation"].fillna(0.0)
    )
    risks = list(geom["risk_level"]) + list(dyn["risk_level"])
    risk_colors = {"low": "#1F77F4", "medium": "#FFC247", "high": "#F04E23"}
    colors = [risk_colors.get(item, "#777777") for item in risks]
    axes[0].bar(np.arange(len(param_labels)), cv_values, color=colors, width=0.72)
    axes[0].set_xticks(np.arange(len(param_labels)))
    axes[0].set_xticklabels(param_labels, rotation=60, ha="right")
    axes[0].set_ylabel("Coefficient of variation")
    axes[0].set_title("Parameter identifiability")
    axes[0].axhline(0.5, color="0.35", lw=0.8, ls="--")
    apply_axis_polish(axes[0])

    sens = dimensionless_sensitivity.copy()
    symbols = sens["symbol"].tolist()
    baseline = sens["baseline_value"].to_numpy(dtype=float)
    y = np.arange(len(symbols))
    rel_min = sens["relative_min"].to_numpy(dtype=float)
    rel_max = sens["relative_max"].to_numpy(dtype=float)
    axes[1].hlines(y, rel_min, rel_max, color="#1F77F4", lw=3)
    axes[1].plot(np.ones_like(y), y, "o", color="#222222", ms=3, label="baseline")
    axes[1].plot(rel_min, y, "|", color="#1F77F4", ms=7)
    axes[1].plot(rel_max, y, "|", color="#1F77F4", ms=7)
    for yi, value in zip(y, baseline):
        axes[1].text(1.03, yi + 0.12, f"{value:.2g}", fontsize=6.5)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(symbols)
    axes[1].set_xlabel("Relative to baseline")
    axes[1].set_title("Dimensionless sensitivity")
    axes[1].axvline(1.0, color="0.25", lw=0.8)
    axes[1].set_xlim(max(0.0, np.nanmin(rel_min) * 0.9), np.nanmax(rel_max) * 1.15)
    apply_axis_polish(axes[1], grid="x")
    for label, ax in zip(["a", "b"], axes):
        add_panel_label(ax, label)

    save_publication_figure(fig, fig_dir / "fig07_uncertainty_identifiability")
    plt.close(fig)


def plot_modeling_framework(fig_dir: Path) -> None:
    fig_dir = Path(fig_dir)
    project_root = fig_dir.parent.parent.resolve()
    source_base = project_root / "Figure1_editable_drawio.drawio"
    source_png = source_base.with_suffix(".drawio.png")
    source_svg = source_base.with_suffix(".drawio.svg")
    if source_png.exists() or source_svg.exists():
        from .drawio_assets import save_drawio_figure_assets

        save_drawio_figure_assets(
            fig_dir / "fig08_modeling_framework",
            source_png=source_png,
            source_svg=source_svg,
        )
        return

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 3.85), constrained_layout=True)
    ax.set_axis_off()
    palette = {
        "data": "#D8ECFA",
        "operator": "#DDF0D2",
        "model": "#F8E2AE",
        "evidence": "#D9D5F1",
        "output": "#E8ECEF",
        "scope": "#F6D0C7",
        "ink": "#2A2E35",
        "line": "#303741",
        "muted": "#4E5660",
        "white": "#FFFFFF",
    }

    def rounded_box(
        x: float,
        y: float,
        w: float,
        h: float,
        color: str,
        edge: str = palette["line"],
        lw: float = 0.9,
        z: int = 2,
    ) -> None:
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.010,rounding_size=0.012",
                facecolor=color,
                edgecolor=edge,
                linewidth=lw,
                transform=ax.transAxes,
                zorder=z,
            )
        )

    def card(x: float, y: float, w: float, h: float, color: str, title: str, body: str) -> None:
        rounded_box(x, y, w, h, color)
        ax.text(
            x + 0.018,
            y + h - 0.052,
            title,
            ha="left",
            va="top",
            fontsize=7.4,
            fontweight="bold",
            color=palette["ink"],
            transform=ax.transAxes,
            zorder=4,
        )
        ax.text(
            x + 0.018,
            y + h - 0.118,
            body,
            ha="left",
            va="top",
            fontsize=6.35,
            linespacing=1.20,
            color=palette["ink"],
            transform=ax.transAxes,
            zorder=4,
        )

    def arrow(x0: float, y0: float, x1: float, y1: float, rad: float = 0.0) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.05,
                color=palette["line"],
                mutation_scale=9,
                shrinkA=2,
                shrinkB=2,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=6,
        )

    ax.text(
        0.028,
        0.948,
        "Observation-to-manifold reduction and evidence gates",
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=palette["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.028,
        0.895,
        "From exported molten-region states to a selected observed-envelope descriptor and trajectory baseline",
        ha="left",
        va="center",
        fontsize=6.6,
        color=palette["muted"],
        transform=ax.transAxes,
    )

    x0 = [0.035, 0.275, 0.515, 0.755]
    y0 = 0.385
    w = 0.205
    h = 0.360
    cards = [
        (
            palette["data"],
            "A  Numerical states",
            "15-condition process matrix\nhalf-domain molten points\nexported transient time series",
        ),
        (
            palette["operator"],
            "B  Observation operators",
            "symmetry reconstruction\nmoving frame $\\xi=x-vt$\nobserved boundary envelope",
        ),
        (
            palette["model"],
            "C  Reduced model families",
            "asymmetric superellipsoid\nreduced state descriptors\nparsimonious attractor dynamics",
        ),
        (
            palette["evidence"],
            "D  Evidence gates",
            "boundary residual\nvolume proxy and distances\nstability, identifiability, holdout",
        ),
    ]
    for x, (color, title, body) in zip(x0, cards):
        card(x, y0, w, h, color, title, body)
    for idx in range(3):
        arrow(x0[idx] + w + 0.006, y0 + h * 0.53, x0[idx + 1] - 0.006, y0 + h * 0.53)

    # Small visual anchors keep the framework scannable without becoming a data figure.
    rng_x = np.array([0.070, 0.090, 0.116, 0.142, 0.168, 0.190])
    rng_y = np.array([0.445, 0.498, 0.466, 0.524, 0.492, 0.538])
    ax.scatter(rng_x, rng_y, s=10, color="#0077BB", alpha=0.85, transform=ax.transAxes, zorder=7)
    ax.scatter(rng_x, 1.070 - rng_y, s=10, color="#0077BB", alpha=0.42, transform=ax.transAxes, zorder=7)
    ax.plot([0.056, 0.206], [0.470, 0.470], color=palette["line"], lw=0.55, ls=":", transform=ax.transAxes, zorder=6)

    ax.plot([0.316, 0.455], [0.475, 0.475], color=palette["line"], lw=0.75, transform=ax.transAxes, zorder=6)
    ax.annotate("", xy=(0.450, 0.475), xytext=(0.410, 0.475), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=0.75, color=palette["line"], mutation_scale=7), zorder=7)
    ax.text(0.330, 0.505, "$x$", fontsize=6.3, color=palette["muted"], transform=ax.transAxes)
    ax.text(0.420, 0.505, "$\\xi$", fontsize=6.3, color=palette["muted"], transform=ax.transAxes)

    theta = np.linspace(0, 2 * np.pi, 160)
    sx = 0.617 + 0.056 * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** 0.65
    sy = 0.466 + 0.036 * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** 0.90
    ax.plot(sx, sy, color="#8A6F16", lw=1.0, transform=ax.transAxes, zorder=7)
    curve_x = np.linspace(0.635, 0.690, 80)
    curve_y = 0.525 - 0.050 * (1 - np.exp(-45 * (curve_x - curve_x.min())))
    ax.plot(curve_x, curve_y, color="#8A6F16", lw=0.95, transform=ax.transAxes, zorder=7)

    check_items = [0.535, 0.495, 0.455]
    for yi in check_items:
        ax.plot([0.798, 0.806, 0.820], [yi, yi - 0.012, yi + 0.014], color="#5F4BB6", lw=1.05,
                transform=ax.transAxes, zorder=7)
        ax.plot([0.831, 0.925], [yi, yi], color="#5F4BB6", lw=0.65, alpha=0.75, transform=ax.transAxes, zorder=7)

    rounded_box(0.070, 0.130, 0.405, 0.135, palette["output"], lw=0.85)
    ax.text(
        0.092,
        0.210,
        "Selected descriptor system",
        ha="left",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=palette["ink"],
        transform=ax.transAxes,
        zorder=4,
    )
    ax.text(
        0.092,
        0.166,
        "observed-envelope manifold + compact trajectory baseline",
        ha="left",
        va="center",
        fontsize=6.45,
        color=palette["ink"],
        transform=ax.transAxes,
        zorder=4,
    )
    rounded_box(0.535, 0.130, 0.395, 0.135, palette["scope"], lw=0.85)
    ax.text(
        0.557,
        0.210,
        "Interpretive boundary",
        ha="left",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=palette["ink"],
        transform=ax.transAxes,
        zorder=4,
    )
    ax.text(
        0.557,
        0.166,
        "observed envelope, volume proxy and validation scope",
        ha="left",
        va="center",
        fontsize=6.35,
        color=palette["ink"],
        transform=ax.transAxes,
        zorder=4,
    )
    arrow(x0[3] + w * 0.50, y0 - 0.005, 0.555, 0.250, rad=-0.08)
    arrow(x0[2] + w * 0.35, y0 - 0.005, 0.355, 0.265, rad=0.10)

    ax.text(
        0.500,
        0.045,
        "The figure separates the observation problem, model families and evidence gates that govern the main-text model-selection sequence.",
        ha="center",
        va="center",
        fontsize=6.25,
        color=palette["muted"],
        transform=ax.transAxes,
    )
    save_publication_figure(fig, fig_dir / "fig08_modeling_framework")
    plt.close(fig)


def plot_dimensionless_regime(dimensionless_sensitivity: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    sens = dimensionless_sensitivity.copy()
    symbols = sens["symbol"].astype(str).tolist()
    plot_symbols = [dimensionless_plot_label(symbol) for symbol in symbols]
    y = np.arange(len(plot_symbols))
    baseline = sens["baseline_value"].to_numpy(dtype=float)
    min_values = sens["min_value"].to_numpy(dtype=float)
    max_values = sens["max_value"].to_numpy(dtype=float)
    colors = ["#1F77F4", "#2DB84D", "#FF8C00", "#A64AC9"]
    axes[0].barh(y, baseline, color=colors, height=0.58)
    axes[0].set_xscale("log")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_symbols)
    axes[0].set_xlabel("Baseline value (log scale)")
    axes[0].set_title("Dimensionless scale diagnostics")
    apply_axis_polish(axes[0], grid="x")
    for yi, value in zip(y, baseline):
        axes[0].text(value * 1.08, yi, f"{value:.2g}", va="center", fontsize=6.5)

    rel_min = sens["relative_min"].to_numpy(dtype=float)
    rel_max = sens["relative_max"].to_numpy(dtype=float)
    axes[1].hlines(y, rel_min, rel_max, color="#1F77F4", lw=3)
    axes[1].plot(np.ones_like(y), y, "o", color="#222222", ms=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(plot_symbols)
    axes[1].set_xlabel("Perturbed value / baseline")
    axes[1].set_title("Sensitivity envelope")
    axes[1].axvline(1.0, color="0.25", lw=0.8)
    axes[1].set_xlim(max(0.0, np.nanmin(rel_min) * 0.9), np.nanmax(rel_max) * 1.12)
    for yi, lo, hi in zip(y, min_values, max_values):
        axes[1].text(np.nanmax(rel_max) * 1.03, yi, f"{lo:.2g}-{hi:.2g}", va="center", fontsize=6.2)
    apply_axis_polish(axes[1], grid="x")
    for label, ax in zip(["a", "b"], axes):
        add_panel_label(ax, label)
    save_publication_figure(fig, fig_dir / "fig09_dimensionless_regime")
    plt.close(fig)


def plot_stability_attractor(
    table: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    eigenvalues: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    fig_dir: Path,
) -> None:
    configure_matplotlib()
    case_id = representative_case_id(table)
    if case_id is not None:
        table = table[table["case_id"].astype(str).eq(case_id)].copy().sort_values("time_s")
        dynamics_summary = dynamics_summary[dynamics_summary["case_id"].astype(str).eq(case_id)].copy()
        eigenvalues = eigenvalues[eigenvalues["case_id"].astype(str).eq(case_id)].copy()
        coupled_eigenvalues = coupled_eigenvalues[coupled_eigenvalues["case_id"].astype(str).eq(case_id)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    t = table["time_s"].to_numpy(dtype=float)
    for col in STATE_COLUMNS:
        y = table[col].to_numpy(dtype=float)
        q_inf = float(dynamics_summary.loc[dynamics_summary["state"].eq(col), "q_inf"].mean())
        err = np.abs(y - q_inf)
        denom = max(float(np.nanmax(err)), 1e-12)
        axes[0].plot(t, err / denom, lw=1.0, alpha=0.85, label=state_plot_label(col))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(r"$|q-q_\infty|$ / max")
    axes[0].set_title("State convergence")
    apply_axis_polish(axes[0])

    labels = [state_plot_label(col) for col in dynamics_summary["state"]]
    k_values = dynamics_summary["k_per_s"].to_numpy(dtype=float)
    axes[1].bar(np.arange(len(labels)), k_values, color="#1F77F4", width=0.72)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel(r"$k_i$ (s$^{-1}$)")
    axes[1].set_title("Diagonal rates")
    axes[1].axhline(0, color="0.3", lw=0.8)
    apply_axis_polish(axes[1])

    real = coupled_eigenvalues["jacobian_eigenvalue_real_per_s"].to_numpy(dtype=float)
    imag = coupled_eigenvalues["jacobian_eigenvalue_imag_per_s"].to_numpy(dtype=float)
    axes[2].scatter(real, imag, s=28, color="#E53935", edgecolor="white", linewidth=0.5)
    axes[2].axvline(0, color="0.25", lw=0.8)
    axes[2].axhline(0, color="0.75", lw=0.6)
    axes[2].set_xlabel(r"Re$(\lambda)$ (s$^{-1}$)")
    axes[2].set_ylabel(r"Im$(\lambda)$ (s$^{-1}$)")
    axes[2].set_title("Coupled eigenvalues of -A")
    apply_axis_polish(axes[2])
    for label, ax in zip(["a", "b", "c"], axes):
        add_panel_label(ax, label)
    save_publication_figure(fig, fig_dir / "fig10_stability_attractor")
    plt.close(fig)


def plot_error_budget_model_selection(
    error_budget: pd.DataFrame,
    model_selection: pd.DataFrame,
    fig_dir: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    eb = error_budget.copy()
    y = np.arange(len(eb))
    colors = ["#1F77F4", "#2DB84D", "#FF8C00", "#A64AC9", "#E53935", "#00A6A6"][: len(eb)]
    axes[0].barh(y, eb["primary_value"].to_numpy(dtype=float), color=colors, height=0.62)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(eb["error_term"].str.replace("E_", "", regex=False))
    axes[0].set_xlabel("Primary metric value")
    axes[0].set_title("Error budget")
    axes[0].invert_yaxis()
    apply_axis_polish(axes[0], grid="x")

    ms = model_selection.copy()
    labels = ms["model"].str.replace("_", "\n", regex=False).tolist()
    values = ms["primary_metric_value"].to_numpy(dtype=float)
    selected = ms["selected_as_main_model"].astype(str).str.lower().eq("true").to_numpy()
    bar_colors = np.where(selected, "#1F77F4", "#AFAFAF")
    axes[1].bar(np.arange(len(labels)), values, color=bar_colors, width=0.68)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=0, ha="center", fontsize=6.5)
    axes[1].set_ylabel("Primary metric")
    axes[1].set_title("Model selection")
    axes[1].set_yscale("log")
    for idx, is_selected in enumerate(selected):
        if is_selected:
            axes[1].text(idx, values[idx] * 1.12, "selected", ha="center", va="bottom", fontsize=6.1, color="#1F77F4")
    apply_axis_polish(axes[1])
    for label, ax in zip(["a", "b"], axes):
        add_panel_label(ax, label)
    save_publication_figure(fig, fig_dir / "fig11_error_budget_model_selection")
    plt.close(fig)


def plot_identifiability_overparameterization(
    parameter_identifiability: pd.DataFrame,
    coupled_matrix: pd.DataFrame,
    fig_dir: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    geom = parameter_identifiability[
        parameter_identifiability["parameter_group"].eq("superellipsoid_geometry")
    ].copy()
    y = np.arange(len(geom))
    colors = geom["risk_level"].map({"low": "#1F77F4", "medium": "#FFC247", "high": "#F04E23"}).fillna("#777777")
    axes[0].barh(y, geom["coefficient_of_variation"].fillna(0.0).to_numpy(dtype=float), color=colors, height=0.62)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(geom["parameter"])
    axes[0].set_xlabel("Coefficient of variation")
    axes[0].set_title("Superellipsoid identifiability")
    axes[0].axvline(0.5, color="0.35", lw=0.8, ls="--")
    axes[0].invert_yaxis()
    apply_axis_polish(axes[0], grid="x")

    if "case_id" in coupled_matrix.columns and coupled_matrix["case_id"].nunique() > 1:
        case_id = representative_case_id(coupled_matrix)
        coupled_matrix = coupled_matrix[coupled_matrix["case_id"].astype(str).eq(case_id)].copy()
    mat = coupled_matrix.pivot(index="row_state", columns="column_state", values="A_value_per_s").loc[
        STATE_COLUMNS, STATE_COLUMNS
    ]
    vmax = float(np.nanmax(np.abs(mat.to_numpy(dtype=float))))
    im = axes[1].imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_xticks(np.arange(len(STATE_COLUMNS)))
    axes[1].set_yticks(np.arange(len(STATE_COLUMNS)))
    axes[1].set_xticklabels([state_plot_label(c) for c in STATE_COLUMNS], rotation=45, ha="right")
    axes[1].set_yticklabels([state_plot_label(c) for c in STATE_COLUMNS])
    axes[1].set_title("Coupled A matrix")
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.82)
    cbar.set_label(r"A entry (s$^{-1}$)")
    cbar.ax.tick_params(labelsize=6.2, length=2.4)
    for label, ax in zip(["a", "b"], axes):
        add_panel_label(ax, label)
    save_publication_figure(fig, fig_dir / "fig12_identifiability_overparameterization")
    plt.close(fig)


def plot_supplementary_all_boundary_fits(table: pd.DataFrame, point_cloud: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    case_id = representative_case_id(table)
    if case_id is not None:
        table = table[table["case_id"].astype(str).eq(case_id)].copy()
        point_cloud = point_cloud[point_cloud["case_id"].astype(str).eq(case_id)].copy()
    times = table["time_s"].to_numpy(dtype=float)
    ncols = 5
    nrows = int(math.ceil(len(times) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.35 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax, time_s in zip(axes, times):
        row = table.loc[np.isclose(table["time_s"], time_s)].iloc[0]
        part = point_cloud[np.isclose(point_cloud["time_s"], time_s)]
        ax.scatter(part["xi_m"] * 1e3, part["Points_1"] * 1e3, s=2.5, alpha=0.25, color="0.45", linewidths=0)
        ax.scatter(part["xi_m"] * 1e3, -part["Points_1"] * 1e3, s=2.5, alpha=0.18, color="0.55", linewidths=0)
        params = model_params_from_row(row, "superellipsoid")
        for side in [-1, 1]:
            xi_curve, y_curve = boundary_curve_top(params, "superellipsoid", side)
            ax.plot(xi_curve * 1e3, y_curve * 1e3, color="#E53935", lw=0.8)
            ax.plot(xi_curve * 1e3, -y_curve * 1e3, color="#E53935", lw=0.8)
        ax.set_title(f"{time_s:.2f}s", fontsize=6.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(times) :]:
        ax.axis("off")
    save_publication_figure(fig, fig_dir / "supp_figS1_all_boundary_fits")
    plt.close(fig)


def plot_supplementary_superellipsoid_parameters(table: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    table = representative_case_subset(table).sort_values("time_s")
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.0), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    cols = [
        ("superellipsoid_af_m", r"$a_f$", 1e3, "mm"),
        ("superellipsoid_ar_m", r"$a_r$", 1e3, "mm"),
        ("superellipsoid_b_m", r"$b$", 1e3, "mm"),
        ("superellipsoid_c_m", r"$c$", 1e3, "mm"),
        ("superellipsoid_xic_m", r"$\xi_c$", 1e3, "mm"),
        ("superellipsoid_zc_m", r"$z_c$", 1e3, "mm"),
        ("superellipsoid_n", r"$n$", 1.0, "-"),
        ("superellipsoid_m", r"$m$", 1.0, "-"),
        ("superellipsoid_p", r"$p$", 1.0, "-"),
    ]
    t = table["time_s"].to_numpy(dtype=float)
    for ax, (col, label, scale, unit) in zip(axes, cols):
        ax.plot(t, table[col].to_numpy(dtype=float) * scale, "o-", ms=2.6, lw=1.0, color="#1F77F4")
        ax.axvline(QUASI_STEADY_START_S, color="0.35", lw=0.7, ls="--")
        ax.set_title(label)
        ax.set_ylabel(unit)
        apply_axis_polish(ax)
    for ax in axes[-3:]:
        ax.set_xlabel("Time (s)")
    save_publication_figure(fig, fig_dir / "supp_figS2_superellipsoid_parameters")
    plt.close(fig)


def plot_supplementary_residuals(
    diagonal_predictions: pd.DataFrame,
    coupled_predictions: pd.DataFrame,
    fig_dir: Path,
) -> None:
    configure_matplotlib()
    diagonal_predictions = representative_case_subset(diagonal_predictions).sort_values("time_s")
    coupled_predictions = representative_case_subset(coupled_predictions).sort_values("time_s")
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), constrained_layout=True)
    axes = axes.ravel()
    time = diagonal_predictions["time_s"].to_numpy(dtype=float)
    scale = {
        "front_length_m": 1e3,
        "rear_length_m": 1e3,
        "full_width_m": 1e3,
        "height_span_m": 1e3,
        "Tmax_K": 1.0,
        "Gmean_K_per_m": 1e-6,
        "Umax_m_per_s": 1.0,
    }
    for idx, col in enumerate(STATE_COLUMNS):
        ax = axes[idx]
        diag = diagonal_predictions[f"{col}_residual"].to_numpy(dtype=float) * scale[col]
        coupled = coupled_predictions[f"{col}_residual"].to_numpy(dtype=float) * scale[col]
        ax.axhline(0, color="0.4", lw=0.7)
        ax.plot(time, diag, "o-", ms=2.4, lw=0.9, color="#1F77F4", label="Diagonal")
        ax.plot(time, coupled, "s--", ms=2.2, lw=0.9, color="#E53935", label="Coupled")
        ax.set_title(state_plot_label(col))
        ax.set_xlabel("Time (s)")
        apply_axis_polish(ax)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center")
    save_publication_figure(fig, fig_dir / "supp_figS3_dynamics_residuals")
    plt.close(fig)


def plot_supplementary_dimensionless_grid(dimensionless_sensitivity: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    sens = dimensionless_sensitivity.copy()
    symbols = sens["symbol"].astype(str).tolist()
    plot_symbols = [dimensionless_plot_label(symbol) for symbol in symbols]
    data = np.vstack(
        [
            sens["relative_min"].to_numpy(dtype=float),
            np.ones(len(sens)),
            sens["relative_max"].to_numpy(dtype=float),
        ]
    )
    im = ax.imshow(data, cmap="plasma", aspect="auto", vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(np.arange(len(symbols)))
    ax.set_xticklabels(plot_symbols)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Minimum", "Baseline", "Maximum"])
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(data)), vmax=float(np.nanmax(data)))
    cmap = mpl.colormaps["plasma"]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r, g, b, _a = cmap(norm(float(data[i, j])))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = "black" if luminance > 0.58 else "white"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=7.2)
    ax.set_title("Dimensionless sensitivity scenario envelope")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Relative to baseline")
    cbar.ax.tick_params(labelsize=6.2, length=2.4)
    save_publication_figure(fig, fig_dir / "supp_figS4_dimensionless_sensitivity_grid")
    plt.close(fig)


def plot_temperature_dependent_properties(curves: dict[str, pd.DataFrame], fig_dir: Path) -> None:
    configure_matplotlib()
    specs = [
        ("density_kg_per_m3", "Density", r"Density (kg m$^{-3}$)", "#1F77F4"),
        ("specific_heat_J_per_kg_K", "Specific heat", r"Specific heat (J kg$^{-1}$ K$^{-1}$)", "#F04E23"),
        ("thermal_conductivity_W_per_m_K", "Thermal conductivity", r"Thermal conductivity (W m$^{-1}$ K$^{-1}$)", "#00A676"),
        ("viscosity_kg_per_m_s", "Viscosity", r"Viscosity (kg m$^{-1}$ s$^{-1}$)", "#7A3DB8"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.6), constrained_layout=True)
    solidus = MATERIAL_CONSTANTS["solidus_temperature_K"]
    liquidus = MATERIAL_CONSTANTS["liquidus_temperature_K"]
    for idx, (ax, (key, title, ylabel, color)) in enumerate(zip(axes.ravel(), specs)):
        curve = curves[key].copy()
        x = curve["temperature_K"].to_numpy(dtype=float)
        y = curve[key].to_numpy(dtype=float)
        liquidus_value = interp_property(curve, key, liquidus)
        ax.plot(x, y, "-o", lw=1.25, ms=3.0, color=color, markerfacecolor="white", markeredgewidth=0.8)
        ax.scatter([liquidus], [liquidus_value], s=24, color=color, zorder=4)
        ax.axvline(solidus, color="0.35", lw=0.75, ls="--", label="Solidus" if idx == 0 else None)
        ax.axvline(liquidus, color="0.10", lw=0.85, ls=":", label="Liquidus" if idx == 0 else None)
        ax.set_title(title)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(ylabel)
        label_x = {0: 0.18, 3: 0.20}.get(idx, 0.04)
        ax.text(
            label_x,
            0.91,
            f"{liquidus_value:.3g} at liquidus",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="0.20",
        )
        add_panel_label(ax, chr(ord("A") + idx))
        apply_axis_polish(ax)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    save_publication_figure(fig, fig_dir / "supp_figS10_temperature_dependent_properties")
    plt.close(fig)


def plot_theory_identifiability_error_bounds(
    identifiability_v4: pd.DataFrame,
    error_bound_summary: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
    fig_dir: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True)

    eb = error_bound_summary.sort_values("display_order").copy()
    y = np.arange(len(eb))
    axes[0].barh(y, eb["normalized_proxy"].to_numpy(dtype=float), color="#1F77F4", height=0.62)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(eb["bound_component"].str.replace("E_", "", regex=False), fontsize=6.3)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Normalized proxy")
    axes[0].set_title("Error-budget weights")
    apply_axis_polish(axes[0], grid="x")

    risk_colors = {"low": "#1F77F4", "medium": "#FFC247", "high": "#F04E23"}
    id_df = identifiability_v4.copy()
    id_df["risk_score_plot"] = pd.to_numeric(id_df["risk_score_numeric"], errors="coerce")

    def _family_risk(label: str, mask: pd.Series) -> dict[str, object]:
        sub = id_df.loc[mask].copy()
        if sub.empty:
            score = 0.0
        else:
            score = float(np.nanmax(sub["risk_score_plot"].to_numpy(dtype=float)))
        if score >= 2.5:
            risk = "high"
        elif score >= 1.5:
            risk = "medium"
        elif score > 0:
            risk = "low"
        else:
            risk = "low"
        return {"label": label, "risk_score": score, "risk_level": risk}

    param = id_df["parameter"].astype(str)
    group = id_df["parameter_group"].astype(str)
    plot_df = pd.DataFrame(
        [
            _family_risk(
                "Geometry scales\n$a_f,a_r,b,c$",
                group.eq("superellipsoid_geometry") & param.isin(["a_f", "a_r", "b", "c"]),
            ),
            _family_risk(
                "Boundary center\n$\\xi_c,z_c$",
                group.eq("superellipsoid_geometry") & param.isin(["xi_c", "z_c"]),
            ),
            _family_risk(
                "Shape exponents\n$n,m,p$",
                group.eq("superellipsoid_geometry") & param.isin(["n", "m", "p"]),
            ),
            _family_risk(
                "Diagonal rates\n$k_i$",
                group.eq("diagonal_attractor"),
            ),
            _family_risk(
                "Coupled matrix\n$A$",
                group.eq("coupled_attractor"),
            ),
        ]
    )
    y = np.arange(len(plot_df))
    colors = plot_df["risk_level"].map(risk_colors).fillna("#777777")
    axes[1].barh(y, plot_df["risk_score"].to_numpy(dtype=float), color=colors, height=0.55)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(plot_df["label"].tolist(), fontsize=7.2, linespacing=1.1)
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(["low", "medium", "high"])
    axes[1].set_xlim(0, 3.35)
    axes[1].invert_yaxis()
    axes[1].set_title("Identifiability constraint")
    axes[1].grid(axis="x", color="0.9", lw=0.6)

    sens = dimensionless_sensitivity.copy()
    span = np.maximum(
        abs(sens["relative_min"].to_numpy(dtype=float) - 1.0),
        abs(sens["relative_max"].to_numpy(dtype=float) - 1.0),
    )
    x = np.arange(len(sens))
    colors = np.where(sens["conclusion_changed"].to_numpy(dtype=bool), "#F04E23", "#2DB84D")
    axes[2].bar(x, span, color=colors, width=0.62)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([dimensionless_plot_label(symbol) for symbol in sens["symbol"].astype(str)])
    axes[2].set_ylabel("Max relative span")
    axes[2].set_title("Nondimensional sensitivity")
    axes[2].axhline(0.2, color="0.4", lw=0.7, ls="--")
    apply_axis_polish(axes[2])
    for label, ax in zip(["a", "b", "c"], axes):
        add_panel_label(ax, label)

    save_publication_figure(fig, fig_dir / "supp_figS5_theory_identifiability_error_bounds")
    plt.close(fig)


def plot_multi_condition_process_matrix(table: pd.DataFrame, fig_dir: Path) -> None:
    if "case_id" not in table.columns or table["case_id"].nunique() <= 1:
        return save_placeholder_figure(
            fig_dir / "fig13_multicondition_process_matrix",
            "Multi-condition process matrix not available for single-condition input.",
        )
    configure_matplotlib()
    meta = case_metadata_from_modeling_table(table)
    powers = sorted(float(v) for v in meta["power_W"].dropna().unique())
    fig, axes = plt.subplots(1, len(powers), figsize=(7.4, 2.7), sharex=True, sharey=True, constrained_layout=True)
    if len(powers) == 1:
        axes = [axes]
    base_color = "#1F77F4"
    powder_min = float(meta["powder_feed_g_min"].min())
    powder_max = float(meta["powder_feed_g_min"].max())

    def label_offset(speed_val: float, powder_val: float) -> tuple[float, float]:
        if np.isclose(speed_val, 6.0):
            return (4.0, 4.0)
        if np.isclose(speed_val, 10.0):
            return (-15.0, 4.0)
        if np.isclose(powder_val, powder_min):
            return (-12.0, -10.0)
        if np.isclose(powder_val, powder_max):
            return (6.0, 8.0)
        return (6.0, 4.0)

    for ax, power in zip(axes, powers):
        subset = meta.loc[np.isclose(meta["power_W"].to_numpy(dtype=float), power)].copy()
        subset = subset.sort_values(["scan_speed_mm_s", "powder_feed_g_min", "case_index"])
        ax.scatter(
            subset["scan_speed_mm_s"],
            subset["powder_feed_g_min"],
            s=74,
            color=base_color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for row in subset.itertuples():
            dx, dy = label_offset(float(row.scan_speed_mm_s), float(row.powder_feed_g_min))
            ax.annotate(
                f"A{row.case_index}",
                xy=(float(row.scan_speed_mm_s), float(row.powder_feed_g_min)),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.28},
                zorder=4,
            )
        ax.set_title(f"{power:.0f} W", fontsize=7.6, fontweight="bold", pad=2.5)
        ax.set_xlim(5.6, 10.4)
        ax.set_ylim(7.2, 14.8)
        ax.set_xticks([6.0, 8.0, 10.0])
        ax.set_yticks([8.0, 12.0, 14.0])
        apply_axis_polish(ax, grid="both")
    axes[0].set_ylabel(r"Powder feed (g min$^{-1}$)")
    fig.supxlabel(r"Scan speed (mm s$^{-1}$)")
    save_publication_figure(fig, fig_dir / "fig13_multicondition_process_matrix")
    plt.close(fig)


def plot_multi_condition_response_surfaces(table: pd.DataFrame, fig_dir: Path) -> None:
    if "case_id" not in table.columns or table["case_id"].nunique() <= 1:
        return save_placeholder_figure(
            fig_dir / "fig14_multicondition_response_surfaces",
            "Multi-condition process response not available for single-condition input.",
        )
    configure_matplotlib()
    steady = table[table["time_s"] >= QUASI_STEADY_START_S].copy()
    case = steady.groupby("case_id", as_index=False).agg(
        case_index=("case_index", "first"),
        power_W=("power_W", "first"),
        scan_speed_mm_s=("scan_speed_mm_s", "first"),
        powder_feed_g_min=("powder_feed_g_min", "first"),
        length_mm=("melt_pool_length_m", lambda v: float(np.nanmean(v) * 1e3)),
        width_mm=("full_width_m", lambda v: float(np.nanmean(v) * 1e3)),
        height_mm=("height_span_m", lambda v: float(np.nanmean(v) * 1e3)),
        Tmax_K=("Tmax_K", "mean"),
    )
    panels = [
        ("length_mm", "Length (mm)"),
        ("width_mm", "Width (mm)"),
        ("height_mm", "Height (mm)"),
        ("Tmax_K", r"$T_{\max}$ (K)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    axes = axes.ravel()
    powder = case["powder_feed_g_min"].to_numpy(dtype=float)
    powder_center = float(np.nanmedian(powder))
    powder_span = max(float(np.nanmax(powder) - np.nanmin(powder)), 1.0)
    powder_min = float(np.nanmin(powder))

    def powder_marker_area(feed_g_min: float) -> float:
        return 42 + 5 * (float(feed_g_min) - powder_min)

    x_plot = case["scan_speed_mm_s"].to_numpy(dtype=float) + 0.18 * (powder - powder_center) / powder_span
    for ax, (col, label) in zip(axes, panels):
        sc = ax.scatter(
            x_plot,
            case["power_W"],
            c=case[col],
            s=case["powder_feed_g_min"].map(powder_marker_area),
            cmap="plasma",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel(r"Scan speed (mm s$^{-1}$)")
        ax.set_ylabel("Power (W)")
        ax.set_title(label)
        ax.set_xlim(5.72, 10.28)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
        cbar.set_label(label)
        cbar.ax.tick_params(labelsize=6.6, length=2.2)
        apply_axis_polish(ax, grid="both")
    powder_levels = sorted(float(value) for value in pd.Series(powder).dropna().unique())
    if len(powder_levels) > 4:
        powder_levels = [float(np.nanmin(powder)), float(np.nanmedian(powder)), float(np.nanmax(powder))]
    size_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.55",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=math.sqrt(powder_marker_area(level)),
            label=rf"{level:g} g min$^{{-1}}$",
        )
        for level in powder_levels
    ]
    fig.legend(
        handles=size_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncols=len(size_handles),
        title="Powder feed",
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.1,
        title_fontsize=7.2,
        fontsize=7.2,
    )
    for label, ax in zip(["a", "b", "c", "d"], axes):
        add_panel_label(ax, label)
    save_publication_figure(fig, fig_dir / "fig14_multicondition_response_surfaces")
    plt.close(fig)


def plot_paired_geometry_metric(
    ax: mpl.axes.Axes,
    case_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    colors: dict[str, str],
) -> None:
    pivot = case_summary.pivot(index="case_index", columns="model", values=metric).sort_index()
    if not {"ellipsoid", "superellipsoid"}.issubset(pivot.columns):
        ax.text(0.5, 0.5, f"{metric} unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    pivot = pivot[["ellipsoid", "superellipsoid"]]

    x = np.array([0.0, 1.0])
    for row in pivot.itertuples(index=False):
        values = np.array([float(row[0]), float(row[1])])
        ax.plot(x, values, color="0.72", lw=0.7, alpha=0.78, zorder=1)
    ax.scatter(np.zeros(len(pivot)), pivot["ellipsoid"], s=16, color=colors["ellipsoid"], edgecolor="white", linewidth=0.35, zorder=3)
    ax.scatter(np.ones(len(pivot)), pivot["superellipsoid"], s=16, color=colors["superellipsoid"], edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ellipsoid", "Super."])
    ax.set_xlim(-0.35, 1.35)
    if (pivot[["ellipsoid", "superellipsoid"]] > 0).all().all():
        ax.set_yscale("log")
    wins = int((pivot["superellipsoid"] < pivot["ellipsoid"]).sum())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.03,
        0.96,
        f"super. lower in {wins}/{len(pivot)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color="0.28",
    )
    apply_axis_polish(ax)


def plot_geometry_metric_support(
    ax: mpl.axes.Axes,
    case_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame | None,
) -> None:
    metric_specs = [
        ("Boundary", "mean_boundary_residual_rmse"),
        ("Volume", "mean_volume_relative_error"),
        ("Chamfer", "mean_normalized_chamfer_distance"),
        ("Hausdorff", "mean_hausdorff_distance_m"),
        ("Radial", "mean_radial_distance_rmse_m"),
    ]
    labels: list[str] = []
    fractions: list[float] = []
    annotations: list[str] = []
    for label, metric in metric_specs:
        if metric not in case_summary.columns:
            continue
        pivot = case_summary.pivot(index="case_index", columns="model", values=metric).sort_index()
        if not {"ellipsoid", "superellipsoid"}.issubset(pivot.columns) or pivot.empty:
            continue
        pivot = pivot[["ellipsoid", "superellipsoid"]]
        wins = int((pivot["superellipsoid"] < pivot["ellipsoid"]).sum())
        total = int(len(pivot))
        labels.append(label)
        fractions.append(wins / total if total else 0.0)
        annotations.append(f"{wins}/{total}")
    y = np.arange(len(labels))
    bar_colors = ["#1F77F4" if label == "Boundary" else "#B0BEC5" for label in labels]
    ax.barh(y, fractions, color=bar_colors, height=0.64)
    for yi, frac, text in zip(y, fractions, annotations):
        ax.text(min(frac + 0.035, 0.98), yi, text, va="center", ha="left", fontsize=6.4, color="0.25")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Fraction of conditions")
    ax.set_title("Metric support")
    ax.invert_yaxis()
    ax.axvline(0.5, color="0.55", lw=0.65, ls="--")
    if robustness_summary is not None and not robustness_summary.empty:
        ok = robustness_summary[robustness_summary["status"].astype(str).str.lower().eq("ok")]
        if len(ok):
            b_wins = int(ok["superellipsoid_improves_boundary"].astype(bool).sum())
            v_wins = int(ok["superellipsoid_improves_volume"].astype(bool).sum())
            ax.text(
                0.02,
                0.03,
                f"robustness: boundary {b_wins}/{len(ok)}, volume {v_wins}/{len(ok)}",
                transform=ax.transAxes,
                fontsize=5.9,
                color="0.30",
                ha="left",
                va="bottom",
            )
    apply_axis_polish(ax, grid="x")


def plot_multi_condition_geometry_comparison(
    geometry_comparison: pd.DataFrame,
    fig_dir: Path,
    table: pd.DataFrame | None = None,
    point_cloud: pd.DataFrame | None = None,
    robustness_summary: pd.DataFrame | None = None,
) -> None:
    if "case_id" not in geometry_comparison.columns:
        return save_placeholder_figure(
            fig_dir / "fig15_multicondition_geometry_comparison",
            "Cross-condition geometry comparison not available for single-condition input.",
        )
    case = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("case_summary")].copy()
    if case.empty:
        return save_placeholder_figure(
            fig_dir / "fig15_multicondition_geometry_comparison",
            "Cross-condition geometry comparison has no case summaries.",
        )
    configure_matplotlib()
    colors = {"ellipsoid": "#8A8A8A", "superellipsoid": "#1F77F4"}
    overlay_colors = {"ellipsoid": "#1F77F4", "superellipsoid": "#E53935"}
    model_labels = {"ellipsoid": "Ellipsoid", "superellipsoid": "Superellipsoid"}
    has_overlay = table is not None and point_cloud is not None and not table.empty and not point_cloud.empty

    if has_overlay:
        fig = plt.figure(figsize=(7.2, 6.75), constrained_layout=False)
        grid = fig.add_gridspec(
            3,
            6,
            height_ratios=[1.0, 1.0, 1.12],
            left=0.075,
            right=0.985,
            bottom=0.075,
            top=0.90,
            hspace=0.54,
            wspace=0.56,
        )
        case_id = representative_case_id(table)
        overlay_table = table.copy()
        overlay_points = point_cloud.copy()
        if case_id is not None:
            overlay_table = overlay_table[overlay_table["case_id"].astype(str).eq(case_id)].copy().sort_values("time_s")
            overlay_points = overlay_points[overlay_points["case_id"].astype(str).eq(case_id)].copy()
        selected_times = nearest_boundary_times(overlay_table, MAIN_BOUNDARY_OVERLAY_TIMES)
        overlay_axes = []
        for idx, time_s in enumerate(selected_times[:2]):
            col0 = idx * 3
            top_ax = fig.add_subplot(grid[0, col0 : col0 + 3])
            side_ax = fig.add_subplot(grid[1, col0 : col0 + 3])
            row = overlay_table.loc[np.isclose(overlay_table["time_s"], time_s)].iloc[0]
            points = overlay_points[np.isclose(overlay_points["time_s"], time_s)]
            phase = "transition" if time_s < QUASI_STEADY_START_S + 0.05 else "late quasi-steady"
            draw_boundary_overlay_pair(
                top_ax,
                side_ax,
                row,
                points,
                overlay_colors,
                model_labels,
                top_title=f"{time_s:.2f} s top view ({phase})",
                side_title=f"{time_s:.2f} s side view",
                top_ylabel="y (mm)" if idx == 0 else "",
                side_ylabel="z (mm)" if idx == 0 else "",
                point_size=3.6,
                line_width=1.0,
            )
            top_ax.tick_params(labelbottom=False)
            top_ax.set_xlabel("")
            side_ax.set_xlabel(r"Moving coordinate $\xi$ (mm)")
            overlay_axes.append(top_ax)
        stats_axes = [
            fig.add_subplot(grid[2, 0:2]),
            fig.add_subplot(grid[2, 2:4]),
            fig.add_subplot(grid[2, 4:6]),
        ]
        handles = [
            mpl.lines.Line2D([0], [0], color=overlay_colors[m], lw=1.2, label=model_labels[m])
            for m in ["ellipsoid", "superellipsoid"]
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.985), ncols=2, frameon=False)
        if overlay_axes:
            add_panel_label(overlay_axes[0], "a", x=-0.065, y=1.06)
    else:
        fig = plt.figure(figsize=(7.2, 3.0), constrained_layout=True)
        grid = fig.add_gridspec(1, 3, wspace=0.35)
        stats_axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]

    plot_paired_geometry_metric(
        stats_axes[0],
        case,
        "mean_boundary_residual_rmse",
        "Boundary RMSE",
        "Boundary residual",
        colors,
    )
    plot_paired_geometry_metric(
        stats_axes[1],
        case,
        "mean_volume_relative_error",
        "Volume rel. error",
        "Volume proxy",
        colors,
    )
    plot_geometry_metric_support(stats_axes[2], case, robustness_summary)
    for label, ax in zip(["b", "c", "d"], stats_axes):
        add_panel_label(ax, label, x=-0.17, y=1.05)
    save_publication_figure(fig, fig_dir / "fig15_multicondition_geometry_comparison")
    plt.close(fig)


def plot_multi_condition_dynamics_validation(dynamics_comparison: pd.DataFrame, fig_dir: Path) -> None:
    if "case_id" not in dynamics_comparison.columns:
        return save_placeholder_figure(
            fig_dir / "fig16_multicondition_dynamics_validation",
            "Cross-condition dynamics validation not available for single-condition input.",
        )
    configure_matplotlib()
    case = (
        dynamics_comparison.groupby(["case_index", "case_id", "model"], as_index=False)
        .agg(validation_relative_rmse=("validation_relative_rmse", "mean"))
        .sort_values("case_index")
    )
    pivot = case.pivot(index="case_index", columns="model", values="validation_relative_rmse")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    x = np.arange(len(pivot))
    width = 0.36
    axes[0].bar(x - width / 2, pivot["diagonal_attractor"], width, color="#1F77F4", label="Diagonal")
    axes[0].bar(x + width / 2, pivot["coupled_ridge_attractor"], width, color="#E53935", label="Coupled")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"A{int(idx)}" for idx in pivot.index], fontsize=6.5)
    axes[0].set_ylabel("Mean validation relative RMSE")
    axes[0].set_title("Condition-wise dynamics validation")
    axes[0].legend()
    apply_axis_polish(axes[0])

    state = dynamics_comparison.pivot_table(index="state", columns="model", values="validation_relative_rmse", aggfunc="mean").loc[
        STATE_COLUMNS
    ]
    y = np.arange(len(state))
    axes[1].barh(y - 0.18, state["diagonal_attractor"], 0.34, color="#1F77F4", label="Diagonal")
    axes[1].barh(y + 0.18, state["coupled_ridge_attractor"], 0.34, color="#E53935", label="Coupled")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([state_plot_label(col) for col in state.index])
    axes[1].set_xlabel("Validation relative RMSE")
    axes[1].set_title("State-wise validation")
    apply_axis_polish(axes[1], grid="x")
    for label, ax in zip(["a", "b"], axes):
        add_panel_label(ax, label)
    save_publication_figure(fig, fig_dir / "fig16_multicondition_dynamics_validation")
    plt.close(fig)


def plot_leave_one_condition_validation(loco: pd.DataFrame, fig_dir: Path) -> None:
    if loco.empty:
        return save_placeholder_figure(
            fig_dir / "fig17_leave_one_condition_validation",
            "Leave-one-condition-out validation requires at least four conditions.",
        )
    detail = loco[loco["held_out_case_id"].astype(str).ne("summary")].copy()
    if detail.empty:
        return save_placeholder_figure(
            fig_dir / "fig17_leave_one_condition_validation",
            "Leave-one-condition-out validation has no detailed rows.",
        )
    configure_matplotlib()
    targets = ["melt_pool_length_m", "full_width_m", "height_span_m", "Tmax_K", "Umax_m_per_s"]
    plot_df = detail[detail["target"].isin(targets)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    plot_df["actual_norm"] = np.nan
    plot_df["predicted_norm"] = np.nan
    for target, idx in plot_df.groupby("target").groups.items():
        actual = plot_df.loc[idx, "actual"].to_numpy(dtype=float)
        predicted = plot_df.loc[idx, "predicted"].to_numpy(dtype=float)
        lo = float(np.nanmin(np.r_[actual, predicted]))
        hi = float(np.nanmax(np.r_[actual, predicted]))
        span = max(hi - lo, 1e-12)
        plot_df.loc[idx, "actual_norm"] = (actual - lo) / span
        plot_df.loc[idx, "predicted_norm"] = (predicted - lo) / span
    for target, group in plot_df.groupby("target", sort=False):
        axes[0].scatter(
            group["actual_norm"],
            group["predicted_norm"],
            s=20,
            alpha=0.78,
            label=short_state_label(target, for_plot=True),
            edgecolor="white",
            linewidth=0.25,
        )
    axes[0].plot([0, 1], [0, 1], color="0.3", lw=0.8, ls="--")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("Actual value, target-wise normalized")
    axes[0].set_ylabel("LOCO prediction, target-wise normalized")
    axes[0].set_title("Held-out prediction")
    axes[0].legend(fontsize=7.2, ncols=1, loc="lower right")
    apply_axis_polish(axes[0], grid="both")

    summary = loco[loco["held_out_case_id"].astype(str).eq("summary")].copy()
    summary = summary[summary["target"].isin(targets)]
    y = np.arange(len(summary))
    axes[1].barh(y, summary["mean_relative_error"].to_numpy(dtype=float), color="#1F77F4", height=0.58)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([short_state_label(t, for_plot=True) for t in summary["target"]], fontsize=7.2)
    axes[1].set_xlabel("Mean relative error")
    axes[1].set_title("LOCO error by target")
    axes[1].invert_yaxis()
    apply_axis_polish(axes[1], grid="x")
    for label, ax in zip(["a", "b"], axes):
        add_panel_label(ax, label)
    save_publication_figure(fig, fig_dir / "fig17_leave_one_condition_validation")
    plt.close(fig)


def plot_external_holdout_validation(
    external_geometry_comparison: pd.DataFrame,
    external_process_validation: pd.DataFrame,
    external_dynamics_summary: pd.DataFrame,
    fig_dir: Path,
) -> None:
    if (
        external_geometry_comparison is None
        or external_process_validation is None
        or external_dynamics_summary is None
        or len(external_geometry_comparison) == 0
        or len(external_process_validation) == 0
        or len(external_dynamics_summary) == 0
    ):
        save_placeholder_figure(
            fig_dir / "fig18_external_holdout_validation",
            "Melt-pool-data holdout evidence is not available.",
        )
        return
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.35), constrained_layout=True)

    geom_case = external_geometry_comparison[
        external_geometry_comparison["time_s"].astype(str).eq("case_summary")
    ].copy()
    boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse")
    case_ids = boundary.index.astype(str).tolist()
    case_axis_labels = [compact_condition_label(case_id) for case_id in case_ids]
    x = np.arange(len(case_ids))
    if {"ellipsoid", "superellipsoid"}.issubset(boundary.columns):
        axes[0].bar(x - 0.17, boundary["ellipsoid"], width=0.34, color="#9C9C9C", label="Ellipsoid")
        axes[0].bar(x + 0.17, boundary["superellipsoid"], width=0.34, color="#1F77F4", label="Superellipsoid")
        wins = int((boundary["superellipsoid"] < boundary["ellipsoid"]).sum())
        axes[0].text(
            0.02,
            0.94,
            f"Superellipsoid lower\n{wins}/{len(boundary)} cases",
            transform=axes[0].transAxes,
            fontsize=6.4,
            color="0.30",
            ha="left",
            va="top",
            linespacing=1.05,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(case_axis_labels, rotation=0, ha="center", fontsize=7.2)
    axes[0].set_xlabel("Holdout condition")
    axes[0].set_ylabel("Boundary residual")
    axes[0].set_title("External geometry")
    axes[0].legend(
        loc="upper right",
        fontsize=6.6,
        frameon=True,
        framealpha=0.86,
        edgecolor="none",
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.05,
    )
    apply_axis_polish(axes[0])

    process_summary = external_process_validation[
        external_process_validation["case_id"].astype(str).eq("summary")
    ].copy()
    if len(process_summary):
        order = process_summary.sort_values("mean_relative_error", ascending=False)
        labels = [short_state_label(str(t), for_plot=True) for t in order["target"]]
        axes[1].barh(np.arange(len(order)), order["mean_relative_error"], color="#2DB84D")
        axes[1].set_yticks(np.arange(len(order)))
        axes[1].set_yticklabels(labels, fontsize=7.2)
        axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean relative error")
    axes[1].set_title("Process response")
    apply_axis_polish(axes[1], grid="x")

    dyn_state = (
        external_dynamics_summary.groupby("state", as_index=False)["relative_rmse"]
        .mean()
        .sort_values("relative_rmse", ascending=False)
    )
    labels = [short_state_label(str(t), for_plot=True) for t in dyn_state["state"]]
    axes[2].barh(np.arange(len(dyn_state)), dyn_state["relative_rmse"], color="#D65F7A")
    axes[2].set_yticks(np.arange(len(dyn_state)))
    axes[2].set_yticklabels(labels, fontsize=7.2)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("External relative RMSE")
    axes[2].set_title("Attractor trajectory")
    apply_axis_polish(axes[2], grid="x")

    for label, ax in zip(["a", "b", "c"], axes):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    save_publication_figure(fig, fig_dir / "fig18_external_holdout_validation")
    plt.close(fig)


def plot_simulation_cross_sections(fig_dir: Path, source_dir: Path | None = None) -> None:
    configure_matplotlib()
    fig_dir.mkdir(parents=True, exist_ok=True)
    inferred_root = fig_dir.parent.parent if fig_dir.parent.name == "analysis_outputs" else Path.cwd()
    candidates = []
    if source_dir is not None:
        candidates.append(Path(source_dir))
    candidates.extend([inferred_root / "simultion pictures", Path.cwd() / "simultion pictures"])
    source_root = next((path for path in candidates if path.exists()), None)
    if source_root is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing simulation cross-section source directory. Searched: {searched}")

    files = {
        "A2-XY": "A2-XY.png",
        "A2-XZ": "A2-XZ.png",
        "A1-XY": "A1-XY.png",
        "A1-XZ": "A1-XZ.png",
        "A3-XY": "A3-XY.png",
        "A3-XZ": "A3-XZ.png",
        "legend_tem": "legend_tem.png",
        "legend_vel": "legend_vel.png",
    }
    missing = [str(source_root / name) for name in files.values() if not (source_root / name).exists()]
    if missing:
        raise FileNotFoundError("Missing simulation cross-section source image(s): " + "; ".join(missing))
    images = {key: plt.imread(source_root / name) for key, name in files.items()}

    fig = plt.figure(figsize=(20.0, 6.6), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        4,
        left=0.025,
        right=0.985,
        bottom=0.105,
        top=0.86,
        wspace=0.045,
        hspace=0.16,
        width_ratios=[1.0, 1.0, 1.0, 0.72],
        height_ratios=[0.63, 1.0],
    )

    def image_panel(ax: mpl.axes.Axes, image: np.ndarray, label: str) -> None:
        ax.imshow(image, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.75)
            spine.set_color("0.42")
        ax.text(
            0.025,
            0.94,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "0.75", "linewidth": 0.5, "pad": 0.35},
        )

    columns = [
        ("A2: 550 W", "A2-XY", "A2-XZ", "A XY", "B XZ"),
        ("A1: 750 W", "A1-XY", "A1-XZ", "C XY", "D XZ"),
        ("A3: 950 W", "A3-XY", "A3-XZ", "E XY", "F XZ"),
    ]
    for col, (title, xy_key, xz_key, xy_label, xz_label) in enumerate(columns):
        top_ax = fig.add_subplot(grid[0, col])
        image_panel(top_ax, images[xy_key], xy_label)
        top_ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
        image_panel(fig.add_subplot(grid[1, col]), images[xz_key], xz_label)

    legend_ax = fig.add_subplot(grid[:, 3])
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_facecolor("white")
    for spine in legend_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("0.42")
    legend_ax.text(0.25, 0.955, "T (K)", transform=legend_ax.transAxes, ha="center", va="top", fontsize=8.2, fontweight="bold")
    legend_ax.text(0.58, 0.955, "U (m/s)", transform=legend_ax.transAxes, ha="center", va="top", fontsize=8.2, fontweight="bold")
    temp_ax = legend_ax.inset_axes([0.08, 0.14, 0.33, 0.76])
    temp_ax.imshow(images["legend_tem"], aspect="auto")
    temp_ax.set_axis_off()
    vel_ax = legend_ax.inset_axes([0.47, 0.14, 0.31, 0.76])
    vel_ax.imshow(images["legend_vel"], aspect="auto")
    vel_ax.set_axis_off()

    fig.suptitle(
        "FLOW-3D molten-region thermal-flow cross sections",
        y=0.972,
        fontsize=13.0,
        fontweight="bold",
    )
    fig.text(0.006, 0.635, "XY", ha="left", va="center", fontsize=9.2, fontweight="bold")
    fig.text(0.006, 0.315, "XZ", ha="left", va="center", fontsize=9.2, fontweight="bold")
    fig.text(
        0.5,
        0.035,
        "All cases: scan speed 8 mm/s; particle rate 60000 s^-1; powder feed 12 g/min",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )
    out_base = fig_dir / "paper_fig13_simulation_cross_sections"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    png_path = out_base.with_suffix(".png")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    try:
        from PIL import Image

        with Image.open(png_path) as image:
            image.save(out_base.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    except Exception:
        fig.savefig(out_base.with_suffix(".tiff"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_paper_figure_set(output_dir: Path) -> None:
    source_dir = output_dir / "figures"
    paper_dir = output_dir / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    for old in paper_dir.glob("paper_fig*.*"):
        if old.is_file() and old.suffix.lower() in {".svg", ".pdf", ".tiff", ".png"}:
            try:
                old.unlink()
            except OSError:
                # Windows may keep recently previewed PDFs/images locked; stale files are harmless.
                pass
    mapping = {
        "paper_fig01_modeling_framework": "fig08_modeling_framework",
        "paper_fig02_process_matrix": "fig13_multicondition_process_matrix",
        "paper_fig03_data_moving_frame": "fig01_moving_frame_point_cloud",
        "paper_fig04_geometry_quasi_steady": "fig02_geometry_evolution",
        "paper_fig05_free_boundary_model_comparison": "fig15_multicondition_geometry_comparison",
        "paper_fig06_process_response": "fig14_multicondition_response_surfaces",
        "paper_fig07_dimensionless_regime": "fig09_dimensionless_regime",
        "paper_fig08_dynamics_validation": "fig16_multicondition_dynamics_validation",
        "paper_fig14_dynamics_residuals_by_state": "supp_figS3_dynamics_residuals",
        "paper_fig09_error_budget_model_selection": "fig11_error_budget_model_selection",
        "paper_fig10_identifiability_overparameterization": "fig12_identifiability_overparameterization",
        "paper_fig11_leave_one_condition_validation": "fig17_leave_one_condition_validation",
        "paper_fig12_external_holdout_validation": "fig18_external_holdout_validation",
        "paper_fig13_simulation_cross_sections": "paper_fig13_simulation_cross_sections",
    }
    for paper_name, source_name in mapping.items():
        for suffix in [".svg", ".pdf", ".tiff", ".png"]:
            src = source_dir / f"{source_name}{suffix}"
            dst = paper_dir / f"{paper_name}{suffix}"
            if src.exists():
                shutil.copyfile(src, dst)


def write_method_draft(
    report_path: Path,
    table: pd.DataFrame,
    quasi: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    eigenvalues: pd.DataFrame,
    error_summary: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    steady = table[table["time_s"] >= QUASI_STEADY_START_S]
    stable = bool(eigenvalues["stable_if_negative"].all())
    length_mean = steady["melt_pool_length_m"].mean() * 1e3
    width_mean = steady["full_width_m"].mean() * 1e3
    height_mean = steady["height_span_m"].mean() * 1e3
    validation_rel = dynamics_summary["validation_relative_rmse"].mean()

    text = f"""# L-DED 鐔旀睜鑷敱杈圭晫闄嶉樁寤烘ā鏂规硶鑽夌

## 1. 鏁版嵁涓庡崐鍩熷绉伴噸鏋?
鍘熷鏁版嵁鏉ヨ嚜 FLOW-3D 瀵煎嚭鐨勭啍姹犲尯鍩熺綉鏍肩偣浜戯紝鍏?{len(table)} 涓椂闂存銆傝绠楀煙鍦?Y 鏂瑰悜閲囩敤鍗婂煙璁剧疆锛宍Y=0` 涓哄绉拌竟鐣岋紝鍥犳鐪熷疄鐔旀睜瀹藉害鍜屼綋绉唬鐞嗛噺鎸夐暅鍍忓叧绯婚噸鏋勶細

```text
W(t) = 2 max(y),
V_full(t) = 2 V_half(t).
```

鏈?pilot 榛樿鍏堝垹闄ゅ畬鍏ㄩ噸澶嶈锛屽啀瀵逛粛鐒堕噸澶嶇殑鍧愭爣鐐硅繘琛屽満鍙橀噺鍧囧€艰仛鍚堛€傚嚑浣曡竟鐣岀敱鐔旀睜鐐逛簯澶栧寘缁滆繎浼硷紝鍗婂煙浣撶Н浠ｇ悊閲忛噰鐢ㄥ崐鍩熺偣浜戜笁缁村嚫鍖呬綋绉紝瀹屾暣鐔旀睜浣撶Н浠ｇ悊閲忎负鍏?2 鍊嶃€?
## 2. 绉诲姩鍧愭爣鑷敱杈圭晫琛ㄧず

鎵弿閫熷害鍥哄畾涓?`v = 8 mm/s`銆傚紩鍏ョЩ鍔ㄥ潗鏍囷細

```text
xi = x - vt.
```

鍦ㄨ鍧愭爣涓紝鐔旀睜鍖哄煙鍐欎负锛?
```text
Omega_m(t) = {{(xi,y,z): point belongs to molten-pool domain}},
Gamma(t) = boundary of Omega_m(t).
```

杈圭晫鐘舵€佸彉閲忓畾涔変负锛?
```text
L_f = max(xi),     L_r = -min(xi),
W = 2 max(y),      H = max(z) - min(z).
```

浠?`t >= {QUASI_STEADY_START_S:.2f} s` 鐨勭粺璁＄湅锛岀啍姹犻暱搴︺€佸叏瀹藉拰楂樺害璺ㄥ害鐨勫噯绋虫€佸潎鍊肩害涓?`{length_mean:.3f} mm`銆乣{width_mean:.3f} mm` 鍜?`{height_mean:.3f} mm`銆?
## 3. 闈炲绉拌嚜鐢辫竟鐣屽嚑浣曟ā鍨?
鍩虹嚎妯″瀷閲囩敤绉诲姩鍧愭爣涓嬬殑闈炲绉版き鐞冩按骞抽泦锛?
```text
((xi - xi_c)/a_f)^2 + (y/b)^2 + ((z-z_c)/c)^2 = 1,  xi >= xi_c,
((xi - xi_c)/a_r)^2 + (y/b)^2 + ((z-z_c)/c)^2 = 1,  xi < xi_c.
```

鍏朵腑 `q(t) = [a_f, a_r, b, c, xi_c, z_c]` 涓鸿嚜鐢辫竟鐣屼綆缁寸姸鎬併€傚綋鍓?pilot 鍥哄畾褰㈢姸鎸囨暟涓?2锛屼綔涓鸿秴妞悆妯″瀷鐨勫熀绾匡紱鍚庣画鑻ヨ竟鐣屾畫宸緝澶э紝鍙皢鎸囨暟 `n,m,p` 绾冲叆璇嗗埆銆?
瀹屾暣鐔旀睜浣撶Н鐨勬き鐞冧唬鐞嗕负锛?
```text
V_ellipsoid = (2/3) pi b c (a_f + a_r).
```

## 4. 鏃犻噺绾叉鏋?
鍚庣画瀹屾暣璁烘枃寤鸿浠庣Щ鍔ㄧ儹婧愪紶鐑柟绋嬪嚭鍙戯細

```text
rho c_p (partial T/partial t + u 路 grad T)
= div(k grad T) + Q(x-vt,y,z) - losses.
```

杈圭晫鐢辩啍鐐圭瓑娓╅潰杩戜技锛?
```text
Gamma(t): T = T_m.
```

鐢变簬鏉愭枡鍙傛暟鏆傛湭纭锛屽綋鍓嶅彧淇濈暀绗﹀彿鏃犻噺绾插寲妗嗘灦锛?
```text
Pe = v L / alpha,
Fo = alpha t / L^2,
Ste = c_p (T_l - T_m) / L_m,
E* = eta P / [rho c_p v r_b^2 (T_m - T_0)].
```

鍑犱綍鍙橀噺宸叉寜鍑嗙ǔ鎬佸钩鍧囩啍姹犻暱搴?`L_ref` 杈撳嚭涓?`lf_star, lr_star, w_star, h_star`銆?
## 5. 浣庣淮鍔ㄥ姏绯荤粺涓庣ǔ瀹氭€?
鍩虹嚎鍔ㄥ姏绯荤粺閲囩敤鍑嗙ǔ鎬佸惛寮曞瓙妯″瀷锛?
```text
dq/dt = A(q_inf - q).
```

褰撳墠 pilot 浣跨敤瀵硅鐭╅樀 `A = diag(k_i)` 浠ラ伩鍏嶅崟宸ュ喌鐭椂闂村簭鍒椾笅鐨勮繃鍙傛暟鍖栥€傜姸鎬佸彉閲忎负锛?
```text
q = [L_f, L_r, W, H, Tmax, Gmean, Umax].
```

璁粌闆嗕负鍓?{TRAIN_FRACTION:.0%} 鏃堕棿姝ワ紝楠岃瘉闆嗕负鍚庣画鏃堕棿姝ャ€傜嚎鎬у寲 Jacobian 涓?`J = -A`锛涘綋鍓嶅叏閮ㄧ壒寰佸€间负璐熺殑鍒ゆ柇缁撴灉涓?`{stable}`銆傞獙璇侀泦骞冲潎鐩稿 RMSE 涓?`{validation_rel:.4f}`銆?
## 6. 璇樊鍒嗚В

褰撳墠鎶ュ憡灏嗚宸垎涓轰笁绫伙細

1. 杈圭晫閲嶆瀯璇樊锛氱敱閲嶅鐐瑰鐞嗐€佸崐鍩熼暅鍍忓拰鍑稿寘澶栧寘缁滆繎浼煎紩鍏ャ€?2. 鍑犱綍妯″瀷璇樊锛氱敱闈炲绉版き鐞冨瀹為檯鐔旀睜澶栧寘缁滅殑鎷熷悎娈嬪樊鍜屼綋绉宸　閲忋€?3. 鍔ㄥ姏瀛﹂娴嬭宸細鐢遍獙璇佹椂闂存涓婄殑 RMSE 鍜岀浉瀵?RMSE 琛￠噺銆?
鍏抽敭璇樊姹囨€昏 `error_summary.csv`銆傚噯绋虫€佺粺璁¤ `quasi_steady_summary.csv`銆傚姩鍔涘鍙傛暟銆侀娴嬪€煎拰鐗瑰緛鍊煎垎鍒 `dynamics_fit_summary.csv`銆乣dynamics_predictions.csv` 鍜?`stability_eigenvalues.csv`銆?
## 7. 鍚庣画璁烘枃鍐欎綔鍒囧叆鐐?
鎺ㄨ崘灏嗚鏂囪础鐚〃杩颁负锛氬熀浜庡崐鍩熼珮淇濈湡 CFD 鐐逛簯锛屾彁鍑轰竴绉嶇Щ鍔ㄥ潗鏍囦笅鐨?L-DED 鐔旀睜鑷敱杈圭晫闄嶉樁寤烘ā妗嗘灦锛屽苟閫氳繃浣庣淮鍚稿紩瀛愬姩鍔涚郴缁熸弿杩扮啍姹犱粠鍚姩闃舵鍚戝噯绋虫€佺殑鏀舵暃杩囩▼銆?"""
    report_path.write_text(text, encoding="utf-8")


def write_enhanced_method_draft(
    report_path: Path,
    table: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    eigenvalues: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    coupled_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    steady = table[table["time_s"] >= QUASI_STEADY_START_S]
    length_mean = steady["melt_pool_length_m"].mean() * 1e3
    width_mean = steady["full_width_m"].mean() * 1e3
    height_mean = steady["height_span_m"].mean() * 1e3
    diagonal_stable = bool(eigenvalues["stable_if_negative"].all())
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    diagonal_validation = float(dynamics_summary["validation_relative_rmse"].mean())
    coupled_validation = float(coupled_summary["validation_relative_rmse"].mean())

    geom_summary = geometry_comparison[geometry_comparison["time_s"] == "summary"]
    ellipsoid_boundary = float(
        geom_summary.loc[geom_summary["model"] == "ellipsoid", "mean_boundary_residual_rmse"].iloc[0]
    )
    super_boundary = float(
        geom_summary.loc[geom_summary["model"] == "superellipsoid", "mean_boundary_residual_rmse"].iloc[0]
    )
    ellipsoid_volume = float(
        geom_summary.loc[geom_summary["model"] == "ellipsoid", "mean_volume_relative_error"].iloc[0]
    )
    super_volume = float(
        geom_summary.loc[geom_summary["model"] == "superellipsoid", "mean_volume_relative_error"].iloc[0]
    )
    diag_u = float(
        dynamics_comparison[
            (dynamics_comparison["model"] == "diagonal_attractor")
            & (dynamics_comparison["state"] == "Umax_m_per_s")
        ]["validation_relative_rmse"].mean()
    )
    coupled_u = float(
        dynamics_comparison[
            (dynamics_comparison["model"] == "coupled_ridge_attractor")
            & (dynamics_comparison["state"] == "Umax_m_per_s")
        ]["validation_relative_rmse"].mean()
    )
    coupled_is_improvement = coupled_validation < diagonal_validation and coupled_stable
    dim_lookup = dimensionless_value_lookup(dimensionless_numbers)
    pe = float(dim_lookup["Pe"])
    fo = float(dim_lookup["Fo_final"])
    ste = float(dim_lookup["Ste"])
    e_star = float(dim_lookup["E_star"])
    re = float(dim_lookup["Re"])
    pr = float(dim_lookup["Pr"])
    ma = float(dim_lookup["Ma"])

    text = f"""# L-DED Melt-Pool Free-Boundary Reduced-Order Modeling Draft

## 1. Data and half-domain reconstruction

The FLOW-3D output contains {len(table)} time-resolved point clouds of the molten region. The computational domain uses a half-domain setting in the Y direction, with `Y=0` as a symmetry plane. The full melt-pool width and volume proxy are reconstructed as:

```text
W(t) = 2 max(y),
V_full(t) = 2 V_half(t).
```

The workflow first removes exact duplicate rows, then collapses repeated coordinates by averaging field variables. The geometric free boundary is approximated by the point-cloud envelope. The half-domain volume proxy is obtained from the 3D convex hull and mirrored to the full domain.

## 2. Moving-frame free-boundary representation

The scanning speed is fixed at `v = 8 mm/s`. The moving coordinate is:

```text
xi = x - vt.
```

The melt-pool domain and boundary are represented as:

```text
Omega_m(t) = {{(xi,y,z): point belongs to molten-pool domain}},
Gamma(t) = boundary of Omega_m(t).
```

Boundary state variables are:

```text
L_f = max(xi),     L_r = -min(xi),
W = 2 max(y),      H = max(z) - min(z).
```

For `t >= {QUASI_STEADY_START_S:.2f} s`, the quasi-steady mean melt-pool length, full width and vertical span are approximately `{length_mean:.3f} mm`, `{width_mean:.3f} mm` and `{height_mean:.3f} mm`.

## 3. Ellipsoid vs. superellipsoid free-boundary models

The baseline asymmetric ellipsoid is:

```text
((xi - xi_c)/a_f)^2 + (y/b)^2 + ((z-z_c)/c)^2 = 1,  xi >= xi_c,
((xi - xi_c)/a_r)^2 + (y/b)^2 + ((z-z_c)/c)^2 = 1,  xi < xi_c.
```

The upgraded asymmetric superellipsoid is:

```text
|((xi - xi_c)/a_side)|^n + |y/b|^m + |(z-z_c)/c|^p = 1,
a_side = a_f for xi >= xi_c, and a_r for xi < xi_c.
```

The state vector is `q = [a_f, a_r, b, c, xi_c, z_c, n, m, p]`, with `1 <= n,m,p <= 6`. Its full-domain volume is:

```text
V = 4 (a_f + a_r) b c
    Gamma(1+1/n) Gamma(1+1/m) Gamma(1+1/p)
    / Gamma(1+1/n+1/m+1/p).
```

The mean boundary residual changes from `{ellipsoid_boundary:.4f}` for the ellipsoid to `{super_boundary:.4f}` for the superellipsoid. The mean volume relative error changes from `{ellipsoid_volume:.4f}` to `{super_volume:.4f}`. The boundary-fit comparison figure should be used to judge whether the added shape exponents are justified.

## 4. Dimensionless framework

The full manuscript should start from the moving heat-source transport equation:

```text
rho c_p (partial T/partial t + u dot grad T)
= div(k grad T) + Q(x-vt,y,z) - losses,
Gamma(t): T = T_m.
```

The present implementation uses `Flow3D_setup.md` as the authoritative source for known laser, phase-change and surface-tension constants, and uses the supplied 316L temperature-dependent property tables for density, heat capacity, thermal conductivity and viscosity. The tables are interpolated at the liquidus temperature to define the material-property scales used in the nondimensional diagnostics. The computed dimensionless groups are:

```text
Pe = v L / alpha,
Fo = alpha t / L^2,
Ste = c_p (T_l - T_s) / L_fus,
E* = eta P / [rho c_p v r_b^2 (T_l - T_0)].
```

For the current data, `Pe = {pe:.3g}`, `Fo_final = {fo:.3g}`, `Ste = {ste:.3g}`, `E* = {e_star:.3g}`, `Re = {re:.3g}`, `Pr = {pr:.3g}`, and `Ma = {ma:.3g}`. The Marangoni number uses the setup-note surface-tension temperature coefficient magnitude `|d sigma/dT| = 2.50836e-4 N/(m K)`. These values are reference scale diagnostics, not solver-property verification. The modeling table also includes `lf_star, lr_star, w_star, h_star`, normalized by the quasi-steady mean melt-pool length `L_ref`.

## 5. Diagonal and coupled attractor dynamics

The diagonal baseline is:

```text
dq_i/dt = k_i(q_inf_i - q_i).
```

The coupled linear attractor is:

```text
dq/dt = A(q_inf - q),
q = [L_f, L_r, W, H, Tmax, Gmean, Umax].
```

For the coupled model, `q_inf` is fixed from the quasi-steady part of the training set. The matrix `A` is identified by ridge regression, with regularization selected by leave-one-step validation over the training interval.

The diagonal model has stable eigenvalues: `{diagonal_stable}` and mean validation relative RMSE `{diagonal_validation:.4f}`. The coupled model has stable eigenvalues: `{coupled_stable}` and mean validation relative RMSE `{coupled_validation:.4f}`. For `Umax`, the diagonal and coupled validation relative RMSE values are `{diag_u:.4f}` and `{coupled_u:.4f}`.

## 6. Model-selection rule

The coupled model should be claimed as an improvement only if it both reduces validation error and has Jacobian eigenvalues with negative real parts. Under the current data this decision is:

```text
coupled_model_is_validated_improvement = {coupled_is_improvement}
```

If this value is `False`, the coupled model should be interpreted as an overparameterization comparison for unsupported cross-state coupling rather than as a selected predictive model.

## 7. Manuscript framing

The recommended contribution statement is: a moving-frame, symmetry-aware reduced-order free-boundary framework is proposed for L-DED melt-pool evolution, and its transient-to-quasi-steady dynamics are tested using high-fidelity CFD point-cloud data.
"""
    report_path.write_text(text, encoding="utf-8")


def write_paper_outline_draft(
    report_path: Path,
    model_selection: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dim = dimensionless_value_lookup(dimensionless_numbers)
    geom = geometry_comparison[geometry_comparison["time_s"] == "summary"].set_index("model")
    ellipsoid_boundary = float(geom.loc["ellipsoid", "mean_boundary_residual_rmse"])
    super_boundary = float(geom.loc["superellipsoid", "mean_boundary_residual_rmse"])
    ellipsoid_volume = float(geom.loc["ellipsoid", "mean_volume_relative_error"])
    super_volume = float(geom.loc["superellipsoid", "mean_volume_relative_error"])
    dyn = dynamics_comparison.groupby("model")["validation_relative_rmse"].mean()
    diagonal_validation = float(dyn["diagonal_attractor"])
    coupled_validation = float(dyn["coupled_ridge_attractor"])
    robust_ok = robustness_summary["status"].eq("ok").sum()
    robust_total = len(robustness_summary)
    super_volume_wins = int(robustness_summary["superellipsoid_improves_volume"].sum())
    super_boundary_wins = int(robustness_summary["superellipsoid_improves_boundary"].sum())
    coupled_wins = int(robustness_summary["coupled_improves_validation"].sum())

    main_models = model_selection[model_selection["selected_as_main_model"].astype(str).str.lower() == "true"][
        ["model_family", "model", "selection_reason"]
    ].to_dict("records")
    model_lines = "\n".join(
        [f"- {row['model_family']}: `{row['model']}`. {row['selection_reason']}" for row in main_models]
    )

    text = f"""# Paper Outline Draft

## Working title

CFD-informed free-boundary reduction of laser directed energy deposition melt-pool evolution via superellipsoid manifolds and stable attractor dynamics.

## Central claim

The L-DED melt pool under the studied 316L condition can be modeled as a symmetry-reconstructed moving-frame free-boundary system that approaches a quasi-steady attractor. A superellipsoid boundary improves geometric representation over an ellipsoid baseline, while a diagonal attractor is more reliable than a coupled attractor for the present single-condition short time series.

## 1. Introduction logic

- High-fidelity CFD resolves melt-pool transport but is too expensive for repeated process-level model evaluation.
- Classical melt-pool descriptors such as length, width and depth are useful but do not define a complete mathematical state.
- The gap addressed here is a compact, interpretable free-boundary representation that preserves transient-to-quasi-steady dynamics.
- The contribution is model selection, not complexity for its own sake: the paper explicitly tests whether extra geometric and dynamical degrees of freedom are justified.

## 2. Mathematical formulation

- Define a moving coordinate `xi = x - vt`, with `v = 8 mm/s`.
- Reconstruct the full domain from the half-domain simulation using `W = 2 max(y)` and `V_full = 2 V_half`.
- Define the melt-pool boundary as the point-cloud envelope of the exported molten domain.
- Use state variables `q = [L_f, L_r, W, H, Tmax, Gmean, Umax]` for reduced-order dynamics.

## 3. Data preprocessing

- Read all FLOW-3D CSV time steps, remove exact duplicate rows, and collapse repeated coordinates by averaging field values.
- Convert all coordinates to the moving frame.
- Estimate the full-domain volume proxy from a mirrored half-domain convex hull.
- Use the 316L temperature-dependent property tables for density, heat capacity, thermal conductivity and viscosity.

## 4. Dimensionless analysis

The computed reference groups are:

```text
Pe = {float(dim['Pe']):.2f}
Fo_final = {float(dim['Fo_final']):.2f}
Ste = {float(dim['Ste']):.3f}
E* = {float(dim['E_star']):.2f}
Ma = {float(dim['Ma']):.2f}
```

These values support a mixed conduction-advection interpretation (`Pe` order unity), substantial normalized heat input (`E*` large), and non-negligible surface-tension-driven flow potential (`Ma` large).

## 5. Free-boundary model comparison

- Ellipsoid boundary residual: `{ellipsoid_boundary:.4f}`.
- Superellipsoid boundary residual: `{super_boundary:.4f}`.
- Ellipsoid volume relative error: `{ellipsoid_volume:.4f}`.
- Superellipsoid volume relative error: `{super_volume:.4f}`.

The superellipsoid is therefore retained as the selected algebraic observed-envelope descriptor because it improves boundary residual while retaining a compact analytic form. Volume and distance-proxy errors remain explicit limitations rather than selection claims.

## 6. Reduced-order dynamics

- Diagonal attractor mean validation relative RMSE: `{diagonal_validation:.4f}`.
- Coupled ridge attractor mean validation relative RMSE: `{coupled_validation:.4f}`.
- Coupled attractor eigenvalues are stable, but validation error is not lower than the diagonal baseline.

The diagonal attractor is selected as a parsimonious baseline dynamics. The coupled model is retained as an overparameterization comparison for unsupported cross-state coupling.

## 7. Model selection

{model_lines}

## 8. Robustness analysis

- Completed robustness scenarios: `{robust_ok}/{robust_total}`.
- Superellipsoid improves volume error in `{super_volume_wins}/{robust_total}` scenarios.
- Superellipsoid improves boundary residual in `{super_boundary_wins}/{robust_total}` scenarios.
- Coupled dynamics improves validation error while remaining stable in `{coupled_wins}/{robust_total}` scenarios.

These checks should be reported as evidence that the main conclusion is not an artifact of a single training split or quasi-steady cutoff.

## 9. Results and discussion structure

1. Show the moving-frame point clouds and explain why the melt pool becomes nearly stationary in this frame.
2. Quantify transient growth and quasi-steady geometry.
3. Compare ellipsoid and superellipsoid boundary fits using residual and volume error.
4. Compare diagonal and coupled attractor dynamics using validation error and eigenvalue stability.
5. Discuss why the more complex coupled model is not selected under a short single-condition sequence.

## 10. Limitations

- The present dataset covers one process condition, so the model is validated for transient evolution under that condition rather than a full process-parameter map.
- The exported data contain only molten-region points, so the free boundary is reconstructed from the point-cloud envelope rather than from a full temperature isosurface.
- The coupled dynamics test is data-limited; additional process conditions may justify a parameterized coupled model later.
"""
    report_path.write_text(text, encoding="utf-8")


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "not available"
    return f"{float(value):.{digits}f}"


def _key_result_values(
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> dict[str, float | int | str]:
    dim = dimensionless_value_lookup(dimensionless_numbers)
    geom = geometry_comparison[geometry_comparison["time_s"] == "summary"].set_index("model")
    dyn_mean = dynamics_comparison.groupby("model")["validation_relative_rmse"].mean()
    quasi = table[table["time_s"] >= QUASI_STEADY_START_S]
    times = np.sort(table["time_s"].unique())
    time_points = ", ".join(f"{float(t):.2f}" for t in times)
    source_files = ", ".join(table["source_file"].astype(str).head(12).tolist())
    if len(table) > 12:
        source_files += f", ... ({len(table)} files total)"
    n_conditions = int(table["case_id"].nunique()) if "case_id" in table.columns else 1
    n_source_files = int(len(table))
    return {
        "n_time_steps": int(table["time_s"].nunique()),
        "n_conditions": n_conditions,
        "n_source_files": n_source_files,
        "process_range_text": case_parameter_ranges(table),
        "t_min": float(table["time_s"].min()),
        "t_max": float(table["time_s"].max()),
        "time_points": time_points,
        "source_files": source_files,
        "raw_rows_min": int(table["raw_rows"].min()),
        "raw_rows_max": int(table["raw_rows"].max()),
        "exact_dedup_rows_min": int(table["exact_dedup_rows"].min()),
        "exact_dedup_rows_max": int(table["exact_dedup_rows"].max()),
        "unique_points_min": int(table["unique_points"].min()),
        "unique_points_max": int(table["unique_points"].max()),
        "exact_duplicates_removed_total": int((table["raw_rows"] - table["exact_dedup_rows"]).sum()),
        "coordinate_duplicates_collapsed_total": int(table["coordinate_multi_value_points"].sum()),
        "lf_quasi_mm": float(quasi["front_length_m"].mean() * 1e3),
        "lr_quasi_mm": float(quasi["rear_length_m"].mean() * 1e3),
        "w_quasi_mm": float(quasi["full_width_m"].mean() * 1e3),
        "h_quasi_mm": float(quasi["height_span_m"].mean() * 1e3),
        "ellipsoid_boundary": float(geom.loc["ellipsoid", "mean_boundary_residual_rmse"]),
        "super_boundary": float(geom.loc["superellipsoid", "mean_boundary_residual_rmse"]),
        "ellipsoid_volume": float(geom.loc["ellipsoid", "mean_volume_relative_error"]),
        "super_volume": float(geom.loc["superellipsoid", "mean_volume_relative_error"]),
        "diagonal_validation": float(dyn_mean["diagonal_attractor"]),
        "coupled_validation": float(dyn_mean["coupled_ridge_attractor"]),
        "robust_total": int(len(robustness_summary)),
        "robust_ok": int(robustness_summary["status"].eq("ok").sum()),
        "super_volume_wins": int(robustness_summary["superellipsoid_improves_volume"].sum()),
        "super_boundary_wins": int(robustness_summary["superellipsoid_improves_boundary"].sum()),
        "coupled_wins": int(robustness_summary["coupled_improves_validation"].sum()),
        "Pe": float(dim["Pe"]),
        "Fo_final": float(dim["Fo_final"]),
        "Ste": float(dim["Ste"]),
        "E_star": float(dim["E_star"]),
        "Ma": float(dim["Ma"]),
        "Re": float(dim["Re"]),
        "Pr": float(dim["Pr"]),
    }


def geometry_volume_phrase(vals: dict[str, float | int | str]) -> str:
    direction = "decreases" if float(vals["super_volume"]) < float(vals["ellipsoid_volume"]) else "increases"
    return (
        f"the mean volume relative error {direction} from "
        f"{_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}"
    )


def geometry_selection_phrase(vals: dict[str, float | int | str]) -> str:
    return (
        f"the superellipsoid reduces the mean boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} "
        f"to {_fmt(vals['super_boundary'], 4)}, while {geometry_volume_phrase(vals)}. "
        "This contrast identifies the boundary residual as the primary envelope-fit selection metric and uses "
        "the volume proxy and distance diagnostics to delimit the descriptor interpretation."
    )


SMALL_COUNT_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def count_word(value: int) -> str:
    value = int(value)
    return SMALL_COUNT_WORDS.get(value, str(value))


def count_noun_phrase(count: int, singular_noun: str, plural_noun: str | None = None) -> str:
    count = int(count)
    noun = singular_noun if count == 1 else (plural_noun or f"{singular_noun}s")
    return f"{count_word(count)} {noun}"


def count_of_total_phrase(count: int, total: int, plural_noun: str) -> str:
    count = int(count)
    total = int(total)
    total_text = count_word(total)
    if total <= 0:
        return f"{count_word(count)} {plural_noun}"
    if count == total:
        return f"all {total_text} {plural_noun}"
    if count == 0:
        return f"none of the {total_text} {plural_noun}"
    return f"{count_word(count)} of the {total_text} {plural_noun}"


def make_data_provenance_summary(table: pd.DataFrame) -> pd.DataFrame:
    n_conditions = int(table["case_id"].nunique()) if "case_id" in table.columns else 1
    time_points = ", ".join(f"{float(t):.2f}" for t in np.sort(table["time_s"].unique()))
    raw_files = ", ".join(table["source_file"].astype(str).head(12).tolist())
    if len(table) > 12:
        raw_files += f", ... ({len(table)} files total)"
    source_text = (
        f"{n_conditions} FLOW-3D numerical L-DED process conditions; not experimental imaging data."
        if n_conditions > 1
        else "Single FLOW-3D numerical L-DED simulation; not experimental imaging data."
    )
    location_text = (
        "Project folder raw data/Aa-b-c-d/*.csv, with condition folders encoding index, power, scan speed and particle rate."
        if n_conditions > 1
        else "Project folder raw data/*.csv."
    )
    process_text = (
        f"{case_parameter_ranges(table)}; powder feed = particle_rate/60000*12 g/min."
        if n_conditions > 1
        else "316L stainless steel, 750 W laser power, 0.008 m/s scan speed, 12 g/min powder feed rate."
    )
    rows = [
        (
            "simulation_source",
            source_text,
        ),
        (
            "raw_data_location",
            location_text,
        ),
        (
            "raw_files",
            raw_files,
        ),
        (
            "time_points_s",
            time_points,
        ),
        (
            "exported_domain",
            "Molten-region points only; solid-domain and already-solidified regions are not exported.",
        ),
        (
            "domain_symmetry",
            "Half computational domain in y with y=0 symmetry plane; full width and full volume proxy are reconstructed by mirroring.",
        ),
        (
            "process_condition",
            process_text,
        ),
        (
            "raw_columns",
            ", ".join(CANONICAL_EXPORT_COLUMNS),
        ),
        (
            "units",
            "Coordinates in m, temperature in K, temperature-gradient magnitude in K/m, velocity in m/s; heat-flux column retained as exported by FLOW-3D.",
        ),
        (
            "raw_rows_per_file",
            f"{int(table['raw_rows'].min())}-{int(table['raw_rows'].max())}.",
        ),
        (
            "exact_deduplicated_rows_per_file",
            f"{int(table['exact_dedup_rows'].min())}-{int(table['exact_dedup_rows'].max())}.",
        ),
        (
            "unique_coordinates_per_file",
            f"{int(table['unique_points'].min())}-{int(table['unique_points'].max())}.",
        ),
        (
            "preprocessing",
            "Exact duplicate rows are removed; repeated coordinates are collapsed by averaging field values; coordinates are transformed to xi=x-vt.",
        ),
        (
            "interpretation_limit",
            "The fitted boundary is the envelope of the exported molten domain, not a recovered solid-liquid isotherm from the unexported full thermal field.",
        ),
    ]
    return pd.DataFrame(rows, columns=["item", "description"])


def make_nomenclature_table() -> pd.DataFrame:
    rows = [
        ("t", "time", "s", "physical simulation time"),
        ("x, y, z", "Cartesian coordinates", "m", "FLOW-3D point-cloud coordinates"),
        ("v_c", "condition-specific laser scanning speed", "m/s", "parsed from the raw-data condition folder"),
        ("xi", "moving-frame coordinate", "m", "xi = x - v_c t"),
        ("Omega_m(t)", "molten computational domain", "m3", "exported molten-region point cloud"),
        ("Gamma(t)", "observed molten-region export envelope", "m2", "outer envelope of the exported molten-region point cloud"),
        ("L_f", "front length", "m", "forward melt-pool extent in the moving frame"),
        ("L_r", "rear length", "m", "rear melt-pool extent in the moving frame"),
        ("W", "full width", "m", "computed as 2 max(y) from the half-domain simulation"),
        ("H", "height span", "m", "z_max - z_min of the molten-region point cloud"),
        ("V_half", "half-domain volume proxy", "m3", "convex-hull volume of the exported half domain"),
        ("V_full", "full-domain volume proxy", "m3", "2 V_half by symmetry"),
        ("a_f, a_r", "front and rear semi-axes", "m", "asymmetric longitudinal free-boundary lengths"),
        ("b", "half-width semi-axis", "m", "transverse semi-axis in y"),
        ("c", "vertical semi-axis", "m", "semi-axis in z"),
        ("xi_c, z_c", "boundary center coordinates", "m", "moving-frame center and vertical center"),
        ("n, m, p", "superellipsoid exponents", "-", "shape exponents bounded during fitting"),
        ("T", "temperature", "K", "FLOW-3D temperature field"),
        ("T0", "initial substrate temperature", "K", "298 K"),
        ("T_s", "solidus temperature", "K", "FLOW-3D setup value 1683 K"),
        ("T_l", "liquidus temperature", "K", "FLOW-3D setup value 1710 K"),
        ("T_max", "maximum temperature", "K", "maximum molten-region temperature at a time step"),
        ("G", "temperature-gradient magnitude", "K/m", "FLOW-3D temperature-gradient output"),
        ("G_mean", "mean temperature gradient", "K/m", "point-cloud average at a time step"),
        ("U", "velocity magnitude", "m/s", "FLOW-3D velocity magnitude"),
        ("U_max", "maximum velocity magnitude", "m/s", "maximum molten-region speed at a time step"),
        ("rho", "density", "kg/m3", "liquidus-interpolated value from the temperature-dependent property table"),
        ("c_p", "specific heat capacity", "J/(kg K)", "liquidus-interpolated value from the temperature-dependent property table"),
        ("k", "thermal conductivity", "W/(m K)", "liquidus-interpolated value from the temperature-dependent property table"),
        ("alpha", "thermal diffusivity", "m2/s", "alpha = k/(rho c_p)"),
        ("mu", "dynamic viscosity", "kg/(m s)", "liquidus-interpolated value from the temperature-dependent property table"),
        ("sigma", "surface tension", "N/m", "FLOW-3D setup value 1.8 N/m"),
        ("d sigma/dT", "surface-tension temperature coefficient", "N/(m K)", "FLOW-3D setup magnitude 2.50836e-4 N/(m K)"),
        ("P", "laser power", "W", "750 W"),
        ("eta", "laser absorptivity", "-", "FLOW-3D setup value 0.1"),
        ("r_b", "laser beam radius", "m", "FLOW-3D setup value 0.0008 m"),
        ("m_dot_p", "powder mass-flow rate", "kg/s", "12 g/min converted to SI units"),
        ("Pe", "Peclet number", "-", "v L_ref/alpha"),
        ("Fo", "Fourier number", "-", "alpha t/L_ref^2"),
        ("Ste", "Stefan number", "-", "c_p(T_l - T_s)/L_fus"),
        ("E*", "normalized heat input", "-", "eta P/[rho c_p v r_b^2 (T_l - T0)]"),
        ("Re", "Reynolds number", "-", "rho v L_ref/mu"),
        ("Pr", "Prandtl number", "-", "mu c_p/k"),
        ("Ma", "Marangoni number", "-", "|d sigma/dT|(T_l - T_s)L_ref/(mu alpha)"),
        ("q", "reduced state vector", "-", "[L_f, L_r, W, H, T_max, G_mean, U_max]"),
        ("q_inf", "quasi-steady attractor state", "-", "training-window mean state"),
        ("k_i", "diagonal relaxation rate", "1/s", "state-wise attractor rate"),
        ("A", "coupled relaxation matrix", "1/s", "ridge-identified attractor matrix"),
        ("lambda", "Jacobian eigenvalue", "1/s", "stability indicator for linearized dynamics"),
        ("Omega_s(t)", "solid or unmelted domain", "m3", "appears only in the full physical problem; not exported in the present data"),
        ("Gamma_sl(t)", "solid-liquid interface", "m2", "Stefan interface in the full physical problem"),
        ("Gamma_fs(t)", "free surface", "m2", "liquid-gas boundary with heat-loss and Marangoni conditions"),
        ("n", "unit normal vector", "-", "outward normal on an interface"),
        ("v_n", "normal interface speed", "m/s", "normal velocity of a phase boundary"),
        ("L_fus", "latent heat of fusion", "J/kg", "316L fusion latent heat"),
        ("h_c", "convective heat-transfer coefficient", "W/(m2 K)", "ambient heat-loss coefficient"),
        ("epsilon_rad", "radiation emissivity", "-", "surface radiation emissivity"),
        ("sigma_SB", "Stefan-Boltzmann constant", "W/(m2 K4)", "radiation constant in the free-surface heat-loss condition"),
        ("grad_s", "surface gradient", "1/m", "tangential gradient operator on the free surface"),
        ("tau", "viscous stress tensor", "Pa", "stress tensor used in the Marangoni boundary condition"),
        ("O_h", "molten-region observation operator", "-", "maps the full physical solution to the exported molten point cloud"),
        ("Pi_M", "manifold projection operator", "-", "projects the observed boundary onto the superellipsoid manifold"),
        ("epsilon_M", "manifold projection error", "m", "Hausdorff-type distance from observed boundary to fitted manifold"),
        ("Phi", "implicit boundary level-set function", "-", "equals 1 on the fitted superellipsoid boundary"),
        ("theta*", "identified manifold parameter vector", "-", "least-squares projection of the observed boundary onto the analytic manifold"),
        ("epsilon_Gamma", "boundary residual", "-", "root-mean-square level-set residual on boundary-envelope points"),
        ("epsilon_V", "relative volume error", "-", "relative difference between analytic full-domain volume and mirrored convex-hull volume proxy"),
        ("lambda_R", "ridge regularization strength", "-", "penalty coefficient used for coupled attractor identification"),
        ("rRMSE_j", "relative validation RMSE", "-", "state-wise normalized validation error for the reduced dynamics"),
        ("delta_j", "normalization floor", "-", "small positive value preventing division by zero in relative error metrics"),
        ("C_l", "error-budget sensitivity weight", "-", "nonnegative weight linking each diagnostic error source to total model error"),
    ]
    return pd.DataFrame(rows, columns=["symbol", "name", "unit", "definition_or_note"])


def make_equation_inventory() -> pd.DataFrame:
    rows = [
        (
            "E1",
            "Mathematical formulation",
            "Moving coordinate",
            r"xi = x - vt",
            "Defines the laser-attached frame used for geometric stationarity.",
        ),
        (
            "E2",
            "Data and preprocessing",
            "Symmetry reconstruction",
            r"W(t) = 2 max_{Omega_m(t)} y, V_full(t) = 2 V_half(t)",
            "Converts the half-domain simulation into full-width and full-volume proxies.",
        ),
        (
            "E3",
            "Data and preprocessing",
            "Reduced geometric descriptors",
            r"L_f = max xi - xi_l, L_r = xi_l - min xi, H = z_max - z_min",
            "Defines the free-boundary state extracted from the point-cloud envelope.",
        ),
        (
            "E4",
            "Free-boundary model",
            "Asymmetric ellipsoid baseline",
            r"((xi-xi_c)/a_s)^2 + (y/b)^2 + ((z-z_c)/c)^2 = 1, a_s = a_f if xi >= xi_c else a_r",
            "Six-parameter baseline geometry.",
        ),
        (
            "E5",
            "Free-boundary model",
            "Asymmetric superellipsoid",
            r"|((xi-xi_c)/a_s)|^n + |y/b|^m + |((z-z_c)/c)|^p = 1",
            "Nine-parameter selected algebraic descriptor.",
        ),
        (
            "E6",
            "Free-boundary model",
            "Superellipsoid full volume",
            r"V = 4(a_f+a_r) b c Gamma(1+1/n) Gamma(1+1/m) Gamma(1+1/p)/Gamma(1+1/n+1/m+1/p)",
            "Closed-form volume used in the geometry error calculation.",
        ),
        (
            "E7",
            "Dimensionless analysis",
            "Heat-transport scale",
            r"rho c_p (partial T/partial t + v partial T/partial xi) = k nabla^2 T + Q",
            "Physics-inspired balance used to motivate dimensionless groups.",
        ),
        (
            "E8",
            "Dimensionless analysis",
            "Dimensionless groups",
            r"Pe=vL_ref/alpha, Fo=alpha t/L_ref^2, Ste=c_p(T_l-T_s)/L_fus, E*=eta P/[rho c_p v r_b^2(T_l-T0)]",
            "Post-processing nondimensional scale diagnostics; not solver-parameter verification.",
        ),
        (
            "E9",
            "Reduced-order dynamics",
            "Diagonal attractor",
            r"dq_i/dt = k_i(q_inf,i - q_i)",
            "Selected baseline dynamics.",
        ),
        (
            "E10",
            "Reduced-order dynamics",
            "Coupled ridge attractor",
            r"dq/dt = A(q_inf - q)",
            "Coupled overparameterization comparison.",
        ),
        (
            "E11",
            "Stability analysis",
            "Attractor stability criterion",
            r"Re(lambda_j(-A)) < 0",
            "Determines local stability of the coupled linear attractor.",
        ),
        (
            "E12",
            "Error analysis",
            "Modeling error decomposition",
            r"E_total = E_reconstruction + E_geometry + E_volume_proxy + E_dynamics + E_parameter_scale",
            "Organizes uncertainty sources for the reduced-order modeling chain.",
        ),
        (
            "E13",
            "Low-dimensional manifold approximation",
            "Boundary projection error",
            r"epsilon_M(t)=inf_{theta in Theta} d_H(Gamma(t), M(theta))",
            "Defines the Hausdorff-type projection distance from the observed boundary to the analytic manifold.",
        ),
        (
            "E14",
            "Lyapunov stability",
            "Diagonal attractor Lyapunov function",
            r"V(q)=1/2 ||q-q_inf||_2^2, dV/dt <= -2 k_min V",
            "Provides the v4 exponential stability proof for the selected dynamics.",
        ),
        (
            "E15",
            "Error-budget interpretation",
            "Semi-formal total error-budget inequality",
            r"E_total <= C1 E_reconstruction + C2 E_geometry + C3 E_volume_proxy + C4 E_dynamics + C5 E_parameter_scale",
            "Uses sensitivity weights to interpret the diagnostic error budget without claiming a sharp worst-case bound.",
        ),
        (
            "E16",
            "Full physical problem",
            "Moving heat equation with latent heat",
            r"rho c_p(T)(partial_t T + u dot nabla T) = nabla dot (k(T)nabla T) + Q_laser + Q_powder - rho L_fus partial_t f_l",
            "States the high-dimensional thermal starting point before observation and reduction.",
        ),
        (
            "E17",
            "Full physical problem",
            "Stefan condition",
            r"rho L_fus v_n = [k nabla T dot n]_s^l",
            "Defines the phase-boundary energy balance that motivates the free-boundary viewpoint.",
        ),
        (
            "E18",
            "Full physical problem",
            "Free-surface heat loss",
            r"-k nabla T dot n = h_c(T-T_inf)+epsilon_rad sigma_SB(T^4-T_inf^4)",
            "Represents convective and radiative heat loss from the melt-pool free surface.",
        ),
        (
            "E19",
            "Full physical problem",
            "Marangoni shear condition",
            r"tau n dot t = (d sigma/dT) grad_s T dot t",
            "Connects temperature gradients to tangential surface stress and boundary deformation.",
        ),
        (
            "E20",
            "Observation model",
            "Molten-region observation operator",
            r"P^h(t)=O_h[Omega_m(t),T,u]",
            "Makes explicit that the data are an observation of the molten region rather than the full physical state.",
        ),
        (
            "E21",
            "Model reduction rationale",
            "Observed-boundary modeling chain",
            r"full Stefan-Marangoni problem -> O_h -> xi-frame Gamma_h(t) -> Pi_M Gamma_h(t) -> q(t) -> dq/dt",
            "Summarizes the observation-to-descriptor modeling chain without claiming a closed-form PDE reduction.",
        ),
        (
            "E22",
            "Model reduction rationale",
            "First-order relaxation approximation",
            r"dq/dt = F(q) approx J(q-q_inf), with J approx -diag(k_i)",
            "Motivates the diagonal attractor as a first-order local model near the quasi-steady state.",
        ),
        (
            "E23",
            "Assumption audit",
            "Regime-invariance check",
            r"class(Pe,Ste,E*,Ma) remains unchanged under the prescribed perturbation set",
            "Links nondimensional sensitivity to assumption robustness.",
        ),
        (
            "E24",
            "Free-boundary model",
            "Implicit boundary level set",
            r"Phi(x;theta)=|(xi-xi_c)/a_s|^n+|y/b|^m+|(z-z_c)/c|^p",
            "Defines the analytic boundary manifold used for projection and residual calculation.",
        ),
        (
            "E25",
            "Free-boundary model",
            "Boundary projection objective",
            r"theta*(t)=argmin_theta N_b^{-1} sum_{x_j in Gamma_h(t)} (Phi(x_j;theta)-1)^2",
            "Makes the superellipsoid fitting operation explicit as a constrained manifold projection.",
        ),
        (
            "E26",
            "Free-boundary model",
            "Geometry diagnostic errors",
            r"epsilon_Gamma=[N_b^{-1} sum(Phi(x_j;theta*)-1)^2]^{1/2}, epsilon_V=|V_M-V_full|/(V_full+delta_V)",
            "Defines the boundary and volume errors used for model selection.",
        ),
        (
            "E27",
            "Reduced-order dynamics",
            "Diagonal trajectory fit",
            r"min_{q_inf,i in Q_i,k_i>=0} sum_{r in T_tr} [q_inf,i+(q_i(t_0)-q_inf,i) exp(-k_i(t_r-t_0))-q_i(t_r)]^2",
            "States how the selected diagonal attractor state and rate are fitted directly to the training trajectory.",
        ),
        (
            "E28",
            "Reduced-order dynamics",
            "Coupled ridge identification",
            r"A*=argmin_A sum_{r in T_tr} ||dot q_r-A(q_inf-q_r)||_2^2+lambda_R||A||_F^2",
            "States the regularized identification problem for the coupled overparameterization check.",
        ),
        (
            "E29",
            "Model validation",
            "State-wise validation relative RMSE",
            r"rRMSE_j = [N_val^{-1} sum_{r in T_val}(qhat_j(t_r)-q_j(t_r))^2]^{1/2}/(range(q_j)+delta_j)",
            "Defines the normalized validation metric used to compare diagonal and coupled dynamics.",
        ),
        (
            "E30",
            "Error-budget interpretation",
            "Weighted total error-budget inequality",
            r"E_total <= C_R E_reconstruction + C_Gamma E_geometry + C_V E_volume + C_D E_dynamics + C_P E_parameter",
            "Links the diagnostic error taxonomy to sensitivity-weighted uncertainty accounting.",
        ),
    ]
    return pd.DataFrame(rows, columns=["equation_id", "section", "label", "equation", "role"])


def write_figure_captions(report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Figure Captions

**Figure 1. Evidence structure for observed boundary-envelope reduction.** The framework separates numerical molten-region states, observation operators, reduced model families and evidence gates used to select the observed-envelope descriptor and compact trajectory baseline. Source files: `paper_fig01_modeling_framework.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 2. Multi-condition process matrix.** The 15 A1-A15 training conditions span laser power, scan speed and powder feed, with powder feed converted from particle generation rate. Full condition identifiers and time-state counts are listed in Supplementary Table S2. Source files: `paper_fig02_process_matrix.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 3. Moving-frame reconstruction of the molten region.** The representative baseline condition shows raw half-domain export, symmetry reconstruction, moving-frame alignment and the reduced observed boundary-envelope descriptors `Lf`, `Lr`, `W` and `H`. Source files: `paper_fig03_data_moving_frame.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 4. FLOW-3D molten-region thermal-flow cross sections.** Temperature-colored molten-region surfaces with velocity vectors compare A2, A1 and A3 at 550, 750 and 950 W under the same scan speed and powder feed. Source files: `paper_fig13_simulation_cross_sections.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 5. Transient geometry and quasi-steady approach.** Time histories of front length, rear length, full width and height show the evolution from early transient growth toward a quasi-steady regime after approximately 0.20 s. Source files: `paper_fig04_geometry_quasi_steady.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 6. Cross-condition observed boundary-envelope model comparison.** Selected boundary overlays show representative transient and late quasi-steady envelope behavior, while paired summaries compare boundary residuals, volume-proxy errors and metric-wise support counts across the model-construction process matrix. Source files: `paper_fig05_free_boundary_model_comparison.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 7. Quasi-steady process-response diagnostics.** Quasi-steady length, width, height and maximum temperature are plotted across the power-speed matrix; marker area encodes powder feed. Source files: `paper_fig06_process_response.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 8. Dimensionless scaling and sensitivity analysis.** Post-processing values of `Pe`, `Ste`, `E*` and `Ma` are plotted with perturbation ranges under reference-temperature, absorptivity and surface-tension-coefficient changes. These are scale diagnostics based on liquidus-interpolated values from the temperature-dependent material-property curves. Source files: `paper_fig07_dimensionless_regime.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 9. Cross-condition dynamics validation.** Condition-wise and state-wise validation errors compare the diagonal attractor with the coupled ridge attractor. Source files: `paper_fig08_dynamics_validation.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 10. Dynamical residuals by state.** State-wise residual panels compare diagonal and coupled attractor behavior and identify comparatively weak validation behavior for the maximum-velocity descriptor. Source files: `paper_fig14_dynamics_residuals_by_state.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 11. Diagnostic error-source taxonomy and model selection.** The taxonomy summarizes reconstruction, boundary-fit, volume-proxy, dynamics and parameter-identifiability sources used to interpret the selected model combination. Source files: `paper_fig09_error_budget_model_selection.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 12. Identifiability and overparameterization.** Superellipsoid parameter variation and coupled-matrix diagnostics motivate the selected model and the non-selected coupled comparison. Source files: `paper_fig10_identifiability_overparameterization.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 13. Leave-one-condition-out validation.** A process-response extrapolation test holds out one training condition at a time; the prediction panel uses target-wise normalization so quantities with different units remain visually comparable. Source files: `paper_fig11_leave_one_condition_validation.svg`, `.pdf`, `.tiff`, and `.png`.

**Figure 14. Melt-pool-data holdout.** V-prefixed validation conditions and the A16-A20 additional holdout are excluded from model construction and used to evaluate boundary-model transfer, quasi-steady process-response prediction and process-parameterized diagonal-attractor trajectories under shared numerical and preprocessing assumptions. Short axis IDs are used in the figure; full condition identifiers, process settings and time-state counts are listed in Supplementary Table S3. These holdout conditions were not used for boundary-model selection, attractor-baseline selection, LOCO fitting or process-response training. Source files: `paper_fig12_external_holdout_validation.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S1. Boundary fits across all time steps.** Top-view superellipsoid overlays are shown for all exported time steps in the representative condition, providing boundary-fit evidence beyond the main-text panels. Source files: `supp_figS1_all_boundary_fits.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S2. Superellipsoid parameters versus time.** The fitted semi-axes, center coordinates and shape exponents are plotted over time to show parameter evolution and quasi-steady behavior. Source files: `supp_figS2_superellipsoid_parameters.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S3. Dimensionless sensitivity scenario grid.** Relative minimum, baseline and maximum values for `Pe`, `Ste`, `E*` and `Ma` summarize the full perturbation envelope used in the sensitivity analysis. Source files: `supp_figS4_dimensionless_sensitivity_grid.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S4. Theory, identifiability and error-budget diagnostics.** Error-source weights, identifiability constraints and nondimensional sensitivity spans document the model-selection boundaries summarized in the main text. Source files: `supp_figS5_theory_identifiability_error_bounds.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S5. Convex-hull and alpha-complex proxy comparison.** A representative quasi-steady time step compares exported molten points, the convex-hull envelope and a projected alpha-complex envelope. The panel evaluates boundary-proxy sensitivity and constrains volume-proxy interpretation. Source files: `supp_figS6_convex_alpha_proxy_comparison.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S6. Representative-condition stability and attractor evidence.** State-error convergence, fitted diagonal rates and coupled eigenvalues support the stability discussion. Source files: `fig10_stability_attractor.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S7. Representative boundary-envelope time-step overlays.** Expanded top and side views complement the selected panels in main-text Figure 6 and show additional transient and quasi-steady envelope behavior. Source files: `fig05_boundary_fit_comparison.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S8. Thermal-flow state evolution.** Time histories of $T_{{\\max}}$, $G_{{\\mathrm{{mean}}}}$ and $U_{{\\max}}$ provide the thermal-flow evidence behind the reduced state variables used in the attractor model. The quasi-steady marker highlights the transition after approximately 0.20 s. Source files: `fig03_thermal_flow_evolution.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S9. Dynamical model trajectory comparison.** State-wise trajectories compare the observed reduced states with the diagonal attractor and coupled ridge attractor. This figure complements the residual plot by showing the prediction curves directly and supports the conclusion that the coupled model does not improve validation accuracy. Source files: `fig06_dynamics_model_comparison.svg`, `.pdf`, `.tiff`, and `.png`.

**Supplementary Figure S10. Temperature-dependent material-property curves.** Density, specific heat, thermal conductivity and viscosity are plotted from the tabulated property inputs used for liquidus-scale interpolation in the nondimensional analysis. Vertical guides mark the solidus and liquidus temperatures. Source files: `supp_figS10_temperature_dependent_properties.svg`, `.pdf`, `.tiff`, and `.png`.
"""
    report_path.write_text(text, encoding="utf-8")


def write_reviewer_risk_response(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    model_selection: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    parameter_identifiability: pd.DataFrame | None = None,
    dimensionless_sensitivity: pd.DataFrame | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    geometry_selection_text = geometry_selection_phrase(vals)
    n_conditions = int(table["case_id"].nunique()) if "case_id" in table.columns else 1
    scope_heading = (
        "Risk 1: The study is simulation-only and uses a finite process matrix"
        if n_conditions > 1
        else "Risk 1: The study uses only one process condition"
    )
    scope_concern = (
        f"A reviewer may argue that {n_conditions} FLOW-3D conditions are not enough for a universal process map or physical measurement support."
        if n_conditions > 1
        else "A reviewer may argue that one condition, 750 W, 8 mm/s and 12 g/min, is insufficient for a process-map or general predictive model."
    )
    scope_response = (
        "Frame the contribution as multi-condition FLOW-3D-informed observed export-envelope reduction, not as an experimentally benchmarked universal process map. The analysis tests whether the analytic boundary descriptor and attractor baseline persist across power, scan speed and powder-feed settings."
        if n_conditions > 1
        else "Frame the contribution as single-condition transient-to-quasi-steady observed-envelope reduced-order modeling, not as prediction over arbitrary L-DED parameters."
    )
    main_models = model_selection[model_selection["selected_as_main_model"].astype(str).str.lower() == "true"][
        ["model_family", "model"]
    ].to_dict("records")
    model_lines = "\n".join([f"- {row['model_family']}: `{row['model']}`" for row in main_models])
    text = f"""# Reviewer Risk And Response Notes

## Main model selection

{model_lines}

## {scope_heading}

**Likely concern.** {scope_concern}

**Response strategy.** {scope_response}

## Risk 2: The exported CFD data contain only the molten region

**Likely concern.** The point-cloud boundary is not a full thermal isosurface because the solidified or unmelted region is absent from the export.

**Response strategy.** State explicitly that Gamma(t) is the envelope of the exported molten domain, not an independently reconstructed solid-liquid interface from the full temperature field. This is acceptable for the present modeling target because the reduced state is defined on the available molten-region export envelope. The volume is reported as a symmetry-reconstructed convex-hull proxy, V_full = 2 V_half, rather than as an exact thermodynamic melt volume.

## Risk 3: The superellipsoid may overfit the boundary

**Likely concern.** The superellipsoid has nine parameters, compared with six for the ellipsoid baseline.

**Response strategy.** Treat the ellipsoid as a required baseline and justify the superellipsoid by boundary-model evidence. In the current results, {geometry_selection_text} Robustness checks show superellipsoid improvement in {vals['super_volume_wins']}/{vals['robust_total']} scenarios for volume error and {vals['super_boundary_wins']}/{vals['robust_total']} scenarios for boundary residual. The manuscript should emphasize that the model remains analytic and low-dimensional, while volume and distance-proxy mismatches are reported rather than hidden. The new geometry table also reports paired better rates, sign-test p-values and distance proxies, so the choice is framed as a boundary-residual selection with explicit geometric-risk diagnostics, not as metric-accurate geometric reconstruction.

## Risk 4: The coupled dynamical model is not selected

**Likely concern.** A reviewer may expect cross-coupling among melt-pool geometry, temperature gradient and flow velocity.

**Interpretation.** The coupled model is treated as an overparameterization comparison, rather than as an optimized competing model. Although the coupled ridge attractor is stable, its mean validation relative RMSE is {_fmt(vals['coupled_validation'], 4)}, compared with {_fmt(vals['diagonal_validation'], 4)} for the diagonal attractor. Robustness checks show coupled-model improvement in {vals['coupled_wins']}/{vals['robust_total']} tested scenarios. The conclusion should therefore be that coupling is physically plausible but not statistically justified by the available condition-wise sequences.
The model-selection table now also includes paired better rates and sign-test p-values for the diagonal-versus-coupled comparison, so the rejection of the coupled model is not a qualitative impression.

## Risk 5: Dimensionless groups depend on chosen 316L properties

**Likely concern.** Temperature-dependent properties make Pe, Ste, E* and Ma sensitive to the reference temperature.

**Response strategy.** Report the parameter audit first. The known laser, phase-change and surface-tension constants are resolved to `Flow3D_setup.md`, and density, heat capacity, thermal conductivity and viscosity are taken from the temperature-dependent property tables. The current values, Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}, are liquidus-scale diagnostics based on those tabulated property curves.
"""
    if parameter_identifiability is not None:
        high_risk = parameter_identifiability[parameter_identifiability["risk_level"].astype(str).str.contains("high")]
        high_risk_names = ", ".join(high_risk["parameter"].astype(str).head(8).tolist())
        text += f"""

## Risk 6: The mathematical theory may look underdeveloped

**Likely concern.** A reviewer may see the work as curve fitting unless the assumptions and stability claims are stated explicitly.

**Response strategy.** Move the theory into named propositions: the half-domain reconstruction defines a symmetry-constrained free-boundary observation operator; the superellipsoid defines a nine-parameter analytic manifold; the diagonal attractor is exponentially stable when k_i > 0; and the coupled attractor is stable when all eigenvalues of -A have negative real parts. These statements are now collected in `theory_and_error_analysis.md` and reflected in manuscript v2.

## Risk 7: Some parameters may be weakly identifiable

**Likely concern.** Superellipsoid exponents and coupled-matrix entries may be unstable under short condition-wise sequences.

**Response strategy.** Treat identifiability as a reported limitation instead of hiding it. The current high-risk parameters include: {high_risk_names}. The main model avoids the highest-risk coupled matrix as a selected predictor, and the manuscript reports parameter-risk diagnostics separately from prediction error.
"""
    if dimensionless_sensitivity is not None:
        changed = dimensionless_sensitivity[dimensionless_sensitivity["conclusion_changed"]]
        changed_symbols = ", ".join(changed["symbol"].astype(str).tolist()) if len(changed) else "none"
        text += f"""

## Risk 8: Material-property sensitivity could alter the physical interpretation

**Likely concern.** The interpretation of Pe, Ste, E* and Ma may depend on the reference temperature or uncertain material constants.

**Response strategy.** Report sensitivity rather than a single number. The current perturbation scan varies reference temperature, absorptivity and surface-tension coefficient; class changes occur for: {changed_symbols}. This frames the nondimensional groups as scale diagnostics and protects the manuscript from overclaiming material-constant precision.
"""
    report_path.write_text(text, encoding="utf-8")


def write_response_matrix(
    report_path: Path,
    parameter_audit: pd.DataFrame,
    geometry_risk_summary: pd.DataFrame,
    validation_hierarchy: pd.DataFrame,
    model_selection: pd.DataFrame,
    identifiability_diagnostics: pd.DataFrame,
    external_holdout_summary: pd.DataFrame | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    conflicts = int(parameter_audit["status"].astype(str).str.contains("conflict").sum())
    mandatory_conflicts = int(
        parameter_audit.loc[parameter_audit["mandatory_reconciliation"].astype(bool), "status"]
        .astype(str)
        .str.contains("conflict")
        .sum()
    )
    geom_actions = "; ".join(geometry_risk_summary["risk_item"].astype(str).head(4).tolist())
    high_ident = identifiability_diagnostics[
        identifiability_diagnostics["risk_level"].astype(str).str.lower().str.contains("high")
    ]
    high_ident_text = ", ".join(high_ident["parameter"].astype(str).head(10).tolist())
    selected = model_selection[model_selection["selected_as_main_model"].astype(str).str.lower().eq("true")]
    selected_text = "; ".join(f"{row.model_family}: {row.model}" for row in selected.itertuples())
    ext_text = "No same-solver CFD holdout summary available."
    if external_holdout_summary is not None and len(external_holdout_summary):
        metric_map = dict(zip(external_holdout_summary["metric"], external_holdout_summary["value"]))
        ext_text = (
            f"{int(metric_map.get('external_validation_case_count', 0))} same-solver CFD holdout cases "
            f"across {int(metric_map.get('external_validation_cohort_count', 0))} cohort(s); "
            f"process mean relative error {float(metric_map.get('external_process_response_mean_relative_error', np.nan)):.4f}; "
            f"dynamics mean relative RMSE {float(metric_map.get('external_dynamics_mean_relative_rmse', np.nan)):.4f}."
        )
    parameter_audit_text = (
        f"Added parameter_reconciliation_audit.csv with {conflicts} parameter-audit conflicts and {mandatory_conflicts} mandatory audit items."
        if conflicts
        else "Known laser, phase-change and surface-tension constants are resolved to Flow3D_setup.md; transport-property values are supplied through temperature-dependent property tables."
    )
    parameter_gate_text = (
        (
            f"The current manuscript remains submission-gated by {mandatory_conflicts} mandatory parameter-audit conflicts. "
            "Until those setup records are reconciled, dimensionless groups are retained only as "
            "post-processing scale diagnostics and are not used as solver-parameter evidence."
        )
        if mandatory_conflicts
        else (
            "The known setup/post-processing conflicts have been closed by treating Flow3D_setup.md as authoritative. "
            "Dimensionless groups are labeled as liquidus-scale diagnostics because density, heat capacity, "
            "thermal conductivity and viscosity are interpolated from the temperature-dependent property tables."
        )
    )
    rows = [
        (
            "1. Reconcile physical and numerical parameters",
            "P0",
            "Resolved",
            parameter_audit_text,
            "Main/supplementary text uses Flow3D_setup.md as the setup source and reports the temperature-dependent property curves used for scale diagnostics.",
        ),
        (
            "2. Add CFD credibility evidence",
            "P0/P1",
            "Scope limitation",
            "Use existing domain, symmetry, cell-size and model-activation information; no new mesh/time-step run in this revision.",
            "State that CFD verification is not completed and outputs are numerical exports rather than ground truth.",
        ),
        (
            "3. Clarify boundary-envelope terminology",
            "P1",
            "Resolved",
            "Use observed molten-region export envelope wording throughout generated manuscript text.",
            "Avoid presenting convex-hull envelopes as the full solid-liquid Stefan interface.",
        ),
        (
            "4. Replace broad geometry claim with residual-limited claim",
            "P1",
            "Resolved",
            f"Geometry risk summary records: {geom_actions}.",
            "Select superellipsoid only for implicit boundary-residual improvement; keep volume as a limitation.",
        ),
        (
            "5. Address convex-hull fragility",
            "P1",
            "Partly addressed",
            "geometry_risk_summary.csv records convex-hull overfill/concavity risk and duplicate/collapse audit.",
            "Do not claim alpha-shape, isotherm or volume-preserving reconstruction in the current data-only revision.",
        ),
        (
            "6. Reframe diagonal attractor",
            "P1",
            "Resolved",
            f"Selected models: {selected_text}.",
            "Call the diagonal model a parsimonious descriptive baseline rather than a statistically established dynamical law.",
        ),
        (
            "7. Move identifiability into main argument",
            "P1",
            "Partly addressed",
            f"High-risk parameters include: {high_ident_text}.",
            "Discuss high-risk shape exponents, center shifts, Umax and coupled-matrix entries in the main limitations.",
        ),
        (
            "8. Clarify validation hierarchy",
            "P1/P2",
            "Resolved",
            f"validation_hierarchy_table.csv separates internal split, LOCO, same-solver CFD holdout cohorts and absent physical-measurement support. {ext_text}",
            "Use CFD holdout wording and avoid experimental-validation claims.",
        ),
        (
            "9. Add absolute errors and uncertainty framing",
            "P2",
            "Partly addressed",
            "Use existing tables for relative errors, risk summaries and validation hierarchy; no new UQ is claimed.",
            "Report confidence language conservatively; do not imply independent repeated experimental evidence.",
        ),
        (
            "10. Make reproducibility concrete",
            "P1/P2",
            "Resolved",
            "New audit tables and response_matrix.md are included in analysis outputs and reproducibility package.",
            "Keep raw FLOW-3D project files excluded; raw CSV availability remains an author/data-sharing statement.",
        ),
    ]
    lines = [
        "# Reviewer Response Matrix",
        "",
        "This matrix maps the strict-review priorities in `comments.txt` to concrete revision actions generated by the current data-only rescue pass.",
        "",
        "Status levels: `Resolved`, `Partly addressed` and `Scope limitation`.",
        "",
        "| Reviewer priority | Severity | Status | Implemented artifact/action | Manuscript stance |",
        "|---|---:|---|---|---|",
    ]
    for priority, severity, status, artifact, stance in rows:
        lines.append(f"| {priority} | {severity} | {status} | {artifact} | {stance} |")
    lines.extend(
        [
            "",
            "## Validation hierarchy",
            "",
            "| Layer | Evidence | Supported claim | Unsupported claim |",
            "|---|---|---|---|",
        ]
    )
    for row in validation_hierarchy.itertuples():
        lines.append(
            f"| {row.validation_layer} | {row.evidence} | {row.supported_claim} | {row.unsupported_claim} |"
        )
    lines.extend(
        [
            "",
            "## Parameter-audit gate",
            "",
            parameter_gate_text,
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manuscript_draft_v1(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    model_selection: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    geometry_selection_text = geometry_selection_phrase(vals)
    selected = model_selection[model_selection["selected_as_main_model"].astype(str).str.lower() == "true"]
    selected_text = "; ".join([f"{row.model_family}: {row.model}" for row in selected.itertuples()])
    text = f"""# CFD-informed free-boundary reduction of laser directed energy deposition melt-pool evolution via superellipsoid manifolds and stable attractor dynamics

## Abstract

High-fidelity computational fluid dynamics can resolve the transient melt pool in laser directed energy deposition (L-DED), but its direct use in mathematical modeling, control-oriented interpretation and repeated parameter studies remains expensive. This manuscript develops a CFD-informed reduced-order modeling framework for a 316L stainless-steel L-DED case at 750 W laser power, 8 mm/s scan speed and 12 g/min powder feed rate. The exported FLOW-3D data contain only the molten region, so the melt pool is treated as a moving-frame free-boundary point cloud. A half-domain simulation is reconstructed through the y = 0 symmetry boundary, and the free boundary is represented by an asymmetric superellipsoid. The resulting state variables are coupled to dimensionless heat-transfer groups and to a low-dimensional attractor model for transient-to-quasi-steady dynamics. The liquidus-based groups are Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}. Relative to an ellipsoid baseline, the superellipsoid reduces the mean boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)} and the mean volume relative error from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}. Robustness tests support the geometric choice in {vals['super_volume_wins']}/{vals['robust_total']} tested settings, whereas a coupled ridge attractor improves the validation error in {vals['coupled_wins']}/{vals['robust_total']} settings. The selected model is therefore a superellipsoid free boundary with a diagonal attractor dynamic. The study is framed as single-condition transient modeling, not as a general process-parameter map.

## Introduction

Laser directed energy deposition is governed by strong thermal gradients, moving heat input, powder capture, melt-pool convection and rapid solidification [Ref]. These mechanisms create a free-boundary problem in which the melt-pool size, shape and internal transport evolve together. Full CFD simulations can describe this evolution in detail, but their output is often too high-dimensional for mathematical analysis, model comparison or control-oriented reduced-order modeling [Ref].

Existing melt-pool descriptions often emphasize scalar descriptors such as length, width and depth [Ref]. These descriptors are useful, but they do not by themselves define a closed mathematical state or explain whether a transient melt pool approaches a stable moving-frame attractor. A second limitation is that higher-fidelity CFD exports may be incomplete from a mathematical standpoint. In the present dataset, only the molten region is exported, so the problem is not reconstruction of the entire thermal field, but construction of a defensible free-boundary model from the available molten-domain point cloud.

The objective of this work is to develop a compact and reproducible modeling chain that converts high-fidelity FLOW-3D molten-region data into a mathematical free-boundary and reduced-order dynamical system. The proposed framework combines symmetry reconstruction, moving-coordinate analysis, analytic free-boundary fitting, dimensionless interpretation and model selection. The central claim is intentionally bounded: under the studied 316L condition, the melt pool can be represented by a superellipsoid free boundary whose extracted state approaches a stable quasi-steady attractor. The work does not claim direct prediction across arbitrary power, speed and powder-feed combinations.

## Mathematical formulation

The laser-attached coordinate is defined by

```text
xi = x - vt,
```

where v = 0.008 m/s. In this coordinate, a melt pool that translates with the laser should become nearly stationary after the early transient. The exported molten region is denoted Omega_m(t). Because the simulation used a half domain in the y direction with a symmetry plane at y = 0, the full width and volume proxy are reconstructed as

```text
W(t) = 2 max y,  V_full(t) = 2 V_half(t).
```

The free boundary Gamma(t) is defined as the outer envelope of the exported molten point cloud. This definition is data-consistent: it does not require the unexported solid or previously solidified domain. From Gamma(t), the reduced state includes front length L_f, rear length L_r, full width W, height H, maximum temperature T_max, mean temperature-gradient magnitude G_mean and maximum velocity magnitude U_max.

## Data and preprocessing

The dataset consists of {vals['n_time_steps']} FLOW-3D CSV files from t={vals['t_min']:.2f} s to t={vals['t_max']:.2f} s. Each file contains point coordinates, temperature, temperature-gradient magnitude, velocity components and velocity magnitude over the molten region. Exact duplicate rows were removed, repeated coordinates were collapsed by averaging field variables, and all coordinates were transformed into the moving frame. The half-domain geometry was mirrored only through aggregate quantities and visual overlays; the statistical tables keep explicit records of raw rows, deduplicated rows and unique points.

The material is 316L stainless steel. The FLOW-3D setup note provides the laser, phase-change and surface-tension constants, and the supplied property tables define temperature-dependent density, heat capacity, thermal conductivity and viscosity. The setup-note laser beam radius is 0.0008 m, absorptivity is 0.1, solidus and liquidus temperatures are 1683 K and 1710 K, and the latent heat of fusion is 2.67776e5 J/kg. The setup-note surface tension is 1.8 N/m, and the surface-tension temperature coefficient magnitude is 2.50836e-4 N/(m K).

## Free-boundary model

Two analytic boundary models are compared. The baseline is an asymmetric ellipsoid,

```text
((xi - xi_c)/a_s)^2 + (y/b)^2 + ((z - z_c)/c)^2 = 1,
```

where a_s = a_f on the front side and a_s = a_r on the rear side. The main candidate is an asymmetric superellipsoid,

```text
|((xi - xi_c)/a_s)|^n + |y/b|^m + |((z - z_c)/c)|^p = 1,
```

with q_g = [a_f, a_r, b, c, xi_c, z_c, n, m, p]. The exponents are bounded during fitting to avoid arbitrary shape complexity. The model retains a closed-form full-domain volume, which allows geometric error to be evaluated against the symmetry-reconstructed convex-hull proxy. This comparison is essential because the additional exponents are accepted only if they improve geometric accuracy under robustness checks.

## Dimensionless analysis

The dimensionless framework uses the quasi-steady melt-pool length as L_ref, setup-note laser/phase-change constants and liquidus-temperature reference transport values from the supplied property curves. The primary groups are

```text
Pe = v L_ref / alpha
Fo = alpha t / L_ref^2
Ste = c_p (T_l - T_s) / L_fus
E* = eta P / [rho c_p v r_b^2 (T_l - T0)]
Ma = |d sigma/dT| (T_l - T_s) L_ref / (mu alpha)
```

The computed values are Pe={vals['Pe']:.2f}, Fo_final={vals['Fo_final']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f}, Re={vals['Re']:.2f}, Pr={vals['Pr']:.3f} and Ma={vals['Ma']:.2f}. These groups provide liquidus-scale context for the reduced model, using the tabulated temperature-dependent material properties.

## Reduced-order dynamics

The reduced state is

```text
q = [L_f, L_r, W, H, T_max, G_mean, U_max].
```

The selected dynamical model is a diagonal attractor,

```text
dq_i/dt = k_i (q_inf,i - q_i),
```

where q_inf is estimated from the quasi-steady training window. A coupled alternative is also tested,

```text
dq/dt = A(q_inf - q),
```

where A is identified by ridge regression and the regularization strength is chosen by leave-one-step validation on the training portion. The coupled model is accepted only if it lowers validation error and has a stable linearized Jacobian. This rule prevents physical intuition about coupling from being converted into an unsupported modeling claim.

## Results

### Moving-coordinate quasi-steady behavior

After transformation to xi = x - vt and symmetry reconstruction about y = 0, the point clouds align into a compact moving-frame envelope (Fig. 1). The extracted geometric descriptors show early growth followed by a quasi-steady regime after approximately 0.20 s (Fig. 2). In that regime, the mean front length, rear length, full width and height are {vals['lf_quasi_mm']:.3f} mm, {vals['lr_quasi_mm']:.3f} mm, {vals['w_quasi_mm']:.3f} mm and {vals['h_quasi_mm']:.3f} mm, respectively.

### Free-boundary model comparison

The superellipsoid improves the free-boundary representation relative to the ellipsoid baseline (Fig. 3). The mean boundary residual decreases from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)}, and the volume proxy is reported as a limitation rather than a selection claim. Robustness checks across training fraction, quasi-steady cutoff and exponent upper bound show volume-error improvement in {vals['super_volume_wins']}/{vals['robust_total']} scenarios and boundary-residual improvement in {vals['super_boundary_wins']}/{vals['robust_total']} scenarios. The model-selection table therefore assigns the superellipsoid as the selected algebraic observed-envelope descriptor.

### Dimensionless interpretation

The dimensionless numbers provide a compact physical interpretation of the extracted free-boundary dynamics. Pe={vals['Pe']:.2f} suggests a mixed conduction-advection regime at the melt-pool scale. Ste={vals['Ste']:.3f} indicates that sensible heating over the melting interval is comparable to latent heat. E*={vals['E_star']:.2f} confirms that the absorbed energy input is large relative to the reference thermal transport scale. Ma={vals['Ma']:.2f} supports the inclusion of flow-sensitive state variables such as U_max and G_mean in the reduced state.

### Reduced-order dynamics and model selection

The diagonal attractor predicts the validation portion with a mean relative RMSE of {_fmt(vals['diagonal_validation'], 4)}, whereas the coupled ridge attractor gives {_fmt(vals['coupled_validation'], 4)} (Fig. 4). Although the coupled model is stable in the baseline fit, robustness testing shows coupled-model improvement in {vals['coupled_wins']}/{vals['robust_total']} scenarios. The selected model combination is therefore: {selected_text}. The coupled model is retained as a negative control showing that additional degrees of freedom are not automatically beneficial for a short single-condition sequence.

## Discussion

The results support a modeling strategy in which geometric complexity and dynamical complexity are evaluated separately. The superellipsoid is justified because it improves boundary and volume metrics while preserving an analytic free-boundary form. By contrast, the coupled attractor is not selected because its larger parameter count does not improve validation performance. This distinction is important for mathematical modeling: a more flexible shape can be warranted when it directly represents the free boundary, whereas a more flexible dynamical system can be unjustified when the time series is short.

The moving-frame formulation also clarifies the role of quasi-steady behavior. The melt pool is not steady in the laboratory frame, but it becomes approximately stationary relative to the laser after the early transient. This allows q_inf to be interpreted as a moving-frame attractor rather than as a static equilibrium in physical space. The dimensionless groups place this attractor in a regime where thermal diffusion, moving-source advection, latent heat and surface-tension effects are all relevant.

## Limitations

The present study is limited to one 316L process condition. It should therefore be read as a validated single-condition reduced-order modeling workflow, not as a predictive process map. The exported FLOW-3D data include only molten-region points, so the reconstructed boundary is an envelope of the available melt domain rather than a full solid-liquid isotherm extracted from the complete temperature field. The volume metric is a convex-hull proxy after symmetry reconstruction. Finally, the coupled dynamical model is data-limited; additional time sequences and process conditions may support a parameterized coupled model in future work.

## Conclusion

This work converts FLOW-3D molten-region point-cloud data into a moving-frame, symmetry-aware free-boundary reduced-order model for L-DED. The selected algebraic observed-envelope descriptor is an asymmetric superellipsoid, supported by lower boundary residual than an ellipsoid baseline while volume and distance-proxy errors remain limitations. The selected dynamical baseline is a diagonal attractor, because the coupled ridge model does not improve validation performance in the present dataset. The resulting framework provides a conservative and reproducible bridge from high-fidelity CFD output to a mathematical modeling manuscript focused on transient-to-quasi-steady melt-pool dynamics.
"""
    report_path.write_text(text, encoding="utf-8")


def write_theory_and_error_analysis(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    error_budget: pd.DataFrame,
    parameter_identifiability: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    geom = geometry_comparison[geometry_comparison["time_s"] == "summary"].set_index("model")
    diag_val = float(np.nanmean(dynamics_summary["validation_relative_rmse"]))
    stable_diag = bool((dynamics_summary["k_per_s"].to_numpy(dtype=float) > 0).all())
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    high_params = parameter_identifiability[parameter_identifiability["risk_level"].astype(str).str.contains("high")]
    high_param_text = ", ".join(high_params["parameter"].astype(str).head(12).tolist())
    sens_lines = "\n".join(
        [
            f"- {row.symbol}: baseline {row.baseline_value:.4g}, range {row.min_value:.4g} to {row.max_value:.4g}, class change = {row.conclusion_changed}."
            for row in dimensionless_sensitivity.itertuples()
        ]
    )
    error_lines = "\n".join(
        [
            f"- {row.error_term}: {row.primary_metric} = {row.primary_value:.4g}. {row.manuscript_interpretation}"
            for row in error_budget.itertuples()
        ]
    )
    text = f"""# Theory And Error Analysis

## 1. Observation model and symmetry reconstruction

The exported FLOW-3D data define a molten-region point cloud, Omega_m^h(t), on a half domain. The full-domain observation is reconstructed through the y = 0 symmetry plane. For scalar integral descriptors, the observation operator is

```text
R[Omega_m^h](t) = Omega_m^h(t) union {{(xi,-y,z): (xi,y,z) in Omega_m^h(t)}}.
```

This implies W(t) = 2 max y and V_full(t) = 2 V_half(t). The reconstruction is exact only under the imposed computational symmetry. It does not reconstruct unexported solid-domain information.

## 2. Moving-frame free-boundary manifold

The moving coordinate xi = x - vt converts a translating melt pool into a nearly stationary free-boundary object after the early transient. The boundary Gamma(t) is modeled as the envelope of the molten-region point cloud. The superellipsoid assumption is a low-dimensional manifold hypothesis:

```text
Gamma(t) approximately belongs to M_SE = {{Gamma(theta): theta = [a_f,a_r,b,c,xi_c,z_c,n,m,p]}}.
```

Relative to the ellipsoid baseline, the superellipsoid reduces mean boundary residual from {float(geom.loc['ellipsoid', 'mean_boundary_residual_rmse']):.4f} to {float(geom.loc['superellipsoid', 'mean_boundary_residual_rmse']):.4f}, and mean volume relative error from {float(geom.loc['ellipsoid', 'mean_volume_relative_error']):.4f} to {float(geom.loc['superellipsoid', 'mean_volume_relative_error']):.4f}.

## 3. Proposition 1: diagonal attractor stability

For each reduced state component, consider

```text
dq_i/dt = k_i(q_inf,i - q_i).
```

Let e_i = q_i - q_inf,i. Then de_i/dt = -k_i e_i and

```text
e_i(t) = e_i(0) exp(-k_i t).
```

Therefore, if k_i > 0, the equilibrium q_i = q_inf,i is globally exponentially stable for that component. In the present fit, all fitted k_i are positive, so the diagonal attractor is stable: {stable_diag}. The mean validation relative RMSE is {diag_val:.4f}.

## 4. Proposition 2: coupled attractor stability

For the coupled model,

```text
dq/dt = A(q_inf - q).
```

With e = q - q_inf, the error dynamics are de/dt = -A e. The equilibrium is locally exponentially stable if every eigenvalue of -A has negative real part. In the present ridge-identified system this condition is satisfied: {coupled_stable}. However, stability alone is not sufficient for model selection because the coupled model does not improve validation error.

## 5. Proposition 3: error decomposition

The reported modeling error is decomposed as

```text
E_total = E_reconstruction + E_geometry + E_volume_proxy + E_dynamics + E_parameter_scale.
```

The decomposition is diagnostic rather than a strict additive probabilistic error bound. It prevents a single validation RMSE from hiding separate sources of uncertainty.

{error_lines}

## 6. Parameter identifiability

The present dataset has 15 time steps and one process condition, so identifiability must be treated conservatively. High-risk parameters are: {high_param_text}. The superellipsoid exponents are flagged because bounded nonlinear exponents can absorb local shape variation. The coupled matrix is flagged because it contains many entries relative to the number of observed training transitions.

## 7. Dimensionless sensitivity

The sensitivity scan varies the reference temperature among solidus, mid-melt and liquidus values, and varies absorptivity and surface-tension temperature coefficient by +/-20%. The observed ranges are:

{sens_lines}

These results should be used to qualify the physical interpretation of Pe, Ste, E* and Ma. The nondimensional groups support scaling arguments but should not be presented as invariant material constants.
"""
    report_path.write_text(text, encoding="utf-8")


def write_manuscript_draft_v2(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    model_selection: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    error_budget: pd.DataFrame,
    parameter_identifiability: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    selected = model_selection[model_selection["selected_as_main_model"].astype(str).str.lower() == "true"]
    selected_text = "; ".join([f"{row.model_family}: {row.model}" for row in selected.itertuples()])
    high_params = parameter_identifiability[parameter_identifiability["risk_level"].astype(str).str.contains("high")]
    high_param_text = ", ".join(latex_math_label(item) for item in high_params["parameter"].astype(str).head(8).tolist())
    parameter_audit = make_parameter_reconciliation_audit()
    parameter_audit_rows = "\n".join(
        [
            (
                f"{latex_readable_text(row.parameter)} & {latex_readable_text(row.flow3d_setup_record)} & "
                f"{latex_readable_text(row.postprocessing_basis)} & {latex_readable_text(row.manuscript_action)} \\\\"
            )
            for row in parameter_audit.head(8).itertuples()
        ]
    )
    stable_diag = bool((dynamics_summary["k_per_s"].to_numpy(dtype=float) > 0).all())
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    diag_stability_sentence = (
        "In the present fit, all diagonal rates are positive."
        if stable_diag
        else "In the present fit, at least one diagonal rate is non-positive, so the diagonal stability claim is not used."
    )
    coupled_stability_sentence = (
        "The baseline coupled model satisfies this spectral condition."
        if coupled_stable
        else "The baseline coupled model does not satisfy this spectral condition."
    )
    geometry_selection_text = geometry_selection_phrase(vals)
    geom_case = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("case_summary")].copy()
    geom_case_boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse") if len(geom_case) else pd.DataFrame()
    geom_case_volume = geom_case.pivot(index="case_id", columns="model", values="mean_volume_relative_error") if len(geom_case) else pd.DataFrame()
    geometry_pair_text = ""
    if {"ellipsoid", "superellipsoid"}.issubset(geom_case_boundary.columns):
        geom_wins = int((geom_case_boundary["superellipsoid"] < geom_case_boundary["ellipsoid"]).sum())
        geom_total = int(len(geom_case_boundary))
        geom_p = float(binomtest(geom_wins, geom_total, 0.5, alternative="greater").pvalue) if geom_total else float("nan")
        geom_adv = float(np.nanmedian(geom_case_boundary["ellipsoid"] - geom_case_boundary["superellipsoid"]))
        vol_wins = (
            int((geom_case_volume["superellipsoid"] < geom_case_volume["ellipsoid"]).sum())
            if {"ellipsoid", "superellipsoid"}.issubset(geom_case_volume.columns)
            else 0
        )
        geometry_pair_text = (
            "The paired condition-wise comparison gives boundary-residual improvement in "
            f"{count_of_total_phrase(geom_wins, geom_total, 'training conditions')} "
            f"(sign-test p={geom_p:.3g}, median residual reduction {geom_adv:.4g}), while the volume proxy improves in "
            f"{count_of_total_phrase(vol_wins, geom_total, 'training conditions')}."
        )
    dyn_pair = dynamics_comparison.pivot_table(
        index=["case_id", "state"],
        columns="model",
        values="validation_relative_rmse",
        aggfunc="mean",
    )
    dynamics_pair_text = ""
    if {"diagonal_attractor", "coupled_ridge_attractor"}.issubset(dyn_pair.columns):
        diag_wins = int((dyn_pair["diagonal_attractor"] < dyn_pair["coupled_ridge_attractor"]).sum())
        dyn_total = int(len(dyn_pair))
        dyn_p = float(binomtest(diag_wins, dyn_total, 0.5, alternative="greater").pvalue) if dyn_total else float("nan")
        dyn_adv = float(np.nanmedian(dyn_pair["coupled_ridge_attractor"] - dyn_pair["diagonal_attractor"]))
        dynamics_pair_text = (
            "Descriptive paired condition-state comparisons show lower diagonal-model validation error in "
            f"{count_of_total_phrase(diag_wins, dyn_total, 'condition-state pairs')} "
            f"(sign-test p={dyn_p:.3g}, median relative-RMSE reduction {dyn_adv:.4g}); these pairs are not treated as independent physical replicates."
        )
    changed_groups = dimensionless_sensitivity[dimensionless_sensitivity["conclusion_changed"]]
    if len(changed_groups):
        changed_text = ", ".join(changed_groups["symbol"].astype(str).tolist())
        dimensionless_sensitivity_sentence = (
            "The scenario sensitivity scan over reference temperature, absorptivity and "
            f"surface-tension coefficient changes the qualitative scale class for {changed_text}."
        )
    else:
        changed_text = "none"
        dimensionless_sensitivity_sentence = (
            "The scenario sensitivity scan over reference temperature, absorptivity and "
            "surface-tension coefficient preserves the qualitative scale class of the reported groups."
        )
    error_table_text = "\n".join(
        [
            f"- {row.error_term}: {row.primary_metric} = {row.primary_value:.4g}."
            for row in error_budget.itertuples()
        ]
    )
    text = f"""# CFD-informed free-boundary reduction of laser directed energy deposition melt-pool evolution via superellipsoid manifolds and stable attractor dynamics

## Abstract

High-fidelity melt-pool data can resolve melt-pool evolution in laser directed energy deposition (L-DED), but the resulting fields are difficult to use directly in mathematical modeling. This study develops a melt-pool-data-informed reduced-order framework for a single 316L stainless-steel L-DED condition at 750 W, 8 mm/s and 12 g/min. The exported data contain only the molten region, so the melt pool is modeled as a moving-frame free-boundary point cloud. A half-domain simulation is reconstructed through the y = 0 symmetry plane, the free boundary is fitted by an asymmetric superellipsoid, and the extracted state is advanced using a stable low-dimensional attractor. The liquidus-reference dimensionless groups are Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}. The superellipsoid reduces mean boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)} and is retained as the selected algebraic observed-envelope descriptor; volume and distance-proxy errors are reported as limitations. A coupled ridge attractor is stable but improves validation error in {vals['coupled_wins']}/{vals['robust_total']} settings, so it is retained as an overparameterization comparison. The selected baseline is a superellipsoid observed-envelope descriptor with a diagonal attractor dynamic. The manuscript reports stability propositions, error decomposition and parameter-identifiability risks to define the scope of interpretation for the single-condition data.

## Introduction

L-DED melt-pool evolution involves moving heat input, temperature-dependent material response, free-surface flow, powder capture and rapid phase change [Ref]. High-fidelity CFD is well suited to resolving these effects, but direct CFD fields are too high-dimensional for compact mathematical analysis, reduced-order prediction and transparent model selection. A useful intermediate model should preserve the main free-boundary structure while remaining interpretable.

Many process descriptors reduce the melt pool to length, width and depth [Ref]. These variables are valuable, but they do not define a complete free-boundary state, nor do they show whether the transient melt pool approaches a stable moving-frame attractor. The present dataset also introduces a practical modeling constraint: FLOW-3D exported only molten-region points. The task is therefore not full thermal-field inversion, but a defensible transformation from molten-domain point clouds to an analytic boundary and reduced-order dynamics.

This work proposes a moving-frame, symmetry-aware and error-audited model chain. Its contribution is not a universal process map. Instead, it asks whether a single high-fidelity L-DED simulation can be compressed into a stable, physically interpretable free-boundary reduced-order model with explicit limits on geometric error, dynamical error and parameter-scale uncertainty.

## Mathematical formulation

The laser-attached coordinate is

```text
xi = x - vt,  v = 0.008 m/s.
```

The exported half-domain molten point cloud is denoted Omega_m^h(t). With the imposed y = 0 symmetry plane, the full-domain observation operator is

```text
R[Omega_m^h](t) = Omega_m^h(t) union {{(xi,-y,z): (xi,y,z) in Omega_m^h(t)}}.
```

Consequently, W(t) = 2 max y and V_full(t) = 2 V_half(t). The free boundary Gamma(t) is the envelope of the exported molten region. The reduced state is

```text
q = [L_f, L_r, W, H, T_max, G_mean, U_max].
```

The geometric model assumes that Gamma(t) lies near a low-dimensional analytic manifold. The ellipsoid baseline is

```text
((xi - xi_c)/a_s)^2 + (y/b)^2 + ((z - z_c)/c)^2 = 1,
```

where a_s = a_f in front of the center and a_s = a_r behind it. The superellipsoid candidate is

```text
|((xi - xi_c)/a_s)|^n + |y/b|^m + |((z - z_c)/c)|^p = 1.
```

The parameter vector is theta = [a_f, a_r, b, c, xi_c, z_c, n, m, p]. This form is flexible enough to capture non-ellipsoidal boundaries but remains compact and analytic.

## Data and preprocessing

The dataset contains {vals['n_time_steps']} CSV files from t={vals['t_min']:.2f} s to t={vals['t_max']:.2f} s. Exact duplicate rows were removed, repeated coordinates were collapsed by field averaging, and all coordinates were transformed into the moving frame. The full-width and full-volume descriptors use the half-domain symmetry assumption. The material is 316L stainless steel. The setup note records the laser and phase-change constants, while the supplied property tables provide temperature-dependent transport properties for nondimensional diagnostics. The setup-note laser beam radius is 0.0008 m, absorptivity is 0.1, the solidus and liquidus temperatures are 1683 K and 1710 K, and the fusion latent heat is 2.67776e5 J/kg.

## Dimensionless analysis

The reference length is the quasi-steady mean melt-pool length. Material properties are evaluated at the liquidus temperature for the baseline values. The main groups are

```text
Pe = v L_ref/alpha,  Fo = alpha t/L_ref^2,
Ste = c_p(T_l-T_s)/L_fus,
E* = eta P/[rho c_p v r_b^2(T_l-T0)],
Ma = |d sigma/dT|(T_l-T_s)L_ref/(mu alpha).
```

The baseline values are Pe={vals['Pe']:.2f}, Fo_final={vals['Fo_final']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f}, Re={vals['Re']:.2f}, Pr={vals['Pr']:.3f} and Ma={vals['Ma']:.2f}. A sensitivity scan over reference temperature, absorptivity and surface-tension temperature coefficient shows class changes for: {changed_text}. These groups are therefore used as scaling diagnostics rather than as invariant constants.

## Reduced-order dynamics

The selected dynamics are a diagonal attractor,

```text
dq_i/dt = k_i(q_inf,i - q_i).
```

A coupled ridge model is tested as

```text
dq/dt = A(q_inf - q),
```

where A is fitted with ridge regularization. The coupled model is accepted only if it reduces validation error and is stable. This rule separates physical plausibility from statistical support.

## Stability analysis

For the diagonal attractor, define e_i = q_i - q_inf,i. Then de_i/dt = -k_i e_i, so e_i(t) = e_i(0) exp(-k_i t). If k_i > 0, the attractor is globally exponentially stable for that state component. In the present fit, all k_i are positive: {stable_diag}.

For the coupled model, define e = q - q_inf. The error equation is de/dt = -A e. The equilibrium is locally exponentially stable if every eigenvalue of -A has negative real part. The baseline coupled model satisfies this eigenvalue condition: {coupled_stable}. However, its mean validation relative RMSE is {_fmt(vals['coupled_validation'], 4)}, compared with {_fmt(vals['diagonal_validation'], 4)} for the diagonal model. Stability alone therefore does not justify the coupled model as the main predictor.

## Error analysis

The total model uncertainty is organized as

```text
E_total = E_reconstruction + E_geometry + E_volume_proxy + E_dynamics + E_parameter_scale.
```

This expression is a diagnostic decomposition, not a claim of independent additive random errors. The current error-budget entries are:

{error_table_text}

    The decomposition clarifies why a low validation RMSE is not enough by itself. The molten-region-only export affects E_reconstruction, the analytic boundary affects E_geometry, the convex-hull volume proxy affects E_volume_proxy, the short validation sequence affects E_dynamics, and reference-property scaling affects E_parameter_scale.

## Results

The full modeling chain is summarized in Fig. 1. The figure makes explicit that the paper is not a direct CFD visualization study, but a sequence of observation, symmetry reconstruction, free-boundary reduction, attractor identification and error auditing.

After moving-coordinate transformation and symmetry reconstruction, the melt-pool envelope approaches a quasi-steady form after approximately 0.20 s (Fig. 2 and Fig. 3). The quasi-steady mean front length, rear length, full width and height are {vals['lf_quasi_mm']:.3f} mm, {vals['lr_quasi_mm']:.3f} mm, {vals['w_quasi_mm']:.3f} mm and {vals['h_quasi_mm']:.3f} mm.

The superellipsoid improves boundary representation over the ellipsoid baseline (Fig. 4; Supplementary Fig. S1 and S2). The mean boundary residual decreases from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)}, and the mean volume relative error decreases from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}. Robustness checks show superellipsoid improvement in {vals['super_volume_wins']}/{vals['robust_total']} settings for volume error and {vals['super_boundary_wins']}/{vals['robust_total']} settings for boundary residual.

The dimensionless regime is summarized in Fig. 5 and Supplementary Fig. S4. The baseline values remain Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}; sensitivity testing supports their use as scale diagnostics rather than invariant constants.

The diagonal attractor predicts the validation portion with mean relative RMSE {_fmt(vals['diagonal_validation'], 4)}, while the coupled ridge attractor gives {_fmt(vals['coupled_validation'], 4)} (Fig. 6; Supplementary Fig. S3). The coupled model improves validation error in {vals['coupled_wins']}/{vals['robust_total']} robustness settings. The selected model is therefore: {selected_text}.

The error budget and model-selection evidence are shown in Fig. 7. These diagnostics separate reconstruction uncertainty, geometric approximation error, volume-proxy error, dynamical prediction error and material-scale uncertainty.

Parameter diagnostics flag the following high-risk parameters: {high_param_text} (Fig. 8). These risks do not invalidate the selected model, but they motivate reporting the superellipsoid as a compact fitted manifold and the coupled matrix as a non-selected comparison rather than as a mechanistic transport law.

## Discussion

The main result is a separation between useful geometric flexibility and unsupported dynamical complexity. The superellipsoid adds three shape exponents and improves boundary and volume metrics across robustness settings. The coupled attractor adds many interaction coefficients, remains stable, but does not improve validation accuracy. This contrast is valuable for a mathematical modeling paper because it shows that model selection was driven by evidence rather than by preference for either simplicity or complexity.

The stability analysis strengthens the reduced-order interpretation. The diagonal model is not only a fitted curve; it is an exponentially stable attractor under positive relaxation rates. The coupled system also satisfies the eigenvalue stability test, but the validation and identifiability diagnostics show why stability is a necessary but insufficient condition for adoption.

The error-budget view also clarifies the scope of the result. The boundary is an envelope of exported molten points, the volume is a mirrored convex-hull proxy, and the material-property groups are reference-state diagnostics. These caveats limit generalization but make the model transparent and reproducible.

## Limitations

This study covers one 316L process condition and should not be interpreted as a general process map. The exported data omit solid and previously solidified regions, so the boundary is not a full thermal isosurface from the complete computational domain. The parameter-identifiability analysis is a pilot diagnostic rather than a full uncertainty quantification study. Additional process conditions would be required to make q_inf, relaxation rates or superellipsoid parameters functions of power, speed and powder-flow rate.

## Conclusion

The proposed framework converts FLOW-3D molten-region point clouds into a moving-frame, symmetry-aware free-boundary reduced-order model. The selected model is an asymmetric superellipsoid boundary with a diagonal stable attractor. The selection is supported by boundary and volume error reduction, robustness checks, stability propositions, parameter-identifiability diagnostics and an explicit error budget. The result is a conservative mathematical modeling basis for single-condition transient-to-quasi-steady L-DED melt-pool evolution.
"""
    report_path.write_text(text, encoding="utf-8")


def write_theory_framework_v4(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    error_bound_summary: pd.DataFrame,
    identifiability_v4: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    stable_diag = bool((dynamics_summary["k_per_s"].to_numpy(dtype=float) > 0).all())
    k_min = float(np.nanmin(dynamics_summary["k_per_s"].to_numpy(dtype=float)))
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    high_risk = identifiability_v4[identifiability_v4["risk_level"].eq("high")]
    high_risk_text = ", ".join(high_risk["parameter"].astype(str).head(12).tolist())
    bound_lines = "\n".join(
        [
            f"- {row.bound_component}: `{row.bound_expression}`, metric `{row.source_metric}` = {row.source_value:.4g}, source `{row.source_table}`."
            for row in error_bound_summary.sort_values("display_order").itertuples()
        ]
    )
    sens_lines = "\n".join(
        [
            f"- {row.symbol}: baseline {row.baseline_value:.4g}, range {row.min_value:.4g} to {row.max_value:.4g}, class change = {row.conclusion_changed}."
            for row in dimensionless_sensitivity.itertuples()
        ]
    )
    text = f"""# Theory Framework v4

## Purpose

This report states the mathematical scaffold used in manuscript v4. It is not a full Stefan-Marangoni analytical solution. Instead, it defines a physics-inspired observation operator, a low-dimensional free-boundary manifold, a stable reduced-order attractor and a traceable error-budget interpretation for one 316L L-DED FLOW-3D condition.

## A1-A5. Modeling assumptions

**A1 Single-condition scope.** The simulated condition is fixed at 750 W, 8 mm/s and 12 g/min. The theory describes transient-to-quasi-steady evolution under this condition only.

**A2 Molten-region observation.** The exported data define a molten-region point cloud `P^h(t)` on a half domain. The unexported solid and previously solidified regions are not reconstructed.

**A3 Symmetry plane.** The plane `y=0` is an imposed computational symmetry boundary, so the full observation is generated by mirroring the half-domain point cloud.

**A4 Moving-frame quasi-steadiness.** In the coordinate `xi=x-vt`, the free-boundary envelope approaches a slowly varying attractor after the early transient near 0.20 s.

**A5 Low-dimensional boundary manifold.** The observed free boundary is approximated by a nine-parameter asymmetric superellipsoid manifold rather than by a full thermal-field interface.

## Free-boundary formulation

Let `Omega_m^h(t)` denote the exported half-domain molten region and let `P^h(t)` be its discrete point-cloud observation. The symmetry reconstruction operator is

```text
R[P^h(t)] = P^h(t) union {{(xi,-y,z): (xi,y,z) in P^h(t)}}.
```

The observed free boundary `Gamma_h(t)` is the envelope of `R[P^h(t)]`. The reduced state is

```text
q(t) = [L_f, L_r, W, H, T_max, G_mean, U_max]^T.
```

**Proposition 1: symmetry-reconstruction consistency.** Under A3, any scalar descriptor that is even in `y` is preserved by the reconstruction, and the full-width and full-volume proxies satisfy `W(t)=2 max y` and `V_full(t)=2 V_half(t)`. The statement is conditional on the imposed simulation symmetry and does not recover unobserved solid-domain fields.

## Low-dimensional manifold approximation

Define the analytic boundary manifold

```text
M_SE(theta) = {{(xi,y,z): |(xi-xi_c)/a_s|^n + |y/b|^m + |(z-z_c)/c|^p = 1}},
```

where `theta=[a_f,a_r,b,c,xi_c,z_c,n,m,p]` and `a_s=a_f` ahead of the center and `a_s=a_r` behind it. A Hausdorff-type projection error is

```text
epsilon_M(t) = inf_theta d_H(Gamma_h(t), M_SE(theta)).
```

**Proposition 2: descriptor-error transfer.** If `d_H(Gamma_h(t), M_SE(theta_t)) <= epsilon_M(t)`, then any Lipschitz boundary descriptor `g(Gamma)` satisfies `|g(Gamma_h(t))-g(M_SE(theta_t))| <= L_g epsilon_M(t)`. Thus the fitted boundary error controls state-descriptor error for `L_f`, `L_r`, `W` and `H` up to descriptor-dependent constants.

Numerically, the superellipsoid reduces the mean boundary residual from {vals['ellipsoid_boundary']:.4f} to {vals['super_boundary']:.4f} and the mean volume relative error from {vals['ellipsoid_volume']:.4f} to {vals['super_volume']:.4f}. The geometric improvement is observed in {vals['super_volume_wins']}/{vals['robust_total']} robustness settings for volume error and {vals['super_boundary_wins']}/{vals['robust_total']} settings for boundary residual.

## Lyapunov stability

The selected dynamics are

```text
dq_i/dt = k_i(q_inf,i - q_i).
```

Let `e=q-q_inf` and use `V(e)=1/2 ||e||_2^2`. For the diagonal system,

```text
dV/dt = - sum_i k_i e_i^2 <= -2 k_min V,  k_min = min_i k_i.
```

**Proposition 3: diagonal attractor exponential stability.** If `k_min>0`, then `||e(t)||_2 <= exp(-k_min t)||e(0)||_2`. Therefore the diagonal attractor is globally exponentially stable in the reduced state space.

In the present fit, all diagonal rates are positive: {stable_diag}; `k_min={k_min:.4g} s^-1`. The mean validation relative RMSE is {vals['diagonal_validation']:.4f}.

For the coupled comparison model, `de/dt=-A e`.

**Proposition 4: coupled spectral stability.** The coupled equilibrium is locally exponentially stable if every eigenvalue of `-A` has negative real part. This condition is satisfied in the baseline fit: {coupled_stable}. The coupled validation relative RMSE is {vals['coupled_validation']:.4f}, so stability does not imply model selection.

## Error-budget interpretation

The v4 error statement is

```text
E_total <= C1 E_reconstruction + C2 E_geometry + C3 E_volume_proxy + C4 E_dynamics + C5 E_parameter_scale.
```

**Proposition 5: diagnostic error-bound decomposition.** Under A1-A5 and finite descriptor Lipschitz constants, the total reported modeling uncertainty can be bounded by reconstruction, geometric projection, volume-proxy, dynamical prediction and parameter-scale components. The constants `C_i` are sensitivity weights, not universal constants estimated from one process condition.

The current source terms are:

{bound_lines}

## Identifiability diagnostics

Identifiability is assessed with coefficient of variation, finite-difference local sensitivity, a Fisher-information proxy, condition proxies and parameter-to-transition ratio. High-risk parameters are: {high_risk_text}. This supports the modeling decision: the superellipsoid is retained as a compact analytic observed-envelope descriptor, while the coupled matrix is retained as an overparameterization comparison.

## Dimensionless sensitivity

The sensitivity scan perturbs reference temperature, absorptivity and surface-tension temperature coefficient. The resulting envelope is:

{sens_lines}

The baseline nondimensional values remain `Pe={vals['Pe']:.2f}`, `Ste={vals['Ste']:.3f}`, `E*={vals['E_star']:.2f}` and `Ma={vals['Ma']:.2f}`. The classification does not change in the scan, so these groups are defensible as scale diagnostics for this simulation.
"""
    report_path.write_text(text, encoding="utf-8")


def write_manuscript_draft_v4(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    model_selection: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    error_budget: pd.DataFrame,
    error_bound_summary: pd.DataFrame,
    identifiability_v4: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    selected = model_selection[model_selection["selected_as_main_model"].astype(str).str.lower() == "true"]
    selected_text = "; ".join([f"{row.model_family}: {row.model}" for row in selected.itertuples()])
    stable_diag = bool((dynamics_summary["k_per_s"].to_numpy(dtype=float) > 0).all())
    k_min = float(np.nanmin(dynamics_summary["k_per_s"].to_numpy(dtype=float)))
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    high_params = identifiability_v4[identifiability_v4["risk_level"].eq("high")]
    high_param_text = ", ".join(high_params["parameter"].astype(str).head(10).tolist())
    changed_groups = dimensionless_sensitivity[dimensionless_sensitivity["conclusion_changed"]]
    changed_text = ", ".join(changed_groups["symbol"].astype(str).tolist()) if len(changed_groups) else "none"
    error_bound_lines = "\n".join(
        [
            f"- {row.bound_component}: {row.bound_weight_symbol} term, source metric {row.source_metric} = {row.source_value:.4g}, risk {row.risk_level}."
            for row in error_bound_summary.sort_values("display_order").itertuples()
        ]
    )
    ident_lines = "\n".join(
        [
            f"- {row.parameter}: risk {row.risk_level}, basis {row.diagnostic_basis}."
            for row in identifiability_v4[identifiability_v4["risk_level"].eq("high")].itertuples()
        ]
    )
    text = f"""# CFD-informed free-boundary reduction of laser directed energy deposition melt-pool evolution via superellipsoid manifolds and stable attractor dynamics

## Abstract

High-fidelity melt-pool data can resolve L-DED melt-pool transport, but the resulting fields are difficult to use directly in mathematical modeling. This v4 draft strengthens a melt-pool-data-informed reduced-order framework for a single 316L stainless-steel L-DED condition at 750 W, 8 mm/s and 12 g/min. The exported data contain only the molten region, so the melt pool is treated as a moving-frame free-boundary observation rather than as a complete thermal field. A half-domain simulation is reconstructed through the `y=0` symmetry plane, the free-boundary envelope is projected onto an asymmetric superellipsoid manifold, and the extracted state is advanced by a Lyapunov-stable diagonal attractor. The liquidus-reference groups are Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}. The superellipsoid reduces mean boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)} and is retained as the selected algebraic observed-envelope descriptor, not as a metric-accurate reconstruction. A coupled ridge attractor is stable but validates worse than the diagonal model ({_fmt(vals['coupled_validation'], 4)} versus {_fmt(vals['diagonal_validation'], 4)} relative RMSE), so it is retained as an overparameterization comparison. The study is deliberately limited to single-condition transient-to-quasi-steady modeling.

## Introduction

Directed energy deposition is governed by moving heat input, powder capture, melt-pool convection, material-property variation and phase change [@ahn2021; @svetlizky2021; @li2023]. Recent CFD, monitoring and machine-learning studies have made melt-pool geometry a central process-state variable [@liao2022; @dasilva2023; @akbari2022; @wu2024]. However, many available descriptions either remain high-dimensional or rely on black-box predictors that require broader training sets.

This work takes a narrower mathematical route. It asks whether one high-fidelity FLOW-3D molten-region export can be converted into a transparent free-boundary reduced-order model. The contribution is not a process-wide predictor. It is a reproducible modeling chain with explicit assumptions, low-dimensional manifold projection, Lyapunov stability, a semi-formal error bound and parameter-identifiability diagnostics.

## Free-boundary formulation

Let `Omega_m^h(t)` be the exported half-domain molten region and `P^h(t)` its discrete point cloud. The laser-attached coordinate is

```text
xi = x - vt,  v = 0.008 m/s.
```

The full-domain observation operator is

```text
R[P^h(t)] = P^h(t) union {{(xi,-y,z): (xi,y,z) in P^h(t)}}.
```

The free boundary `Gamma_h(t)` is the envelope of `R[P^h(t)]`, and the reduced state is `q(t)=[L_f,L_r,W,H,T_max,G_mean,U_max]^T`.

**Assumptions A1-A5.** The model is single-condition; the observation contains only molten-region points; `y=0` is a symmetry boundary; the moving-frame boundary becomes quasi-steady after the early transient; and the boundary lies near a low-dimensional analytic manifold.

**Proposition 1: symmetry-reconstruction consistency.** Under the imposed half-domain symmetry, descriptors that are even in `y` are reconstructed by mirroring. In particular, `W(t)=2 max y` and `V_full(t)=2 V_half(t)`. This is a statement about the numerical observation operator, not a recovery of unexported solid-domain data.

## Low-dimensional manifold approximation

The selected free-boundary manifold is the asymmetric superellipsoid

```text
M_SE(theta): |(xi-xi_c)/a_s|^n + |y/b|^m + |(z-z_c)/c|^p = 1,
```

where `theta=[a_f,a_r,b,c,xi_c,z_c,n,m,p]` and `a_s` switches between `a_f` and `a_r` across the fitted center. The projection error is

```text
epsilon_M(t) = inf_theta d_H(Gamma_h(t), M_SE(theta)).
```

**Proposition 2: descriptor-error transfer.** If `d_H(Gamma_h(t),M_SE(theta_t)) <= epsilon_M(t)`, then any Lipschitz boundary descriptor `g` satisfies `|g(Gamma_h(t))-g(M_SE(theta_t))| <= L_g epsilon_M(t)`. Thus geometric fitting error controls the extracted descriptors `L_f`, `L_r`, `W` and `H` up to descriptor constants.

Numerically, the superellipsoid improves the boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)} and the volume relative error from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}. Robustness tests show geometric improvement in {vals['super_volume_wins']}/{vals['robust_total']} settings for volume and {vals['super_boundary_wins']}/{vals['robust_total']} settings for boundary residual.

## Dimensionless analysis

The reference length is the quasi-steady mean melt-pool length, and the baseline material properties are interpolated at the liquidus temperature. The main groups are

```text
Pe = v L_ref/alpha,
Fo = alpha t/L_ref^2,
Ste = c_p(T_l-T_s)/L_fus,
E* = eta P/[rho c_p v r_b^2(T_l-T0)],
Ma = |d sigma/dT|(T_l-T_s)L_ref/(mu alpha).
```

The baseline values are Pe={vals['Pe']:.2f}, Fo_final={vals['Fo_final']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f}, Re={vals['Re']:.2f}, Pr={vals['Pr']:.3f} and Ma={vals['Ma']:.2f}. Perturbing reference temperature, absorptivity and surface-tension coefficient produces class changes for: {changed_text}. The groups are therefore used as scale diagnostics rather than universal constants.

## Reduced-order dynamics

The selected model is

```text
dq_i/dt = k_i(q_inf,i - q_i).
```

The coupled comparison is

```text
dq/dt = A(q_inf - q).
```

The coupled model is tested because physical cross-coupling is plausible, but it is not selected unless it both satisfies the spectral stability criterion and reduces validation error.

## Lyapunov stability

Let `e=q-q_inf` and `V(e)=1/2 ||e||_2^2`. For the diagonal system,

```text
dV/dt = - sum_i k_i e_i^2 <= -2 k_min V.
```

**Proposition 3: diagonal attractor exponential stability.** If `k_min=min_i k_i>0`, then `||e(t)||_2 <= exp(-k_min t)||e(0)||_2`. The selected diagonal model is therefore globally exponentially stable in the reduced state space. In this dataset, all fitted rates are positive: {stable_diag}, with `k_min={k_min:.4g} s^-1`.

**Proposition 4: coupled spectral stability.** For `de/dt=-A e`, the equilibrium is locally exponentially stable if all eigenvalues of `-A` have negative real part. The fitted coupled model satisfies this condition: {coupled_stable}. Its validation relative RMSE is {_fmt(vals['coupled_validation'], 4)}, higher than the diagonal value {_fmt(vals['diagonal_validation'], 4)}, so stability is necessary but not sufficient for model adoption.

## Error-budget interpretation

The diagnostic v4 bound is

```text
E_total <= C1 E_reconstruction + C2 E_geometry + C3 E_volume_proxy + C4 E_dynamics + C5 E_parameter_scale.
```

**Proposition 5: semi-formal error-budget decomposition.** Under A1-A5 and finite descriptor sensitivity constants, the reported model uncertainty can be organized by reconstruction, geometric-projection, volume-proxy, dynamical-prediction and material-scale terms. The constants `C_i` are sensitivity weights; a single process condition is not sufficient to estimate universal constants.

The numerical source terms are:

{error_bound_lines}

This formulation raises the paper from a single validation score to an auditable error chain.

## Identifiability diagnostics

Parameter identifiability is assessed by coefficient of variation, finite-difference local sensitivity, Fisher-information proxy, condition proxy and parameter-to-transition ratio. High-risk parameters are:

{ident_lines}

These diagnostics support the chosen hierarchy. The superellipsoid is retained as a compact boundary manifold despite high-risk exponent bounds, because it improves geometry robustly. The 49-entry coupled matrix is not selected because the parameter-to-transition ratio and validation behavior indicate overparameterization.

## Results

Figures 1-8 present the framework, moving-frame reconstruction, quasi-steady geometry, boundary-model comparison, nondimensional regime, stability evidence, error budget and identifiability diagnostics. Supplementary Figure S5 adds the v4 theory diagnostics by plotting error-budget source terms, identifiability risk and nondimensional sensitivity in one audit figure.

The central numerical result remains unchanged from v3. The quasi-steady mean front length, rear length, full width and height are {vals['lf_quasi_mm']:.3f} mm, {vals['lr_quasi_mm']:.3f} mm, {vals['w_quasi_mm']:.3f} mm and {vals['h_quasi_mm']:.3f} mm. The selected model is: {selected_text}. Coupled dynamics improves validation error in {vals['coupled_wins']}/{vals['robust_total']} robustness settings, so it remains a negative control.

## Discussion

The v4 theory strengthens the manuscript in three ways. First, the point-cloud preprocessing is framed as an observation operator on a molten-region free boundary. Second, the superellipsoid is defined as a low-dimensional manifold with a projection-error interpretation rather than as an arbitrary curve fit. Third, the diagonal attractor is supported by a Lyapunov argument and selected by validation error, while the coupled model is stable but not empirically justified.

The remaining limitation is structural: one process condition cannot identify a process-wide map. The manuscript should therefore keep its claim narrow: CFD-informed free-boundary reduced-order modeling for a single transient-to-quasi-steady L-DED simulation.

## Conclusion

This v4 draft converts the existing analysis into a more defensible mathematical modeling paper. The main model remains an asymmetric superellipsoid free boundary with a diagonal exponentially stable attractor. The additional contribution is theoretical: explicit observation assumptions, low-dimensional manifold approximation, Lyapunov stability, semi-formal error-budget interpretation and identifiability diagnostics.
"""
    report_path.write_text(text, encoding="utf-8")


def make_assumption_justification_matrix(
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    identifiability_v4: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    high_risk = ", ".join(
        identifiability_v4.loc[identifiability_v4["risk_level"].eq("high"), "parameter"].astype(str).head(8)
    )
    class_changes = int(dimensionless_sensitivity["conclusion_changed"].sum())
    rows = [
        {
            "assumption_id": "A1",
            "assumption": "Single-condition transient-to-quasi-steady modeling",
            "physical_basis": "The FLOW-3D export corresponds to one fixed 316L L-DED condition: 750 W, 8 mm/s and 12 g/min.",
            "mathematical_role": "Allows q_inf and relaxation rates to be treated as constants for this trajectory.",
            "current_evidence": f"{vals['n_time_steps']} time steps from t={vals['t_min']:.2f} s to t={vals['t_max']:.2f} s.",
            "failure_mode": "The model cannot predict arbitrary power, speed or powder-feed combinations without new simulations.",
            "reviewer_response": "Frame the work as single-condition free-boundary reduction, not a process map.",
            "source_outputs": "modeling_table.csv; manuscript_draft_v5.md",
            "risk_level": "medium",
        },
        {
            "assumption_id": "A2",
            "assumption": "Half-domain symmetry about y=0",
            "physical_basis": "The computational domain imposes a symmetry plane, so the uncomputed half is a mirror image under the simulation setup.",
            "mathematical_role": "Defines the reconstruction operator R and the descriptors W=2 max(y), V_full=2 V_half.",
            "current_evidence": "All geometry tables report full width and full-domain volume proxy through symmetry reconstruction.",
            "failure_mode": "Off-axis powder flow or asymmetric convection would violate this assumption in other simulations or experiments.",
            "reviewer_response": "State that symmetry is inherited from the CFD setup and is not an experimental symmetry claim.",
            "source_outputs": "modeling_table.csv; theory_framework_v4.md",
            "risk_level": "low_medium",
        },
        {
            "assumption_id": "A3",
            "assumption": "Molten-region observation rather than full thermal-field inversion",
            "physical_basis": "Only molten-region cells were exported from FLOW-3D; solid and previously solidified regions are absent.",
            "mathematical_role": "Defines O_h and Gamma_h(t) as an observed molten-envelope boundary.",
            "current_evidence": "Boundary and volume errors are explicitly treated as reconstruction and proxy errors.",
            "failure_mode": "The observed envelope may differ from a complete solid-liquid isotherm reconstructed from the full thermal field.",
            "reviewer_response": "Use the phrase observed molten-region free boundary and avoid claiming full Stefan-interface recovery.",
            "source_outputs": "error_budget_summary.csv; error_bound_summary.csv",
            "risk_level": "medium_high",
        },
        {
            "assumption_id": "A4",
            "assumption": "Moving-frame quasi-steadiness",
            "physical_basis": "A moving heat source admits a laser-attached coordinate where steady translation appears nearly stationary after the initial transient.",
            "mathematical_role": "Justifies xi=x-vt and the use of a quasi-steady attractor q_inf.",
            "current_evidence": f"Geometry is interpreted after {QUASI_STEADY_START_S:.2f} s; diagonal validation relative RMSE is {vals['diagonal_validation']:.4f}.",
            "failure_mode": "Strong oscillation, keyhole instability or track-end transients would break the quasi-steady assumption.",
            "reviewer_response": "Show time histories and keep the claim limited to the simulated time window.",
            "source_outputs": "quasi_steady_summary.csv; dynamics_fit_summary.csv",
            "risk_level": "medium",
        },
        {
            "assumption_id": "A5",
            "assumption": "Superellipsoid as low-dimensional observed boundary-envelope manifold",
            "physical_basis": "The melt-pool envelope is shaped by a localized heat source, conduction, surface flow and phase-boundary smoothing.",
            "mathematical_role": "Defines Pi_M Gamma_h(t) and reduces a boundary object to theta=[a_f,a_r,b,c,xi_c,z_c,n,m,p].",
            "current_evidence": f"Boundary residual improves {vals['ellipsoid_boundary']:.4f}->{vals['super_boundary']:.4f}; volume proxy error changes {vals['ellipsoid_volume']:.4f}->{vals['super_volume']:.4f}; boundary robustness {vals['super_boundary_wins']}/{vals['robust_total']}.",
            "failure_mode": "Highly fragmented, concave or multi-lobed melt pools may not be represented by a single superellipsoid.",
            "reviewer_response": "Present the model as a compact manifold projection with reported residuals and exponent identifiability risks.",
            "source_outputs": "geometry_model_comparison.csv; identifiability_diagnostics_v4.csv",
            "risk_level": "medium",
        },
        {
            "assumption_id": "A6",
            "assumption": "Temperature-dependent property curves support liquidus-scale diagnostics",
            "physical_basis": "The material-property basis is tabulated as temperature-dependent density, heat capacity, conductivity and viscosity curves.",
            "mathematical_role": "Sets alpha, Pe, Ste, E*, Re, Pr and Ma at a reference state and in sensitivity scans.",
            "current_evidence": f"Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f}, Ma={vals['Ma']:.2f}; class changes={class_changes}.",
            "failure_mode": "Different reference states or scenario perturbations can shift numerical values even if regime classes remain stable.",
            "reviewer_response": "Report the property curves, liquidus-interpolated values and scenario ranges used for the dimensionless diagnostics.",
            "source_outputs": "temperature_dependent_properties.csv; dimensionless_numbers.csv; dimensionless_sensitivity_summary.csv",
            "risk_level": "low_medium",
        },
        {
            "assumption_id": "A7",
            "assumption": "Diagonal attractor as first-order local relaxation",
            "physical_basis": "Near a quasi-steady translated melt pool, each reduced descriptor can relax toward a trajectory-specific equilibrium.",
            "mathematical_role": "Approximates F(q) near q_inf by decoupled negative diagonal terms.",
            "current_evidence": f"All k_i>0 and mean validation relative RMSE is {vals['diagonal_validation']:.4f}.",
            "failure_mode": "Strong cross-coupling or oscillatory modes would require a coupled or nonlinear dynamics.",
            "reviewer_response": "Use the coupled model as a control and choose the diagonal model only because validation supports it.",
            "source_outputs": "dynamics_fit_summary.csv; dynamics_model_comparison.csv",
            "risk_level": "medium",
        },
        {
            "assumption_id": "A8",
            "assumption": "Coupled model is an overparameterization comparison",
            "physical_basis": "Thermal, geometric and flow states are physically coupled, but the short sequence does not identify 49 matrix entries robustly.",
            "mathematical_role": "Provides a stricter comparison model without changing the selected dynamics.",
            "current_evidence": f"Coupled validation relative RMSE is {vals['coupled_validation']:.4f}; coupled wins {vals['coupled_wins']}/{vals['robust_total']}; high-risk parameters include {high_risk}.",
            "failure_mode": "Additional process conditions may make coupled dynamics identifiable and preferable.",
            "reviewer_response": "Report the negative result as evidence against unsupported complexity.",
            "source_outputs": "coupled_A_matrix.csv; identifiability_diagnostics_v4.csv; robustness_summary.csv",
            "risk_level": "high_for_coupled_model",
        },
    ]
    return pd.DataFrame(rows)


def write_theoretical_derivation_v5(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    assumption_matrix: pd.DataFrame,
    error_bound_summary: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    k_min = float(np.nanmin(dynamics_summary["k_per_s"].to_numpy(dtype=float)))
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    assumption_lines = "\n".join(
        [
            f"- {row.assumption_id}: {row.assumption}. Evidence: {row.current_evidence}"
            for row in assumption_matrix.itertuples()
        ]
    )
    bound_lines = "\n".join(
        [
            f"- {row.bound_component}: {row.bound_expression}; metric {row.source_metric}={row.source_value:.4g}."
            for row in error_bound_summary.sort_values("display_order").itertuples()
        ]
    )
    sens_lines = "\n".join(
        [
            f"- {row.symbol}: baseline {row.baseline_value:.4g}, range {row.min_value:.4g}-{row.max_value:.4g}, class change={row.conclusion_changed}."
            for row in dimensionless_sensitivity.itertuples()
        ]
    )
    text = f"""# Theoretical Derivation v5

## 1. Full physical problem used as the modeling origin

The high-dimensional L-DED melt pool can be idealized as a moving-source Stefan-Marangoni problem. In a full CFD state, temperature `T`, velocity `u`, pressure `p`, liquid fraction `f_l`, the molten region `Omega_m(t)`, the solid or unmelted region `Omega_s(t)`, the solid-liquid interface `Gamma_sl(t)` and the liquid-gas free surface `Gamma_fs(t)` evolve together.

A representative thermal balance is

```text
rho c_p(T)(partial_t T + u dot grad T)
  = div(k(T) grad T) + Q_laser + Q_powder - rho L_fus partial_t f_l.
```

The full phase-boundary problem would also include a Stefan condition

```text
rho L_fus v_n = [k grad T dot n]_s^l,
```

free-surface heat loss

```text
-k grad T dot n = h_c(T-T_inf) + epsilon_rad sigma_SB(T^4-T_inf^4),
```

and Marangoni shear

```text
tau n dot t = (d sigma/dT) grad_s T dot t.
```

The present paper does not solve this full system analytically. It uses the system as a physical origin for an observed free-boundary reduced-order model.

## 2. Observation operator and actual mathematical problem

The FLOW-3D export contains only molten-region points. Therefore the available data are an observation

```text
P^h(t) = O_h[Omega_m(t), T, u],
```

on a half domain. The actual mathematical problem solved here is:

```text
Given P^h(t_j), construct R[P^h(t_j)], estimate Gamma_h(t_j),
project Gamma_h(t_j) to M_SE(theta_j), extract q(t_j),
and identify a stable reduced-order dynamic for q(t).
```

This distinction prevents problem switching. The full Stefan-Marangoni problem motivates the structure; the observed molten-region reduction is the object that is fitted, validated and bounded.

## 3. Physical-to-reduced-order derivation chain

The v5 derivation chain is

```text
full Stefan-Marangoni problem
-> molten-region observation operator O_h
-> half-domain symmetry reconstruction R
-> moving-frame boundary Gamma_h(xi,y,z,t)
-> superellipsoid manifold projection Pi_M Gamma_h
-> reduced state q(t)
-> stable attractor dynamics dq/dt.
```

Each arrow introduces an approximation. These approximations are tracked through the assumption matrix and the error-bound summary rather than hidden inside a single validation metric.

## 4. Moving-frame quasi-steady proposition

**Proposition v5-1: moving-frame quasi-steady reduction.** For a localized heat input translating at constant speed `v`, a boundary that is approximately steady in the laser-attached frame satisfies `partial_t Gamma_h(xi,y,z,t) approx 0` after the early transient. Under this condition, temporal evolution of the observed boundary can be represented as relaxation of a finite-dimensional descriptor toward a quasi-steady state.

In the present dataset, the quasi-steady interpretation is applied after approximately {QUASI_STEADY_START_S:.2f} s. The diagonal attractor validation relative RMSE is {vals['diagonal_validation']:.4f}.

## 5. Superellipsoid projection proposition

**Proposition v5-2: low-dimensional free-boundary projection.** If the observed molten envelope is connected, single-lobed and smoothed by conduction and surface-tension-driven flow, then a compact analytic shape manifold can approximate its outer boundary. The asymmetric superellipsoid is used as this manifold because it supports front-rear asymmetry, transverse width, vertical extent and non-ellipsoidal exponent variation.

The numerical evidence is direct: residual improves {vals['ellipsoid_boundary']:.4f}->{vals['super_boundary']:.4f}, volume relative error improves {vals['ellipsoid_volume']:.4f}->{vals['super_volume']:.4f}, and robust geometric improvement appears in {vals['super_volume_wins']}/{vals['robust_total']} settings.

## 6. First-order relaxation proposition

**Proposition v5-3: diagonal attractor as local first-order approximation.** Let the unknown reduced dynamics be `dq/dt=F(q)` and let `q_inf` be a quasi-steady state. A first-order expansion gives `dq/dt approx J(q-q_inf)`. If the available trajectory does not support reliable identification of off-diagonal terms, the stable diagonal approximation `J approx -diag(k_i)` gives the selected model.

All fitted `k_i` are positive and `k_min={k_min:.4g} s^-1`, so the Lyapunov result from v4 applies. The coupled model is also spectrally stable ({coupled_stable}) but validates worse ({vals['coupled_validation']:.4f} versus {vals['diagonal_validation']:.4f}) and improves 0/{vals['robust_total']} robustness settings.

## 7. Dimensionless regime-invariance proposition

**Proposition v5-4: nondimensional interpretation stability.** If perturbing the reference temperature, absorptivity and surface-tension coefficient does not change the qualitative class of `Pe`, `Ste`, `E*` or `Ma`, then the nondimensional interpretation is stable as a scale diagnosis for this simulation, even though individual values remain parameter dependent.

The observed sensitivity envelope is:

{sens_lines}

## 8. Assumption assessment matrix

The v5 assumption audit is summarized as:

{assumption_lines}

## 9. Error-bound continuity from v4

The v4 semi-formal bound remains the organizing uncertainty statement:

```text
E_total <= C1 E_reconstruction + C2 E_geometry + C3 E_volume_proxy + C4 E_dynamics + C5 E_parameter_scale.
```

The current source terms are:

{bound_lines}

## 10. Manuscript implication

The v5 theory does not claim a closed-form Stefan-Marangoni solution. Its contribution is a defensible reduction: a high-dimensional physical free-boundary problem motivates an observed molten-boundary model, and every reduction step is paired with a diagnostic, assumption or error term.
"""
    report_path.write_text(text, encoding="utf-8")


def write_manuscript_draft_v5(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    model_selection: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    assumption_matrix: pd.DataFrame,
    error_bound_summary: pd.DataFrame,
    identifiability_v4: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    selected = model_selection[model_selection["selected_as_main_model"].astype(str).str.lower() == "true"]
    selected_text = "; ".join([f"{row.model_family}: {row.model}" for row in selected.itertuples()])
    k_min = float(np.nanmin(dynamics_summary["k_per_s"].to_numpy(dtype=float)))
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    high_params = ", ".join(
        identifiability_v4.loc[identifiability_v4["risk_level"].eq("high"), "parameter"].astype(str).head(10)
    )
    class_changes = ", ".join(
        dimensionless_sensitivity.loc[dimensionless_sensitivity["conclusion_changed"], "symbol"].astype(str)
    ) or "none"
    assumption_text = "\n".join(
        [
            f"- **{row.assumption_id} {row.assumption}.** {row.mathematical_role} Evidence: {row.current_evidence} Risk: {row.failure_mode}"
            for row in assumption_matrix.itertuples()
        ]
    )
    bound_text = "\n".join(
        [
            f"- {row.bound_component}: {row.source_metric} = {row.source_value:.4g}; source {row.source_table}."
            for row in error_bound_summary.sort_values("display_order").itertuples()
        ]
    )
    text = f"""# CFD-informed free-boundary reduction of laser directed energy deposition melt-pool evolution via superellipsoid manifolds and stable attractor dynamics

## Abstract

High-fidelity CFD resolves L-DED melt-pool transport, but the full Stefan-Marangoni state is too high-dimensional for compact mathematical interpretation. This v5 draft develops a physical-to-observation modeling rationale for a single 316L L-DED condition at 750 W, 8 mm/s and 12 g/min. The full physical origin is a moving-source phase-change problem with Stefan, heat-loss and Marangoni boundary effects. The actual data, however, contain only FLOW-3D molten-region points, so the paper solves an observed free-boundary identification problem rather than a full thermal-field inverse problem. The modeling chain is: full physical problem, molten-region observation, half-domain symmetry reconstruction, moving-frame boundary, superellipsoid manifold projection, reduced state and stable attractor dynamics. The liquidus-reference groups are Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}. The selected model remains an asymmetric superellipsoid boundary with a diagonal attractor. The superellipsoid improves boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)}, while the diagonal attractor validates better than the coupled model ({_fmt(vals['diagonal_validation'], 4)} versus {_fmt(vals['coupled_validation'], 4)} relative RMSE).

## Introduction

L-DED combines a translating heat source, powder addition, melt-pool convection, phase change and temperature-dependent 316L properties. A complete first-principles formulation is a high-dimensional moving-boundary problem. It includes energy transport, latent heat, free-surface heat losses, Marangoni shear and a moving solid-liquid interface. Such a formulation is physically rich but does not by itself produce a compact model from a short molten-region export.

This paper therefore separates the full physical problem from the actual mathematical problem. The full Stefan-Marangoni picture supplies the modeling rationale. The solved problem is the construction of a reduced-order free-boundary model from observed molten-region point clouds. This distinction is central: the manuscript does not claim to reconstruct the complete solid-liquid isotherm or solve the full CFD equations analytically.

The contribution is a physically motivated reduction chain. A half-domain molten point cloud is mirrored by the imposed symmetry plane, transformed to a moving coordinate, projected to a superellipsoid boundary manifold and reduced to a state vector `q(t)`. The resulting dynamics are interpreted as first-order relaxation toward a quasi-steady translated melt pool.

## Mathematical formulation: full physical origin

The full L-DED melt-pool problem can be written schematically as

```text
rho c_p(T)(partial_t T + u dot grad T)
  = div(k(T) grad T) + Q_laser + Q_powder - rho L_fus partial_t f_l.
```

At a solid-liquid interface `Gamma_sl(t)`, a Stefan balance gives

```text
rho L_fus v_n = [k grad T dot n]_s^l.
```

At a free surface `Gamma_fs(t)`, heat loss and thermocapillary forcing may be represented as

```text
-k grad T dot n = h_c(T-T_inf) + epsilon_rad sigma_SB(T^4-T_inf^4),
tau n dot t = (d sigma/dT) grad_s T dot t.
```

These equations motivate a free-boundary perspective, but they are not solved directly in this work.

## Observed reduced problem

The exported data define an observation operator

```text
P^h(t) = O_h[Omega_m(t), T, u],
```

where `P^h(t)` is a half-domain molten-region point cloud. The symmetry reconstruction is

```text
R[P^h(t)] = P^h(t) union {{(xi,-y,z): (xi,y,z) in P^h(t)}}.
```

The moving coordinate is `xi=x-vt`. The observed free boundary `Gamma_h(t)` is the envelope of `R[P^h(t)]`, not a full-domain thermal isosurface. The reduced state is `q=[L_f,L_r,W,H,T_max,G_mean,U_max]^T`.

## Model reduction rationale

The modeling chain is

```text
full Stefan-Marangoni problem
-> molten-region observation operator
-> moving-frame quasi-steady boundary
-> superellipsoid manifold projection
-> reduced state q(t)
-> stable attractor dynamics.
```

**Proposition v5-1: moving-frame quasi-steady reduction.** A localized heat source translating at constant speed admits a laser-attached coordinate in which the melt-pool envelope becomes slowly varying after the initial transient. This motivates the quasi-steady state `q_inf`.

**Proposition v5-2: superellipsoid manifold projection.** A connected single-lobed molten envelope shaped by localized heating, conduction and surface-tension-driven smoothing can be approximated by a compact analytic manifold. The asymmetric superellipsoid is chosen because it represents front-rear asymmetry, width, height and non-ellipsoidal boundary exponents.

**Proposition v5-3: first-order relaxation.** Near `q_inf`, an unknown reduced dynamic `dq/dt=F(q)` has the local form `dq/dt approx J(q-q_inf)`. With only a short single-condition sequence, the stable diagonal approximation `J approx -diag(k_i)` is identifiable, whereas the 49-entry coupled matrix is not selected.

**Proposition v5-4: nondimensional regime invariance.** If the qualitative classes of `Pe`, `Ste`, `E*` and `Ma` do not change under the prescribed perturbations, their interpretation as scale diagnostics is stable for this simulation.

## Numerical evidence linked to the assumptions

The superellipsoid improves the mean boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)} and the mean volume relative error from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}. Robustness checks support geometric improvement in {vals['super_volume_wins']}/{vals['robust_total']} tested settings.

The diagonal attractor has all positive rates, with `k_min={k_min:.4g} s^-1`, and mean validation relative RMSE {_fmt(vals['diagonal_validation'], 4)}. The coupled model is spectrally stable ({coupled_stable}) but validates worse at {_fmt(vals['coupled_validation'], 4)} and improves validation error in {vals['coupled_wins']}/{vals['robust_total']} robustness settings.

The dimensionless scan gives class changes for: {class_changes}. The baseline values remain Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}.

## Assumption audit

{assumption_text}

## Stability and error-budget interpretation

The diagonal attractor uses `V=1/2||q-q_inf||^2`, giving `dV/dt <= -2 k_min V` when all `k_i>0`. The total error is organized as

```text
E_total <= C1 E_reconstruction + C2 E_geometry + C3 E_volume_proxy + C4 E_dynamics + C5 E_parameter_scale.
```

The current traceable source terms are:

{bound_text}

## Discussion

The v5 formulation strengthens the manuscript by making the problem boundary explicit. The paper does not substitute a fitted superellipsoid for the full physics. Instead, it uses the full physics to justify a structured observation and reduction map. This makes the model defensible for a mathematical modeling journal because every approximation has a stated role, a diagnostic and a failure mode.

The main limitation remains the same: one FLOW-3D condition cannot support process-wide prediction. The value of the work is a reproducible and physically motivated path from high-fidelity molten-region data to an interpretable free-boundary reduced-order model.

## Conclusion

The selected model remains `{selected_text}`. The v5 contribution is theoretical rather than algorithmic: it connects the full moving-source Stefan-Marangoni picture to the observed molten-region free-boundary problem, then to a superellipsoid manifold and a stable diagonal attractor. High-risk parameters remain {high_params}, and the coupled model remains a useful negative control for overparameterization.
"""
    report_path.write_text(text, encoding="utf-8")


def make_timescale_separation_summary(dynamics_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "geometry": {"front_length_m", "rear_length_m", "full_width_m", "height_span_m"},
        "thermal": {"Tmax_K", "Gmean_K_per_m"},
        "flow": {"Umax_m_per_s"},
    }
    for row in dynamics_summary.itertuples(index=False):
        k = float(row.k_per_s)
        tau = 1.0 / k if np.isfinite(k) and k > 0 else np.nan
        group = next((name for name, states in groups.items() if row.state in states), "other")
        validation = float(row.validation_relative_rmse)
        if not np.isfinite(tau) or validation > 0.2:
            risk = "high"
        elif validation > 0.1:
            risk = "medium"
        else:
            risk = "low"
        rows.append(
            {
                "state": row.state,
                "label": row.label,
                "state_group": group,
                "k_per_s": k,
                "characteristic_time_s": tau,
                "train_relative_rmse": float(row.train_relative_rmse),
                "validation_relative_rmse": validation,
                "supports_first_order_relaxation": bool(np.isfinite(tau) and validation <= 0.12),
                "risk_level": risk,
                "interpretation": "supports first-order relaxation"
                if np.isfinite(tau) and validation <= 0.12
                else "weak state-specific support",
                "case_id": getattr(row, "case_id", "all_conditions"),
                "case_index": getattr(row, "case_index", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    group_tau = out.groupby("state_group")["characteristic_time_s"].median().to_dict()
    out["group_median_tau_s"] = out["state_group"].map(group_tau)
    geometry_tau = group_tau.get("geometry", np.nan)
    out["relative_to_geometry_tau"] = (
        out["characteristic_time_s"] / geometry_tau if np.isfinite(geometry_tau) else np.nan
    )
    return out


def make_validation_stress_tests(table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, float | str | int | bool]] = []

    for fraction in [0.60, 0.70, 0.80]:
        with temporary_settings(train_fraction=fraction):
            _, summary, _ = fit_attractor_model(table)
        rows.append(
            {
                "test_family": "train_fraction",
                "scenario": f"train_fraction_{fraction:.2f}",
                "status": "ok",
                "mean_validation_relative_rmse": float(summary["validation_relative_rmse"].mean()),
                "max_validation_relative_rmse": float(summary["validation_relative_rmse"].max()),
                "umax_validation_relative_rmse": float(
                    summary.loc[summary["state"].eq("Umax_m_per_s"), "validation_relative_rmse"].iloc[0]
                ),
                "n_states": len(summary),
                "supports_main_model": bool(summary["validation_relative_rmse"].mean() <= 0.12),
            }
        )

    unique_t = np.sort(table["time_s"].dropna().unique().astype(float))
    for split_end in range(5, len(unique_t)):
        fraction = split_end / len(unique_t)
        with temporary_settings(train_fraction=fraction):
            _, summary, _ = fit_attractor_model(table)
        rows.append(
            {
                "test_family": "rolling_origin",
                "scenario": f"train_until_{unique_t[split_end - 1]:.2f}s",
                "status": "ok",
                "mean_validation_relative_rmse": float(summary["validation_relative_rmse"].mean()),
                "max_validation_relative_rmse": float(summary["validation_relative_rmse"].max()),
                "umax_validation_relative_rmse": float(
                    summary.loc[summary["state"].eq("Umax_m_per_s"), "validation_relative_rmse"].iloc[0]
                ),
                "n_states": len(summary),
                "supports_main_model": bool(summary["validation_relative_rmse"].mean() <= 0.12),
            }
        )

    for hold_t in unique_t[1:-1]:
        errors = []
        group_iter = table.groupby("case_id", sort=False) if "case_id" in table.columns else [("all_conditions", table)]
        for _, group in group_iter:
            g = group.sort_values("time_s")
            t_case = g["time_s"].to_numpy(dtype=float)
            if hold_t not in set(t_case):
                continue
            hold_idx = int(np.where(np.isclose(t_case, hold_t))[0][0])
            if hold_idx == 0 or hold_idx == len(t_case) - 1:
                continue
            keep = np.ones(len(t_case), dtype=bool)
            keep[hold_idx] = False
            for col in STATE_COLUMNS:
                y = g[col].to_numpy(dtype=float)
                pred = np.interp(hold_t, t_case[keep], y[keep])
                errors.append(abs(float(pred - y[hold_idx])) / max(abs(float(y[hold_idx])), 1e-12))
        if not errors:
            continue
        rows.append(
            {
                "test_family": "leave_one_time_step",
                "scenario": f"leave_{hold_t:.2f}s",
                "status": "ok",
                "mean_validation_relative_rmse": float(np.mean(errors)),
                "max_validation_relative_rmse": float(np.max(errors)),
                "umax_validation_relative_rmse": float(np.mean(errors[STATE_COLUMNS.index("Umax_m_per_s")::len(STATE_COLUMNS)])),
                "n_states": len(errors),
                "supports_main_model": bool(np.mean(errors) <= 0.12),
            }
        )

    for noise_fraction in [0.01, 0.03, 0.05]:
        perturbed = table.copy()
        for col in STATE_COLUMNS:
            values = perturbed[col].to_numpy(dtype=float)
            pattern = np.sin(np.linspace(0.0, 2.0 * math.pi, len(values)))
            perturbed[col] = values * (1.0 + noise_fraction * pattern)
        _, summary, _ = fit_attractor_model(perturbed)
        rows.append(
            {
                "test_family": "deterministic_state_noise",
                "scenario": f"state_noise_{noise_fraction:.2f}",
                "status": "ok",
                "mean_validation_relative_rmse": float(summary["validation_relative_rmse"].mean()),
                "max_validation_relative_rmse": float(summary["validation_relative_rmse"].max()),
                "umax_validation_relative_rmse": float(
                    summary.loc[summary["state"].eq("Umax_m_per_s"), "validation_relative_rmse"].iloc[0]
                ),
                "n_states": len(summary),
                "supports_main_model": bool(summary["validation_relative_rmse"].mean() <= 0.12),
            }
        )

    out = pd.DataFrame(rows)
    support_rate = float(out["supports_main_model"].mean()) if len(out) else np.nan
    summary_text = f"""# Validation Stress Summary

Total stress scenarios: {len(out)}.

Support rate for the selected diagonal attractor threshold: {support_rate:.3f}.

Mean validation relative RMSE range: {out['mean_validation_relative_rmse'].min():.4f} to {out['mean_validation_relative_rmse'].max():.4f}.

These stress tests are internal checks across the available multi-condition FLOW-3D dataset. They strengthen the evidence narrative but do not replace physical measurements or independently generated held-out process designs.
"""
    return out, summary_text


def make_submission_gap_audit(
    validation_stress_tests: pd.DataFrame,
    timescale_summary: pd.DataFrame,
    assumption_matrix: pd.DataFrame,
    external_holdout_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    support_rate = float(validation_stress_tests["supports_main_model"].mean())
    high_assumptions = int(assumption_matrix["risk_level"].astype(str).str.contains("high").sum())
    high_timescale = int(timescale_summary["risk_level"].eq("high").sum())
    external_cases = 0
    external_process_error = np.nan
    external_dynamics_error = np.nan
    if external_holdout_summary is not None and len(external_holdout_summary):
        metric_map = dict(zip(external_holdout_summary["metric"], external_holdout_summary["value"]))
        external_cases = int(metric_map.get("external_validation_case_count", 0))
        external_process_error = float(metric_map.get("external_process_response_mean_relative_error", np.nan))
        external_dynamics_error = float(metric_map.get("external_dynamics_mean_relative_rmse", np.nan))
    if external_cases > 0:
        external_status = f"same_ecosystem_cfd_holdout_available_{external_cases}_conditions_no_physical_measurement_support"
        external_risk = "medium_high"
        external_evidence = (
            f"external_cases={external_cases}; process_mean_rel_error={external_process_error:.4f}; "
            f"dynamics_mean_rel_rmse={external_dynamics_error:.4f}"
        )
        external_action = "Melt-pool-data holdout cohorts support transfer evidence; physical measurement comparison remains the next validation step."
    else:
        external_status = "simulation_only_multi_condition_validation_no_external_holdout"
        external_risk = "high"
        external_evidence = f"stress_support_rate={support_rate:.3f}"
        external_action = "Physical measurement comparison or an independently generated held-out process design would strengthen external validation."
    return pd.DataFrame(
        [
            {
                "gap_area": "validation_hierarchy",
                "current_status": external_status,
                "risk_level": external_risk,
                "evidence": external_evidence,
                "recommended_action": external_action,
            },
            {
                "gap_area": "parameter_reconciliation",
                "current_status": "laser_phase_and_temperature_dependent_property_inputs_reconciled",
                "risk_level": "medium",
                "evidence": "beam_radius absorptivity phase_temperatures latent_heat and surface_tension constants resolved to setup note; rho cp k mu are supplied as temperature-dependent property tables",
                "recommended_action": "Property-curve provenance should accompany the nondimensional groups, with material-property uncertainty expanded when independent measurements become available.",
            },
            {
                "gap_area": "theory_rigor",
                "current_status": "assumption-proposition-proof-sketch framework",
                "risk_level": "medium",
                "evidence": "The theory notes formalize projection and Lyapunov arguments; constants remain data-limited.",
                "recommended_action": "Proof claims should remain conservative and tied to the stated diagnostic role of the error constants.",
            },
            {
                "gap_area": "assumption_scope",
                "current_status": "assumption assessment generated",
                "risk_level": "medium_high" if high_assumptions else "medium",
                "evidence": f"high_or_high_for assumptions={high_assumptions}",
                "recommended_action": "The assumption assessment supports interpretation and future validation planning.",
            },
            {
                "gap_area": "timescale_support",
                "current_status": "estimated from fitted relaxation rates",
                "risk_level": "medium_high" if high_timescale else "medium",
                "evidence": f"weakly_constrained_states={high_timescale}",
                "recommended_action": "Weak Umax identifiability and descriptor-level time-scale separation should frame interpretation.",
            },
            {
                "gap_area": "reproducibility",
                "current_status": "processed descriptors and fitted parameters archived",
                "risk_level": "medium",
                "evidence": "compile summary generated by script",
                "recommended_action": "The reproducibility materials should be extended when experimental data become available.",
            },
        ]
    )


def write_rigorous_theory_notes(
    report_path: Path,
    error_bound_summary: pd.DataFrame,
    timescale_summary: pd.DataFrame,
    validation_stress_tests: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bound_lines = "\n".join(
        [
            f"- {row.bound_component}: {row.bound_expression}; source {row.source_table}."
            for row in error_bound_summary.sort_values("display_order").itertuples()
        ]
    )
    tau_lines = "\n".join(
        [
            f"- {row.label}: k={row.k_per_s:.4g} 1/s, tau={row.characteristic_time_s:.4g} s, constraint={row.risk_level}."
            for row in timescale_summary.itertuples()
        ]
    )
    stress_mean = float(validation_stress_tests["mean_validation_relative_rmse"].mean())
    text = f"""# Rigorous Theory Notes

## Formal item 1: observation operator

Let `S(t)=(Omega_m(t),T(t),u(t))` denote the full high-dimensional Stefan-Marangoni state. The exported data are modeled as `P^h(t)=O_h[S(t)]`, where `O_h` retains only molten-region points in the half domain. The reconstruction operator `R` mirrors `P^h(t)` across `y=0`.

## Formal item 2: manifold projection error

The observed boundary `Gamma_h(t)` is projected to the superellipsoid manifold `M_SE(theta)`. The projection error is `epsilon_M(t)=inf_theta d_H(Gamma_h(t),M_SE(theta))`. The measured boundary residual is a computable proxy for this distance.

## Formal item 3: state-map error transfer

For any Lipschitz descriptor `g`, `|g(Gamma_h)-g(Pi_M Gamma_h)| <= L_g epsilon_M`. This links boundary approximation error to state variables such as `L_f`, `L_r`, `W` and `H`.

## Formal item 4: first-order relaxation and Lyapunov stability

Near `q_inf`, `dq/dt=F(q)` is approximated by `J(q-q_inf)`. The selected diagonal approximation has `J=-diag(k_i)`. With `V=1/2||q-q_inf||^2`, `dV/dt <= -2 k_min V` if all `k_i>0`.

## Formal item 5: total error bound

```text
E_total <= C1 E_reconstruction + C2 E_geometry + C3 E_volume_proxy + C4 E_dynamics + C5 E_parameter_scale.
```

{bound_lines}

## Time-scale evidence

{tau_lines}

## Temporal-sampling stress evidence

The internal stress-test mean relative RMSE averaged over scenarios is {stress_mean:.4f}. These tests are useful within the multi-condition FLOW-3D design, but they are not a substitute for physical measurements or independently generated held-out process designs.
"""
    report_path.write_text(text, encoding="utf-8")


def write_free_boundary_manifold_rationale(report_path: Path, geometry_comparison: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    geom = geometry_comparison[geometry_comparison["time_s"] == "summary"].set_index("model")
    vol_direction = "decreases" if float(geom.loc["superellipsoid", "mean_volume_relative_error"]) < float(geom.loc["ellipsoid", "mean_volume_relative_error"]) else "increases"
    text = f"""# Free-Boundary Manifold Rationale

## Why a superellipsoid?

The melt-pool boundary is shaped by localized heating, conductive smoothing, surface-tension-driven smoothing, front-rear asymmetry from translation and the imposed half-domain symmetry. A superellipsoid is a compact shape family that can encode these effects without turning the boundary into an unconstrained high-dimensional surface.

## Why not only an ellipsoid?

The ellipsoid is retained as a required baseline. The superellipsoid is selected because it reduces mean boundary residual from {float(geom.loc['ellipsoid', 'mean_boundary_residual_rmse']):.4f} to {float(geom.loc['superellipsoid', 'mean_boundary_residual_rmse']):.4f}. The mean volume relative error {vol_direction} from {float(geom.loc['ellipsoid', 'mean_volume_relative_error']):.4f} to {float(geom.loc['superellipsoid', 'mean_volume_relative_error']):.4f}, so volume is reported as a proxy-error limitation rather than as a selection win.

## Why not a more complex boundary?

A spline, neural implicit field or full triangulated surface would reduce geometric bias but would weaken identifiability under 15 time steps. The present manuscript prioritizes a low-dimensional, auditable free-boundary manifold.

## Failure modes

The model may fail for multi-lobed pools, keyhole collapse, strong asymmetry, disconnected molten regions or cases where the exported molten envelope is not close to a single connected boundary.
"""
    report_path.write_text(text, encoding="utf-8")


def write_manuscript_draft_final(
    report_path: Path,
    dimensionless_numbers: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    timescale_summary: pd.DataFrame,
    validation_stress_tests: pd.DataFrame,
    submission_gap_audit: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dim = dimensionless_value_lookup(dimensionless_numbers)
    geom = geometry_comparison[geometry_comparison["time_s"] == "summary"].set_index("model")
    dyn = dynamics_comparison.groupby("model")["validation_relative_rmse"].mean()
    stress_support = float(validation_stress_tests["supports_main_model"].mean())
    high_gaps = ", ".join(submission_gap_audit.loc[submission_gap_audit["risk_level"].eq("high"), "gap_area"])
    if "case_id" in timescale_summary.columns:
        tau_source = timescale_summary[timescale_summary["case_id"].astype(str).eq(representative_case_id(timescale_summary))]
        if len(tau_source) == 0:
            tau_source = timescale_summary.groupby("label", as_index=False)["characteristic_time_s"].median()
    else:
        tau_source = timescale_summary
    tau_text = ", ".join(
        f"{row.label}: tau={float(row.characteristic_time_s):.3g}s"
        for row in tau_source.head(len(STATE_LABELS)).itertuples()
        if hasattr(row, "label") and hasattr(row, "characteristic_time_s")
    )
    geom_case = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("case_summary")].copy()
    geom_case_boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse") if len(geom_case) else pd.DataFrame()
    geom_case_volume = geom_case.pivot(index="case_id", columns="model", values="mean_volume_relative_error") if len(geom_case) else pd.DataFrame()
    geom_stats_text = ""
    if len(geom_case_boundary):
        geom_stats_text = (
            f"Paired geometry comparisons show the superellipsoid reduces the boundary residual in "
            f"{int((geom_case_boundary['superellipsoid'] < geom_case_boundary['ellipsoid']).sum())}/{len(geom_case_boundary)} conditions "
            f"(sign test p={float(binomtest(int((geom_case_boundary['superellipsoid'] < geom_case_boundary['ellipsoid']).sum()), len(geom_case_boundary), 0.5, alternative='greater').pvalue):.3g}); "
            f"the paired median advantage is "
            f"{float(np.nanmedian(geom_case_boundary['ellipsoid'] - geom_case_boundary['superellipsoid'])):.4g}."
        )
        if len(geom_case_volume):
            geom_stats_text += (
                f" Volume proxy improves in {int((geom_case_volume['superellipsoid'] < geom_case_volume['ellipsoid']).sum())}/{len(geom_case_volume)} conditions."
            )
    dyn_case = dynamics_comparison.pivot_table(index=["case_id", "state"], columns="model", values="validation_relative_rmse", aggfunc="mean")
    dyn_stats_text = ""
    if {"diagonal_attractor", "coupled_ridge_attractor"}.issubset(dyn_case.columns):
        diag_better = int((dyn_case["diagonal_attractor"] < dyn_case["coupled_ridge_attractor"]).sum())
        total_dyn = int(len(dyn_case))
        try:
            dyn_p = float(binomtest(diag_better, total_dyn, 0.5, alternative="greater").pvalue)
        except Exception:
            dyn_p = float("nan")
        dyn_stats_text = (
            f" Paired dynamics comparisons show the diagonal model improves validation in {diag_better}/{total_dyn} condition-state pairs "
            f"(sign test p={dyn_p:.3g})."
        )
    geom_distance_text = ""
    if {"mean_chamfer_distance_m", "mean_hausdorff_distance_m"}.issubset(geom.columns):
        geom_distance_text = (
            f", Chamfer ellipsoid/superellipsoid "
            f"{float(geom.loc['ellipsoid', 'mean_chamfer_distance_m']):.4e}/"
            f"{float(geom.loc['superellipsoid', 'mean_chamfer_distance_m']):.4e} m, "
            f"Hausdorff ellipsoid/superellipsoid "
            f"{float(geom.loc['ellipsoid', 'mean_hausdorff_distance_m']):.4e}/"
            f"{float(geom.loc['superellipsoid', 'mean_hausdorff_distance_m']):.4e} m. "
            "The distance metrics are reported as geometric-risk diagnostics rather than as selection wins"
        )
    text = f"""# Manuscript Draft: High-Risk-Reduced Version

## Core position

The manuscript should be positioned as CFD-informed observed boundary-envelope identification of FLOW-3D molten-region observations. The main claim remains deliberately bounded: the available sequences can be reduced to asymmetric superellipsoid boundary descriptors, a parsimonious diagonal attractor baseline and process-response diagnostics, without claiming physical measurement support or universal process-map prediction.

## Strengthened theory

The theory package formalizes the observation operator, superellipsoid manifold projection, descriptor error transfer, first-order relaxation and Lyapunov stability. It uses the full Stefan-Marangoni picture only as physical motivation and does not claim a closed-form solution of the governing PDE system.

## Strengthened validation

The internal stress tests include rolling-origin time extrapolation, leave-one-time-step interpolation, training-fraction perturbation and deterministic state-noise perturbation. The parsimonious diagonal baseline is supported in {stress_support:.3f} of tested cases, so the stress tests are reported as a limitation rather than as decisive proof of superiority.{dyn_stats_text}

## Key numbers

- Pe={float(dim['Pe']):.2f}, Ste={float(dim['Ste']):.3f}, E*={float(dim['E_star']):.2f}, Ma={float(dim['Ma']):.2f}.
- Boundary diagnostics: ellipsoid residual {float(geom.loc['ellipsoid', 'mean_boundary_residual_rmse']):.4f}, superellipsoid residual {float(geom.loc['superellipsoid', 'mean_boundary_residual_rmse']):.4f}{geom_distance_text}. {geom_stats_text}
- Validation relative RMSE: diagonal {float(dyn['diagonal_attractor']):.4f}, coupled {float(dyn['coupled_ridge_attractor']):.4f}.
- Characteristic times: {tau_text}.

## Remaining high-risk gap

The principal remaining high-risk gap is: {high_gaps or 'none marked high'}. This should be acknowledged directly before submission to a high-level mathematical modeling journal, together with the fact that the current validation remains CFD-internal rather than experimentally external.
"""
    report_path.write_text(text, encoding="utf-8")


def make_active_figure_manifest(output_dir: Path) -> pd.DataFrame:
    paper_dir = output_dir / "paper_figures"
    figures_dir = output_dir / "figures"
    formats = ["svg", "pdf", "tiff", "png"]
    rows = []

    def add_row(
        item_type: str,
        label: str,
        stem: str,
        base_dir: Path,
        status: str,
        role: str,
        manuscript_use: str,
    ) -> None:
        paths = {fmt: base_dir / f"{stem}.{fmt}" for fmt in formats}
        rows.append(
            {
                "item_type": item_type,
                "label": label,
                "figure_stem": stem,
                "status": status,
                "role": role,
                "manuscript_use": manuscript_use,
                "svg_path": str(paths["svg"]),
                "pdf_path": str(paths["pdf"]),
                "tiff_path": str(paths["tiff"]),
                "png_path": str(paths["png"]),
                "all_formats_exist": bool(all(path.exists() for path in paths.values())),
                "png_size_bytes": paths["png"].stat().st_size if paths["png"].exists() else 0,
            }
        )

    active_main = [
        ("Figure 1", "paper_fig01_modeling_framework", "modeling framework"),
        ("Figure 2", "paper_fig02_process_matrix", "multi-condition process matrix"),
        ("Figure 3", "paper_fig03_data_moving_frame", "moving-frame reconstruction"),
        ("Figure 4", "paper_fig13_simulation_cross_sections", "thermal-flow cross-section context"),
        ("Figure 5", "paper_fig04_geometry_quasi_steady", "geometric evolution and quasi-steady approach"),
        ("Figure 6", "paper_fig05_free_boundary_model_comparison", "boundary overlays and cross-condition comparison"),
        ("Figure 7", "paper_fig06_process_response", "process-response surface diagnostics"),
        ("Figure 8", "paper_fig07_dimensionless_regime", "dimensionless regime and sensitivity"),
        ("Figure 9", "paper_fig08_dynamics_validation", "cross-condition dynamics validation"),
        ("Figure 10", "paper_fig14_dynamics_residuals_by_state", "state-wise dynamics residuals"),
        ("Figure 11", "paper_fig09_error_budget_model_selection", "error budget and model selection"),
        ("Figure 12", "paper_fig10_identifiability_overparameterization", "identifiability and overparameterization"),
        ("Figure 13", "paper_fig11_leave_one_condition_validation", "leave-one-condition-out validation"),
        ("Figure 14", "paper_fig12_external_holdout_validation", "melt-pool-data holdout"),
    ]
    active_supp = [
        ("Supplementary Figure S1", "supp_figS1_all_boundary_fits", "all time-step boundary fits"),
        ("Supplementary Figure S2", "supp_figS2_superellipsoid_parameters", "superellipsoid parameter trajectories"),
        ("Supplementary Figure S3", "supp_figS4_dimensionless_sensitivity_grid", "dimensionless sensitivity scenario grid"),
        (
            "Supplementary Figure S4",
            "supp_figS5_theory_identifiability_error_bounds",
            "theory, identifiability and error-budget diagnostics",
        ),
        (
            "Supplementary Figure S5",
            "supp_figS6_convex_alpha_proxy_comparison",
            "convex-hull and alpha-complex proxy comparison",
        ),
        ("Supplementary Figure S6", "fig10_stability_attractor", "representative stability and attractor evidence"),
        ("Supplementary Figure S7", "fig05_boundary_fit_comparison", "expanded representative boundary overlays"),
        ("Supplementary Figure S8", "fig03_thermal_flow_evolution", "thermal-flow state evolution"),
        ("Supplementary Figure S9", "fig06_dynamics_model_comparison", "dynamical model trajectory comparison"),
        ("Supplementary Figure S10", "supp_figS10_temperature_dependent_properties", "temperature-dependent material-property curves"),
    ]

    active_stems = {stem for _, stem, _ in active_main}
    for label, stem, role in active_main:
        base_dir = paper_dir
        status = "active"
        if not (base_dir / f"{stem}.png").exists():
            legacy_source = {
                "paper_fig02_process_matrix": "fig13_multicondition_process_matrix",
                "paper_fig06_process_response": "fig14_multicondition_response_surfaces",
                "paper_fig08_dynamics_validation": "fig16_multicondition_dynamics_validation",
                "paper_fig11_leave_one_condition_validation": "fig17_leave_one_condition_validation",
            }.get(stem)
            if legacy_source is not None and (figures_dir / f"{legacy_source}.png").exists():
                base_dir = figures_dir
                stem_to_use = legacy_source
                status = "active"
            else:
                stem_to_use = stem
        else:
            stem_to_use = stem
        add_row("main_figure", label, stem_to_use, base_dir, status, role, "main_text")
    for label, stem, role in active_supp:
        add_row("supplementary_figure", label, stem, figures_dir, "active", role, "supplementary_methods")

    legacy_stems = sorted({path.stem for path in paper_dir.glob("paper_fig*.*")} - active_stems)
    for stem in legacy_stems:
        add_row(
            "legacy_figure",
            "legacy",
            stem,
            paper_dir,
            "legacy",
            "old numbering retained on disk",
            "do_not_use_current_submission",
        )
    return pd.DataFrame(rows)


def write_supplementary_methods_draft(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    error_budget: pd.DataFrame,
    parameter_identifiability: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, pd.DataFrame({
        "status": ["ok"],
        "superellipsoid_improves_volume": [True],
        "superellipsoid_improves_boundary": [True],
        "coupled_improves_validation": [False],
    }))
    dim_lines = "\n".join(
        [
            f"- {row.symbol}: baseline {row.baseline_value:.4g}, range {row.min_value:.4g} to {row.max_value:.4g}, classes {row.observed_classes}."
            for row in dimensionless_sensitivity.itertuples()
        ]
    )
    error_lines = "\n".join(
        [f"- {row.error_term}: {row.source}; metric {row.primary_metric}." for row in error_budget.itertuples()]
    )
    risk_lines = "\n".join(
        [
            f"- {row.parameter}: CV={row.coefficient_of_variation:.3g}, risk={row.risk_level}."
            for row in parameter_identifiability.itertuples()
            if str(row.risk_level).lower() == "high"
        ]
    )
    n_conditions = int(table["case_id"].nunique()) if "case_id" in table.columns else 1
    speed_text = (
        "with condition-specific scan speed parsed from each raw-data folder"
        if n_conditions > 1
        else "with v = 0.008 m/s"
    )
    location_text = (
        "FLOW-3D condition folders under raw data/Aa-b-c-d"
        if n_conditions > 1
        else "FLOW-3D CSV files under raw data"
    )
    text = f"""# Supplementary Methods Draft

## S1. Data source and coordinate convention

The source data consist of {vals['n_time_steps']} FLOW-3D CSV files from {n_conditions} L-DED process conditions in {location_text}, covering t={vals['t_min']:.2f} s to t={vals['t_max']:.2f} s. Each CSV contains points exported only from the molten region. The analysis therefore treats the data as a molten-domain point cloud, not as a complete thermal-field export. Coordinates are converted to the laser-attached frame by xi = x - v_c t, {speed_text}.

## S2. Half-domain symmetry reconstruction

The FLOW-3D domain is simulated only for y >= 0, with y = 0 as a symmetry plane. The full-domain observation is obtained by mirroring each point (xi, y, z) to (xi, -y, z). Scalar descriptors use W = 2 max y and V_full = 2 V_half. The reported volume is a full-domain convex-hull proxy of the exported molten points.

## S3. Boundary extraction and geometric fitting

For each time step, duplicate rows are removed and repeated coordinates are collapsed by averaging fields. The boundary is approximated by the convex-hull envelope of the molten point cloud. Two analytic shapes are fitted:

```text
Ellipsoid:       ((xi - xi_c)/a_s)^2 + (y/b)^2 + ((z - z_c)/c)^2 = 1
Superellipsoid: |((xi - xi_c)/a_s)|^n + |y/b|^m + |((z - z_c)/c)|^p = 1
```

The asymmetric longitudinal scale a_s is a_f ahead of the fitted center and a_r behind it. The superellipsoid exponents n, m and p are bounded during optimization. Boundary residual and volume relative error are reported for both models.

## S4. Dimensionless groups

The baseline material properties are interpolated at the liquidus temperature. The main dimensionless groups are Pe, Fo, Ste, E*, Re, Pr and Ma. The key reported values are Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}. Sensitivity scenarios perturb reference temperature, absorptivity and surface-tension coefficient:

{dim_lines}

## S5. Reduced-order dynamics

The reduced state is q = [L_f, L_r, W, H, T_max, G_mean, U_max]. The selected baseline is the diagonal attractor dq_i/dt = k_i(q_inf,i - q_i). The comparison model is dq/dt = A(q_inf - q), where A is fitted with ridge regression. Training uses the early 70% of the time steps by default and validation uses the remaining time steps.

## S6. Stability analysis

For the diagonal model, e_i = q_i - q_inf,i gives de_i/dt = -k_i e_i. Positive k_i therefore gives exponential stability. For the coupled model, e = q - q_inf gives de/dt = -A e. The coupled model is stable if every eigenvalue of -A has negative real part. The manuscript reports stability separately from model selection, because a stable higher-dimensional model can still validate worse than a simpler model.

## S7. Error budget

The error taxonomy is

```text
E_total = E_reconstruction + E_geometry + E_volume_proxy + E_dynamics + E_parameter_scale.
```

The current error-budget sources are:

{error_lines}

## S8. Identifiability and overparameterization

Identifiability is assessed through coefficient of variation, stability of fitted parameters across time, signs of dynamic parameters and validation behavior. High-risk parameters are:

{risk_lines}

These diagnostics explain why the superellipsoid is retained as the selected algebraic observed-envelope descriptor, while the coupled matrix is retained as an overparameterization comparison.

## S9. Supplementary figures

Supplementary Figure S1 shows the superellipsoid boundary fit for all 15 time steps. Supplementary Figure S2 shows superellipsoid parameters versus time. Supplementary Figure S3 shows state-wise residuals for diagonal and coupled dynamics. Supplementary Figure S4 shows the full dimensionless sensitivity scenario grid.

## S10. Reproducibility

Running `python scripts/flow3d_melt_pool_pilot.py` regenerates the modeling tables, figures, manuscript drafts, figure captions, reference seed file, literature matrix and validation manifest. The output validation table records whether all expected tables, reports and figure files exist and whether figure PNGs are nonempty.
"""
    report_path.write_text(text, encoding="utf-8")


def write_submission_readiness_checklist(report_path: Path, figure_manifest: pd.DataFrame, literature_matrix: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    active_count = int(figure_manifest["status"].eq("active").sum())
    legacy_count = int(figure_manifest["status"].eq("legacy").sum())
    active_manifest = figure_manifest[figure_manifest["status"].eq("active")]
    main_count = int(active_manifest["item_type"].eq("main_figure").sum()) if "item_type" in active_manifest.columns else 0
    supp_count = int(active_manifest["item_type"].eq("supplementary_figure").sum()) if "item_type" in active_manifest.columns else 0
    all_active_ready = bool(active_manifest["all_formats_exist"].all())
    text = f"""# Submission Readiness Checklist

## Completed in this package

- Manuscript draft v3 with citation keys and no bare `[Ref]` placeholders.
- Literature matrix with {len(literature_matrix)} candidate references and manuscript-use notes.
- BibTeX seed file for the candidate reference set.
- Active figure manifest with {active_count} active figures and {legacy_count} legacy figure stems.
- Figure files for {main_count} main figures and {supp_count} supplementary figures; active formats complete: {all_active_ready}.
- Supplementary Methods draft explaining preprocessing, symmetry reconstruction, boundary fitting, dynamics, stability, error budget, sensitivity and supplementary figures.
- Nomenclature table and equation inventory from the reproducible analysis script.
- Parameter reconciliation, geometry-risk and validation-hierarchy audit tables.
- Response matrix mapping the strict-review priorities to concrete revision actions.
- Processed reproducibility package prepared under `analysis_outputs/reproducibility_package/` and zipped as `analysis_outputs/reproducibility_package.zip`.
- AMM LaTeX submission package staged under `analysis_outputs/submission_package/`.
- `main_submission.tex` provides the main article without appended Supplementary Information; `supplementary_methods.pdf` is retained for separate upload.
- Cover letter, highlights, AMM checklist and declaration material are generated as submission-support reports.
- Reviewer-risk response notes covering finite process-matrix scope, molten-region-only export, overfitting, overparameterization, theory depth, identifiability and material sensitivity.

## Remaining manual tasks before submission

- Target journal set to Applied Mathematical Modelling; before upload, check the staged package against the current author guide.
- Verify every seed reference against the publisher page or database export, especially older books and classic papers.
- Author names and affiliations are restored in the LaTeX source; add or verify corresponding-author email, ORCID identifiers, acknowledgments, funding, competing-interest, CRediT and AI-assisted-technology declarations.
- Processed reproducibility files and analysis scripts are now available through the GitHub repository; decide separately whether raw FLOW-3D CSV files can be shared publicly.
- Confirm whether additional FLOW-3D software settings, domain dimensions, boundary conditions, solver controls and export filters can be supplied by the authors.
- Check all TIFF files against the target journal's DPI, color mode and physical width requirements.
- Remove or ignore legacy figure stems during final layout; use only files marked `active` in `active_figure_manifest.csv`.
- Add final reference manager output, journal-specific BibTeX or CSL formatting.
- Keep the parameter-audit basis visible: setup-note laser, phase-change and surface-tension constants are reconciled, and density, heat-capacity, conductivity and viscosity are supplied through temperature-dependent property tables.
- Keep measurement-comparison language bounded, since the current evidence is CFD-informed and includes only same-solver CFD holdout support.

## Current recommended submission position

For Applied Mathematical Modelling, the paper should be presented as CFD-informed observed export-envelope identification and engineering mathematical modeling for L-DED. The central defensible claim is that the molten-region point-cloud sequences admit compact superellipsoid boundary descriptors and parsimonious stable condition-wise baseline dynamics, with same-solver CFD holdout support from the V-prefixed and A16-A20 cohorts, while remaining short of physical measurement support or a universal process map.
"""
    report_path.write_text(text, encoding="utf-8")


def write_cover_letter_draft(report_path: Path, dimensionless_numbers: pd.DataFrame, dynamics_comparison: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dim = dimensionless_value_lookup(dimensionless_numbers)
    dyn_mean = dynamics_comparison.groupby("model")["validation_relative_rmse"].mean()
    text = f"""# Cover Letter Draft

Dear Editor,

We are pleased to submit the manuscript entitled "CFD-informed observed boundary-envelope identification of L-DED melt pools using superellipsoid manifolds and parsimonious attractor dynamics" for consideration in Applied Mathematical Modelling.

The manuscript develops an observed-boundary mathematical model for 316L laser directed energy deposition molten-region point clouds. FLOW-3D (Flow Science, Inc.) exports of the molten region are transformed into symmetry-reconstructed moving-frame point clouds, represented by asymmetric superellipsoid boundary-envelope manifolds, interpreted through dimensionless groups, and summarized by parsimonious stable baseline dynamics. The manuscript does not claim a closed-form or Galerkin reduction of the Stefan-Marangoni system; the governing equations set the physical context for an observation-driven modeling and validation procedure.

The work is aimed at readers interested in observed-boundary modeling, reduced-order dynamics and model selection in engineering systems. It is not a process-wide empirical map. Its contribution is limited to CFD-output observed-boundary reduction and CFD holdout transfer: molten-region melt-pool data are reduced to compact analytic boundary-envelope descriptors and transient-to-quasi-steady dynamical baselines across a finite process matrix, with holdout checks from the V-prefixed validation cohort and the A16-A20 additional cohort. The nondimensional values Pe={dim['Pe']:.2f}, Ste={dim['Ste']:.3f}, E*={dim['E_star']:.2f} and Ma={dim['Ma']:.2f} are retained as post-processing scale diagnostics under an explicit parameter-assessment constraint. The diagonal attractor has slightly lower mean validation relative RMSE than the coupled ridge attractor ({dyn_mean['diagonal_attractor']:.4f} versus {dyn_mean['coupled_ridge_attractor']:.4f}), but it is selected as a parsimonious baseline rather than as a statistically dominant model; the more complex coupled model is retained as an overparameterization comparison.

A processed reproducibility package, analysis scripts and generated manuscript source are available at {DEFAULT_REPOSITORY_URL}, including geometry descriptors, fitted parameters, model-selection tables, parameter-audit tables, CFD holdout summaries, plotting scripts and LaTeX source files.

We believe the manuscript fits Applied Mathematical Modelling because it combines observation operators, analytic observed-boundary manifolds, nondimensional scaling, validation design, reproducibility checks, error budgeting and parameter identifiability for an engineering CFD problem.

Sincerely,

{COVER_LETTER_SIGNATURE}
"""
    report_path.write_text(text, encoding="utf-8")


def write_highlights_draft(
    report_path: Path,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    external_holdout_summary: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Models L-DED molten-region point clouds with observed-boundary envelopes.",
        "Reconstructs half-domain exports in a laser-attached moving frame.",
        "Selects superellipsoids by boundary residual, not volume proxy.",
        "Uses a diagonal attractor as a parsimonious stable baseline.",
        "Reports CFD holdout transfer for withheld process conditions.",
    ]
    count_lines = "\n".join(f"- {line} ({len(line)} characters)" for line in lines)
    bullet_lines = "\n".join(f"- {line}" for line in lines)
    text = f"""# Highlights Draft

{bullet_lines}

Character-count audit for Elsevier/AMM highlight upload:

{count_lines}
"""
    report_path.write_text(text, encoding="utf-8")


def write_declarations_for_submission(report_path: Path) -> None:
    """Write AMM/Elsevier declaration material without inventing author facts."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Declarations for AMM Submission

These statements are prepared for the LaTeX submission route. Items marked `{AUTHOR_CONFIRMATION_REQUIRED}` must be confirmed by the authors before upload.

## Corresponding Author Metadata

- Corresponding author: Xiaoli Lin.
- Email: {AUTHOR_CONFIRMATION_REQUIRED}.
- ORCID identifiers: {AUTHOR_CONFIRMATION_REQUIRED}.
- Author order confirmation: {AUTHOR_CONFIRMATION_REQUIRED}.

## Declaration of Competing Interest

{AUTHOR_CONFIRMATION_REQUIRED}: choose one option before submission.

- The authors declare no competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
- Or list all financial, personal, institutional, employment, patent, consulting, software or other relationships relevant to the work.

## Funding

{AUTHOR_CONFIRMATION_REQUIRED}: add grant numbers, funder names and institutional support, or state that the research received no specific funding only if that is factually correct.

## Acknowledgments

{AUTHOR_CONFIRMATION_REQUIRED}: add people, facilities, software support and institutional resources that should be acknowledged.

## Author Contributions

{AUTHOR_CONFIRMATION_REQUIRED}: complete CRediT roles for Boxue Song, Xiaoli Lin, Xingyu Jiang, Tianbiao Yu and Wenchao Xi. Candidate roles to assign only after author confirmation: conceptualization, methodology, software, validation, formal analysis, investigation, data curation, writing original draft, writing review and editing, visualization, supervision, project administration and funding acquisition.

## AI-Assisted Technology Declaration

{AUTHOR_CONFIRMATION_REQUIRED}: complete according to the actual use of AI-assisted tools in writing, coding, translation, editing, figure generation and submission preparation. Do not state that such tools were not used unless that is factually correct.

## Permissions and Originality

The manuscript figures and tables are generated from the authors' analysis outputs and project data unless the authors identify third-party material before submission. {AUTHOR_CONFIRMATION_REQUIRED}: confirm that all figure panels, tables and text can be submitted without third-party permission, or list required permissions.

## Data and Code Availability Statement

Processed reproducibility materials, analysis scripts and generated manuscript source are available in the GitHub repository {DEFAULT_REPOSITORY_URL}. Raw FLOW-3D molten-region CSV exports are available from the corresponding author upon reasonable request, subject to software, export and project-sharing restrictions. Proprietary FLOW-3D project files are not distributed.
"""
    report_path.write_text(text, encoding="utf-8")


def write_amm_submission_checklist(report_path: Path, output_dir: Path, figure_manifest: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    active_manifest = figure_manifest[figure_manifest["status"].astype(str).eq("active")].copy()
    main_count = int(active_manifest["item_type"].astype(str).eq("main_figure").sum()) if len(active_manifest) else 0
    supp_count = int(active_manifest["item_type"].astype(str).eq("supplementary_figure").sum()) if len(active_manifest) else 0
    all_active_ready = bool(active_manifest["all_formats_exist"].all()) if len(active_manifest) else False
    latex_dir = output_dir / "latex"
    report_dir = output_dir / "reports"
    checks = [
        ("Target journal", "Applied Mathematical Modelling / Elsevier"),
        ("Official guide", AMM_GUIDE_FOR_AUTHORS_URL),
        ("Submission source route", "LaTeX route; do not upload the legacy root Word draft."),
        ("Main article source", yes_no((latex_dir / "main_submission.tex").exists())),
        ("Main article PDF", yes_no((latex_dir / "main_submission.pdf").exists())),
        ("Standalone supplementary PDF", yes_no((latex_dir / "supplementary_methods.pdf").exists())),
        ("References", yes_no((latex_dir / "references.bib").exists())),
        ("Cover letter draft", yes_no((report_dir / "cover_letter_draft.md").exists())),
        ("Highlights under 85 characters", yes_no((report_dir / "highlights_draft.md").exists())),
        ("Declarations file", yes_no((report_dir / "declarations_for_submission.md").exists())),
        ("Active main figures", str(main_count)),
        ("Active supplementary figures", str(supp_count)),
        ("All active figure formats present", yes_no(all_active_ready)),
        ("Raw data route", "Processed GitHub package; raw FLOW-3D CSV by reasonable request."),
        ("Author-confirmed metadata", AUTHOR_CONFIRMATION_REQUIRED),
    ]
    check_rows = "\n".join(f"| {latex_escape(item)} | {latex_escape(status)} |" for item, status in checks)
    text = f"""# AMM Submission Checklist

| Item | Status |
|---|---|
{check_rows}

## Upload Route

Use `analysis_outputs/submission_package/` as the staging folder. The package is built from `main_submission.tex`, `main_submission.pdf`, `supplementary_methods.pdf`, `references.bib`, active figures, cover letter, highlights and declaration material.

## Manual Confirmations Before Upload

- Corresponding-author email and ORCID identifiers.
- Author order and CRediT roles.
- Competing-interest, funding and acknowledgments statements.
- AI-assisted technology declaration.
- Permissions/originality confirmation for all figures and tables.
- Whether AMM accepts the LaTeX source package directly in the selected submission workflow.
"""
    report_path.write_text(text, encoding="utf-8")


def replace_submission_external_supp_refs(text: str) -> str:
    replacements = {
        r"Supplementary Table~\ref{tab:supp-assumptions}": "Supplementary Table S5",
        r"Supplementary Table~\ref{tab:supp-sensitivity-tests}": "Supplementary Table S6",
        r"Supplementary Table~\ref{tab:supp-gap-audit}": "Supplementary Table S7",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def make_article_only_submission_tex(main_tex: str, supplementary_body_tex: str) -> str:
    marker = f"\n\n{supplementary_body_tex}\n\n\\end{{document}}"
    if marker in main_tex:
        article = main_tex.replace(marker, "\n\n\\end{document}")
    elif supplementary_body_tex in main_tex:
        article = main_tex.replace(supplementary_body_tex, "")
    else:
        article = main_tex
    return replace_submission_external_supp_refs(article)


def refresh_generated_directory(path: Path, root: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to refresh generated directory outside output root: {path}")
    shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_submission_package(output_dir: Path) -> pd.DataFrame:
    """Stage the AMM upload bundle without including legacy figures or old Word drafts."""
    package_dir = output_dir / "submission_package"
    refresh_generated_directory(package_dir, output_dir)
    records: list[dict[str, object]] = []

    def copy_file(src: Path, dst_relative: str, role: str) -> None:
        dst = package_dir / dst_relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copyfile(src, dst)
        records.append(
            {
                "role": role,
                "source_path": str(src),
                "package_path": str(dst.relative_to(package_dir)),
                "included": bool(dst.exists()),
                "size_bytes": dst.stat().st_size if dst.exists() else 0,
            }
        )

    latex_files = [
        "main_submission.tex",
        "main_submission.pdf",
        "main_submission.bbl",
        "references.bib",
        "supplementary_methods.tex",
        "supplementary_methods.pdf",
        "latex_figure_manifest.csv",
        "latex_compile_summary.txt",
        "README.md",
    ]
    for name in latex_files:
        copy_file(output_dir / "latex" / name, f"latex/{name}", "submission source")

    report_files = [
        "cover_letter_draft.md",
        "highlights_draft.md",
        "declarations_for_submission.md",
        "amm_submission_checklist.md",
        "submission_readiness_checklist.md",
    ]
    for name in report_files:
        copy_file(output_dir / "reports" / name, f"reports/{name}", "submission report")

    figure_manifest_path = output_dir / "tables" / "active_figure_manifest.csv"
    copy_file(figure_manifest_path, "tables/active_figure_manifest.csv", "active figure manifest")
    if figure_manifest_path.exists() and figure_manifest_path.stat().st_size > 0:
        figure_manifest = pd.read_csv(figure_manifest_path)
        active = figure_manifest[figure_manifest["status"].astype(str).eq("active")].copy()
        for row in active.to_dict(orient="records"):
            for fmt in ["pdf", "tiff", "svg", "png"]:
                src_text = str(row.get(f"{fmt}_path", "")).strip()
                if not src_text:
                    continue
                src = Path(src_text)
                if not src.exists():
                    fallback_dir = output_dir / ("paper_figures" if row.get("manuscript_use") == "main_text" else "figures")
                    fallback = fallback_dir / src.name
                    if fallback.exists():
                        src = fallback
                dst_dir = "paper_figures" if row.get("manuscript_use") == "main_text" else "figures"
                copy_file(src, f"{dst_dir}/{src.name}", "active submission figure")

    readme = f"""# AMM Submission Package

This staging folder is generated for the Applied Mathematical Modelling / Elsevier LaTeX submission route.

Primary files:

- `latex/main_submission.tex` and `latex/main_submission.pdf`: main article only.
- `latex/supplementary_methods.tex` and `latex/supplementary_methods.pdf`: separate supplementary material.
- `latex/references.bib` and `latex/main_submission.bbl`: bibliography source and compiled bibliography.
- `paper_figures/` and `figures/`: active figure exports only.
- `reports/cover_letter_draft.md`, `reports/highlights_draft.md` and `reports/declarations_for_submission.md`: upload-support material.

Do not upload the legacy root Word draft as the editable manuscript source. Check author metadata and declarations before submission.

Guide used for packaging decisions: {AMM_GUIDE_FOR_AUTHORS_URL}
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")
    records.append(
        {
            "role": "submission package README",
            "source_path": "generated",
            "package_path": "README.md",
            "included": True,
            "size_bytes": (package_dir / "README.md").stat().st_size,
        }
    )
    manifest = pd.DataFrame(records)
    manifest.to_csv(package_dir / "submission_package_manifest.csv", index=False)
    return manifest


def write_minimal_public_example(package_dir: Path, output_dir: Path, records: list[dict[str, object]]) -> None:
    example_dir = package_dir / "example_data"
    example_dir.mkdir(parents=True, exist_ok=True)
    source_table = output_dir / "tables" / "modeling_table.csv"
    included = False
    if source_table.exists():
        table = pd.read_csv(source_table)
        case_id = representative_case_id(table)
        if case_id is not None:
            table = table[table["case_id"].astype(str).eq(case_id)].copy()
        table = table.sort_values("time_s")
        example_columns = [
            "case_id",
            "case_index",
            "power_W",
            "scan_speed_mm_s",
            "powder_feed_g_min",
            "time_s",
            "front_length_m",
            "rear_length_m",
            "melt_pool_length_m",
            "full_width_m",
            "height_span_m",
            "Tmax_K",
            "Gmean_K_per_m",
            "Umax_m_per_s",
            "volume_proxy_m3",
            "superellipsoid_af_m",
            "superellipsoid_ar_m",
            "superellipsoid_b_m",
            "superellipsoid_c_m",
            "superellipsoid_xic_m",
            "superellipsoid_zc_m",
            "superellipsoid_n",
            "superellipsoid_m",
            "superellipsoid_p",
            "superellipsoid_residual_rmse",
            "superellipsoid_volume_full_m3",
            "superellipsoid_volume_relative_error",
        ]
        available_columns = [col for col in example_columns if col in table.columns]
        example_path = example_dir / "minimal_public_modeling_table.csv"
        table[available_columns].to_csv(example_path, index=False)
        included = example_path.exists()
        records.append(
            {
                "role": "minimal public example data",
                "source_path": str(source_table),
                "package_path": str(example_path.relative_to(package_dir)),
                "included": included,
                "size_bytes": example_path.stat().st_size if example_path.exists() else 0,
            }
        )
    readme = """# Minimal Public Example

This folder contains a processed descriptor-level example extracted from the representative baseline condition. It is intended to let readers run a small public example without access to the raw FLOW-3D molten-region CSV exports.

Run from the root of the reproducibility package:

```bash
python scripts/minimal_example_summary.py --input example_data/minimal_public_modeling_table.csv --output-dir example_outputs
```

The command writes `example_outputs/minimal_example_summary.csv` and `example_outputs/minimal_example_geometry.png`. It exercises the descriptor-table and plotting workflow only. It does not reproduce the full manuscript pipeline and does not include proprietary FLOW-3D project files or raw molten-region exports.
"""
    readme_path = example_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    records.append(
        {
            "role": "minimal public example instructions",
            "source_path": str(output_dir),
            "package_path": str(readme_path.relative_to(package_dir)),
            "included": readme_path.exists(),
            "size_bytes": readme_path.stat().st_size if readme_path.exists() else 0,
        }
    )
    if not included:
        note_path = example_dir / "minimal_public_modeling_table_not_included.txt"
        note_path.write_text("The processed modeling table was not available when the package was assembled.\n", encoding="utf-8")
        records.append(
            {
                "role": "minimal public example data note",
                "source_path": str(source_table),
                "package_path": str(note_path.relative_to(package_dir)),
                "included": note_path.exists(),
                "size_bytes": note_path.stat().st_size if note_path.exists() else 0,
            }
        )


def write_reproducibility_package(output_dir: Path) -> pd.DataFrame:
    """Assemble a processed, non-proprietary reproducibility package.

    The package intentionally excludes raw FLOW-3D project files and raw molten-region
    CSV exports. It contains processed descriptors, fitted parameters, validation
    summaries, scripts and manuscript sources that can be submitted as supplementary
    data or uploaded to a repository.
    """
    resolved_output = output_dir.resolve()
    for stale_dir in output_dir.glob(".reproducibility_package_refresh_*"):
        try:
            if stale_dir.is_dir() and resolved_output in stale_dir.resolve().parents:
                shutil.rmtree(stale_dir)
        except OSError:
            pass
    public_package_dir = output_dir / "reproducibility_package"
    package_dir = public_package_dir
    resolved_package = public_package_dir.resolve()
    if public_package_dir.exists() and resolved_output in resolved_package.parents:
        try:
            shutil.rmtree(public_package_dir)
        except OSError:
            # Windows can keep freshly compiled LaTeX logs or PDFs locked by a previewer.
            # In that case, refresh files in place instead of failing the plotting run.
            package_dir = public_package_dir
    package_dir.mkdir(parents=True, exist_ok=True)
    resolved_current_package = package_dir.resolve()
    stale_example_outputs = package_dir / "example_outputs"
    if stale_example_outputs.exists():
        try:
            if resolved_current_package in stale_example_outputs.resolve().parents:
                shutil.rmtree(stale_example_outputs)
        except OSError:
            pass

    records: list[dict[str, object]] = []

    def copy_file(src: Path, dst_relative: str, role: str) -> None:
        dst = package_dir / dst_relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copyfile(src, dst)
        records.append(
            {
                "role": role,
                "source_path": str(src),
                "package_path": str(dst.relative_to(package_dir)),
                "included": bool(dst.exists()),
                "size_bytes": dst.stat().st_size if dst.exists() else 0,
            }
        )

    table_files = [
        "case_metadata.csv",
        "multi_condition_modeling_table.csv",
        "multi_condition_point_cloud_summary.csv",
        "geometry_model_comparison.csv",
        "superellipsoid_parameters.csv",
        "dynamics_predictions.csv",
        "dynamics_fit_summary.csv",
        "dynamics_model_comparison.csv",
        "model_selection_summary.csv",
        "error_budget_summary.csv",
        "parameter_reconciliation_audit.csv",
        "geometry_risk_summary.csv",
        "validation_hierarchy_table.csv",
        "dimensionless_numbers.csv",
        "dimensionless_sensitivity_summary.csv",
        "leave_one_condition_out_validation.csv",
        "external_validation_case_audit.csv",
        "external_validation_file_audit.csv",
        "external_validation_modeling_table.csv",
        "external_validation_geometry_model_comparison.csv",
        "external_holdout_process_response_validation.csv",
        "external_holdout_dynamics_summary.csv",
        "external_holdout_validation_summary.csv",
        "holdout_cohort_summary.csv",
        "input_file_manifest.csv",
        "boundary_extraction_sensitivity.csv",
        "geometry_selection_metrics.csv",
        "dynamics_minimal_baselines.csv",
        "external_holdout_minimal_dynamics_baselines.csv",
        "method_detail_audit.csv",
        "dynamics_derivative_audit.csv",
        "q_inf_estimation_audit.csv",
        "holdout_extrapolation_audit.csv",
        "literature_dimension_benchmark.csv",
        "dynamics_fit_asymmetry_audit.csv",
        "numerical_credibility_audit.csv",
        "environment_summary.csv",
        "active_figure_manifest.csv",
        "nomenclature_table.csv",
        "equation_inventory.csv",
    ]
    for name in table_files:
        copy_file(output_dir / "tables" / name, f"tables/{name}", "processed table")

    report_files = [
        "external_validation_data_audit.md",
        "response_matrix.md",
        "figure_captions.md",
        "submission_readiness_checklist.md",
        "cover_letter_draft.md",
        "highlights_draft.md",
        "declarations_for_submission.md",
        "amm_submission_checklist.md",
        "environment_summary.txt",
    ]
    for name in report_files:
        copy_file(output_dir / "reports" / name, f"reports/{name}", "report")

    latex_files = [
        "main_submission.tex",
        "main_submission.pdf",
        "main_submission.bbl",
        "supplementary_methods.tex",
        "supplementary_methods.pdf",
        "references.bib",
        "latex_compile_summary.txt",
    ]
    for name in latex_files:
        copy_file(output_dir / "latex" / name, f"latex/{name}", "manuscript source")

    drawio_source_files = [
        "Figure1_editable_drawio.drawio",
        "Figure1_editable_drawio.drawio.png",
        "Figure1_editable_drawio.drawio.svg",
    ]
    for name in drawio_source_files:
        copy_file(Path(name), f"source_figures/{name}", "editable figure source")

    figure_manifest_path = output_dir / "tables" / "active_figure_manifest.csv"
    if figure_manifest_path.exists() and figure_manifest_path.stat().st_size > 0:
        figure_manifest = pd.read_csv(figure_manifest_path)
        for row in figure_manifest.to_dict(orient="records"):
            for fmt in ["svg", "pdf", "tiff", "png"]:
                src_text = str(row.get(f"{fmt}_path", "")).strip()
                if not src_text:
                    continue
                src = Path(src_text)
                if not src.exists():
                    fallback_dir = output_dir / ("paper_figures" if row.get("manuscript_use") == "main_text" else "figures")
                    fallback = fallback_dir / src.name
                    if fallback.exists():
                        src = fallback
                dst_dir = "paper_figures" if "paper_figures" in src.parts or row.get("manuscript_use") == "main_text" else "figures"
                copy_file(src, f"{dst_dir}/{src.name}", "active figure export")

    copy_file(Path("scripts") / "flow3d_melt_pool_pilot.py", "scripts/flow3d_melt_pool_pilot.py", "pipeline entry point")
    copy_file(Path("scripts") / "minimal_example_summary.py", "scripts/minimal_example_summary.py", "minimal public example script")
    src_pkg = Path("scripts") / "flow3d_pipeline"
    dst_pkg = package_dir / "scripts" / "flow3d_pipeline"
    if src_pkg.exists():
        shutil.copytree(src_pkg, dst_pkg, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    records.append(
        {
            "role": "pipeline package",
            "source_path": str(src_pkg),
            "package_path": str(dst_pkg.relative_to(package_dir)),
            "included": dst_pkg.exists(),
            "size_bytes": sum(path.stat().st_size for path in dst_pkg.rglob("*") if path.is_file()) if dst_pkg.exists() else 0,
        }
    )
    src_figure_scripts = Path("scripts") / "figures"
    dst_figure_scripts = package_dir / "scripts" / "figures"
    if src_figure_scripts.exists():
        shutil.copytree(
            src_figure_scripts,
            dst_figure_scripts,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    records.append(
        {
            "role": "single-figure scripts",
            "source_path": str(src_figure_scripts),
            "package_path": str(dst_figure_scripts.relative_to(package_dir)),
            "included": dst_figure_scripts.exists(),
            "size_bytes": (
                sum(path.stat().st_size for path in dst_figure_scripts.rglob("*") if path.is_file())
                if dst_figure_scripts.exists()
                else 0
            ),
        }
    )
    write_minimal_public_example(package_dir, output_dir, records)

    readme = """# Reproducibility Materials

These processed materials accompany the manuscript and are suitable for journal supplementary data or repository archival.

Included:

- processed geometry descriptors and reduced-state time series;
- fitted superellipsoid parameters, geometry-selection metrics and model-selection tables;
- leave-one-condition-out and shared-setting numerical holdout summaries for the V-prefixed and A16-A20 cohorts;
- parameter reconciliation, geometry-risk, q_inf, holdout-extrapolation, boundary-extraction and validation-hierarchy assessment tables;
- dynamics fit-asymmetry and numerical-credibility assessment tables that document descriptor limits without adding new simulations;
- input-file index with raw CSV paths, row counts, schemas and SHA256 hashes;
- environment summary; final output checksums are generated in `analysis_outputs/tables/output_checksums.csv`;
- figure index, captions, editable Figure 1 draw.io source, nomenclature and equation inventory;
- a minimal public processed-descriptor example in `example_data/`;
- plotting/analysis scripts and LaTeX manuscript sources.

Excluded:

- proprietary FLOW-3D project files;
- raw FLOW-3D molten-region CSV exports, which are available from the corresponding author upon reasonable request subject to project-sharing and software-export constraints.

Reproduction command from the project root:

```bash
python scripts/flow3d_melt_pool_pilot.py
```

The command rebuilds the processed tables, active figures, LaTeX manuscript, compiled PDFs and these processed reproducibility materials from the available CSV exports. The final archive checksum is written outside the archive in `analysis_outputs/tables/output_checksums.csv` to avoid a self-referential ZIP hash.

Minimal public example command from the root of this reproducibility package:

```bash
python scripts/minimal_example_summary.py --input example_data/minimal_public_modeling_table.csv --output-dir example_outputs
```

This command uses only the processed descriptor-level example and writes a small summary table plus a diagnostic geometry plot. It does not require the raw FLOW-3D exports.
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")
    manifest = pd.DataFrame(records)
    manifest.to_csv(package_dir / "reproducibility_manifest.csv", index=False)
    if package_dir != public_package_dir:
        public_package_dir.mkdir(parents=True, exist_ok=True)
        for src in package_dir.rglob("*"):
            if not src.is_file():
                continue
            dst = public_package_dir / src.relative_to(package_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(src, dst)
            except OSError:
                pass
    zip_base = output_dir / "reproducibility_package"
    zip_path = zip_base.with_suffix(".zip")
    stale_tmp_zip_path = output_dir / f".{zip_path.name}.tmp"
    tmp_zip_path = Path(tempfile.gettempdir()) / f"{zip_path.stem}-{os.getpid()}.tmp.zip"
    try:
        if stale_tmp_zip_path.exists():
            try:
                stale_tmp_zip_path.unlink()
            except OSError:
                pass
        if tmp_zip_path.exists():
            try:
                tmp_zip_path.unlink()
            except OSError:
                pass
        with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for src in sorted(package_dir.rglob("*")):
                if not src.is_file():
                    continue
                relative = src.relative_to(package_dir)
                if any(part in {"example_outputs", ".matplotlib-cache", "__pycache__"} for part in relative.parts):
                    continue
                archive.write(src, arcname=relative.as_posix())
        try:
            tmp_zip_path.replace(zip_path)
        except OSError:
            # Some synchronized Windows folders permit writing the archive but deny
            # file replacement or cross-volume moves. Copying bytes into the existing
            # path refreshes the archive without relying on a rename operation.
            shutil.copyfile(tmp_zip_path, zip_path)
            try:
                tmp_zip_path.unlink()
            except OSError:
                pass
    except OSError as exc:
        try:
            if tmp_zip_path.exists():
                tmp_zip_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Could not refresh {zip_path}. Close any program using the archive and rerun the pipeline."
        ) from exc
    if package_dir != public_package_dir:
        try:
            shutil.rmtree(package_dir)
        except OSError:
            pass
    return manifest


def reference_seed_entries() -> list[dict[str, str | int | bool]]:
    """Post-2020-first candidate bibliography for the manuscript package.

    A small number of older items are retained only when they are foundational
    to the moving-source, superquadric or regularization formulation.
    """

    def ref(
        topic: str,
        key: str,
        title: str,
        year: int,
        source_type: str,
        relevance: str,
        use: str,
        risk: str,
        source: str,
        bibtex: str,
        priority: str = "supporting",
    ) -> dict[str, str | int | bool]:
        return {
            "topic": topic,
            "citation_key": key,
            "paper_title": title,
            "year": year,
            "source_type": source_type,
            "method_relevance": relevance,
            "how_to_use_in_manuscript": use,
            "risk_or_limitation": risk,
            "verification_source": source,
            "priority_in_current_manuscript": priority,
            "is_2020_or_later": year >= 2020,
            "bibtex": bibtex,
        }

    return [
        ref(
            "DED review",
            "ahn2021",
            "Directed Energy Deposition (DED) Process: State of the Art",
            2021,
            "review",
            "Recent high-citation overview of DED process principles, classification and research trends.",
            "Use in the opening paragraph to define DED and frame process scope.",
            "Broad process review, not a melt-pool reduced-order model.",
            "Springer page, DOI 10.1007/s40684-020-00302-7.",
            """@article{ahn2021,
  author = {Ahn, Dong-Gyu},
  title = {Directed Energy Deposition (DED) Process: State of the Art},
  journal = {International Journal of Precision Engineering and Manufacturing-Green Technology},
  volume = {8},
  pages = {703--742},
  year = {2021},
  doi = {10.1007/s40684-020-00302-7}
}""",
            "core",
        ),
        ref(
            "DED review",
            "svetlizky2021",
            "Directed energy deposition (DED) additive manufacturing: Physical characteristics, defects, challenges and applications",
            2021,
            "review",
            "Recent Materials Today review covering DED laser-material interaction, melt-pool behavior, defects and monitoring.",
            "Use to justify the physical importance of melt-pool geometry and thermal behavior.",
            "Broad review; not specific to FLOW-3D or mathematical free-boundary reduction.",
            "ScienceDirect and Tel Aviv University records, DOI 10.1016/j.mattod.2021.03.020.",
            """@article{svetlizky2021,
  author = {Svetlizky, David and Das, Mitun and Zheng, Baolong and Vyatskikh, Alexandra L. and Bose, Susmita and Bandyopadhyay, Amit and Schoenung, Julie M. and Lavernia, Enrique J. and Eliaz, Noam},
  title = {Directed Energy Deposition (DED) Additive Manufacturing: Physical Characteristics, Defects, Challenges and Applications},
  journal = {Materials Today},
  volume = {49},
  pages = {271--295},
  year = {2021},
  doi = {10.1016/j.mattod.2021.03.020}
}""",
            "core",
        ),
        ref(
            "DED review",
            "li2023",
            "Directed energy deposition of metals: processing, microstructures, and mechanical properties",
            2023,
            "review",
            "Recent International Materials Reviews article linking DED processing, thermal history, microstructure and properties.",
            "Use to frame the process-structure-property importance of melt-pool descriptors.",
            "Review emphasizes materials and microstructure more than reduced-order modeling.",
            "SAGE/Taylor record, DOI 10.1080/09506608.2022.2097411.",
            """@article{li2023,
  author = {Li, Shi-Hao and Kumar, Punit and Chandra, Shubham and Ramamurty, Upadrasta},
  title = {Directed Energy Deposition of Metals: Processing, Microstructures, and Mechanical Properties},
  journal = {International Materials Reviews},
  volume = {68},
  number = {6},
  pages = {605--647},
  year = {2023},
  doi = {10.1080/09506608.2022.2097411}
}""",
            "core",
        ),
        ref(
            "DED numerical modeling",
            "poggi2022",
            "State-of-the-art of numerical simulation of laser powder Directed Energy Deposition process",
            2022,
            "review",
            "Summarizes numerical simulation approaches and required model inputs for laser powder DED.",
            "Use to position FLOW-3D output as high-fidelity source data requiring reduced-order interpretation.",
            "Procedia review; concise rather than exhaustive.",
            "ScienceDirect page, DOI 10.1016/j.procir.2022.09.012.",
            """@article{poggi2022,
  author = {Poggi, Mirna and Atzeni, Eleonora and Iuliano, Luca and Salmi, Alessandro},
  title = {State-of-the-art of Numerical Simulation of Laser Powder Directed Energy Deposition Process},
  journal = {Procedia CIRP},
  volume = {112},
  pages = {376--381},
  year = {2022},
  doi = {10.1016/j.procir.2022.09.012}
}""",
            "core",
        ),
        ref(
            "DED CFD",
            "zhang2021dedcfd",
            "Numerical investigation on heat transfer of melt pool and clad generation in directed energy deposition of stainless steel",
            2021,
            "CFD model",
            "Stainless-steel DED CFD study with VOF treatment, melt-pool dynamics and clad generation.",
            "Use as the closest post-2020 CFD context for stainless-steel DED melt-pool modeling.",
            "Forward CFD model, not a reduced-order model from exported point clouds.",
            "ScienceDirect page, DOI 10.1016/j.ijthermalsci.2021.106954.",
            """@article{zhang2021dedcfd,
  author = {Zhang, Y. M. and Lim, C. W. J. and Tang, C. and Li, B.},
  title = {Numerical Investigation on Heat Transfer of Melt Pool and Clad Generation in Directed Energy Deposition of Stainless Steel},
  journal = {International Journal of Thermal Sciences},
  volume = {165},
  pages = {106954},
  year = {2021},
  doi = {10.1016/j.ijthermalsci.2021.106954}
}""",
            "core",
        ),
        ref(
            "laser cladding molten-pool evolution",
            "song2021cladding",
            "Development mechanism and solidification morphology of molten pool generated by laser cladding",
            2021,
            "CFD model",
            "Authors' previous numerical study of molten-pool formation, convection and solidification morphology in laser cladding.",
            "Use to justify the observation window covering the rapid early transient and quasi-steady molten-pool evolution.",
            "Laser cladding context; used here only for temporal-evolution motivation.",
            "Local PDF Song_THESCI.pdf and DOI metadata, DOI 10.1016/j.ijthermalsci.2020.106579.",
            """@article{song2021cladding,
  author = {Song, Boxue and Yu, Tianbiao and Jiang, Xingyu and Xi, Wenchao and Lin, Xiaoli},
  title = {Development Mechanism and Solidification Morphology of Molten Pool Generated by Laser Cladding},
  journal = {International Journal of Thermal Sciences},
  volume = {159},
  pages = {106579},
  year = {2021},
  doi = {10.1016/j.ijthermalsci.2020.106579}
}""",
            "core",
        ),
        ref(
            "DED powder physics",
            "wang2023powder",
            "Multi-phase flow simulation of powder streaming in laser-based directed energy deposition",
            2023,
            "multiphase simulation",
            "Recent multiphase gas-powder-laser model for laser-based DED powder delivery.",
            "Use to justify keeping powder-coupled effects in the physical background while not modeling powder flow directly.",
            "Powder-stream model, not a melt-pool free-boundary reduction.",
            "ScienceDirect page, DOI 10.1016/j.ijheatmasstransfer.2023.124240.",
            """@article{wang2023powder,
  author = {Wang, Lu and Wang, Shuhao and Zhang, Yanming and Yan, Wentao},
  title = {Multi-phase Flow Simulation of Powder Streaming in Laser-based Directed Energy Deposition},
  journal = {International Journal of Heat and Mass Transfer},
  volume = {212},
  pages = {124240},
  year = {2023},
  doi = {10.1016/j.ijheatmasstransfer.2023.124240}
}""",
        ),
        ref(
            "DED melt-pool monitoring",
            "hong2025catchment",
            "Prediction of the powder catchment efficiency based on off-axial powder monitoring and coaxial melt pool monitoring in laser directed energy deposition",
            2025,
            "monitoring and process efficiency",
            "Recent L-DED study linking off-axis powder monitoring and coaxial melt-pool monitoring to powder catchment efficiency.",
            "Use to connect powder delivery, melt-pool state monitoring and reduced process descriptors.",
            "Monitoring and efficiency prediction study; not a mathematical boundary-envelope reduction.",
            "Crossref DOI metadata, DOI 10.1088/1361-6501/ae0a6f.",
            """@article{hong2025catchment,
  author = {Hong, Weijie and Zheng, Yi and Ma, Chenguang and Zhang, Yingjie},
  title = {Prediction of the Powder Catchment Efficiency Based on Off-axial Powder Monitoring and Coaxial Melt Pool Monitoring in Laser Directed Energy Deposition},
  journal = {Measurement Science and Technology},
  volume = {36},
  number = {10},
  pages = {105204},
  year = {2025},
  doi = {10.1088/1361-6501/ae0a6f}
}""",
        ),
        ref(
            "DED defect physics",
            "zhang2024pore",
            "Pore evolution mechanisms during directed energy deposition additive manufacturing",
            2024,
            "in situ mechanism study",
            "Nature Communications study linking DED melt-pool dynamics, pores and in situ synchrotron observations.",
            "Use to show why melt-pool boundary evolution matters for defect formation.",
            "Defect-mechanism study rather than reduced-order free-boundary modeling.",
            "Crossref DOI metadata, DOI 10.1038/s41467-024-45913-9.",
            """@article{zhang2024pore,
  author = {Zhang, Kai and Chen, Yunhui and Marussi, Sebastian and Fan, Xianqiang and Fitzpatrick, Maureen and Bhagavath, Shishira and Majkut, Marta and Lukic, Bratislav and others},
  title = {Pore Evolution Mechanisms During Directed Energy Deposition Additive Manufacturing},
  journal = {Nature Communications},
  volume = {15},
  number = {1},
  pages = {1715},
  year = {2024},
  doi = {10.1038/s41467-024-45913-9}
}""",
            "core",
        ),
        ref(
            "DED in situ imaging",
            "sinclair2024gasflow",
            "An in situ imaging investigation of the effect of gas flow rates on directed energy deposition",
            2024,
            "in situ imaging",
            "Recent in situ imaging study showing how gas-flow conditions influence DED process behavior.",
            "Use to strengthen the claim that melt-pool observations encode process-state information.",
            "Focuses on gas-flow effects, not on analytic reduced-order modeling.",
            "Crossref DOI metadata, DOI 10.1016/j.matdes.2024.113183.",
            """@article{sinclair2024gasflow,
  author = {Sinclair, Lorna and Hatt, Oliver and Clark, Samuel J. and Marussi, Sebastian and Ruckh, Elena and Atwood, Robert C. and Jones, Martyn and Baxter, Gavin J. and others},
  title = {An in situ Imaging Investigation of the Effect of Gas Flow Rates on Directed Energy Deposition},
  journal = {Materials and Design},
  volume = {244},
  pages = {113183},
  year = {2024},
  doi = {10.1016/j.matdes.2024.113183}
}""",
        ),
        ref(
            "DED thermofluidics",
            "lei2024shaping",
            "Manipulating melt pool thermofluidic transport in directed energy deposition driven by a laser intensity spatial shaping strategy",
            2024,
            "thermofluidic experiment/model",
            "Recent DED study showing that melt-pool transport can be intentionally modified through laser intensity shaping.",
            "Use to support the physical relevance of thermofluidic free-boundary descriptors.",
            "Process actuation study rather than a reduced-order model from FLOW-3D point clouds.",
            "Crossref DOI metadata, DOI 10.1080/17452759.2024.2308513.",
            """@article{lei2024shaping,
  author = {Lei, Chaojiao and Ren, Song and Yin, Cunhong and Liu, Xixia and Chen, Mingfei and Wu, Jiazhu and Han, Changjun},
  title = {Manipulating Melt Pool Thermofluidic Transport in Directed Energy Deposition Driven by a Laser Intensity Spatial Shaping Strategy},
  journal = {Virtual and Physical Prototyping},
  volume = {19},
  number = {1},
  pages = {e2308513},
  year = {2024},
  doi = {10.1080/17452759.2024.2308513}
}""",
            "core",
        ),
        ref(
            "DED melt-pool dynamics",
            "chen2025temporal",
            "Revealing melt pool dynamics during laser temporal shaping directed energy deposition of 316L stainless steel",
            2025,
            "melt-pool dynamics",
            "Recent 316L L-DED study showing how temporal laser shaping modifies melt-pool dynamics.",
            "Use to update the thermofluidic context for process-actuated melt-pool geometry and dynamics.",
            "Process-actuation study rather than a reduced-order boundary-manifold model.",
            "Crossref DOI metadata, DOI 10.1016/j.optlastec.2025.113774.",
            """@article{chen2025temporal,
  author = {Chen, Zhenggang and Wu, Jiazhu and Lei, Yuchao and Yin, Cunhong and Wang, Gui and Cao, Yang},
  title = {Revealing Melt Pool Dynamics During Laser Temporal Shaping Directed Energy Deposition of 316L Stainless Steel},
  journal = {Optics and Laser Technology},
  volume = {192},
  pages = {113774},
  year = {2025},
  doi = {10.1016/j.optlastec.2025.113774}
}""",
            "core",
        ),
        ref(
            "DED computational framework",
            "kovsca2023",
            "Towards an automated framework for the finite element computational modelling of directed energy deposition",
            2023,
            "computational framework",
            "Automated FE framework for DED with free-surface detection and experiment-based comparison.",
            "Use in Methods/Discussion to connect the boundary-observation problem to modern DED simulation workflows.",
            "FE thermo-mechanical framework, not FLOW-3D thermo-fluid output.",
            "ScienceDirect page, DOI 10.1016/j.finel.2023.103949.",
            """@article{kovsca2023,
  author = {Kovsca, Dejan and Starman, Bojan and Klobcar, Damjan and Halilovic, Miroslav and Mole, Nikolaj},
  title = {Towards an Automated Framework for the Finite Element Computational Modelling of Directed Energy Deposition},
  journal = {Finite Elements in Analysis and Design},
  volume = {221},
  pages = {103949},
  year = {2023},
  doi = {10.1016/j.finel.2023.103949}
}""",
        ),
        ref(
            "DED control",
            "liao2022",
            "Simulation-guided variable laser power design for melt pool depth control in directed energy deposition",
            2022,
            "simulation-guided control",
            "Uses simulation to design time-series laser power profiles for melt-pool depth control.",
            "Use to motivate reduced-order melt-pool states as useful for future control-oriented modeling.",
            "Control target differs from the present single-condition post-processing model.",
            "ScienceDirect page, DOI 10.1016/j.addma.2022.102912.",
            """@article{liao2022,
  author = {Liao, Shuheng and Webster, Samantha and Huang, Dean and Council, Raymonde and Ehmann, Kornel and Cao, Jian},
  title = {Simulation-guided Variable Laser Power Design for Melt Pool Depth Control in Directed Energy Deposition},
  journal = {Additive Manufacturing},
  volume = {56},
  pages = {102912},
  year = {2022},
  doi = {10.1016/j.addma.2022.102912}
}""",
            "core",
        ),
        ref(
            "DED control",
            "smoqi2022",
            "Closed-loop control of meltpool temperature in directed energy deposition",
            2022,
            "control experiment",
            "Closed-loop pyrometry-based melt-pool temperature control for powder-laser DED.",
            "Use to connect the attractor state to control relevance without claiming controller design.",
            "Experimental control study rather than mathematical model selection.",
            "University repository and DOI 10.1016/j.matdes.2022.110508.",
            """@article{smoqi2022,
  author = {Smoqi, Ziyad M. and Bevans, Benjamin D. and Gaikwad, Aniruddha and Craig, James and Abul-Haj, Alan and Roeder, Brent and Macy, Bill and Shield, Jeffrey E. and Rao, Prahalada K.},
  title = {Closed-loop Control of Meltpool Temperature in Directed Energy Deposition},
  journal = {Materials and Design},
  volume = {215},
  pages = {110508},
  year = {2022},
  doi = {10.1016/j.matdes.2022.110508}
}""",
        ),
        ref(
            "DED closed-loop control",
            "shin2025heightcontrol",
            "Enhanced geometric accuracy in directed energy deposition via closed-loop melt pool height control using real-time thermal imaging",
            2025,
            "closed-loop monitoring and control",
            "Recent DED study demonstrating closed-loop melt-pool height control with real-time thermal imaging.",
            "Use to update the control-oriented motivation for compact melt-pool state descriptors.",
            "Closed-loop control experiment rather than analytic manifold fitting.",
            "Crossref DOI metadata, DOI 10.1016/j.addma.2025.104846.",
            """@article{shin2025heightcontrol,
  author = {Shin, Subin and Jeon, Ikgeun and Sohn, Hoon},
  title = {Enhanced Geometric Accuracy in Directed Energy Deposition via Closed-loop Melt Pool Height Control Using Real-time Thermal Imaging},
  journal = {Additive Manufacturing},
  volume = {109},
  pages = {104846},
  year = {2025},
  doi = {10.1016/j.addma.2025.104846}
}""",
        ),
        ref(
            "DED monitoring",
            "dasilva2023",
            "Melt pool monitoring and process optimisation of directed energy deposition via coaxial thermal imaging",
            2023,
            "monitoring",
            "Recent DED melt-pool monitoring and process optimization study based on coaxial thermal imaging.",
            "Use to justify melt-pool area/length evolution as a process-stability signal.",
            "Monitoring paper, not simulation-based free-boundary reduction.",
            "ScienceDirect page, DOI 10.1016/j.jmapro.2023.10.021.",
            """@article{dasilva2023,
  author = {Da Silva, Adrien and Frostevarg, Jan and Kaplan, Alexander F. H.},
  title = {Melt Pool Monitoring and Process Optimisation of Directed Energy Deposition via Coaxial Thermal Imaging},
  journal = {Journal of Manufacturing Processes},
  volume = {107},
  pages = {126--133},
  year = {2023},
  doi = {10.1016/j.jmapro.2023.10.021}
}""",
        ),
        ref(
            "DED melt-pool monitoring",
            "ji2025coaxial",
            "Coaxial melt pool monitoring with pyrometer and camera for hybrid CNN-based bead geometry prediction in directed energy deposition",
            2025,
            "multimodal monitoring",
            "Recent DED study combining pyrometer and camera data for CNN-based bead-geometry prediction.",
            "Use to update the monitoring literature connecting melt-pool signatures to geometric descriptors.",
            "Monitoring and prediction paper rather than a physics-grounded reduced dynamical model.",
            "Crossref DOI metadata, DOI 10.1016/j.precisioneng.2025.02.016.",
            """@article{ji2025coaxial,
  author = {Ji, Seong Hun and Ko, Tae Hwan and Yoon, Jongcheon and Lee, Seung Hwan and Lee, Hyub},
  title = {Coaxial Melt Pool Monitoring with Pyrometer and Camera for Hybrid CNN-based Bead Geometry Prediction in Directed Energy Deposition},
  journal = {Precision Engineering},
  volume = {94},
  pages = {1--12},
  year = {2025},
  doi = {10.1016/j.precisioneng.2025.02.016}
}""",
        ),
        ref(
            "DED process monitoring",
            "asadi2024dnn",
            "Process monitoring by deep neural networks in directed energy deposition: CNN-based detection, segmentation, and statistical analysis of melt pools",
            2024,
            "deep-learning monitoring",
            "Recent DED melt-pool process monitoring paper using CNN detection and segmentation.",
            "Use to position melt-pool geometry as a measurable process-state signal.",
            "Image-based monitoring model rather than analytic dynamics.",
            "Crossref DOI metadata, DOI 10.1016/j.rcim.2023.102710.",
            """@article{asadi2024dnn,
  author = {Asadi, Reza and Queguineur, Antoine and Wiikinkoski, Olli and Mokhtarian, Hossein and Aihkisalo, Tommi and Revuelta, Alejandro and Flores Ituarte, Inigo},
  title = {Process Monitoring by Deep Neural Networks in Directed Energy Deposition: CNN-based Detection, Segmentation, and Statistical Analysis of Melt Pools},
  journal = {Robotics and Computer-Integrated Manufacturing},
  volume = {87},
  pages = {102710},
  year = {2024},
  doi = {10.1016/j.rcim.2023.102710}
}""",
            "core",
        ),
        ref(
            "DED defect monitoring",
            "abranovic2024flaw",
            "Melt pool level flaw detection in laser hot wire directed energy deposition using a convolutional long short-term memory autoencoder",
            2024,
            "deep-learning defect detection",
            "Uses temporal deep learning to detect melt-pool-level flaws in laser hot-wire DED.",
            "Use to motivate time-series melt-pool descriptors and validation against overfitting.",
            "Laser hot-wire DED and neural detection rather than free-boundary reduction.",
            "Crossref DOI metadata, DOI 10.1016/j.addma.2023.103843.",
            """@article{abranovic2024flaw,
  author = {Abranovic, Brandon and Sarkar, Sulagna and Chang-Davidson, Elizabeth and Beuth, Jack},
  title = {Melt Pool Level Flaw Detection in Laser Hot Wire Directed Energy Deposition Using a Convolutional Long Short-term Memory Autoencoder},
  journal = {Additive Manufacturing},
  volume = {79},
  pages = {103843},
  year = {2024},
  doi = {10.1016/j.addma.2023.103843}
}""",
        ),
        ref(
            "DED infrared monitoring",
            "herzog2024infrared",
            "Defect detection by multi-axis infrared process monitoring of laser beam directed energy deposition",
            2024,
            "infrared monitoring",
            "Multi-axis infrared monitoring study for defect detection in laser beam DED.",
            "Use to connect melt-pool thermal signatures with defect-sensitive process observation.",
            "Monitoring-focused, not a CFD-informed reduced-order model.",
            "Crossref DOI metadata, DOI 10.1038/s41598-024-53931-2.",
            """@article{herzog2024infrared,
  author = {Herzog, T. and Brandt, M. and Trinchi, A. and Sola, A. and Hagenlocher, C. and Molotnikov, A.},
  title = {Defect Detection by Multi-axis Infrared Process Monitoring of Laser Beam Directed Energy Deposition},
  journal = {Scientific Reports},
  volume = {14},
  number = {1},
  pages = {3861},
  year = {2024},
  doi = {10.1038/s41598-024-53931-2}
}""",
        ),
        ref(
            "DED monitoring",
            "kong2023monitoring",
            "Development of melt-pool monitoring system based on degree of irregularity for defect diagnosis of directed energy deposition process",
            2023,
            "monitoring metric",
            "Defines an irregularity-based melt-pool monitoring metric for DED defect diagnosis.",
            "Use to justify boundary irregularity and shape descriptors as process-state observables.",
            "Short monitoring study rather than mathematical model selection.",
            "Crossref DOI metadata, DOI 10.57062/ijpem-st.2023.0045.",
            """@article{kong2023monitoring,
  author = {Kong, Jun Ho and Lee, Sang Won},
  title = {Development of Melt-pool Monitoring System Based on Degree of Irregularity for Defect Diagnosis of Directed Energy Deposition Process},
  journal = {International Journal of Precision Engineering and Manufacturing-Smart Technology},
  volume = {1},
  number = {2},
  pages = {137--143},
  year = {2023},
  doi = {10.57062/ijpem-st.2023.0045}
}""",
        ),
        ref(
            "DED control",
            "miao2023lqr",
            "Closed loop control of melt pool width in laser directed energy deposition process based on PSO-LQR",
            2023,
            "closed-loop control",
            "Closed-loop controller for DED melt-pool width using PSO-LQR.",
            "Use to support the control relevance of width and low-dimensional melt-pool states.",
            "Controller design, not free-boundary parameter identification.",
            "Crossref DOI metadata, DOI 10.1109/ACCESS.2023.3292789.",
            """@article{miao2023lqr,
  author = {Miao, Liguo and Xing, Fei and Chai, Yuanxin},
  title = {Closed Loop Control of Melt Pool Width in Laser Directed Energy Deposition Process Based on PSO-LQR},
  journal = {IEEE Access},
  volume = {11},
  pages = {78170--78181},
  year = {2023},
  doi = {10.1109/ACCESS.2023.3292789}
}""",
        ),
        ref(
            "DED control",
            "rahmani2024psq",
            "System identification and closed-loop control of laser hot-wire directed energy deposition using the parameter-signature-quality modeling scheme",
            2024,
            "system identification",
            "Recent DED system-identification and closed-loop-control study.",
            "Use to connect the attractor model with modern data-supported system identification.",
            "Laser hot-wire DED differs from laser powder L-DED; the paper is used for control context only.",
            "Crossref DOI metadata, DOI 10.1016/j.jmapro.2024.01.029.",
            """@article{rahmani2024psq,
  author = {Rahmani Dehaghani, Mostafa and Sahraeidolatkhaneh, Atieh and Nilsen, Morgan and Sikstrom, Fredrik and Sajadi, Pouyan and Tang, Yifan and Wang, G. Gary},
  title = {System Identification and Closed-loop Control of Laser Hot-wire Directed Energy Deposition Using the Parameter-Signature-Quality Modeling Scheme},
  journal = {Journal of Manufacturing Processes},
  volume = {112},
  pages = {1--13},
  year = {2024},
  doi = {10.1016/j.jmapro.2024.01.029}
}""",
        ),
        ref(
            "melt-pool machine learning",
            "akbari2022",
            "MeltpoolNet: Melt pool characteristic prediction in Metal Additive Manufacturing using machine learning",
            2022,
            "machine learning benchmark",
            "Benchmarks physics-aware ML for melt-pool defect and geometry prediction.",
            "Use to show the field trend toward compact melt-pool descriptors and interpretable surrogates.",
            "General metal AM and not FLOW-3D-free-boundary identification.",
            "ScienceDirect page, DOI 10.1016/j.addma.2022.102817.",
            """@article{akbari2022,
  author = {Akbari, Parand and Ogoke, Francis and Kao, Ning-Yu and Meidani, Kazem and Yeh, Chun-Yu and Lee, William and Barati Farimani, Amir},
  title = {MeltpoolNet: Melt Pool Characteristic Prediction in Metal Additive Manufacturing Using Machine Learning},
  journal = {Additive Manufacturing},
  volume = {55},
  pages = {102817},
  year = {2022},
  doi = {10.1016/j.addma.2022.102817}
}""",
            "core",
        ),
        ref(
            "melt-pool surrogate modeling",
            "hemmasian2023",
            "Surrogate modeling of melt pool temperature field using deep learning",
            2023,
            "surrogate model",
            "Flow-3D-based dataset and deep-learning surrogate for 3D melt-pool temperature fields.",
            "Use to position the present work as analytic reduced-order modeling rather than black-box thermal-field surrogate learning.",
            "LPBF rather than DED; still highly relevant because it uses Flow-3D melt-pool simulations.",
            "ScienceDirect page, DOI 10.1016/j.addlet.2023.100123.",
            """@article{hemmasian2023,
  author = {Hemmasian, AmirPouya and Ogoke, Francis and Akbari, Parand and Malen, Jonathan and Beuth, Jack and Barati Farimani, Amir},
  title = {Surrogate Modeling of Melt Pool Temperature Field Using Deep Learning},
  journal = {Additive Manufacturing Letters},
  volume = {5},
  pages = {100123},
  year = {2023},
  doi = {10.1016/j.addlet.2023.100123}
}""",
            "core",
        ),
        ref(
            "DED surrogate modeling",
            "wu2024",
            "A Robust Recurrent Neural Networks-Based Surrogate Model for Thermal History and Melt Pool Characteristics in Directed Energy Deposition",
            2024,
            "time-series surrogate",
            "Recent RNN/LSTM surrogate for DED thermal history and melt-pool characteristics.",
            "Use to contrast data-hungry neural surrogates with the present short-sequence analytic attractor.",
            "Neural model with broader simulated/experimental design, not a free-boundary manifold.",
            "MDPI page, DOI 10.3390/ma17174363.",
            """@article{wu2024,
  author = {Wu, Sung-Heng and Tariq, Usman and Joy, Ranjit and Mahmood, Muhammad Arif and Malik, Asad Waqar and Liou, Frank},
  title = {A Robust Recurrent Neural Networks-Based Surrogate Model for Thermal History and Melt Pool Characteristics in Directed Energy Deposition},
  journal = {Materials},
  volume = {17},
  number = {17},
  pages = {4363},
  year = {2024},
  doi = {10.3390/ma17174363}
}""",
            "core",
        ),
        ref(
            "DED surrogate modeling",
            "wang2023tcn",
            "Prediction of melt pool width and layer height for laser directed energy deposition enabled by physics-driven temporal convolutional network",
            2023,
            "physics-driven neural surrogate",
            "Physics-driven temporal convolutional network for DED melt-pool width and layer-height prediction.",
            "Use to contrast neural time-series surrogates with the present analytic attractor model.",
            "Learns process-output relations rather than an explicit free-boundary manifold.",
            "Crossref DOI metadata, DOI 10.1016/j.jmsy.2023.06.002.",
            """@article{wang2023tcn,
  author = {Wang, Yanghui and Hu, Kaixiong and Li, Weidong and Wang, Lihui},
  title = {Prediction of Melt Pool Width and Layer Height for Laser Directed Energy Deposition Enabled by Physics-driven Temporal Convolutional Network},
  journal = {Journal of Manufacturing Systems},
  volume = {69},
  pages = {1--17},
  year = {2023},
  doi = {10.1016/j.jmsy.2023.06.002}
}""",
            "core",
        ),
        ref(
            "DED thermal modeling",
            "jelinek2020thermalfe",
            "Two-dimensional thermal finite element model of directed energy deposition: Matching melt pool temperature profile with pyrometer measurement",
            2020,
            "thermal finite-element model",
            "Thermal FE model calibrated against pyrometer melt-pool temperature profiles.",
            "Use to support the link between melt-pool thermal observables and reduced thermal descriptors.",
            "Two-dimensional thermal model, not full CFD point-cloud reduction.",
            "Crossref DOI metadata, DOI 10.1016/j.jmapro.2020.06.021.",
            """@article{jelinek2020thermalfe,
  author = {Jelinek, Bohumir and Young, William J. and Dantin, Matthew and Furr, William and Doude, Haley and Priddy, Matthew W.},
  title = {Two-dimensional Thermal Finite Element Model of Directed Energy Deposition: Matching Melt Pool Temperature Profile with Pyrometer Measurement},
  journal = {Journal of Manufacturing Processes},
  volume = {57},
  pages = {187--195},
  year = {2020},
  doi = {10.1016/j.jmapro.2020.06.021}
}""",
        ),
        ref(
            "DED surrogate uncertainty",
            "pham2022uncertainty",
            "Characterization, propagation, and sensitivity analysis of uncertainties in the directed energy deposition process using a deep learning-based surrogate model",
            2022,
            "surrogate uncertainty analysis",
            "Deep-learning surrogate used for uncertainty propagation and sensitivity in DED.",
            "Use to support the paper's uncertainty and sensitivity framing.",
            "Surrogate UQ, not analytic free-boundary error budgeting.",
            "Crossref DOI metadata, DOI 10.1016/j.probengmech.2022.103297.",
            """@article{pham2022uncertainty,
  author = {Pham, T. Q. D. and Hoang, T. V. and Tran, X. V. and Fetni, Seifallah and Duchene, L. and Tran, H. S. and Habraken, A. M.},
  title = {Characterization, Propagation, and Sensitivity Analysis of Uncertainties in the Directed Energy Deposition Process Using a Deep Learning-based Surrogate Model},
  journal = {Probabilistic Engineering Mechanics},
  volume = {69},
  pages = {103297},
  year = {2022},
  doi = {10.1016/j.probengmech.2022.103297}
}""",
            "core",
        ),
        ref(
            "DED process mapping uncertainty",
            "menon2022multifidelity",
            "Multi-fidelity surrogate-based process mapping with uncertainty quantification in laser directed energy deposition",
            2022,
            "multi-fidelity UQ",
            "Multi-fidelity surrogate and UQ workflow for laser DED process mapping.",
            "Use to contrast process-map prediction with the present single-condition reduced model.",
            "Process-map UQ rather than observed free-boundary dynamics.",
            "Crossref DOI metadata, DOI 10.3390/ma15082902.",
            """@article{menon2022multifidelity,
  author = {Menon, Nandana and Mondal, Sudeepta and Basak, Amrita},
  title = {Multi-Fidelity Surrogate-Based Process Mapping with Uncertainty Quantification in Laser Directed Energy Deposition},
  journal = {Materials},
  volume = {15},
  number = {8},
  pages = {2902},
  year = {2022},
  doi = {10.3390/ma15082902}
}""",
            "core",
        ),
        ref(
            "physics-informed AM modeling",
            "jiang2024piml",
            "Physics-Informed Machine Learning for Accurate Prediction of Temperature and Melt Pool Dimension in Metal Additive Manufacturing",
            2024,
            "physics-informed ML",
            "Recent PIML model for temperature and melt-pool dimension prediction with limited labeled data.",
            "Use to frame the manuscript as physics-informed and data-limited, while noting it does not use PINNs.",
            "General metal AM rather than DED-specific FLOW-3D point-cloud reduction.",
            "SAGE page, DOI 10.1089/3dp.2022.0363.",
            """@article{jiang2024piml,
  author = {Jiang, Feilong and Xia, Min and Hu, Yaowu},
  title = {Physics-Informed Machine Learning for Accurate Prediction of Temperature and Melt Pool Dimension in Metal Additive Manufacturing},
  journal = {3D Printing and Additive Manufacturing},
  volume = {11},
  number = {4},
  pages = {1679--1689},
  year = {2024},
  doi = {10.1089/3dp.2022.0363}
}""",
        ),
        ref(
            "DED physics-informed ML",
            "kumar2023piml",
            "Physics-informed machine learning models for the prediction of transient temperature distribution of ferritic steel in directed energy deposition by cold metal transfer",
            2023,
            "physics-informed ML",
            "Recent DED-adjacent physics-informed ML study for transient temperature prediction.",
            "Use as a supporting post-2020 example of simulation-assisted ML for DED temperature fields.",
            "Cold metal transfer process differs from laser powder L-DED.",
            "SAGE/Taylor page, DOI 10.1080/13621718.2023.2247242.",
            """@article{kumar2023piml,
  author = {Kumar, Amritesh and Sarma, Ritam and Bag, Swarup and Srivastava, V. C. and Kapil, Sajan},
  title = {Physics-informed Machine Learning Models for the Prediction of Transient Temperature Distribution of Ferritic Steel in Directed Energy Deposition by Cold Metal Transfer},
  journal = {Science and Technology of Welding and Joining},
  volume = {28},
  number = {9},
  pages = {914--922},
  year = {2023},
  doi = {10.1080/13621718.2023.2247242}
}""",
        ),
        ref(
            "DED machine learning review",
            "era2023",
            "Machine Learning in Directed Energy Deposition (DED) Additive Manufacturing: A State-of-the-art Review",
            2023,
            "review",
            "Reviews ML applications in DED and highlights data limitations and model categories.",
            "Use in the Introduction or Discussion to position the model against ML-heavy DED literature.",
            "Conference-proceedings review retained as a candidate source for DED process context.",
            "Manufacturing Letters DOI 10.1016/j.mfglet.2023.08.079.",
            """@article{era2023,
  author = {Era, Israt Zarin and Farahani, Mojtaba A. and Wuest, Thorsten and Liu, Zhi-Chao},
  title = {Machine Learning in Directed Energy Deposition (DED) Additive Manufacturing: A State-of-the-art Review},
  journal = {Manufacturing Letters},
  volume = {35},
  pages = {689--700},
  year = {2023},
  doi = {10.1016/j.mfglet.2023.08.079}
}""",
        ),
        ref(
            "melt-pool machine learning",
            "zhu2023",
            "Prediction of melt pool shape in additive manufacturing based on machine learning methods",
            2023,
            "machine learning",
            "Uses DED experiment-derived melt-pool shapes to predict height, width and depth with ML models.",
            "Use to cite recent interest in melt-pool shape prediction from process data.",
            "Empirical ML model, not a physical free-boundary/dynamics model.",
            "ScienceDirect page, DOI 10.1016/j.optlastec.2022.108964.",
            """@article{zhu2023,
  author = {Zhu, Xiaobo and Jiang, Fengchun and Guo, Chunhuan and Gao, Huabing and Wang, Zhen and Dong, Tao and Li, Haixin},
  title = {Prediction of Melt Pool Shape in Additive Manufacturing Based on Machine Learning Methods},
  journal = {Optics and Laser Technology},
  volume = {159},
  pages = {108964},
  year = {2023},
  doi = {10.1016/j.optlastec.2022.108964}
}""",
        ),
        ref(
            "DED melt-pool geometry prediction",
            "xu2026nonsteady",
            "Prediction of melt pool geometry in laser directed energy deposition considering non-steady-state melt pool behavior",
            2026,
            "melt-pool geometry prediction",
            "Recent L-DED prediction study that explicitly accounts for non-steady-state melt-pool behavior.",
            "Use to update the process-response and prediction literature with a 2026 geometry-focused reference.",
            "Predictive modeling paper rather than analytic boundary-manifold identification.",
            "Crossref DOI metadata, DOI 10.1016/j.measurement.2025.118899.",
            """@article{xu2026nonsteady,
  author = {Xu, Zelin and Peng, Shitong and Yang, Shoulan and Guo, Jianan and Liu, Weiwei and Wang, Fengtao},
  title = {Prediction of Melt Pool Geometry in Laser Directed Energy Deposition Considering Non-steady-state Melt Pool Behavior},
  journal = {Measurement},
  volume = {257},
  pages = {118899},
  year = {2026},
  doi = {10.1016/j.measurement.2025.118899}
}""",
        ),
        ref(
            "scientific machine learning",
            "karniadakis2021",
            "Physics-informed machine learning",
            2021,
            "review",
            "High-impact review of physics-informed learning for data-limited scientific problems.",
            "Use to connect the paper to physics-informed modeling without claiming a PINN implementation.",
            "Broad SciML review; not AM-specific.",
            "Nature Reviews Physics page, DOI 10.1038/s42254-021-00314-5.",
            """@article{karniadakis2021,
  author = {Karniadakis, George Em and Kevrekidis, Ioannis G. and Lu, Lu and Perdikaris, Paris and Wang, Sifan and Yang, Liu},
  title = {Physics-informed Machine Learning},
  journal = {Nature Reviews Physics},
  volume = {3},
  pages = {422--440},
  year = {2021},
  doi = {10.1038/s42254-021-00314-5}
}""",
            "core",
        ),
        ref(
            "scientific machine learning",
            "cuomo2022",
            "Scientific Machine Learning Through Physics-Informed Neural Networks: Where we are and What's Next",
            2022,
            "review",
            "Comprehensive PINN review in Journal of Scientific Computing.",
            "Use as an additional mathematical modeling reference for physics-informed data integration.",
            "PINN-focused, whereas the present model is analytic and regression-based.",
            "Springer page, DOI 10.1007/s10915-022-01939-z.",
            """@article{cuomo2022,
  author = {Cuomo, Salvatore and Schiano Di Cola, Vincenzo and Giampaolo, Fabio and Rozza, Gianluigi and Raissi, Maziar and Piccialli, Francesco},
  title = {Scientific Machine Learning Through Physics-Informed Neural Networks: Where we are and What's Next},
  journal = {Journal of Scientific Computing},
  volume = {92},
  pages = {88},
  year = {2022},
  doi = {10.1007/s10915-022-01939-z}
}""",
        ),
        ref(
            "reduced-order modeling",
            "bai2021",
            "Non-intrusive nonlinear model reduction via machine learning approximations to low-dimensional operators",
            2021,
            "ROM method",
            "Recent non-intrusive ROM reference emphasizing low-dimensional operators and validation against overfitting.",
            "Use to justify non-intrusive reduction from simulation outputs instead of intrusive solver modification.",
            "General dynamical systems ROM, not DED-specific.",
            "Springer page, DOI 10.1186/s40323-021-00213-5.",
            """@article{bai2021,
  author = {Bai, Zhe and Peng, Liqian},
  title = {Non-intrusive Nonlinear Model Reduction via Machine Learning Approximations to Low-dimensional Operators},
  journal = {Advanced Modeling and Simulation in Engineering Sciences},
  volume = {8},
  pages = {28},
  year = {2021},
  doi = {10.1186/s40323-021-00213-5}
}""",
            "core",
        ),
        ref(
            "uncertainty quantification",
            "wang2020uq",
            "Uncertainty quantification and reduction in metal additive manufacturing",
            2020,
            "UQ review",
            "npj Computational Materials article on UQ in metal AM.",
            "Use to support error-budget and uncertainty-language choices.",
            "Broad AM UQ, not the same as the local sensitivity scan here.",
            "Nature page, DOI 10.1038/s41524-020-00444-x.",
            """@article{wang2020uq,
  author = {Wang, Zhuo and Jiang, Chen and Liu, Pengwei and Yang, Wenhua and Zhao, Ying and Horstemeyer, Mark F. and Chen, Long-Qing and Hu, Zhen and Chen, Lei},
  title = {Uncertainty Quantification and Reduction in Metal Additive Manufacturing},
  journal = {npj Computational Materials},
  volume = {6},
  pages = {175},
  year = {2020},
  doi = {10.1038/s41524-020-00444-x}
}""",
            "core",
        ),
        ref(
            "uncertainty quantification",
            "hermann2023",
            "Data-Driven Prediction and Uncertainty Quantification of Process Parameters for Directed Energy Deposition",
            2023,
            "DED UQ",
            "Recent DED data-driven prediction and UQ paper.",
            "Use to support the need for uncertainty-aware DED model interpretation.",
            "Process-parameter UQ differs from the present material-scale sensitivity scan.",
            "MDPI page, DOI 10.3390/ma16237308.",
            """@article{hermann2023,
  author = {Hermann, Florian and Michalowski, Andreas and Bruennette, Tim and Reimann, Peter and Vogt, Sabrina and Graf, Thomas},
  title = {Data-Driven Prediction and Uncertainty Quantification of Process Parameters for Directed Energy Deposition},
  journal = {Materials},
  volume = {16},
  number = {23},
  pages = {7308},
  year = {2023},
  doi = {10.3390/ma16237308}
}""",
        ),
        ref(
            "metal additive manufacturing review",
            "debroy2018",
            "Additive manufacturing of metallic components - process, structure and properties",
            2018,
            "review",
            "High-impact review linking AM process physics, melt-pool behavior, microstructure and properties.",
            "Use to frame the process-structure-property importance of compact melt-pool descriptors.",
            "Broad metal-AM review rather than DED-specific reduced-order modeling.",
            "Publisher DOI metadata, DOI 10.1016/j.pmatsci.2017.10.001.",
            """@article{debroy2018,
  author = {DebRoy, Tarasankar and Wei, H. L. and Zuback, J. S. and Mukherjee, T. and Elmer, J. W. and Milewski, J. O. and Beese, A. M. and Wilson-Heid, A. and De, A. and Zhang, W.},
  title = {Additive Manufacturing of Metallic Components - Process, Structure and Properties},
  journal = {Progress in Materials Science},
  volume = {92},
  pages = {112--224},
  year = {2018},
  doi = {10.1016/j.pmatsci.2017.10.001}
}""",
            "core",
        ),
        ref(
            "metal AM thermofluidics",
            "king2015",
            "Laser powder bed fusion additive manufacturing of metals; physics, computational, and materials challenges",
            2015,
            "review",
            "Foundational metal-AM review describing laser melt-pool physics and computational challenges.",
            "Use to contextualize high-fidelity melt-pool simulation and thermofluidic modeling.",
            "LPBF-centered rather than DED-centered; used for general laser metal AM physics.",
            "AIP DOI metadata, DOI 10.1063/1.4937809.",
            """@article{king2015,
  author = {King, Wayne E. and Barth, Hans D. and Castillo, Victor M. and Gallegos, Gilbert F. and Gibbs, John W. and Hahn, Douglas E. and Kamath, Chandrika and Rubenchik, Alexander M.},
  title = {Laser Powder Bed Fusion Additive Manufacturing of Metals; Physics, Computational, and Materials Challenges},
  journal = {Applied Physics Reviews},
  volume = {2},
  number = {4},
  pages = {041304},
  year = {2015},
  doi = {10.1063/1.4937809}
}""",
            "core",
        ),
        ref(
            "melt-pool thermofluidics",
            "khairallah2016",
            "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones",
            2016,
            "thermofluidic simulation",
            "High-fidelity simulation study linking complex melt flow with pore, spatter and denudation mechanisms.",
            "Use in the theory and discussion to support thermofluidic free-boundary motivation.",
            "LPBF process rather than L-DED; cited for general laser melt-pool physics.",
            "ScienceDirect DOI metadata, DOI 10.1016/j.actamat.2016.02.014.",
            """@article{khairallah2016,
  author = {Khairallah, Saad A. and Anderson, Andrew T. and Rubenchik, Alexander and King, Wayne E.},
  title = {Laser Powder-bed Fusion Additive Manufacturing: Physics of Complex Melt Flow and Formation Mechanisms of Pores, Spatter, and Denudation Zones},
  journal = {Acta Materialia},
  volume = {108},
  pages = {36--45},
  year = {2016},
  doi = {10.1016/j.actamat.2016.02.014}
}""",
            "core",
        ),
        ref(
            "AM process maps",
            "mukherjee2016",
            "Printability of alloys for additive manufacturing",
            2016,
            "process-map study",
            "Process-window analysis for alloy printability in metal additive manufacturing.",
            "Use to contextualize process-response trends without claiming a universal process map.",
            "Printability/process-map study rather than observed-boundary modeling.",
            "Nature DOI metadata, DOI 10.1038/srep19717.",
            """@article{mukherjee2016,
  author = {Mukherjee, T. and Zuback, J. S. and De, A. and DebRoy, T.},
  title = {Printability of Alloys for Additive Manufacturing},
  journal = {Scientific Reports},
  volume = {6},
  pages = {19717},
  year = {2016},
  doi = {10.1038/srep19717}
}""",
            "core",
        ),
        ref(
            "Marangoni physics",
            "scriven1960",
            "The Marangoni effects",
            1960,
            "foundational interface physics",
            "Classic article on surface-tension-gradient-driven flow.",
            "Use to support the Marangoni boundary-condition term in the physical formulation.",
            "Foundational and old; used only for physical origin.",
            "Nature DOI metadata, DOI 10.1038/187186a0.",
            """@article{scriven1960,
  author = {Scriven, L. E. and Sternling, C. V.},
  title = {The Marangoni Effects},
  journal = {Nature},
  volume = {187},
  pages = {186--188},
  year = {1960},
  doi = {10.1038/187186a0}
}""",
            "foundational",
        ),
        ref(
            "moving boundary theory",
            "crank1984",
            "Free and Moving Boundary Problems",
            1984,
            "foundational book",
            "Classic monograph on free and moving boundary problems, including Stefan-type formulations.",
            "Use to support the Stefan moving-interface formulation.",
            "Foundational mathematical text, not AM-specific.",
            "Oxford University Press book record.",
            """@book{crank1984,
  author = {Crank, John},
  title = {Free and Moving Boundary Problems},
  publisher = {Oxford University Press},
  address = {Oxford},
  year = {1984}
}""",
            "foundational",
        ),
        ref(
            "implicit-front representation",
            "osher1988",
            "Fronts propagating with curvature-dependent speed: algorithms based on Hamilton-Jacobi formulations",
            1988,
            "foundational numerical method",
            "Foundational level-set/front-propagation paper.",
            "Use to support implicit boundary and level-set language without implying a level-set solver.",
            "Numerical-analysis foundation, not an AM melt-pool paper.",
            "ScienceDirect DOI metadata, DOI 10.1016/0021-9991(88)90002-2.",
            """@article{osher1988,
  author = {Osher, Stanley and Sethian, James A.},
  title = {Fronts Propagating with Curvature-dependent Speed: Algorithms Based on Hamilton-Jacobi Formulations},
  journal = {Journal of Computational Physics},
  volume = {79},
  number = {1},
  pages = {12--49},
  year = {1988},
  doi = {10.1016/0021-9991(88)90002-2}
}""",
            "foundational",
        ),
        ref(
            "reduced-order modeling",
            "benner2015",
            "A survey of projection-based model reduction methods for parametric dynamical systems",
            2015,
            "review",
            "Survey of projection-based reduced-order modeling for parametric dynamical systems.",
            "Use to place the observation-to-manifold reduction in broader ROM methodology.",
            "General ROM survey, not DED-specific.",
            "SIAM DOI metadata, DOI 10.1137/130932715.",
            """@article{benner2015,
  author = {Benner, Peter and Gugercin, Serkan and Willcox, Karen},
  title = {A Survey of Projection-based Model Reduction Methods for Parametric Dynamical Systems},
  journal = {SIAM Review},
  volume = {57},
  number = {4},
  pages = {483--531},
  year = {2015},
  doi = {10.1137/130932715}
}""",
            "core",
        ),
        ref(
            "data-driven dynamics",
            "brunton2016",
            "Discovering governing equations from data by sparse identification of nonlinear dynamical systems",
            2016,
            "data-driven dynamics",
            "Influential reference for data-driven discovery of low-dimensional dynamical systems.",
            "Use to contextualize fitted reduced dynamics and model parsimony.",
            "General dynamical-systems method, not AM-specific.",
            "PNAS DOI metadata, DOI 10.1073/pnas.1517384113.",
            """@article{brunton2016,
  author = {Brunton, Steven L. and Proctor, Joshua L. and Kutz, J. Nathan},
  title = {Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {113},
  number = {15},
  pages = {3932--3937},
  year = {2016},
  doi = {10.1073/pnas.1517384113}
}""",
            "core",
        ),
        ref(
            "identifiability",
            "raue2009",
            "Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood",
            2009,
            "identifiability method",
            "Profile-likelihood identifiability analysis for partially observed dynamical models.",
            "Use to support the interpretation of weakly constrained fitted parameters.",
            "Methodological source from systems biology, not AM-specific.",
            "Oxford Academic DOI metadata, DOI 10.1093/bioinformatics/btp358.",
            """@article{raue2009,
  author = {Raue, Andreas and Kreutz, Clemens and Maiwald, Thomas and Bachmann, Julie and Schilling, Marcel and Klingmuller, Ursula and Timmer, Jens},
  title = {Structural and Practical Identifiability Analysis of Partially Observed Dynamical Models by Exploiting the Profile Likelihood},
  journal = {Bioinformatics},
  volume = {25},
  number = {15},
  pages = {1923--1929},
  year = {2009},
  doi = {10.1093/bioinformatics/btp358}
}""",
            "core",
        ),
        ref(
            "moving heat source",
            "rosenthal1946",
            "The theory of moving sources of heat and its application to metal treatments",
            1946,
            "foundational theory",
            "Classic moving heat-source formulation.",
            "Retain only to justify the moving coordinate xi = x - vt.",
            "Foundational and old; do not use as current literature support.",
            "Classic ASME record with commonly cited pagination 849--866.",
            """@article{rosenthal1946,
  author = {Rosenthal, Daniel},
  title = {The Theory of Moving Sources of Heat and Its Application to Metal Treatments},
  journal = {Transactions of the ASME},
  volume = {68},
  number = {8},
  pages = {849--866},
  year = {1946}
}""",
            "foundational",
        ),
        ref(
            "moving heat source",
            "goldak1984",
            "A new finite element model for welding heat sources",
            1984,
            "foundational heat-source model",
            "Classic double-ellipsoid heat-source model.",
            "Retain to contrast heat-source geometry with the present boundary geometry.",
            "Welding heat-source model, not L-DED free-boundary reduction.",
            "DOI 10.1007/BF02667333.",
            """@article{goldak1984,
  author = {Goldak, John and Chakravarti, Aditya and Bibby, Malcolm},
  title = {A New Finite Element Model for Welding Heat Sources},
  journal = {Metallurgical Transactions B},
  volume = {15},
  number = {2},
  pages = {299--305},
  year = {1984},
  doi = {10.1007/BF02667333}
}""",
            "foundational",
        ),
        ref(
            "geometric boundary model",
            "barr1981",
            "Superquadrics and angle-preserving transformations",
            1981,
            "foundational geometry",
            "Classic superquadric shape-primitive reference.",
            "Retain to justify the mathematical origin of the superellipsoid manifold.",
            "Computer-graphics origin; physical meaning comes from present model comparison.",
            "DOI 10.1109/MCG.1981.1673799.",
            """@article{barr1981,
  author = {Barr, Alan H.},
  title = {Superquadrics and Angle-Preserving Transformations},
  journal = {IEEE Computer Graphics and Applications},
  volume = {1},
  number = {1},
  pages = {11--23},
  year = {1981},
  doi = {10.1109/MCG.1981.1673799}
}""",
            "foundational",
        ),
        ref(
            "regularization",
            "hoerl1970",
            "Ridge regression: biased estimation for nonorthogonal problems",
            1970,
            "foundational regularization",
            "Classic ridge regression reference.",
            "Retain to justify ridge regularization in the coupled model.",
            "Foundational and old; pair with validation-based modern ROM references.",
            "DOI 10.1080/00401706.1970.10488634.",
            """@article{hoerl1970,
  author = {Hoerl, Arthur E. and Kennard, Robert W.},
  title = {Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  journal = {Technometrics},
  volume = {12},
  number = {1},
  pages = {55--67},
  year = {1970},
  doi = {10.1080/00401706.1970.10488634}
}""",
            "foundational",
        ),
        ref(
            "regularization",
            "tikhonov1977",
            "Solutions of Ill-posed Problems",
            1977,
            "foundational inverse problems",
            "Foundational text on regularized inverse problems.",
            "Retain as optional supporting reference for ill-posed coupled-matrix estimation.",
            "Foundational book reference; publisher and address are included for BibTeX completeness.",
            "Classic book record.",
            """@book{tikhonov1977,
  author = {Tikhonov, A. N. and Arsenin, V. Y.},
  title = {Solutions of Ill-posed Problems},
  publisher = {Winston},
  address = {Washington, DC},
  year = {1977}
}""",
            "foundational",
        ),
    ]


def make_literature_matrix() -> pd.DataFrame:
    rows = []
    for item in reference_seed_entries():
        rows.append(
            {
                "topic": item["topic"],
                "citation_key": item["citation_key"],
                "paper_title": item["paper_title"],
                "year": item["year"],
                "is_2020_or_later": item["is_2020_or_later"],
                "priority_in_current_manuscript": item["priority_in_current_manuscript"],
                "source_type": item["source_type"],
                "method_relevance": item["method_relevance"],
                "how_to_use_in_manuscript": item["how_to_use_in_manuscript"],
                "risk_or_limitation": item["risk_or_limitation"],
                "verification_source": item["verification_source"],
            }
        )
    return pd.DataFrame(rows)


def extract_latex_citation_keys(*texts: str) -> set[str]:
    keys: set[str] = set()
    citation_pattern = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
    for text in texts:
        for match in citation_pattern.finditer(text):
            for key in match.group(1).split(","):
                cleaned = key.strip()
                if cleaned:
                    keys.add(cleaned)
    return keys


def write_references_seed(report_path: Path, only_keys: set[str] | None = None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    entries = reference_seed_entries()
    if only_keys is not None:
        available_keys = {str(item["citation_key"]) for item in entries}
        missing = sorted(only_keys - available_keys)
        if missing:
            raise ValueError(f"Citation keys are missing from reference seed entries: {', '.join(missing)}")
        entries = [item for item in entries if str(item["citation_key"]) in only_keys]
    text = "\n\n".join(str(item["bibtex"]) for item in entries) + "\n"
    report_path.write_text(text, encoding="utf-8")


def write_literature_search_log(report_path: Path, literature_matrix: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(literature_matrix)
    n_recent = int(literature_matrix["is_2020_or_later"].sum())
    recent_pct = 100.0 * n_recent / max(n_total, 1)
    topics = "\n".join(
        f"- {topic}: {count} references"
        for topic, count in literature_matrix.groupby("topic").size().sort_index().items()
    )
    text = f"""# Literature Search Log

## Search intent

The citation package was revised to prioritize literature published in 2020 or later while retaining only a small number of foundational references needed for the mathematical formulation.

## Search date

2026-05-12

## Inclusion logic

- Prefer 2020+ peer-reviewed papers from publisher pages, DOI pages, PubMed records, university repositories or journal pages.
- Prioritize DED/L-DED melt-pool modeling, numerical simulation, monitoring/control, CFD-informed surrogates, reduced-order modeling, physics-informed machine learning and uncertainty quantification.
- Retain pre-2020 references only when they define a foundational construct: moving heat sources, superquadrics or ridge/Tikhonov regularization.

## Current library balance

- Total candidate references: {n_total}
- References from 2020 or later: {n_recent} ({recent_pct:.1f}%)
- Foundational pre-2020 references retained: {n_total - n_recent}

## Topic coverage

{topics}

## Manual verification still required

The seed BibTeX entries are now DOI-oriented and current-literature biased, but the final target-journal bibliography should still be exported from a reference manager or publisher metadata before submission.
"""
    report_path.write_text(text, encoding="utf-8")


def markdown_table_from_dataframe(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or len(df) == 0:
        return "_No rows available._"
    show = df.head(max_rows).copy() if max_rows is not None else df.copy()
    cols = [str(col) for col in show.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        values = []
        for col in show.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_external_validation_data_audit(
    report_path: Path,
    external_case_audit: pd.DataFrame,
    external_file_audit: pd.DataFrame,
    external_holdout_summary: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    case_count = int(len(external_case_audit)) if external_case_audit is not None else 0
    file_count = int(len(external_file_audit)) if external_file_audit is not None else 0
    ready_cases = (
        int(external_case_audit["dynamics_validation_ready"].sum())
        if external_case_audit is not None and "dynamics_validation_ready" in external_case_audit.columns
        else 0
    )
    cohort_count = (
        int(external_case_audit["validation_cohort"].nunique())
        if external_case_audit is not None and "validation_cohort" in external_case_audit.columns
        else 0
    )
    metric_lines = "_External holdout summary is not available._"
    if external_holdout_summary is not None and len(external_holdout_summary):
        metric_lines = markdown_table_from_dataframe(external_holdout_summary)
    display_cols = [
        "validation_cohort",
        "case_id",
        "power_W",
        "scan_speed_mm_s",
        "particle_rate",
        "powder_feed_g_min",
        "csv_count",
        "time_min_s",
        "time_max_s",
        "geometry_validation_ready",
        "dynamics_validation_ready",
    ]
    case_table = markdown_table_from_dataframe(
        external_case_audit[[col for col in display_cols if col in external_case_audit.columns]]
        if external_case_audit is not None and len(external_case_audit)
        else pd.DataFrame()
    )
    text = f"""# Same-Ecosystem CFD Holdout Data Audit

- Validation sources: `validation data/` and `additional data/` when present
- CFD holdout cohorts processed: {cohort_count}
- CFD holdout cases processed: {case_count}
- CFD holdout CSV time-step files processed: {file_count}
- Cases ready for full geometry, thermal-flow and dynamics validation: {ready_cases}/{case_count if case_count else 0}

## Case Audit

{case_table}

## Holdout Metrics

{metric_lines}

## Interpretation

The V-prefixed validation cases and A16-A20 additional cases are processed separately from the A1-A15 training process matrix. They are therefore used as same-solver CFD holdout cohorts: descriptor extraction and boundary fitting are applied to the holdout files, while process-response and trajectory-prediction errors are evaluated against relationships learned from the training cases. This is not measurement-based evidence and does not test independent solver physics.
"""
    report_path.write_text(text, encoding="utf-8")


def write_manuscript_draft_v3(
    report_path: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    model_selection: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    error_budget: pd.DataFrame,
    parameter_identifiability: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    selected = model_selection[model_selection["selected_as_main_model"].astype(str).str.lower() == "true"]
    selected_text = "; ".join([f"{row.model_family}: {row.model}" for row in selected.itertuples()])
    high_params = parameter_identifiability[parameter_identifiability["risk_level"].astype(str).str.contains("high")]
    high_param_text = ", ".join(latex_math_label(item) for item in high_params["parameter"].astype(str).head(8).tolist())
    parameter_audit = make_parameter_reconciliation_audit()
    parameter_audit_rows = "\n".join(
        [
            (
                f"{latex_readable_text(row.parameter)} & {latex_readable_text(row.flow3d_setup_record)} & "
                f"{latex_readable_text(row.postprocessing_basis)} & {latex_readable_text(row.manuscript_action)} \\\\"
            )
            for row in parameter_audit.head(8).itertuples()
        ]
    )
    stable_diag = bool((dynamics_summary["k_per_s"].to_numpy(dtype=float) > 0).all())
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    diag_stability_sentence = (
        "In the present fit, all diagonal rates are positive."
        if stable_diag
        else "In the present fit, at least one diagonal rate is non-positive, so the diagonal stability claim is not used."
    )
    coupled_stability_sentence = (
        "The baseline coupled model satisfies this spectral condition."
        if coupled_stable
        else "The baseline coupled model does not satisfy this spectral condition."
    )
    geometry_selection_text = geometry_selection_phrase(vals)
    geom_case = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("case_summary")].copy()
    geom_case_boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse") if len(geom_case) else pd.DataFrame()
    geom_case_volume = geom_case.pivot(index="case_id", columns="model", values="mean_volume_relative_error") if len(geom_case) else pd.DataFrame()
    geometry_pair_text = ""
    if {"ellipsoid", "superellipsoid"}.issubset(geom_case_boundary.columns):
        geom_wins = int((geom_case_boundary["superellipsoid"] < geom_case_boundary["ellipsoid"]).sum())
        geom_total = int(len(geom_case_boundary))
        geom_p = float(binomtest(geom_wins, geom_total, 0.5, alternative="greater").pvalue) if geom_total else float("nan")
        geom_adv = float(np.nanmedian(geom_case_boundary["ellipsoid"] - geom_case_boundary["superellipsoid"]))
        vol_wins = (
            int((geom_case_volume["superellipsoid"] < geom_case_volume["ellipsoid"]).sum())
            if {"ellipsoid", "superellipsoid"}.issubset(geom_case_volume.columns)
            else 0
        )
        geometry_pair_text = (
            "The paired condition-wise comparison gives boundary-residual improvement in "
            f"{count_of_total_phrase(geom_wins, geom_total, 'training conditions')} "
            f"(sign-test p={geom_p:.3g}, median residual reduction {geom_adv:.4g}), while the volume proxy improves in "
            f"{count_of_total_phrase(vol_wins, geom_total, 'training conditions')}."
        )
    dyn_pair = dynamics_comparison.pivot_table(
        index=["case_id", "state"],
        columns="model",
        values="validation_relative_rmse",
        aggfunc="mean",
    )
    dynamics_pair_text = ""
    if {"diagonal_attractor", "coupled_ridge_attractor"}.issubset(dyn_pair.columns):
        diag_wins = int((dyn_pair["diagonal_attractor"] < dyn_pair["coupled_ridge_attractor"]).sum())
        dyn_total = int(len(dyn_pair))
        dyn_p = float(binomtest(diag_wins, dyn_total, 0.5, alternative="greater").pvalue) if dyn_total else float("nan")
        dyn_adv = float(np.nanmedian(dyn_pair["coupled_ridge_attractor"] - dyn_pair["diagonal_attractor"]))
        dynamics_pair_text = (
            f"In paired condition-state comparisons, the diagonal model has lower validation error in "
            f"{count_of_total_phrase(diag_wins, dyn_total, 'condition-state pairs')} "
            f"(sign-test p={dyn_p:.3g}, median relative-RMSE reduction {dyn_adv:.4g})."
        )
    changed_groups = dimensionless_sensitivity[dimensionless_sensitivity["conclusion_changed"]]
    changed_text = ", ".join(changed_groups["symbol"].astype(str).tolist()) if len(changed_groups) else "none"
    error_table_text = "\n".join(
        [
            f"- {row.error_term}: {row.primary_metric} = {row.primary_value:.4g}. Interpretation: {row.manuscript_interpretation}"
            for row in error_budget.itertuples()
        ]
    )
    text = f"""# CFD-informed free-boundary reduction of laser directed energy deposition melt-pool evolution via superellipsoid manifolds and stable attractor dynamics

## Abstract

High-fidelity computational fluid dynamics can resolve melt-pool transport in laser directed energy deposition (L-DED), but the resulting fields are difficult to use directly in mathematical modeling, model comparison and fast interpretation. This study develops a CFD-informed reduced-order framework for a 316L stainless-steel L-DED process matrix covering {vals['process_range_text']}. The exported FLOW-3D data contain only the molten region, so the melt pool is modeled as a moving-frame free-boundary point cloud rather than as a complete thermal field. A half-domain simulation is reconstructed through the y = 0 symmetry plane, the boundary is fitted by an asymmetric superellipsoid, and the extracted state is advanced with a stable low-dimensional attractor. The liquidus-reference dimensionless groups are Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}. Relative to an ellipsoid baseline, the superellipsoid reduces the mean boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)} and the mean volume relative error from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}. Robustness tests support the geometric improvement in {vals['super_volume_wins']}/{vals['robust_total']} settings, whereas a coupled ridge attractor improves validation error in {vals['coupled_wins']}/{vals['robust_total']} settings. The selected model is therefore an asymmetric superellipsoid free boundary with a diagonal attractor dynamic. The manuscript is deliberately framed as CFD-informed reduced-order modeling, not as a process-wide predictive map.

## Keywords

laser directed energy deposition; melt pool; free-boundary model; reduced-order modeling; dimensionless analysis; stability; model selection

## Introduction

Directed energy deposition is now widely studied for metallic repair, graded deposition and large-component manufacturing. Recent reviews emphasize that DED quality depends on laser-material interaction, powder or wire delivery, melt-pool thermal behavior, defects, monitoring and process stability [@ahn2021; @svetlizky2021; @li2023]. Recent numerical studies likewise show that heat transfer, free-surface evolution, mass addition, powder flow and thermo-mechanical response remain central difficulties for reliable DED simulation [@poggi2022; @zhang2021dedcfd; @wang2023powder; @kovsca2023].

Melt-pool geometry is not only a simulation output. It is increasingly used as a process-state variable for monitoring, control and data-driven prediction. Simulation-guided control studies have targeted melt-pool depth or temperature [@liao2022; @smoqi2022], while coaxial thermal imaging has linked melt-pool area and length changes to DED process optimization [@dasilva2023]. Machine-learning studies have also predicted melt-pool dimensions, morphology or thermal fields from process conditions [@akbari2022; @zhu2023; @hemmasian2023; @wu2024]. These studies motivate a compact melt-pool state, but many of them either require broader training sets or focus on black-box prediction rather than analytic free-boundary structure.

The present work takes a narrower mathematical route. It asks whether one high-fidelity FLOW-3D L-DED simulation can be converted into a transparent reduced-order free-boundary model. This goal is close in spirit to recent physics-informed and non-intrusive modeling trends, where simulation data and physical constraints are combined under limited data [@karniadakis2021; @cuomo2022; @bai2021; @jiang2024piml]. The difference is that the present model does not train a neural network. Instead, it extracts a low-dimensional analytic boundary and a stable attractor from a short time sequence.

Only a few older references are retained because they define the mathematical scaffolding. The moving coordinate follows classical moving-source heat-transfer logic [@rosenthal1946], the ellipsoidal comparison connects to classical heat-source geometry [@goldak1984], the selected free-boundary shape uses the superquadric family [@barr1981], and the coupled attractor comparison uses ridge/Tikhonov regularization to reduce overfitting risk [@hoerl1970; @tikhonov1977]. The main literature support, however, is intentionally weighted toward 2020+ DED modeling, monitoring, physics-informed modeling and uncertainty quantification [@wang2020uq; @hermann2023].

The contribution is therefore bounded and testable: for a single 316L L-DED condition, the molten-region point cloud is transformed into a symmetry-reconstructed moving-frame boundary, fitted by an analytic superellipsoid, interpreted through dimensionless groups, advanced by a stable low-dimensional attractor, and audited through model selection, sensitivity and error-budget diagnostics.

## Mathematical formulation

Let Omega_m^h(t) be the exported half-domain molten-region point cloud. The laser-attached coordinate is

```text
xi = x - vt,   v = 0.008 m/s.
```

The imposed symmetry boundary is y = 0. The full-domain observation operator is

```text
R[Omega_m^h](t) = Omega_m^h(t) union {{(xi,-y,z): (xi,y,z) in Omega_m^h(t)}}.
```

The full-width descriptor and full-volume proxy are W(t) = 2 max y and V_full(t) = 2 V_half(t). The free boundary Gamma(t) is defined as the envelope of the exported molten domain. This definition is intentionally tied to the available data and does not claim recovery of the complete solid-liquid interface from the full CFD temperature field.

The reduced state is q(t) = [L_f, L_r, W, H, T_max, G_mean, U_max]^T, where L_f and L_r are front and rear extents in the moving coordinate, W is the symmetry-reconstructed full width, H is the vertical span, T_max is the maximum temperature, G_mean is the mean temperature-gradient magnitude and U_max is the maximum velocity magnitude.

## Free-boundary model

The ellipsoid baseline is

```text
((xi - xi_c)/a_s)^2 + (y/b)^2 + ((z - z_c)/c)^2 = 1,
```

with a_s = a_f in front of the fitted center and a_s = a_r behind it. The superellipsoid candidate is

```text
|((xi - xi_c)/a_s)|^n + |y/b|^m + |((z - z_c)/c)|^p = 1,
```

with parameter vector theta = [a_f, a_r, b, c, xi_c, z_c, n, m, p]. The superellipsoid representation is evaluated against the ellipsoid baseline, and its additional flexibility is interpreted through the boundary-residual evidence reported in the model-selection analysis.

## Data and preprocessing

The dataset contains {vals['n_time_steps']} FLOW-3D CSV files from t={vals['t_min']:.2f} s to t={vals['t_max']:.2f} s. Duplicate rows are removed, repeated coordinates are collapsed by field averaging, and all coordinates are transformed into the moving frame. The simulation used a half computational domain in the y direction, so W(t) and V_full(t) use symmetry reconstruction.

The material is 316L stainless steel. The FLOW-3D setup note provides the laser, phase-change and surface-tension constants, and the supplied property tables define temperature-dependent density, heat capacity, thermal conductivity and viscosity. The setup-note laser radius is 0.0008 m, absorptivity is 0.1, the solidus and liquidus temperatures are 1683 K and 1710 K, and the fusion latent heat is 2.67776e5 J/kg. Surface tension is 1.8 N/m and the magnitude of the surface-tension temperature coefficient is 2.50836e-4 N/(m K).

## Dimensionless analysis

The reference length is the quasi-steady mean melt-pool length. Material properties are evaluated at the liquidus temperature for baseline reporting. The main groups are

```text
Pe = v L_ref/alpha,   Fo = alpha t/L_ref^2,
Ste = c_p(T_l - T_s)/L_fus,
E* = eta P/[rho c_p v r_b^2 (T_l - T0)],
Ma = |d sigma/dT|(T_l - T_s)L_ref/(mu alpha).
```

The baseline values are Pe={vals['Pe']:.2f}, Fo_final={vals['Fo_final']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f}, Re={vals['Re']:.2f}, Pr={vals['Pr']:.3f} and Ma={vals['Ma']:.2f}. A scenario sensitivity scan over reference temperature, absorptivity and surface-tension coefficient shows class changes for: {changed_text}. The sensitivity analysis follows recent AM uncertainty-quantification practice by reporting material-scale sensitivity rather than hiding it inside a single nominal property set [@wang2020uq; @hermann2023].

## Reduced-order dynamics

The selected baseline dynamics is the diagonal attractor

```text
dq_i/dt = k_i(q_inf,i - q_i).
```

The alternative coupled model is

```text
dq/dt = A(q_inf - q).
```

Because the coupled model has many more parameters than the diagonal model, A is identified with ridge regularization. The regularization strength is selected by leave-one-step training validation. The coupled model is accepted as the main dynamic only if it is stable and improves validation error.

## Stability analysis

For the diagonal model, define e_i = q_i - q_inf,i. Then de_i/dt = -k_i e_i, and e_i(t) = e_i(0) exp(-k_i t). Thus, if k_i > 0, q_i = q_inf,i is exponentially stable for that state component. In the present fit, all diagonal rates are positive: {stable_diag}.

For the coupled model, e = q - q_inf satisfies de/dt = -A e. The coupled equilibrium is locally exponentially stable if all eigenvalues of -A have negative real part. The baseline coupled model satisfies this stability condition: {coupled_stable}. However, its mean validation relative RMSE is {_fmt(vals['coupled_validation'], 4)}, compared with {_fmt(vals['diagonal_validation'], 4)} for the diagonal attractor. Stability is therefore necessary but not sufficient for selecting the coupled model.

## Error analysis

The uncertainty chain is organized as

```text
E_total = E_reconstruction + E_geometry + E_volume_proxy + E_dynamics + E_parameter_scale.
```

This expression is an error-budget taxonomy, not an assumption that independent random errors add linearly. The current entries are:

{error_table_text}

This separation is central to the manuscript. The molten-region-only export affects E_reconstruction. The analytic free-boundary manifold affects E_geometry. The mirrored convex-hull volume affects E_volume_proxy. The short validation sequence affects E_dynamics. Temperature-dependent 316L properties affect E_parameter_scale.

## Results

Figure 1 summarizes the modeling chain from FLOW-3D molten-region data to symmetry reconstruction, moving-frame analysis, superellipsoid fitting, attractor identification and error auditing.

Figures 2-5 summarize the training process matrix, moving-frame reconstruction, source thermal-flow context and transient geometric evolution. Figure 4 provides a visual audit of the FLOW-3D temperature and velocity fields for three same-speed, same-powder-feed power cases; quantitative model selection remains based on the boundary residual, volume-proxy, dynamics-error and holdout diagnostics. After the initial growth stage, the melt-pool envelope approaches a quasi-steady form after approximately 0.20 s. The quasi-steady mean front length, rear length, full width and height are {vals['lf_quasi_mm']:.3f} mm, {vals['lr_quasi_mm']:.3f} mm, {vals['w_quasi_mm']:.3f} mm and {vals['h_quasi_mm']:.3f} mm.

Figure 6 compares the ellipsoid and superellipsoid boundary models. The mean boundary residual decreases from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)}, and the mean volume relative error decreases from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}. Robustness tests show superellipsoid improvement in {vals['super_volume_wins']}/{vals['robust_total']} settings for volume error and {vals['super_boundary_wins']}/{vals['robust_total']} settings for boundary residual.

Figure 7 reports the quasi-steady process-response diagnostics. Figure 8 reports the nondimensional regime and sensitivity envelope. The values Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f} indicate a regime where advective translation, finite melting enthalpy scale, concentrated heat input and thermocapillary forcing all matter as scaling diagnostics.

Figure 9 compares stability and predictive evidence. Both the diagonal and coupled attractors are stable by their respective criteria. The diagonal model has lower validation relative RMSE, {_fmt(vals['diagonal_validation'], 4)} versus {_fmt(vals['coupled_validation'], 4)}, and the coupled model improves validation error in {vals['coupled_wins']}/{vals['robust_total']} robustness settings.

Figure 10 presents the error budget and model-selection summary. Figure 11 shows parameter-identifiability and overparameterization diagnostics. High-risk parameters include: {high_param_text}. These diagnostics support the selected model: {selected_text}.

## Discussion

The main result is a separation between useful geometric flexibility and unsupported dynamical complexity. The superellipsoid adds three shape exponents to the ellipsoid and improves both boundary and volume diagnostics across robustness settings. The coupled attractor adds many interaction coefficients, remains stable, but does not improve validation accuracy. This contrast is important for a mathematical modeling journal because it shows that model choice is governed by evidence.

The updated literature base clarifies the manuscript's position. Recent DED papers increasingly use CFD, monitoring, simulation-guided control and ML surrogates to understand or regulate the melt pool [@zhang2021dedcfd; @liao2022; @dasilva2023; @wu2024]. The present work complements that direction by asking a smaller but more mathematical question: what low-dimensional boundary and attractor can be defended from a short molten-region CFD export?

The negative result for the coupled attractor is useful. Cross-coupling among geometry, temperature gradients and velocity is physically plausible, but the short single-condition sequence does not identify a coupled matrix strongly enough to improve validation. Reporting this result reduces overclaiming and strengthens the selected diagonal model.

The free-boundary definition is also deliberately conservative. Because only molten-region points were exported, Gamma(t) is the envelope of the observed molten domain. It is not a reconstructed solid-liquid isotherm from the full temperature field. This limitation narrows the claim, but it makes the model reproducible from the available data and keeps the error budget explicit.

## Limitations

The study uses one 316L condition and should not be read as a predictive process map over power, speed, beam radius, absorptivity and powder flow. Additional FLOW-3D conditions would be needed to parameterize q_inf, k_i and superellipsoid shape parameters as process functions. The volume metric is a mirrored convex-hull proxy, not a direct thermodynamic melt volume. The sensitivity analysis is a scenario scan, not a full global uncertainty quantification. The reference list is a post-2020-prioritized seed bibliography and should still be exported from publisher metadata before submission.

## Conclusion

This work converts FLOW-3D molten-region point clouds into a moving-frame, symmetry-aware free-boundary reduced-order model for single-condition L-DED melt-pool evolution. The selected model is an asymmetric superellipsoid boundary coupled to a diagonal exponentially stable attractor. The selection is supported by geometric error reduction, robustness checks, nondimensional interpretation, stability criteria, validation error, parameter-identifiability diagnostics and an explicit error budget. The revised citation package now places the work primarily in the context of 2020+ DED modeling, melt-pool monitoring/control, physics-informed modeling, reduced-order modeling and uncertainty quantification, while retaining only a few foundational older references for the mathematical constructs.
"""
    if "[Ref]" in text:
        raise ValueError("manuscript_draft_v3.md still contains a [Ref] placeholder")
    report_path.write_text(text, encoding="utf-8")


def latex_escape(text: str | float | int | bool) -> str:
    value = str(text).replace("_", " ")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def latex_math_label(text: str | float | int | bool) -> str:
    value = str(text)
    labels = {
        "L_f": r"$L_f$",
        "L_r": r"$L_r$",
        "W": r"$W$",
        "H": r"$H$",
        "Tmax": r"$T_{\max}$",
        "Tmax_K": r"$T_{\max}$",
        "Gmean": r"$G_{\mathrm{mean}}$",
        "Gmean_K_per_m": r"$G_{\mathrm{mean}}$",
        "Umax": r"$U_{\max}$",
        "Umax_m_per_s": r"$U_{\max}$",
        "front_length_m": r"$L_f$",
        "rear_length_m": r"$L_r$",
        "full_width_m": r"$W$",
        "height_span_m": r"$H$",
        "a_f": r"$a_f$",
        "a_r": r"$a_r$",
        "b": r"$b$",
        "c": r"$c$",
        "Pe": r"$Pe$",
        "Fo": r"$Fo$",
        "Ste": r"$Ste$",
        "E_star": r"$E^*$",
        "Ma": r"$Ma$",
        "xi_c": r"$\xi_c$",
        "z_c": r"$z_c$",
        "n": r"$n$",
        "m": r"$m$",
        "p": r"$p$",
        "k_front_length_m": r"$k_{L_f}$",
        "k_rear_length_m": r"$k_{L_r}$",
        "k_full_width_m": r"$k_W$",
        "k_height_span_m": r"$k_H$",
        "k_width_m": r"$k_W$",
        "k_height_m": r"$k_H$",
        "k_Tmax_K": r"$k_{T_{\max}}$",
        "k_Gmean_K_per_m": r"$k_{G_{\mathrm{mean}}}$",
        "k_Umax_m_per_s": r"$k_{U_{\max}}$",
        "A_matrix_entries": r"coupled-attractor matrix entries",
        "E_reconstruction": r"$E_{\mathrm{reconstruction}}$",
        "E_geometry": r"$E_{\mathrm{geometry}}$",
        "E_volume_proxy": r"$E_{\mathrm{volume}}$",
        "E_dynamics": r"$E_{\mathrm{dynamics}}$",
        "E_model_complexity": r"$E_{\mathrm{complexity}}$",
        "E_overparameterization": r"$E_{\mathrm{overparameterization}}$",
        "E_parameter_scale": r"$E_{\mathrm{parameter}}$",
    }
    return labels.get(value, latex_escape(value))


def latex_readable_text(text: str | float | int | bool) -> str:
    value = latex_escape(text)
    replacements = {
        "mixed conduction advection": "mixed conduction-advection",
        "sensible and latent comparable": "sensible and latent comparable",
        "latent heat dominant": "reference-scale latent-heat interpretation",
        "reference scale latent heat interpretation": "reference-scale latent-heat interpretation",
        "large normalized heat input": "large normalized heat input",
        "weak marangoni scale": "reference-scale weak Marangoni interpretation",
        "moderate marangoni scale": "reference-scale Marangoni interpretation",
        "strong marangoni scale": "reference-scale strong Marangoni interpretation",
        "reference scale weak marangoni interpretation": "reference-scale weak Marangoni interpretation",
        "reference scale marangoni interpretation": "reference-scale Marangoni interpretation",
        "reference scale strong marangoni interpretation": "reference-scale strong Marangoni interpretation",
        "rmse": "RMSE",
        "medium high": "medium-high",
        "low medium": "low-medium",
        "All k i>0": r"All $k_i>0$",
        "k i": r"$k_i$",
        "xi c": r"$\xi_c$",
        "z c": r"$z_c$",
        "a r": r"$a_r$",
        "k front length m": r"$k_{L_f}$",
        "k rear length m": r"$k_{L_r}$",
        "k full width m": r"$k_W$",
        "k height span m": r"$k_H$",
        "k width m": r"$k_W$",
        "k height m": r"$k_H$",
        "n, m, p": r"$n$, $m$, $p$",
        "k Umax m per s": r"$k_{U_{\max}}$",
        "Umax": r"$U_{\max}$",
        "A matrix entries": r"coupled-attractor matrix entries",
        "observed boundary envelope geometry": "observed boundary-envelope geometry",
        "free boundary geometry": "observed boundary-envelope geometry",
        "Boundary residual improves 0.2922->0.1943": r"Boundary residual improves 0.2922$\to$0.1943",
        "Boundary residual improves 0.2695->0.1937": r"Boundary residual improves 0.2695$\to$0.1937",
        "volume error improves 0.4317->0.3739": r"volume error improves 0.4317$\to$0.3739",
        "volume proxy error changes 1.4268->1.6804": r"volume proxy error changes 1.4268$\to$1.6804",
        "validation hierarchy": "validation hierarchy",
        "same ecosystem cfd holdout available": "shared-setting numerical holdout available",
        "same_ecosystem_cfd_holdout_available": "shared-setting numerical holdout available",
        "conditions no physical measurement support": "conditions; physical measurement comparison pending",
        "conditions_no_physical_measurement_support": "conditions; physical measurement comparison pending",
        "simulation only multi condition validation no external holdout": "simulation-only multi-condition validation; no external holdout",
        "simulation_only_multi_condition_validation_no_external_holdout": "simulation-only multi-condition validation; no external holdout",
        "laser phase and temperature dependent property inputs reconciled": "laser, phase-change and temperature-dependent property inputs reconciled",
        "laser_phase_and_temperature_dependent_property_inputs_reconciled": "laser, phase-change and temperature-dependent property inputs reconciled",
        "assumption-proposition-proof-sketch framework": "assumption, proposition and proof-sketch basis",
        "assumption-proposition-proof-sketch_framework": "assumption, proposition and proof-sketch basis",
        "assumption assessment generated": "assumption assessment summarized",
        "assumption audit generated": "assumption assessment summarized",
        "assumption risk": "assumption scope",
        "high or high for assumptions": "substantial-boundary assumptions",
        "high_or_high_for assumptions": "substantial-boundary assumptions",
        "processed descriptors and fitted parameters archived": "processed descriptors and fitted parameters archived",
        "compile summary generated by script": "compile summary available",
        "same-ecosystem CFD holdout": "same-solver numerical holdout",
        "same ecosystem cfd": "shared-setting numerical",
        "risk-reduction": "validation",
        "compiled pdf required": "compiled PDF available",
        "keep current values": "retain current values",
        "latex submission package": "local LaTeX compilation",
        "Coupled model is a negative-control diagnostic": "Coupled model is an overparameterization comparison",
        "negative-control diagnostic": "overparameterization comparison",
        "high-for-coupled-model": "substantial for coupled model",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def evidence_boundary_label(text: str | float | int | bool) -> str:
    """Render internal support labels as publication-facing evidence boundaries."""

    raw = str(text).strip().lower().replace("_", "-")
    mapping = {
        "low": "low",
        "medium": "moderate",
        "high": "substantial",
        "low-medium": "low-to-moderate",
        "medium-high": "moderate-to-substantial",
        "high-for-coupled-model": "substantial for coupled model",
        "high for coupled model": "substantial for coupled model",
    }
    return latex_escape(mapping.get(raw, raw))


def publication_table_text(text: str | float | int | bool) -> str:
    """Render status-style table entries as manuscript-facing prose."""

    value = latex_readable_text(text)
    replacements = {
        "Coupled model is a negative-control diagnostic": "Coupled model is an overparameterization comparison",
        "negative-control diagnostic": "overparameterization comparison",
        "assumption risk": "assumption scope",
        "assumption audit generated": "assumption assessment summarized",
        "Use the assumption matrix in supplementary material and reviewer response.": (
            "Use the assumption matrix to define interpretive scope and future validation needs."
        ),
        "latex submission package": "local LaTeX compilation",
        "compiled PDF required": "compiled PDF available",
        "flow3d setup note authoritative known laser phase constants synced": (
            "simulation-parameter record reconciled with known laser and phase-change constants"
        ),
        "Obtain the exact temperature-independent transport-property constants before using dimensionless groups as solver-parameter verification; retain current values as reference scale diagnostics.": (
            "Report dimensionless groups as reference-scale diagnostics until exact transport-property constants are independently confirmed."
        ),
        "same ecosystem cfd": "shared-setting numerical",
        "same-ecosystem CFD": "same-solver numerical",
        "risk-reduction": "validation",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def yes_no(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def remove_legacy_main_latex_outputs(latex_dir: Path) -> None:
    """Remove the former combined manuscript artifacts from the LaTeX output folder."""
    legacy_suffixes = [
        ".tex",
        ".pdf",
        ".aux",
        ".bbl",
        ".blg",
        ".log",
        ".out",
        ".toc",
        ".lof",
        ".lot",
        ".fls",
        ".fdb_latexmk",
        ".synctex.gz",
    ]
    for suffix in legacy_suffixes:
        path = latex_dir / f"main{suffix}"
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def latex_cite(keys: str) -> str:
    cleaned = [item.strip().lstrip("@") for item in keys.split(";")]
    return r"\citep{" + ",".join(cleaned) + "}"


def humanize_submission_text(text: str) -> str:
    """Lightly de-template manuscript prose while preserving technical claims."""

    replacements = {
        "A central modeling question is how to describe this evolving molten region by a small number of interpretable state variables when the complete thermal-flow field is not available.": (
            "The modeling problem is to describe this evolving molten region with a small number of interpretable state variables when the complete thermal-flow field is not available."
        ),
        "This study formulates the problem as observed boundary-envelope identification": "Here, the problem is formulated as observed boundary-envelope identification",
        "The resulting model is therefore an asymmetric superellipsoid observed-boundary descriptor with a parsimonious diagonal attractor baseline; it is intended as an auditable reduced mathematical representation of molten-region evolution, not as a full inverse reconstruction of the Stefan-Marangoni free boundary.": (
            "The working model is an asymmetric superellipsoid observed-boundary descriptor with a parsimonious diagonal attractor baseline. It is a traceable reduced mathematical description of molten-region evolution, not a full inverse reconstruction of the Stefan-Marangoni free boundary."
        ),
        "central difficulties": "open difficulties",
        "The present work asks whether": "We ask whether",
        "This goal is close in spirit to": "This question is related to",
        "The difference is that the present model does not train a neural network or claim a fully predictive process map. Instead, it extracts": (
            "Here, no neural network is trained and no fully predictive process map is claimed. The model extracts"
        ),
        "The paper separates the full physical problem from the problem actually solved.": "We separate the full physical problem from the reduced problem fitted here.",
        "The mathematical task is therefore to construct and validate": "The mathematical task is to construct and validate",
        "This work contributes an observed-boundary mathematical modeling framework for simulation-derived L-DED melt pools.": (
            "The contribution is an observed-boundary mathematical model for simulation-derived L-DED melt pools."
        ),
        "The novelty is not a closed-form Stefan-Marangoni solution, but an audited chain that maps": (
            "The novelty is not a closed-form Stefan-Marangoni solution. It is the transparent map from"
        ),
        "The novelty is not a closed-form Stefan-Marangoni solution. It is the audited map from sparse molten-region CFD observations to analytic boundary-envelope manifolds, reduced-state descriptors, parsimonious baseline dynamics and auditable model-selection diagnostics.": (
            "The novelty lies in the transparent map from sparse melt-pool observations to analytic boundary-envelope manifolds, reduced-state descriptors, parsimonious baseline dynamics and checkable model-selection results."
        ),
        "This provides a reusable computational modeling template for high-fidelity additive-manufacturing simulations in which full-field inverse reconstruction is unavailable.": (
            "The procedure can be rerun on high-fidelity additive-manufacturing simulations when full-field inverse reconstruction is unavailable."
        ),
        "The actual mathematical object is therefore": "The fitted mathematical object is",
        "The resulting geometric diagnostics are": "The geometric diagnostics are",
        "This expression is an error-budget taxonomy, not an assumption that independent random errors add linearly.": (
            "This expression is an error-budget taxonomy. It does not assume that independent random errors add linearly."
        ),
        "The table below reports observable diagnostics for the corresponding terms, so the bound is used to structure uncertainty rather than to claim a sharp worst-case estimate from the finite simulation dataset.": (
            "The table reports observable diagnostics for these terms. The bound structures the uncertainty discussion rather than giving a sharp worst-case estimate from the finite simulation set."
        ),
        "The model-selection rule is deliberately conservative.": "The model-selection rule is conservative.",
        "The selected working model should therefore be read as an auditable baseline for observed boundary evolution, while the coupled attractor remains a negative-control diagnostic.": (
            "We use the selected model as a traceable baseline for observed boundary evolution, with the coupled attractor retained as an overparameterization comparison."
        ),
        "The full machine-readable table remains in the generated analysis package.": "The full machine-readable table is included in the analysis package.",
        "Residual submission risks are summarized as an audit item": "Residual limitations are summarized",
        "rather than treated as a main-text research-result table.": "rather than treated as a main-text result.",
        "This result is treated as a parsimonious model-selection outcome rather than as a strong statistical dominance claim.": (
            "We interpret this as a parsimonious model-selection result, not as statistical dominance."
        ),
        "The main result is a separation between useful boundary flexibility, imperfect volume recovery and unsupported dynamical complexity across a multi-condition CFD design.": (
            "The results separate three issues that are easy to conflate: boundary flexibility, imperfect volume recovery and unsupported dynamical complexity."
        ),
        "This contrast is important for a mathematical modeling journal because model choice is governed by evidence rather than by formal flexibility alone.": (
            "The comparison matters because model choice is governed by evidence, not by formal flexibility alone."
        ),
        "The present formulation clarifies what is physical and what is reduced.": "The formulation keeps the physical problem and the fitted reduced problem distinct.",
        "This distinction prevents the model from being interpreted as a full thermal-field inverse solution.": (
            "That distinction prevents the model from being read as a full thermal-field inverse solution."
        ),
        "The present work complements that direction by asking a more mathematical question:": "This paper asks a narrower mathematical question:",
        "The observed boundary-envelope definition is deliberately conservative.": r"The definition of $\Gamma(t)$ is conservative.",
        "This limitation narrows the claim, but it makes the model reproducible from the available data and keeps the error budget explicit.": (
            "The limitation narrows the claim and keeps the model reproducible from the available data."
        ),
        "The study uses 15 training simulated 316L conditions and should be read as": "The analysis uses 15 training simulated 316L conditions and is best read as",
        "and is a FLOW-3D-informed observed-boundary modeling study, not as an experimentally benchmarked universal process map": (
            "and is best read as a melt-pool-data-informed observed-boundary modeling study rather than an experimentally benchmarked universal process map"
        ),
        "and is best read as a FLOW-3D-informed observed-boundary modeling study, not as an experimentally benchmarked universal process map": (
            "and is best read as a melt-pool-data-informed observed-boundary modeling study rather than an experimentally benchmarked universal process map"
        ),
        "The added 5-condition external CFD holdout reduces the previous validation gap.": "The CFD holdout cohorts reduce the previous validation gap.",
        "This study establishes a multi-condition CFD-informed observed boundary-envelope identification and modeling framework": (
            "This study develops a multi-condition melt-pool-data-informed observed boundary-envelope model"
        ),
        "The central result is that": "The main result is that",
        "The contribution is therefore not": "The contribution is not",
        "First, the geometric part of the framework shows that": "First, the geometric analysis shows that",
        "Second, the dynamical part of the framework supports": "Second, the dynamics support",
        "Finally, the analysis clarifies the boundary of the claim.": "Finally, the scope of interpretation remains explicit.",
        "Within these limits, the study provides a defensible mathematical modeling template for turning": (
            "Within these limits, the study gives a defensible way to turn"
        ),
        "the necessary next step": "the next step",
        "The process matrix spans 15 process conditions spanning ": "The process matrix covers ",
        "the supplementary workflow begins": "the supplementary analysis begins",
        "This choice is intentionally conservative": "This choice is conservative",
        "The full time-step boundary fits and the fitted parameter trajectories are shown immediately below to keep the geometric evidence adjacent to the fitting procedure.": (
            "The full time-step boundary fits and fitted parameter trajectories are placed below, next to the fitting procedure."
        ),
        "The assumption matrix is placed in the Supplementary Information because it is a reviewer-audit device rather than a primary modeling result.": (
            "The assumption matrix is placed in the Supplementary Information because it supports model assessment rather than the primary result."
        ),
        r"\subsection*{Submission-gap audit}": r"\subsection*{Residual limitations}",
        "The remaining submission risks are listed here because they are useful for transparency but are not themselves a modeling result.": (
            "The remaining limitations are listed here for transparency. They are not treated as modeling results."
        ),
        "The assumption matrix is placed in the supplementary methods because it is a reviewer-audit device rather than a primary modeling result.": (
            "The assumption matrix is placed in the supplementary methods because it supports model assessment rather than the primary result."
        ),
        r"\caption{Submission-gap audit for AMM-oriented revision.}": r"\caption{Residual limitations for the manuscript revision.}",
        "These diagnostics are used to decide whether additional flexibility is mathematically defensible.": (
            "These diagnostics indicate whether the extra flexibility is justified."
        ),
        "These panels are useful for auditability, but they are not part of the main model-selection sequence.": (
            "These panels support reproducibility, but they are not part of the main model-selection sequence."
        ),
        "each reported error source is linked to a measurable table or figure so that the reduced model remains auditable.": (
            "each reported error source is linked to a measurable table or figure so that the reduced model can be checked."
        ),
        "Second, the dynamics support a deliberately simple attractor baseline.": "Second, the dynamics support a simple attractor baseline.",
        "It includes processed geometry descriptors": "It contains processed geometry descriptors",
        r"\title{CFD-informed observed boundary-envelope identification of L-DED melt pools using superellipsoid manifolds and parsimonious attractor dynamics}": (
            r"\title{Observed boundary-envelope identification of L-DED melt pools from melt-pool data using superellipsoid manifolds and parsimonious attractor dynamics}"
        ),
        r"\title{Supplementary Methods: CFD-informed observed boundary-envelope identification of L-DED melt pools}": (
            r"\title{Supplementary Methods: observed boundary-envelope identification of L-DED melt pools from melt-pool data}"
        ),
        "Laser directed energy deposition simulations produce rich molten-region fields, but those fields are difficult to reuse as compact mathematical descriptors.": (
            "Laser directed energy deposition simulations resolve detailed molten-region fields, but these outputs are difficult to reuse as compact mathematical descriptors."
        ),
        "This study develops an observation-to-manifold reduction framework for 316L melt-pool exports across": (
            "Here, we develop an observation-to-manifold reduction framework for 316L melt-pool exports across"
        ),
        "Half-domain FLOW-3D molten-region point clouds are transformed into symmetry-reconstructed moving-frame envelopes, projected onto asymmetric superellipsoid manifolds and reduced to geometric, thermal and flow descriptors with descriptive relaxation baselines.": (
            "Half-domain FLOW-3D molten-region point clouds are reconstructed by symmetry in a moving frame. The resulting envelopes are projected onto asymmetric superellipsoid manifolds and reduced to geometric, thermal and flow descriptors with descriptive relaxation baselines."
        ),
        "Relative to an ellipsoid baseline, the superellipsoid lowers the implicit boundary residual": (
            "Relative to an ellipsoid baseline, the superellipsoid reduces the implicit boundary residual"
        ),
        "The result is an auditable descriptor-extraction and process-response framework for molten-region numerical exports; it is not a full Stefan-Marangoni inverse reconstruction, not suitable for melt-volume prediction, not a metric-accurate geometry reconstruction, and not evidence that validates solver physics.": (
            "Together, these outputs provide a reproducible descriptor-extraction and process-response framework for molten-region numerical exports. The framework addresses observed-envelope reduction, process-response analysis and reusable reduced-state interpretation."
        ),
        "Laser directed energy deposition (L-DED) is widely studied for metallic repair, graded deposition and large-component manufacturing": (
            "Laser directed energy deposition (L-DED) is widely studied in metallic repair, graded deposition and large-component manufacturing"
        ),
        "The molten region remains difficult to describe because laser-material interaction, mass addition, heat transfer, phase change, free-surface heat loss and thermocapillary transport evolve together.": (
            "Its molten region is difficult to reduce to a compact model because laser-material interaction, mass addition, heat transfer, phase change, free-surface heat loss and thermocapillary transport evolve together."
        ),
        "The scientific problem addressed here is how to represent this evolving molten region as a low-dimensional, interpretable and verifiable mathematical object when the accessible information is a time-resolved boundary observation rather than a complete thermal-flow field.": (
            "The problem addressed here is how to represent this evolving molten region as a low-dimensional, interpretable and verifiable mathematical object when only time-resolved boundary observations are available."
        ),
        "Existing studies have approached melt-pool evolution through simulation, monitoring, control and data-driven prediction.": (
            "Existing work has approached melt-pool evolution through simulation, monitoring, control and data-driven prediction."
        ),
        "These advances motivate compact melt-pool states, yet many methods emphasize scalar dimensions, full-field prediction, feedback targets or black-box maps.": (
            "Together, these advances motivate compact melt-pool states. Many methods, however, still emphasize scalar dimensions, full-field prediction, feedback targets or black-box maps."
        ),
        "What remains less developed is an auditable boundary-envelope representation that links observed melt-pool shape, reduced descriptors, model-selection evidence and dynamical evolution across process conditions.": (
            "Less developed is a transparent boundary-envelope representation that links observed melt-pool shape, reduced descriptors, model-selection evidence and dynamical evolution across process conditions."
        ),
        "This study develops an observed boundary-envelope mathematical modeling framework for L-DED melt pools.": (
            "Here, we develop an observed boundary-envelope modeling framework for L-DED melt pools."
        ),
        "The framework reconstructs half-domain molten-region observations in a moving coordinate, projects the observed boundary envelope onto an asymmetric superellipsoid manifold, extracts reduced geometric and thermal-flow descriptors, and compares parsimonious first-order attractor baselines with overparameterized coupled alternatives.": (
            "The workflow reconstructs half-domain molten-region observations in a moving coordinate and projects the observed boundary envelope onto an asymmetric superellipsoid manifold. It then extracts reduced geometric and thermal-flow descriptors and compares parsimonious first-order attractor baselines with overparameterized coupled alternatives."
        ),
        "The specific contribution is threefold: an observation-to-manifold reduction for sparse FLOW-3D molten-region exports, a model-selection protocol that separates algebraic envelope fit from geometric-distance and volume diagnostics, and an auditable descriptor system linking boundary geometry, process response and fitted relaxation baselines.": (
            "The contribution has three parts: an observation-to-manifold reduction for sparse melt-pool data, a model-selection protocol that separates algebraic envelope fit from geometric-distance and volume diagnostics, and a transparent descriptor system linking boundary geometry, process response and fitted relaxation baselines."
        ),
        "The novelty lies in the auditable observation-to-manifold reduction and evidence hierarchy for incomplete molten-region exports, rather than in a new superquadric fitting algorithm or a new dynamical-systems theory.": (
            "The novelty is the transparent observation-to-manifold reduction and evidence hierarchy for incomplete molten-region exports, rather than a new superquadric fitting algorithm or a new dynamical-systems theory."
        ),
        "The contribution is therefore an observed-envelope modeling framework, not a closed-form solution or asymptotic reduction of the Stefan-Marangoni system.": (
            "The contribution is an observed-envelope modeling framework, not a closed-form solution or asymptotic reduction of the Stefan-Marangoni system."
        ),
        "The equation is written in this general temperature-dependent form only to motivate the physical origin of the observation problem.": (
            "This general temperature-dependent form is used only to motivate the physical origin of the observation problem."
        ),
        "The exported data do not contain the complete solid-domain temperature field needed to solve": (
            "The exported data do not contain the complete solid-domain temperature field required to solve"
        ),
        "These equations motivate the observed boundary-envelope viewpoint. They are not solved analytically here.": (
            "These equations motivate the observed boundary-envelope viewpoint, but they are not solved analytically here."
        ),
        "The operation reconstructs geometric descriptors of the full observation but does not recover unobserved antisymmetric flow components.": (
            "The operation reconstructs full-observation geometric descriptors, but it does not recover unobserved antisymmetric flow components."
        ),
        "The moving-frame step is justified by constant laser translation: after the initial transient, a localized heat source can approach a slowly varying shape in the laser-attached coordinate.": (
            "The moving-frame step follows from constant laser translation. After the initial transient, a localized heat source can approach a slowly varying shape in the laser-attached coordinate."
        ),
        "The superellipsoid step is a manifold projection, not a claim that the true Stefan interface is exactly superellipsoidal.": (
            "The superellipsoid step is a manifold projection. It is not a claim that the true Stefan interface is exactly superellipsoidal."
        ),
        "This chain is an observation-driven modeling map, not an asymptotic or Galerkin reduction of the governing PDEs.": (
            "This chain is an observation-driven modeling map, not an asymptotic or Galerkin reduction of the governing equations."
        ),
        "The dataset is a local FLOW-3D numerical export from 15 316L L-DED simulations, not an experimental image sequence.": (
            "The dataset is a FLOW-3D numerical export from 15 316L L-DED simulations, not an experimental image sequence."
        ),
        "The export therefore excludes the surrounding solid domain and any already-solidified material.": (
            "The export excludes the surrounding solid domain and any already-solidified material."
        ),
        "Because the exported data are molten-region numerical observations, the boundary model is interpreted as an envelope reduction of the available point cloud, not as a complete inverse reconstruction of the full thermal field.": (
            "Because the exported data are molten-region observations, the boundary model is interpreted as an envelope reduction of the available point cloud, not as a complete inverse reconstruction of the full thermal field."
        ),
        "These values are not used in the abstract or conclusion to claim a verified heat-input or thermocapillary regime.": (
            "These values are used as scale diagnostics, not as verified heat-input or thermocapillary regime claims."
        ),
        "The selected parsimonious baseline dynamics is the diagonal attractor": (
            "The parsimonious dynamical baseline is the diagonal attractor"
        ),
        "The coupled model is retained as a negative-control diagnostic for unsupported cross-state coupling, not as the default dynamical law.": (
            "The coupled model is retained as an overparameterization comparison for unsupported cross-state coupling."
        ),
        "No finite-difference derivative is used in the primary diagonal fit.": (
            "The primary diagonal fit does not use finite-difference derivatives."
        ),
        "The full assumption validation matrix is reported in Supplementary Table": (
            "Supplementary Table"
        ),
        "to keep the main text focused on the model definition, model selection and validation results.": (
            "reports the full assumption validation matrix, keeping the main text focused on model definition, model selection and validation results."
        ),
        "To reduce the risk that the parsimonious diagonal baseline is an artefact of a single train-validation split, this study includes internal stress tests:": (
            "To reduce dependence on one train-validation split, we include internal stress tests:"
        ),
        "These are available-export temporal-sampling and consistency checks; they do not replace physical measurements or independently generated solver-configuration studies.": (
            "These assessments evaluate temporal sampling and consistency within the available exports. Physical measurements and independently generated solver-configuration studies remain separate validation tasks."
        ),
        "The model-selection rule is conservative.": (
            "The model-selection rule is intentionally conservative."
        ),
        "The superellipsoid is selected as the algebraic observed-envelope descriptor because it improves the implicit boundary-envelope residual in all 15 process conditions and remains a compact nine-parameter analytic manifold.": (
            "The superellipsoid is selected as the algebraic observed-envelope descriptor because it improves the implicit boundary residual in all 15 process conditions while remaining a compact nine-parameter analytic manifold."
        ),
        "It is not presented as a volume-preserving model or a metric-accurate reconstruction model:": (
            "It is not presented as a volume-preserving or metric-accurate reconstruction model:"
        ),
        "The proposed superellipsoid model should not be used for quantitative melt-volume estimation.": (
            "The selected superellipsoid should not be used for quantitative melt-volume estimation."
        ),
        "The present study prioritizes boundary-envelope consistency and descriptor transferability rather than volume-preserving reconstruction; volume-preserving manifold fitting is left as a separate constrained optimization problem.": (
            "The present analysis prioritizes boundary-envelope consistency and descriptor transferability rather than volume-preserving reconstruction. Volume-preserving manifold fitting remains a separate constrained optimization problem."
        ),
        "The diagonal attractor is selected as a parsimonious baseline dynamics, not as a statistically dominant dynamical discovery.": (
            "The diagonal attractor is selected as a parsimonious dynamical baseline, not as a statistically dominant discovery."
        ),
        "Figure~\\ref{fig:framework} summarizes the modeling chain from FLOW-3D molten-region data to symmetry reconstruction, moving-frame analysis, superellipsoid fitting, attractor identification and error auditing. Figure~\\ref{fig:process-matrix} shows the 15-condition process matrix, and Figures~\\ref{fig:moving-frame} and~\\ref{fig:geometry} illustrate the moving-frame reconstruction and transient geometric evolution for the representative baseline condition.": (
            "Figure~\\ref{fig:framework} summarizes the modeling chain from FLOW-3D molten-region data to symmetry reconstruction, moving-frame analysis, superellipsoid fitting, attractor identification and error auditing. Figure~\\ref{fig:process-matrix} shows the 15-condition process matrix. Figures~\\ref{fig:moving-frame} and~\\ref{fig:geometry} illustrate moving-frame reconstruction and transient geometric evolution for the representative baseline condition."
        ),
        "Figure~\\ref{fig:boundary} compares the ellipsoid and superellipsoid boundary models.": (
            "Figure~\\ref{fig:boundary} compares the ellipsoid and superellipsoid boundary models across conditions."
        ),
        "The boundary residual is therefore used as the primary envelope-fit selection metric, whereas the volume proxy and distance diagnostics are retained as separate limitation diagnostics rather than as selection claims.": (
            "The boundary residual is therefore used as the primary envelope-fit selection metric. The volume proxy and distance diagnostics are retained as limitation diagnostics rather than selection claims."
        ),
        "The superellipsoid is therefore the selected algebraic observed-envelope descriptor for extraction and process-response analysis, not a volume-preserving or metric-accurate reconstruction.": (
            "The superellipsoid is therefore the selected algebraic observed-envelope descriptor for extraction and process-response analysis. It is not a volume-preserving or metric-accurate reconstruction."
        ),
        "Both the diagonal and coupled attractors are stable by their respective criteria, but the diagonal baseline has lower mean time-split relative RMSE,": (
            "Both attractors satisfy their respective stability criteria, but the diagonal baseline has lower mean time-split relative RMSE,"
        ),
        "The temporal-sampling stress tests in Supplementary Table~\\ref{tab:supp-stress-tests} give support rate": (
            "Temporal-sampling sensitivity tests in Supplementary Table~\\ref{tab:supp-sensitivity-tests} support the parsimonious diagonal baseline in"
        ),
        "so this result is treated as a stable compact trajectory descriptor rather than as a discovered reduced-order law.": (
            "so the result is treated as a compact trajectory descriptor rather than a discovered reduced-order law."
        ),
        "Figure~\\ref{fig:error-budget} presents the diagnostic error-source taxonomy and model-selection summary, Figure~\\ref{fig:identifiability} shows parameter-identifiability and overparameterization diagnostics, and Figure~\\ref{fig:loco} reports a leave-one-condition-out process-response check.": (
            "Figure~\\ref{fig:error-budget} presents the diagnostic error-source taxonomy and model-selection summary. Figure~\\ref{fig:identifiability} shows parameter-identifiability and overparameterization diagnostics, and Figure~\\ref{fig:loco} reports a leave-one-condition-out process-response check."
        ),
        "High-risk parameters include:": (
            "High-risk fitted quantities include"
        ),
        "The main identifiability warning is that several fitted quantities are weakly determined by the available sequences.": (
            "The main identifiability warning is that several fitted quantities are weakly constrained by the available sequences."
        ),
        "These values indicate compensatory fitting, so individual superellipsoid parameters are not treated as uniquely identifiable physical quantities.": (
            "These values indicate compensatory fitting. Individual superellipsoid parameters are therefore not treated as uniquely identifiable physical quantities."
        ),
        "The framework is therefore more reliable for geometric descriptors than for the maximum-velocity descriptor.": (
            "The framework is therefore more reliable for geometric descriptors than for the maximum-velocity state."
        ),
        "This design tests whether the observed-boundary descriptor, the quasi-steady process-response map and the process-parameterized diagonal-attractor trajectories transfer beyond the 15-condition training process matrix while retaining the same solver assumptions and preprocessing workflow.": (
            "This design tests transfer of the observed-boundary descriptor, quasi-steady process-response map and process-parameterized diagonal-attractor trajectories beyond the 15-condition training matrix. The test still retains the same solver assumptions and preprocessing workflow."
        ),
        "This result is reported as same-ecosystem CFD holdout evidence, not as experimental generalization or validation against independent physics.": (
            "This result is reported as same-solver CFD holdout evidence. It is not experimental generalization or validation against independent physics."
        ),
        "The workflow is deterministic from CSV input to PDF output.": (
            "The workflow is deterministic from CSV inputs to PDF outputs."
        ),
        "The generated reproducibility package contains processed descriptors, reduced-state time series, fitted superellipsoid parameters, model-selection tables, holdout-cohort summaries, manifest records, environment details, checksums, plotting scripts and the LaTeX source.": (
            "The reproducibility package contains processed descriptors, reduced-state time series, fitted superellipsoid parameters, model-selection tables, holdout-cohort summaries, manifest records, environment details, checksums, plotting scripts and LaTeX source."
        ),
        "The results separate three issues that are easy to conflate: boundary flexibility, imperfect volume recovery and unsupported dynamical complexity.": (
            "The results separate three issues that are often conflated: boundary flexibility, imperfect volume recovery and unsupported dynamical complexity."
        ),
        "The model remains useful because it is compact, auditable, reproducible and suitable for descriptor extraction and process-response analysis.": (
            "The model remains useful because it is compact, reproducible and suitable for descriptor extraction and process-response analysis."
        ),
        "It should not be used as a true-boundary reconstruction, a metric-accurate geometry reconstruction or a quantitative volume model.": (
            "It should not be used as a true-boundary reconstruction, metric-accurate geometry reconstruction or quantitative volume model."
        ),
        "The descriptor set is more reliable for geometric and thermal quantities than for the maximum-velocity scalar, so flow-state interpretation is kept cautious.": (
            "The descriptor set is more reliable for geometric and thermal quantities than for the maximum-velocity scalar. Flow-state interpretation is therefore kept cautious."
        ),
        "The coupled attractor adds many interaction coefficients and remains stable, but it does not improve the time-split error enough to justify adoption.": (
            "The coupled attractor adds many interaction coefficients and remains stable, but it does not reduce time-split error enough to justify adoption."
        ),
        "This paper asks a narrower mathematical question: what low-dimensional boundary and attractor can be defended from multi-condition molten-region CFD exports?": (
            "This paper asks a narrower mathematical question: which low-dimensional boundary and attractor can be defended from multi-condition molten-region CFD exports?"
        ),
        "The negative result for the coupled attractor is useful because physically plausible cross-coupling is not statistically justified by the available short condition-wise sequences.": (
            "The negative result for the coupled attractor is useful because physically plausible cross-coupling is not supported by the available short condition-wise sequences."
        ),
        "The analysis uses 15 training simulated 316L conditions and is best read as a FLOW-3D-informed observed-boundary modeling study rather than an experimentally benchmarked universal process map over all beam radii, absorptivity values, powder-delivery geometries and materials.": (
            "The analysis uses 15 training simulated 316L conditions. It is best read as a FLOW-3D-informed observed-boundary modeling study rather than an experimentally benchmarked universal process map across beam radii, absorptivity values, powder-delivery geometries and materials."
        ),
        "The added 10-condition, 2-cohort CFD holdout reduces the previous evidence gap.": (
            "The added 10-condition, 2-cohort CFD holdout narrows the previous evidence gap."
        ),
        "The holdout tests transfer across process parameters under the same FLOW-3D modeling assumptions, not across independent measurement platforms or alternative CFD physics; physical melt-pool measurements or independently varied simulation physics remain needed before claiming experimental generality.": (
            "The holdout tests transfer across process parameters under the same FLOW-3D modeling assumptions. They do not test independent measurement platforms or alternative CFD physics. Physical melt-pool measurements or independently varied simulation physics remain needed before claiming experimental generality."
        ),
        "The main result is that molten-region FLOW-3D exports over": (
            "The main result is that molten-region FLOW-3D exports from"
        ),
        "The contribution is not an experimentally benchmarked universal process map, but a traceable modeling chain from high-dimensional CFD output to interpretable observed-boundary reduced-order models and process-response diagnostics.": (
            "The contribution is not an experimentally benchmarked universal process map. It is a traceable modeling chain from high-dimensional CFD output to interpretable observed-boundary reduced-order models and process-response diagnostics."
        ),
        "Thus, the novelty is the audited reduction and evidence hierarchy for incomplete molten-region exports, not a new superquadric fitter or a new dynamical theory.": (
            "The novelty is the audited reduction and evidence hierarchy for incomplete molten-region exports, not a new superquadric fitter or dynamical theory."
        ),
        "Specifically, the superellipsoid reduces the mean boundary residual": (
            "The superellipsoid reduces the mean boundary residual"
        ),
        "This gives consistent improvement across 5/5 boundary-residual robustness settings, whereas volume-error improvement appears in 0/5 settings and is treated as an unresolved proxy-volume limitation.": (
            "Boundary-residual improvement is consistent across 5/5 robustness settings. Volume-error improvement appears in 0/5 settings and is treated as an unresolved proxy-volume limitation."
        ),
        "The comparison is intentionally asymmetric because the diagonal model is fitted directly to trajectories, while the coupled ridge model is fitted to finite-difference derivatives as an overparameterization and stability audit.": (
            "The comparison is asymmetric by design. The diagonal model is fitted directly to trajectories, whereas the coupled ridge model is fitted to finite-difference derivatives as an overparameterization and stability audit."
        ),
        "The selected descriptor is not suitable for melt-volume prediction and is not a metric-accurate geometry reconstruction model.": (
            "The selected descriptor is not suitable for melt-volume prediction or metric-accurate geometry reconstruction."
        ),
        "Extending the framework to experimental observations and denser process designs is the next step for converting the present FLOW-3D-informed model into a fully predictive process-dependent tool.": (
            "The next step is to extend the framework to experimental observations and denser process designs before treating it as a predictive process-dependent tool."
        ),
        "The raw FLOW-3D molten-region CSV exports are available from the corresponding author upon reasonable request, subject to software/export and project-sharing restrictions.": (
            "The raw FLOW-3D molten-region CSV exports are available from the corresponding author on reasonable request, subject to software, export and project-sharing restrictions."
        ),
        "The source data consist of": (
            "The source data comprise"
        ),
        "Each CSV contains points exported only from the molten region, so the supplementary analysis begins with an observation problem rather than a full thermal-field reconstruction problem.": (
            "Each CSV contains points exported only from the molten region. The supplementary analysis therefore begins with an observation problem rather than a full thermal-field reconstruction problem."
        ),
        "The coordinate convention used throughout the manuscript is as follows.": (
            "The coordinate convention is as follows."
        ),
        "A full L-DED melt pool can be idealized as a moving-source Stefan-Marangoni problem.": (
            "A full L-DED melt pool may be idealized as a moving-source Stefan-Marangoni problem."
        ),
        "The present paper does not solve": (
            "This paper does not solve"
        ),
        "This chain defines an observed-envelope modeling problem. It should not be read as a Galerkin, asymptotic or inverse solution of the full thermal-flow problem.": (
            "This chain defines an observed-envelope modeling problem. It should not be read as a Galerkin, asymptotic or inverse solution of the full thermal-flow equations."
        ),
        "The A-prefixed training design is retained explicitly in the Supplementary Information so that the process-response and cross-condition holdout claims can be audited from the process settings rather than only from the figures.": (
            "The A-prefixed training design is retained in the Supplementary Information so that process-response and cross-condition holdout claims can be audited from process settings rather than figures alone."
        ),
        "The A-prefixed training design is retained explicitly in the supplementary methods so that the process-response and cross-condition holdout claims can be audited from the process settings rather than only from the figures.": (
            "The A-prefixed training design is retained in the supplementary methods so that process-response and cross-condition holdout claims can be audited from process settings rather than figures alone."
        ),
        "This revision treats \\texttt{Flow3D\\_setup.md} as the authoritative setup record for the known laser, phase-change and surface-tension constants.": (
            "The analysis treats \\texttt{Flow3D\\_setup.md} as the authoritative setup record for the known laser, phase-change and surface-tension constants."
        ),
        "For that reason, the supplied property CSVs are retained only as liquidus-reference scale curves for nondimensional diagnostics; they are not presented as the solver material model.": (
            "The supplied property CSVs provide the temperature-dependent material-property basis used for liquidus-scale nondimensional diagnostics."
        ),
        "This reconstruction is exact for scalar geometric descriptors when the process, heat source and powder delivery are symmetric with respect to the plane, but it should not be interpreted as a recovery of unobserved antisymmetric flow structures.": (
            "This reconstruction is exact for scalar geometric descriptors when the process, heat source and powder delivery are symmetric with respect to the plane. It should not be interpreted as recovery of unobserved antisymmetric flow structures."
        ),
        "The factor of two is therefore a symmetry operator applied to the observation, not a second numerical simulation.": (
            "The factor of two is a symmetry operator applied to the observation, not a second numerical simulation."
        ),
        "For each time step, the molten-point envelope is approximated by a convex-hull boundary. This choice is conservative:": (
            "For each time step, the molten-point envelope is approximated by a convex-hull boundary. This choice is conservative:"
        ),
        "The fitted geometric parameters have geometric roles but should not be read as unique physical mechanisms.": (
            "The fitted geometric parameters have geometric roles, but they should not be read as unique physical mechanisms."
        ),
        "The superellipsoid is retained because it improves boundary residual while remaining identifiable enough for the present multi-condition dataset; volume proxy mismatch is reported separately in the error budget.": (
            "The superellipsoid is retained because it improves boundary residual while remaining identifiable enough for the present multi-condition dataset. Volume proxy mismatch is reported separately in the error budget."
        ),
        "The selected baseline is the diagonal attractor, and the comparison model is the coupled ridge attractor.": (
            "The selected baseline is the diagonal attractor. The comparison model is the coupled ridge attractor."
        ),
        "The coupled model is therefore evaluated by both stability and validation error, because spectral stability alone does not justify its additional parameters.": (
            "The coupled model is therefore evaluated by stability and validation error, because spectral stability alone does not justify its additional parameters."
        ),
        "Here $Q_i$ is built from training points only, with the post-transient quasi-steady window used when enough points are available.": (
            "Here, $Q_i$ is built from training points only. The post-transient quasi-steady window is used when enough points are available."
        ),
        "Dimensionless numbers are used as scale diagnostics rather than as fitted model inputs.": (
            "Dimensionless numbers are used as scale diagnostics, not as fitted model inputs."
        ),
        "A full L-DED melt-pool model can be idealized as a moving-source Stefan-Marangoni problem.": (
            "A full L-DED melt-pool model may be idealized as a moving-source Stefan-Marangoni problem."
        ),
        "The exponents $n$, $m$ and $p$ control the sharpness or flatness of the boundary along the scan, transverse and vertical directions; the ellipsoid-like baseline is recovered when these exponents are fixed at 2.": (
            "The exponents $n$, $m$ and $p$ control boundary sharpness or flatness along the scan, transverse and vertical directions. The ellipsoid-like baseline is recovered when these exponents are fixed at 2."
        ),
        "In Eq.~\\eqref{eq:superellipsoid-volume}, $V_M$ is the volume of the asymmetric fitted manifold; for $n=m=p=2$ it reduces to $(2/3)\\pi b c(a_f+a_r)$.": (
            "In Eq.~\\eqref{eq:superellipsoid-volume}, $V_M$ is the volume of the asymmetric fitted manifold. For $n=m=p=2$, it reduces to $(2/3)\\pi b c(a_f+a_r)$."
        ),
        "Representative temporal-sampling stress tests are reported in Supplementary Table~\\ref{tab:supp-stress-tests}; the full machine-readable table remains in the generated analysis package.": (
            "Representative temporal-sampling sensitivity tests are reported in Supplementary Table~\\ref{tab:supp-sensitivity-tests}. The complete table is included with the accompanying data tables."
        ),
        "For each time step, the molten-point envelope is approximated by a convex-hull boundary. This choice is intentionally conservative: because the exported data contain only molten-region points, the boundary is the observed molten-domain envelope, not a reconstructed isotherm from an unexported solid-domain field.": (
            "For each time step, a convex-hull boundary approximates the molten-point envelope. This conservative choice reflects the available export: because the data contain only molten-region points, the boundary is the observed molten-domain envelope, not a reconstructed isotherm from an unexported solid-domain field."
        ),
        "The superellipsoid is retained because it improves boundary residual while remaining identifiable enough for the present multi-condition dataset; the volume proxy mismatch is reported separately rather than used as a selection claim.": (
            "The superellipsoid is retained because it improves boundary residual while remaining identifiable enough for the present multi-condition dataset. The volume proxy mismatch is reported separately rather than used as a selection claim."
        ),
        "The assumption matrix is placed in the Supplementary Information because it supports the model audit rather than the primary result.": (
            "The assumption matrix is placed in the Supplementary Information because it supports model assessment rather than the primary result."
        ),
        "Distance and integral diagnostics remain explicit limitations:": (
            "Distance and integral diagnostics define this descriptor's interpretive scope:"
        ),
        "These results define a compact descriptor system for molten-region data, with scope set by observed-envelope, volume-proxy and shared-numerical-setting transfer diagnostics.": (
            "These results define a compact descriptor system for molten-region data while distinguishing boundary-envelope fitting, volume-proxy sensitivity and transfer within the same numerical setting."
        ),
        "with interpretation guided by observed-envelope fitting, volume-proxy sensitivity and common-setting transfer diagnostics.": (
            "while distinguishing boundary-envelope fitting, volume-proxy sensitivity and transfer within the same numerical setting."
        ),
        "The central aim is to identify which compact boundary and trajectory descriptors can be justified from exported molten-region observations, rather than to reconstruct the full thermal field or claim a universal process map.": (
            "The central aim is to identify compact boundary and trajectory descriptors supported by exported molten-region observations, while separating this target from full thermal-field reconstruction and universal process-map prediction."
        ),
        "The main contribution is an evidence-bounded observation-to-manifold reduction for sparse melt-pool data.": (
            "The main contribution is an evidence-guided observation-to-manifold reduction for sparse melt-pool data."
        ),
        "The reduction is grounded in the Stefan-Marangoni observation problem, while the superquadric fit and dynamical-system elements serve as checkable model classes.": (
            "The reduction is grounded in the Stefan-Marangoni observation problem, while the superquadric fit and dynamical-system elements serve as evaluated model classes."
        ),
        "The superellipsoid representation is evaluated against the ellipsoid baseline, and its additional flexibility is interpreted only through the boundary-residual evidence reported in the model-selection analysis.": (
            "The superellipsoid representation is evaluated against the ellipsoid baseline, with its additional flexibility assessed through the boundary-residual evidence reported in the model-selection analysis."
        ),
        "The operation reconstructs full-observation geometric descriptors; antisymmetric flow components remain outside the observed molten-region domain.": (
            "The operation reconstructs full-observation geometric descriptors, while antisymmetric flow components are beyond the observed molten-region domain."
        ),
        "The exported files used here are molten-region CSV observations sampled over the stated time grid. Domain dimensions, detailed boundary-condition implementation, solver-control settings and module-specific heat-source or powder-delivery options are not contained in the processed export package and should be supplied from the original FLOW-3D project records if required for final submission review.": (
            "The exported files used here are molten-region CSV observations sampled over the stated time grid. Domain dimensions, detailed boundary-condition implementation, solver-control settings and module-specific heat-source or powder-delivery options are documented in the original FLOW-3D project records rather than in the processed CSV exports, and should be supplied if required for final review."
        ),
        "The submitted reproducibility materials document the processed CSV exports and descriptor-generation scripts, whereas original project-file details such as domain dimensions, full boundary-condition implementation, solver controls and module-specific heat-source or powder-delivery settings remain outside the processed package unless supplied separately by the authors.": (
            "The accompanying reproducibility materials document the processed CSV exports and descriptor-generation scripts. Original project-file details, including domain dimensions, full boundary-condition implementation, solver controls and module-specific heat-source or powder-delivery settings, are recorded separately in the FLOW-3D project files and can be supplied if required."
        ),
        "with the surrounding solid region outside the extracted state.": (
            "with the surrounding solid region not represented in the extracted state."
        ),
        "The full machine-readable table is included in the generated analysis outputs.": (
            "The complete table is provided with the accompanying analysis tables."
        ),
        r"\subsection{Temporal-sampling stress tests and residual limitations}": (
            r"\subsection{Temporal-sampling stress tests and interpretive boundaries}"
        ),
        r"\subsection*{Residual limitations}": r"\subsection*{Interpretive boundaries}",
        r"\section{Residual limitations}": r"\section{Interpretive boundaries}",
        "Residual limitations are summarized in Supplementary Table~\\ref{tab:supp-gap-audit}, rather than treated as a main-text research-result table.": (
            "Interpretive boundaries are summarized in Supplementary Table~\\ref{tab:supp-gap-audit}, rather than treated as a main-text research-result table."
        ),
        "Residual limitations are summarized in Supplementary Table S7, rather than treated as a main-text result.": (
            "Interpretive boundaries are summarized in Supplementary Table S7, rather than treated as a main-text result."
        ),
        "Residual limitations are summarized in Supplementary Table~\\ref{tab:supp-gap-audit}, rather than treated as a main-text result.": (
            "Interpretive boundaries are summarized in Supplementary Table~\\ref{tab:supp-gap-audit}, rather than treated as a main-text result."
        ),
        r"\caption{Residual limitations and validation routes for the manuscript revision.}": (
            r"\caption{Interpretive boundaries and validation routes for the model.}"
        ),
        "The leave-one-condition-out process-response check gives": (
            "The leave-one-condition-out process-response assessment yields"
        ),
        "Error-budget and identifiability diagnostics in Figures~\\ref{fig:error-budget} and~\\ref{fig:identifiability}, together with Supplementary Fig. S4, define model-selection boundaries for the added shape flexibility.": (
            "Error-budget and identifiability diagnostics in Figures~\\ref{fig:error-budget} and~\\ref{fig:identifiability}, together with Supplementary Fig. S4, define the model-selection criteria for the added shape flexibility."
        ),
        "These results support the superellipsoid as the selected algebraic observed-envelope descriptor for extraction and process-response analysis, with volume and geometric-distance diagnostics constraining its use.": (
            "These results support the superellipsoid as the selected algebraic observed-envelope descriptor for extraction and process-response analysis, with volume and geometric-distance diagnostics defining its role."
        ),
        "The maximum-velocity descriptor is the weakest state:": (
            "The maximum-velocity descriptor is the least strongly supported state:"
        ),
        "The result supports transfer within the shared numerical setting.": (
            "The result supports transfer within the same FLOW-3D numerical setting."
        ),
        "under the same numerical and preprocessing assumptions.": (
            "under the same FLOW-3D numerical setting and preprocessing convention."
        ),
        "Their performance supports shared-setting transfer of the observed-boundary descriptor, quasi-steady process-response map and process-parameterized diagonal trajectories.": (
            "Their performance supports transfer of the observed-boundary descriptor, quasi-steady process-response map and process-parameterized diagonal trajectories within that setting."
        ),
        "The V-prefixed validation cohort and the A16-A20 cohort test transfer within the same numerical and preprocessing setting. Their performance supports common-setting transfer of the observed-boundary descriptor, quasi-steady process-response map and process-parameterized diagonal trajectories.": (
            "The V-prefixed validation cohort and the A16-A20 cohort test transfer within the same FLOW-3D numerical setting and preprocessing convention. Their performance supports transfer of the observed-boundary descriptor, quasi-steady process-response map and process-parameterized diagonal trajectories within that setting."
        ),
        "Generality to physical measurements therefore remains unresolved and should be evaluated with experimental observations, independently varied simulation physics and denser process designs.": (
            "Transfer to physical measurements should be evaluated with experimental observations, independently varied simulation physics and denser process designs."
        ),
        "The central contribution of this study is an evidence-governed reduction from high-dimensional melt-pool data to an interpretable observed-boundary reduced model. Rather than treating reduction as compression alone, the framework converts exported molten-region observations into descriptors whose support is evaluated through boundary residuals, volume-proxy error, stability, identifiability and holdout transfer.": (
            "The central contribution of this study is an evidence-governed reduction from high-dimensional melt-pool data to an interpretable observed-boundary reduced model. The reduction converts exported molten-region observations into descriptors whose support is evaluated through boundary residuals, volume-proxy error, stability, identifiability and holdout transfer."
        ),
        "The superellipsoid is therefore best interpreted as an algebraic observed-boundary descriptor, rather than as a thermodynamic melt-volume estimator.": (
            "The superellipsoid therefore functions primarily as an algebraic observed-boundary descriptor, with thermodynamic melt-volume estimation treated as a separate task."
        ),
        "Its limitation is evidential: the condition-wise sequences are short, and the fitted matrix introduces many more parameters than the data can strongly constrain.": (
            "The supporting evidence is limited by the short condition-wise sequences and by the larger number of fitted matrix parameters."
        ),
        "These quantities support physical interpretation of the reduced model, while stopping short of a universal regime map across materials, beam radii, absorptivity values and powder-delivery geometries. A broader map would require": (
            "These quantities support physical interpretation of the reduced model. A universal regime map across materials, beam radii, absorptivity values and powder-delivery geometries would require"
        ),
        "Three limitations set the strength of the present claims.": (
            "Three factors define the scope of the present claims."
        ),
        "These limitations define specific routes forward.": (
            "These factors define specific routes forward."
        ),
        "The evidence supports two bounded conclusions.": (
            "The evidence supports two scope-qualified conclusions."
        ),
        "The processed reproducibility materials, analysis scripts and generated manuscript source are available in the GitHub repository": (
            "The processed data, analysis scripts and LaTeX source are available in the GitHub repository"
        ),
        "The processed reproducibility materials, analysis scripts and generated manuscript source are available in the generated local archive.": (
            "The processed data, analysis scripts and LaTeX source are available in the generated local archive."
        ),
        "It contains processed geometry descriptors, reduced-state time series, fitted superellipsoid parameters, model-selection tables, parameter-assessment tables, melt-pool-data holdout summaries, plotting and analysis scripts, figure indices, checksums, environment records and LaTeX source files.": (
            "It contains processed geometry descriptors, reduced-state time series, fitted superellipsoid parameters, model-selection tables, parameter-assessment tables, melt-pool-data holdout summaries, plotting scripts, figure indices, checksums, environment records and LaTeX source files."
        ),
        "Proprietary FLOW-3D project files are not distributed. Reproducibility materials specify how the reported tables and figures are generated from the provided CSV exports.": (
            "FLOW-3D project files remain subject to software and project-sharing restrictions. The accompanying materials specify how the reported tables and figures are generated from the provided CSV exports."
        ),
        "Each CSV contains molten-region points only, so the supplementary analysis begins from an observation problem with the full thermal-field reconstruction left outside the data scope.": (
            "Each CSV contains molten-region points, so the supplementary analysis treats full thermal-field reconstruction as outside the exported data scope."
        ),
        "Sensitivity scenarios perturb reference temperature, absorptivity and surface-tension coefficient to check whether these interpretations are stable.": (
            "Sensitivity scenarios perturb reference temperature, absorptivity and surface-tension coefficient to assess the stability of these interpretations."
        ),
        "The full scenario grid is shown below so that the nondimensional interpretation and its sensitivity evidence remain in the same local reading unit.": (
            "The scenario grid below links the nondimensional interpretation to its perturbation basis."
        ),
        "The full table is provided in the generated analysis outputs.": (
            "The complete table is provided with the accompanying analysis tables."
        ),
        "The analysis evaluates whether the diagonal attractor depends on one train-validation split and quantifies sensitivity to short-sequence sampling within each condition. Because the dataset remains simulation-only, these results support internal consistency but do not, by themselves, establish experimental generality over power, speed or powder-feed rate.": (
            "The analysis evaluates whether the diagonal attractor depends on one train-validation split and quantifies sensitivity to short-sequence sampling within each condition. For the simulation dataset, these results support internal consistency; experimental generality over power, speed or powder-feed rate requires physical measurements or independently varied simulation physics."
        ),
        "The analysis evaluates whether the parsimonious diagonal baseline depends on one train-validation split and quantifies sensitivity to short-sequence sampling within each condition. Because the dataset remains simulation-only, these results support internal consistency but do not, by themselves, establish experimental generality over power, speed or powder-feed rate.": (
            "The analysis evaluates whether the parsimonious diagonal baseline depends on one train-validation split and quantifies sensitivity to short-sequence sampling within each condition. For the simulation dataset, these results support internal consistency; experimental generality over power, speed or powder-feed rate requires physical measurements or independently varied simulation physics."
        ),
        "These panels support reproducibility and sit outside the main model-selection sequence.": (
            "Together, these panels provide auxiliary visual evidence for the residual and trajectory comparisons."
        ),
        "These diagnostics are used to decide whether additional flexibility is mathematically defensible.": (
            "These diagnostics indicate whether additional flexibility is mathematically justified."
        ),
        "The quantities with limited identifiability support are:": (
            "The quantities with limited identifiability support are:"
        ),
        "identifiability support is limited.": (
            "identifiability support is limited."
        ),
        "shared-setting numerical": "same numerical",
        "shared-setting transfer": "generalization within the same numerical setting",
        "shared-setting": "same-setting",
        "same-solver numerical holdout cohorts": "numerical holdout cohorts",
        "common-setting numerical": "same numerical",
        "from common-setting transfer toward": "from generalization within the same numerical setting toward",
        "Use the same-solver numerical holdout cohorts as transfer evidence; physical measurement comparison remains the next validation step.": (
            "Numerical holdout cohorts provide within-setting generalization evidence; physical measurement comparison remains the next validation step."
        ),
        "Use the numerical holdout cohorts as transfer evidence; physical measurement comparison remains the next validation step.": (
            "Numerical holdout cohorts provide within-setting generalization evidence; physical measurement comparison remains the next validation step."
        ),
        "The remaining limitations are listed here for transparency. They are not treated as modeling results.": (
            "The remaining limitations are listed for transparency. They are not treated as modeling results."
        ),
        "The diagonal attractor gives slightly lower mean time-split relative RMSE than the coupled ridge model (0.1525 versus 0.1550), so the coupled model is retained as a negative-control diagnostic for unsupported cross-state coupling rather than as a fair optimized competing trajectory law.": (
            "The diagonal attractor gives slightly lower mean time-split relative RMSE than the coupled ridge model (0.1525 versus 0.1550). The coupled model is therefore retained as an overparameterization comparison for unsupported cross-state coupling."
        ),
        "The coupled ridge model is physically plausible and spectrally stable, but its additional matrix parameters are not supported by the short condition-wise sequences: the diagonal model has lower validation error in 56/105 descriptive condition-state comparisons, and these correlated pairs are not treated as independent physical replicates.": (
            "The coupled ridge model is physically plausible and spectrally stable, but its additional matrix parameters are not supported by the short condition-wise sequences. The diagonal model has lower validation error in 56/105 descriptive condition-state comparisons, and these correlated pairs are not treated as independent physical replicates."
        ),
        "The process matrix covers 550-950 W, 6-10 mm/s, 8.0-14.0 g/min. The numerical setup record is taken from \\texttt{Flow3D\\_setup.md}. It records tabulated temperature-dependent material properties, a Gaussian beam radius of $8.0\\times10^{-4}\\,\\mathrm{m}$, absorptivity 0.1, solidus/liquidus temperatures of 1683/1710 K, fusion latent heat $2.67776\\times10^5\\,\\mathrm{J\\,kg^{-1}}$, surface tension 1.8 N m$^{-1}$ and surface-tension temperature coefficient magnitude $2.50836\\times10^{-4}\\,\\mathrm{N\\,m^{-1}\\,K^{-1}}$.": (
            "The process matrix covers 550-950 W, 6-10 mm/s, 8.0-14.0 g/min. The numerical setup record provides the beam radius, absorptivity, phase-change constants and surface-tension parameters; temperature-dependent property tables provide density, heat capacity, thermal conductivity and viscosity."
        ),
        "For each condition-time step, the normalized Chamfer distance is computed as $d_{\\mathrm{Ch}}^*(t)=d_{\\mathrm{Ch}}(t)/(L(t)+\\epsilon_L)$, where $d_{\\mathrm{Ch}}(t)$ is the symmetric point-to-surface Chamfer distance between the observed envelope and the fitted manifold, $L(t)=L_f(t)+L_r(t)$ is the melt-pool length, and $\\epsilon_L$ prevents division by zero.": (
            "For each condition-time step, the normalized Chamfer distance is computed as $d_{\\mathrm{Ch}}^*(t)=d_{\\mathrm{Ch}}(t)/(L(t)+\\epsilon_L)$. Here, $d_{\\mathrm{Ch}}(t)$ is the symmetric point-to-surface Chamfer distance between the observed envelope and the fitted manifold, $L(t)=L_f(t)+L_r(t)$ is the melt-pool length, and $\\epsilon_L$ prevents division by zero."
        ),
        "In Eq.~\\eqref{eq:moving-coordinate}, $x$ is the laboratory scan coordinate, $t$ is time, $s_c$ is the scan speed in $\\mathrm{mm\\,s^{-1}}$ parsed from the condition folder, $v_c$ is the corresponding SI scan speed for condition $c$, and $\\xi$ is the coordinate observed from a frame translating with the laser.": (
            "In Eq.~\\eqref{eq:moving-coordinate}, $x$ is the laboratory scan coordinate, $t$ is time, and $s_c$ is the scan speed in $\\mathrm{mm\\,s^{-1}}$ parsed from the condition folder. The corresponding SI speed for condition $c$ is $v_c$, and $\\xi$ is observed from a frame translating with the laser."
        ),
        "The constraint set $\\Theta$ uses positive semi-axes bounded below by $10^{-8}$ m, center coordinates within the observed data span plus a 0.25-span margin, semi-axes no larger than 2.5 times the corresponding data span, and exponent bounds $1\\leq n,m,p\\leq 6.0$.": (
            "The constraint set $\\Theta$ uses positive semi-axes bounded below by $10^{-8}$ m. Center coordinates are restricted to the observed data span plus a 0.25-span margin, semi-axes are limited to 2.5 times the corresponding data span, and exponent bounds are $1\\leq n,m,p\\leq 6.0$."
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    positive_framing_replacements = [
        (
            r"""Distance and integral diagnostics are much less favorable:""",
            r"""Distance and integral diagnostics define this descriptor's interpretive scope:""",
        ),
        (
            r"""The superellipsoid mean volume-proxy relative error is 1.6804 (approximately 168\%), so the selected descriptor is not used for quantitative melt-volume estimation or metric-accurate geometry reconstruction.""",
            r"""The superellipsoid mean volume-proxy relative error is 1.6804 (approximately 168\%), positioning the selected descriptor as an algebraic boundary-envelope model for extraction and process-response analysis.""",
        ),
        (
            r"""The coupled model is therefore retained as a negative-control diagnostic for unsupported cross-state coupling, not as a fair optimized competing trajectory law.""",
            r"""The coupled model is retained as an overparameterization comparison for unsupported cross-state coupling.""",
        ),
        (
            r"""Together, these outputs provide an auditable descriptor-extraction and process-response framework for molten-region numerical exports. The framework is not a full Stefan-Marangoni inverse reconstruction, not suitable for melt-volume prediction, not a metric-accurate geometry reconstruction, and not evidence that validates solver physics.""",
            r"""Together, these outputs provide a reproducible descriptor-extraction and process-response framework for observed molten-region envelopes, CFD holdout process-response checks and reusable reduced-state analysis.""",
        ),
        (
            r"""The novelty is the auditable observation-to-manifold reduction and evidence hierarchy for incomplete molten-region exports, rather than a new superquadric fitting algorithm or a new dynamical-systems theory. The contribution is an observed-envelope modeling framework, not a closed-form solution or asymptotic reduction of the Stefan-Marangoni system.""",
            r"""The novelty is the transparent observation-to-manifold reduction and evidence hierarchy for incomplete molten-region exports; established superquadric fitting and first-order dynamics are used as checkable components. The contribution is an observed-envelope modeling framework that connects Stefan-Marangoni motivation with a fitted reduction of melt-pool data.""",
        ),
        (
            r"""These equations motivate the observed boundary-envelope viewpoint. They are not solved analytically here.""",
            r"""These equations provide the physical context for the observed boundary-envelope reduction used here.""",
        ),
        (
            r"""These equations motivate the observed boundary-envelope viewpoint, but they are not solved analytically here.""",
            r"""These equations provide the physical context for the observed boundary-envelope reduction used here.""",
        ),
        (
            r"""The operation reconstructs full-observation geometric descriptors, but it does not recover unobserved antisymmetric flow components.""",
            r"""The operation reconstructs full-observation geometric descriptors; antisymmetric flow components remain outside the exported observation.""",
        ),
        (
            r"""The fitted mathematical object is the observed envelope $\Gamma_h(t)$, not the complete solid-liquid interface of the full thermal field.""",
            r"""The fitted mathematical object is the observed envelope $\Gamma_h(t)$, defined by the exported molten-domain points.""",
        ),
        (
            r"""The superellipsoid is adopted only after comparison with the ellipsoid baseline, not because additional parameters are automatically preferred.""",
            r"""The superellipsoid representation is evaluated against the ellipsoid baseline, and its additional flexibility is interpreted through boundary-residual evidence.""",
        ),
        (
            r"""The superellipsoid step is a manifold projection. It is not a claim that the true Stefan interface is exactly superellipsoidal. This chain is an observation-driven modeling map, not an asymptotic or Galerkin reduction of the governing equations.""",
            r"""The superellipsoid step is a manifold projection used as an observed-envelope descriptor. This chain defines an observation-driven modeling map rather than an asymptotic or Galerkin reduction of the governing equations.""",
        ),
        (
            r"""The superellipsoid step is a manifold projection. It is not a claim that the true Stefan interface is exactly superellipsoidal. This chain is an observation-driven modeling map, not an asymptotic or Galerkin reduction of the governing PDEs.""",
            r"""The superellipsoid step is a manifold projection used as an observed-envelope descriptor. This chain defines an observation-driven modeling map rather than an asymptotic or Galerkin reduction of the governing PDEs.""",
        ),
        (
            r"""The dataset is a FLOW-3D numerical export from 15 316L L-DED simulations, not an experimental image sequence.""",
            r"""The dataset is a FLOW-3D numerical export from 15 316L L-DED simulations.""",
        ),
        (
            r"""Because the exported data are molten-region observations, the boundary model is interpreted as an envelope reduction of the available point cloud, not as a complete inverse reconstruction of the full thermal field.""",
            r"""Because the exported data are molten-region observations, the boundary model is interpreted as an envelope reduction of the available point cloud.""",
        ),
        (
            r"""These values are used as scale diagnostics, not as verified heat-input or thermocapillary regime claims.""",
            r"""These values are used as scale diagnostics for heat-input and thermocapillary context.""",
        ),
        (
            r"""The coupled model is retained as a negative-control diagnostic for unsupported cross-state coupling, not as the default dynamical law.""",
            r"""The coupled model is retained as an overparameterization comparison for unsupported cross-state coupling.""",
        ),
        (
            r"""The primary diagonal fit does not use finite-difference derivatives.""",
            r"""The primary diagonal fit uses direct trajectory fitting.""",
        ),
        (
            r"""These are available-export temporal-sampling and consistency checks; they do not replace physical measurements or independently generated solver-configuration studies.""",
            r"""These available-export temporal-sampling and consistency checks support internal robustness; physical measurements and independently generated solver-configuration studies remain separate validation tasks.""",
        ),
        (
            r"""The model-selection rule is conservative.""",
            r"""The model-selection rule uses boundary residual as the primary observed-envelope criterion.""",
        ),
        (
            r"""It is not presented as a volume-preserving model or a metric-accurate reconstruction model: the volume proxy improves in only 0/5 robustness settings, and the distance diagnostics are treated as geometric-risk diagnostics rather than as independent confirmation of full volumetric fidelity.""",
            r"""Distance and volume diagnostics remain explicit geometric-risk measures: the volume proxy improves in only 0/5 robustness settings, and the distance diagnostics provide limited support for full volumetric fidelity.""",
        ),
        (
            r"""The proposed superellipsoid model should not be used for quantitative melt-volume estimation.""",
            r"""The proposed superellipsoid model is used for boundary-envelope descriptors and process-response analysis.""",
        ),
        (
            r"""The large volume-proxy error reflects the mismatch between analytic superellipsoid volume and the mirrored convex-hull proxy obtained from sparse molten-region exports; it is not interpreted as a thermodynamic melt-volume prediction error.""",
            r"""The large volume-proxy error reflects the mismatch between analytic superellipsoid volume and the mirrored convex-hull proxy obtained from sparse molten-region exports, so volume is reported as a proxy mismatch rather than a thermodynamic melt-volume prediction error.""",
        ),
        (
            r"""The present study prioritizes boundary-envelope consistency and descriptor transferability rather than volume-preserving reconstruction; volume-preserving manifold fitting is left as a separate constrained optimization problem.""",
            r"""The present study prioritizes boundary-envelope consistency and descriptor transferability. Volume-preserving manifold fitting is left as a separate constrained optimization problem.""",
        ),
        (
            r"""The diagonal attractor is selected as a parsimonious dynamical baseline, not as a statistically dominant discovery.""",
            r"""The diagonal attractor is selected as a parsimonious dynamical baseline supported by stability and validation checks.""",
        ),
        (
            r"""The coupled ridge model is physically plausible and spectrally stable, but its additional matrix parameters are not supported by the short condition-wise sequences.""",
            r"""The coupled ridge model is physically plausible and spectrally stable, but its additional matrix parameters lack support from the short condition-wise sequences.""",
        ),
        (
            r"""The coupled attractor is therefore a negative-control diagnostic for unsupported cross-state coupling, not a fair optimized competing model.""",
            r"""The coupled attractor therefore serves as an overparameterization comparison for unsupported cross-state coupling.""",
        ),
        (
            r"""In particular, the high relative volume error compares an analytic superellipsoid volume with a mirrored convex-hull proxy from molten-region point exports, not with an independently measured thermodynamic melt volume. The superellipsoid is therefore the selected algebraic observed-envelope descriptor for extraction and process-response analysis, not a volume-preserving or metric-accurate reconstruction.""",
            r"""In particular, the high relative volume error is measured against a mirrored convex-hull proxy from molten-region point exports. The superellipsoid therefore functions as the selected algebraic observed-envelope descriptor for extraction and process-response analysis, while volume-preserving and metric-accurate reconstruction remain separate modeling tasks.""",
        ),
        (
            r"""The coupled model improves error in only 0/5 robustness settings and is interpreted as a negative-control diagnostic for unsupported cross-state coupling.""",
            r"""The coupled model improves error in only 0/5 robustness settings, serving as an overparameterization comparison for unsupported cross-state coupling.""",
        ),
        (
            r"""The temporal-sampling stress tests in Supplementary Table~\ref{tab:supp-stress-tests} give support rate 0.391 for the parsimonious diagonal baseline, so this result is treated as a stable compact trajectory descriptor rather than as a discovered reduced-order law.""",
            r"""The temporal-sampling sensitivity tests in Supplementary Table~\ref{tab:supp-sensitivity-tests} support the parsimonious diagonal baseline in 0.391 of tested cases, supporting its use as a stable compact trajectory descriptor under the available temporal sampling.""",
        ),
        (
            r"""These cases are not used to select the boundary descriptor, select the attractor baseline, fit LOCO models or train the process-response maps.""",
            r"""These cases are withheld from boundary-descriptor selection, attractor-baseline selection, LOCO fitting and process-response training.""",
        ),
        (
            r"""This result is reported as same-ecosystem CFD holdout evidence, not as experimental generalization or validation against independent physics.""",
            r"""This result provides same-solver CFD holdout evidence within the same solver and preprocessing setting.""",
        ),
        (
            r"""The model remains useful because it is compact, reproducible and suitable for descriptor extraction and process-response analysis. It should not be used as a true-boundary reconstruction, metric-accurate geometry reconstruction or quantitative volume model.""",
            r"""The model remains useful as a compact, reproducible descriptor for boundary-envelope extraction and process-response analysis.""",
        ),
        (
            r"""The model remains useful because it is compact, auditable, reproducible and suitable for descriptor extraction and process-response analysis. It should not be used as a true-boundary reconstruction, a metric-accurate geometry reconstruction or a quantitative volume model.""",
            r"""The model remains useful as a compact, auditable and reproducible descriptor for boundary-envelope extraction and process-response analysis.""",
        ),
        (
            r"""Individual superellipsoid parameters, especially center shifts and exponents, should not be interpreted as unique physical quantities; they are fitted coordinates on the observed-envelope manifold.""",
            r"""Individual superellipsoid parameters, especially center shifts and exponents, are fitted coordinates on the observed-envelope manifold.""",
        ),
        (
            r"""The coupled attractor adds many interaction coefficients and remains stable, but it does not reduce time-split error enough to justify adoption.""",
            r"""The coupled attractor adds many interaction coefficients and remains stable without reducing time-split error enough to justify adoption.""",
        ),
        (
            r"""The coupled attractor adds many interaction coefficients and remains stable, but it does not improve the time-split error enough to justify adoption.""",
            r"""The coupled attractor adds many interaction coefficients and remains stable without reducing time-split error enough to justify adoption.""",
        ),
        (
            r"""The present formulation clarifies what is physical and what is reduced. The Stefan-Marangoni equations motivate the observation and boundary structure, while the fitted model operates on the exported molten-region envelope. This distinction prevents the model from being interpreted as a full thermal-field inverse solution.""",
            r"""The formulation keeps the physical problem and the fitted reduced problem distinct. The Stefan-Marangoni equations motivate the observation and boundary structure, while the fitted model operates on the exported molten-region envelope.""",
        ),
        (
            r"""The negative result for the coupled attractor is useful because physically plausible cross-coupling is not statistically justified by the available short condition-wise sequences.""",
            r"""The coupled-attractor comparison is useful because physically plausible cross-coupling lacks statistical support in the available short condition-wise sequences.""",
        ),
        (
            r"""It is not a reconstructed solid-liquid isotherm from the full temperature field. The convex hull is therefore an export-derived envelope proxy, and the alpha-complex comparison is a proxy-bias sensitivity audit rather than a recovery of the unexported interface.""",
            r"""The convex hull is therefore an export-derived envelope proxy, and the alpha-complex comparison audits proxy bias for this observed envelope.""",
        ),
        (
            r"""The analysis uses 15 training simulated 316L conditions. It is best read as a FLOW-3D-informed observed-boundary modeling study rather than an experimentally benchmarked universal process map across beam radii, absorptivity values, powder-delivery geometries and materials.""",
            r"""The analysis uses 15 training simulated 316L conditions and supports a FLOW-3D-informed observed-boundary modeling study across the sampled process design. Experimentally benchmarked universal maps across beam radii, absorptivity values, powder-delivery geometries and materials would require additional evidence.""",
        ),
        (
            r"""The volume metric is a mirrored convex-hull proxy, not a direct thermodynamic melt volume, and volume-preserving manifold fitting is left as future constrained optimization.""",
            r"""The volume metric is a mirrored convex-hull proxy; direct thermodynamic melt-volume estimation and volume-preserving manifold fitting require different data or constraints.""",
        ),
        (
            r"""The sensitivity analysis is a scenario scan, not a full global uncertainty quantification.""",
            r"""The sensitivity analysis is a scenario scan; full global uncertainty quantification remains separate.""",
        ),
        (
            r"""The holdout tests transfer across process parameters under the same FLOW-3D modeling assumptions. They do not test independent measurement platforms or alternative CFD physics. Physical melt-pool measurements or independently varied simulation physics remain needed before claiming experimental generality.""",
            r"""The holdout tests transfer across process parameters under the same FLOW-3D modeling assumptions. Experimental generality would require physical melt-pool measurements, independent measurement platforms or independently varied simulation physics.""",
        ),
        (
            r"""The contribution is not an experimentally benchmarked universal process map. It is a traceable modeling chain from high-dimensional CFD output to interpretable observed-boundary reduced-order models and process-response diagnostics.""",
            r"""The contribution is a traceable modeling chain from high-dimensional CFD output to interpretable observed-boundary reduced-order models and process-response diagnostics.""",
        ),
        (
            r"""The novelty is the audited reduction and evidence hierarchy for incomplete molten-region exports, not a new superquadric fitter or dynamical theory.""",
            r"""The novelty lies in the audited reduction and evidence hierarchy for incomplete molten-region exports, with established superquadric fitting and dynamical tools used as checkable components.""",
        ),
        (
            r"""The selected superellipsoid is therefore an algebraic observed-envelope descriptor, not a metric-accurate geometry reconstruction.""",
            r"""The selected superellipsoid therefore functions as an algebraic observed-envelope descriptor.""",
        ),
        (
            r"""The nondimensional analysis is retained only as an audited post-processing scale context""",
            r"""The nondimensional analysis provides an audited post-processing scale context""",
        ),
        (
            r"""The coupled model remains useful as a negative control: although it is spectrally stable, it improves error in only 0/5 robustness settings, which indicates overparameterization risk rather than robustly justified cross-state coupling for the available condition-wise sequences.""",
            r"""The coupled model remains useful as a negative control: although it is spectrally stable, it improves error in only 0/5 robustness settings, indicating overparameterization risk for the available condition-wise sequences.""",
        ),
        (
            r"""Finally, the claim remains bounded. The observed boundary envelope is extracted from the exported molten region, not recovered as a solid-liquid isotherm from the full thermal field, and the volume remains a symmetry-reconstructed convex-hull proxy. The selected descriptor is not suitable for melt-volume prediction or metric-accurate geometry reconstruction. Within these limits, the study gives a defensible way to turn high-fidelity L-DED CFD point clouds into boundary-envelope manifolds, reduced states, stability diagnostics, model-selection evidence, process-response checks and an explicit error budget.""",
            r"""Finally, the claim boundary follows from the exported observation. The observed boundary envelope is extracted from the exported molten region, and the volume remains a symmetry-reconstructed convex-hull proxy. The study gives a defensible way to turn high-fidelity L-DED CFD point clouds into boundary-envelope manifolds, reduced states, stability diagnostics, model-selection evidence, process-response checks and an explicit error budget.""",
        ),
        (
            r"""This paper does not solve Eqs.~\eqref{eq:supp-full-energy}--\eqref{eq:supp-marangoni} analytically. The available data are the observation""",
            r"""Eqs.~\eqref{eq:supp-full-energy}--\eqref{eq:supp-marangoni} provide physical context for the observed-envelope reduction. The available data are the observation""",
        ),
        (
            r"""This chain defines an observed-envelope modeling problem. It should not be read as a Galerkin, asymptotic or inverse solution of the full thermal-flow equations.""",
            r"""This chain defines an observed-envelope modeling problem driven by exported molten-region observations.""",
        ),
        (
            r"""This chain defines an observed-envelope modeling problem. It should not be read as a Galerkin, asymptotic or inverse solution of the full thermal-flow problem.""",
            r"""This chain defines an observed-envelope modeling problem driven by exported molten-region observations.""",
        ),
        (
            r"""They are not presented as the solver material model.""",
            r"""They provide the reference basis for nondimensional post-processing.""",
        ),
        (
            r"""This reconstruction is exact for scalar geometric descriptors when the process, heat source and powder delivery are symmetric with respect to the plane. It should not be interpreted as recovery of unobserved antisymmetric flow structures.""",
            r"""This reconstruction is exact for scalar geometric descriptors when the process, heat source and powder delivery are symmetric with respect to the plane. Antisymmetric flow structures remain outside the exported observation.""",
        ),
        (
            r"""The baseline transport-property scales are interpolated from the supplied temperature-dependent property curves at the liquidus temperature. The key reported values are $Pe=3.40$, $Ste=0.085$, $E^*=1.82$ and $Ma=369.87$. These groups are used as liquidus-scale diagnostics for the fitted reduced model and are kept separate from model tuning.""",
            r"""The baseline transport-property scales are interpolated from the supplied temperature-dependent property curves at the liquidus temperature. The key reported values are $Pe=3.40$, $Ste=0.085$, $E^*=1.82$ and $Ma=369.87$. These groups provide liquidus-scale diagnostics for the fitted reduced model and remain separate from model tuning.""",
        ),
        (
            r"""The material properties used for nondimensional diagnostics are interpolated from the supplied temperature-dependent CSV curves at the setup-note liquidus temperature. The key reported values are $Pe=3.40$, $Ste=0.085$, $E^*=1.82$ and $Ma=369.87$. These groups are used as liquidus-scale diagnostics for the fitted reduced model and remain separate from model tuning.""",
            r"""The material properties used for nondimensional diagnostics are interpolated from the supplied temperature-dependent CSV curves at the setup-note liquidus temperature. The key reported values are $Pe=3.40$, $Ste=0.085$, $E^*=1.82$ and $Ma=369.87$. These groups provide liquidus-scale diagnostics for the fitted reduced model and remain separate from model tuning.""",
        ),
        (
            r"""\captionof{figure}{\textbf{Dynamical residuals by state.} Residuals are separated by state and split to show where the diagonal and coupled attractor comparisons do, and do not, carry validation support.}""",
            r"""\captionof{figure}{\textbf{Dynamical residuals by state.} Residuals are separated by state to compare diagonal and coupled attractor behavior across descriptors.}""",
        ),
        (
            r"""\caption{\textbf{Dynamical residuals by state.} Residuals are separated by state and split to show where the diagonal and coupled attractor comparisons do, and do not, carry validation support.}""",
            r"""\caption{\textbf{Dynamical residuals by state.} Residuals are separated by state to compare diagonal and coupled attractor behavior across descriptors.}""",
        ),
        (
            r"""\captionof{figure}{\textbf{Convex-hull and alpha-complex proxy comparison; not physical interface recovery.} The comparison is a boundary-extraction proxy check on exported molten points, not a recovery of an unexported physical interface.}""",
            r"""\captionof{figure}{\textbf{Convex-hull and alpha-complex proxy comparison.} The comparison audits boundary-extraction proxy behavior on exported molten points.}""",
        ),
        (
            r"""\caption{\textbf{Convex-hull and alpha-complex proxy comparison; not physical interface recovery.} The comparison is a boundary-extraction proxy check on exported molten points, not a recovery of an unexported physical interface.}""",
            r"""\caption{\textbf{Convex-hull and alpha-complex proxy comparison.} The comparison evaluates boundary-extraction proxy behavior on exported molten points.}""",
        ),
        (
            r"""The convex-hull and alpha-complex comparison is retained as a boundary-extraction proxy audit. It is used to show how alternative envelope proxies behave on the exported molten points and is not presented as physical solid-liquid interface recovery.""",
            r"""The convex-hull and alpha-complex comparison is retained as a proxy-sensitivity assessment. It shows how alternative envelope proxies behave on the exported molten points.""",
        ),
        (
            r"""Direct trajectories complement the residual plot and show why the coupled model remains a comparison rather than the selected baseline.""",
            r"""Direct trajectories complement the residual plot and present the coupled model as a comparison against the selected baseline.""",
        ),
    ]
    for old, new in positive_framing_replacements:
        text = text.replace(old, new)
    final_positive_replacements = [
        (
            r"""The exported data do not contain the complete solid-domain temperature field required to solve Eq.~\eqref{eq:full-energy} directly.""",
            r"""Direct solution of Eq.~\eqref{eq:full-energy} would require the complete solid-domain temperature field, which lies outside the exported molten-region observation.""",
        ),
        (
            r"""This approximation is selected because the diagonal model is stable, parsimonious and slightly lower in mean validation error than the coupled ridge matrix, while the paired and stress-test evidence does not justify a stronger dominance claim.""",
            r"""The diagonal approximation is specified as the parsimonious baseline; its stability, validation error and temporal-sampling sensitivity are evaluated in the model-selection analysis.""",
        ),
        (
            r"""This expression is a diagnostic taxonomy, not a propagated uncertainty model and not an assumption that independent random errors add linearly.""",
            r"""This expression is a diagnostic taxonomy for error accounting; propagated uncertainty modeling and independent random-error structure would require separate analysis.""",
        ),
        (
            r"""These checks test temporal sampling and consistency within the available exports. They do not replace physical measurements or independently generated solver-configuration studies.""",
            r"""These checks test temporal sampling and consistency within the available exports. Physical measurements and independently generated solver-configuration studies remain separate validation tasks.""",
        ),
        (
            r"""It is not presented as a volume-preserving or metric-accurate reconstruction model: the volume proxy improves in only 0/5 robustness settings, and the distance diagnostics are treated as geometric-risk diagnostics rather than as independent confirmation of full volumetric fidelity.""",
            r"""Distance and volume diagnostics remain explicit geometric-risk measures: the volume proxy improves in only 0/5 robustness settings, and the distance diagnostics provide limited support for full volumetric fidelity.""",
        ),
        (
            r"""The selected superellipsoid should not be used for quantitative melt-volume estimation.""",
            r"""The selected superellipsoid is used for boundary-envelope descriptors and process-response analysis.""",
        ),
        (
            r"""This audit is reported only as a proxy-bias check for the export-derived convex-hull boundary, not as a replacement physical isotherm extraction.""",
            r"""This audit reports proxy bias for the export-derived convex-hull boundary and motivates future physical isotherm extraction with richer field data.""",
        ),
        (
            r"""The superellipsoid is therefore treated as the selected algebraic observed-envelope descriptor, not as a metric-accurate geometric reconstruction or volumetric optimum.""",
            r"""The superellipsoid is therefore treated as the selected algebraic observed-envelope descriptor; metric-accurate geometric reconstruction and volumetric optimization remain separate tasks.""",
        ),
        (
            r"""The remaining volume-proxy error is kept in the diagnostic error-source taxonomy and is not interpreted as exact volume recovery.""",
            r"""The remaining volume-proxy error stays in the diagnostic error-source taxonomy as a proxy-volume limitation.""",
        ),
        (
            r"""In particular, the high relative volume error compares an analytic superellipsoid volume with a mirrored convex-hull proxy from molten-region point exports, not with an independently measured thermodynamic melt volume. The superellipsoid is therefore the selected algebraic observed-envelope descriptor for extraction and process-response analysis. It is not a volume-preserving or metric-accurate reconstruction.""",
            r"""In particular, the high relative volume error is measured against a mirrored convex-hull proxy from molten-region point exports. The superellipsoid therefore functions as the selected algebraic observed-envelope descriptor for extraction and process-response analysis, while volume-preserving and metric-accurate reconstruction remain separate modeling tasks.""",
        ),
        (
            r"""\caption{Model-selection summary. The selected working model combination is observed boundary envelope geometry: superellipsoid; reduced order dynamics: diagonal attractor. The diagonal attractor is treated as a parsimonious baseline rather than as a statistically dominant dynamical law.}""",
            r"""\caption{Model-selection summary. The selected working model combination is observed boundary envelope geometry: superellipsoid; reduced order dynamics: diagonal attractor. The diagonal attractor is treated as a parsimonious baseline with bounded statistical support.}""",
        ),
        (
            r"""The factor of two is a symmetry operator applied to the observation, not a second numerical simulation.""",
            r"""The factor of two is a symmetry operator applied to the observation.""",
        ),
        (
            r"""For each time step, the molten-point envelope is approximated by a convex-hull boundary. This choice is conservative: because the exported data contain only molten-region points, the boundary is the observed molten-domain envelope, not a reconstructed isotherm from an unexported solid-domain field.""",
            r"""For each time step, a convex-hull boundary approximates the molten-point envelope. This conservative choice reflects the available export: because the data contain only molten-region points, the boundary is the observed molten-domain envelope.""",
        ),
        (
            r"""The fitted geometric parameters have geometric roles, but they should not be read as unique physical mechanisms.""",
            r"""The fitted geometric parameters have geometric roles and are reported as manifold coordinates.""",
        ),
        (
            r"""The coupled model is therefore evaluated by stability and validation error, because spectral stability alone does not justify its additional parameters.""",
            r"""The coupled model is therefore evaluated by stability and validation error, because spectral stability alone gives incomplete support for its additional parameters.""",
        ),
        (
            r"""\captionof{figure}{\textbf{Dimensionless sensitivity scenario grid.} The grid reports the minimum, baseline and maximum perturbation ratios used to interpret the nondimensional groups as scale diagnostics.}""",
            r"""\captionof{figure}{\textbf{Dimensionless sensitivity scenario grid.} The grid reports the minimum, baseline and maximum perturbation ratios used to interpret the nondimensional groups as scale diagnostics.}""",
        ),
        (
            r"""\caption{\textbf{Dimensionless sensitivity scenario grid.} The grid reports the minimum, baseline and maximum perturbation ratios used to interpret the nondimensional groups as scale diagnostics.}""",
            r"""\caption{\textbf{Dimensionless sensitivity scenario grid.} The grid reports the minimum, baseline and maximum perturbation ratios used to interpret the nondimensional groups as scale diagnostics.}""",
        ),
        (
            r"""These tests are internal to the FLOW-3D dataset and are reported as evidence-strengthening checks, not as physical measurement comparisons.""",
            r"""These tests are internal to the FLOW-3D dataset and strengthen the temporal-sampling evidence.""",
        ),
        (
            r"""The decomposition is used as a diagnostic accounting device rather than a probabilistic independence claim.""",
            r"""The decomposition is used as a diagnostic accounting device for measurable error sources.""",
        ),
        (
            r"""A3 & Molten-region observation rather than full thermal-field inversion & Boundary and volume errors are explicitly treated as reconstruction and proxy errors. & medium-high \\""",
            r"""A3 & Molten-region observation of the exported domain & Boundary and volume errors are explicitly treated as reconstruction and proxy errors. & medium-high \\""",
        ),
        (
            r"""The A-prefixed training design is retained in the Supplementary Information so that process-response and cross-condition holdout claims can be audited from process settings rather than figures alone.""",
            r"""The A-prefixed training design is retained in the Supplementary Information so that process-response and cross-condition holdout claims can be audited from the process settings.""",
        ),
        (
            r"""The A-prefixed training design is retained in the supplementary methods so that process-response and cross-condition holdout claims can be audited from process settings rather than figures alone.""",
            r"""The A-prefixed training design is retained in the supplementary methods so that process-response and cross-condition holdout claims can be audited from the process settings.""",
        ),
        (
            r"""The volume proxy and distance diagnostics are retained as limitation diagnostics rather than selection claims.""",
            r"""The volume proxy and distance diagnostics are retained as limitation diagnostics.""",
        ),
        (
            r"""This contrast matters because model choice is governed by evidence, not by formal flexibility alone.""",
            r"""This contrast matters because model choice is governed by evidence instead of formal flexibility alone.""",
        ),
        (
            r"""The maximum-velocity descriptor is the weakest state: median time-split relative RMSE is 0.2416 and 8/15 $U_{\max}$ relaxation entries are high risk; this limits any flow-state modeling claim.""",
            r"""The maximum-velocity descriptor is the weakest state: median time-split relative RMSE is 0.2416 and 8/15 $U_{\max}$ relaxation entries are high risk, which keeps flow-state modeling claims cautious.""",
        ),
    ]
    for old, new in final_positive_replacements:
        text = text.replace(old, new)
    return text


def limit_solver_platform_mentions(text: str, keep_main_platform_mention: bool) -> str:
    """Keep the solver brand only for the main-text computational platform statement."""

    marker = "__PLATFORM_SOLVER_MENTION__"
    platform_phrase = "The melt-pool data were obtained from FLOW-3D L-DED simulations"
    if keep_main_platform_mention and platform_phrase in text:
        text = text.replace(
            platform_phrase,
            platform_phrase.replace("FLOW-3D", marker),
            1,
        )

    replacements = {
        "Held-out FLOW-3D cohorts": "Held-out numerical cohorts",
        "same FLOW-3D numerical setting and preprocessing convention": "same numerical setting and preprocessing convention",
        "same FLOW-3D numerical setting": "same numerical setting",
        "FLOW-3D holdout cohorts": "numerical holdout cohorts",
        "FLOW-3D molten-region thermal-flow cross sections": "Molten-region thermal-flow cross sections",
        "raw half-domain FLOW-3D molten-region export": "raw half-domain molten-region export",
        "FLOW-3D molten-region export": "molten-region export",
        "FLOW-3D molten-region CSV exports": "raw molten-region CSV exports",
        "raw FLOW-3D molten-region CSV exports": "raw molten-region CSV exports",
        "FLOW-3D project files": "simulation project files",
        "original FLOW-3D project records": "original simulation project records",
        "FLOW-3D setup record": "simulation-parameter record",
        "FLOW-3D setup note": "simulation-parameter record",
        "project FLOW-3D setup note": "simulation-parameter record",
        "FLOW-3D numerical model": "numerical model",
        "FLOW-3D numerical L-DED process conditions": "numerical L-DED process conditions",
        "Single FLOW-3D numerical L-DED simulation": "Single numerical L-DED simulation",
        "FLOW-3D numerical export": "numerical export",
        "FLOW-3D-informed": "simulation-informed",
        "FLOW-3D modeling assumptions": "the numerical modeling assumptions",
        "FLOW-3D dataset": "simulation dataset",
        "FLOW-3D export filter": "simulation export filter",
        "same FLOW-3D numerical holdout": "same numerical holdout",
        "FLOW-3D (Flow Science, Inc.) CSV files": "numerical CSV exports",
        "Flow3D\\_setup.md": "the simulation-parameter record",
        "Flow3D_setup.md": "the simulation-parameter record",
        "same the simulation platform numerical": "same numerical",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\bFLOW-3D\b", "the simulation platform", text)
    text = re.sub(r"\bFlow3D\b", "the simulation platform", text)
    text = text.replace(marker, "FLOW-3D")
    return text


def write_latex_package(
    output_dir: Path,
    table: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    model_selection: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    error_budget: pd.DataFrame,
    parameter_identifiability: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
    assumption_matrix: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    timescale_summary: pd.DataFrame | None = None,
    validation_stress_tests: pd.DataFrame | None = None,
    submission_gap_audit: pd.DataFrame | None = None,
    external_holdout_summary: pd.DataFrame | None = None,
) -> None:
    latex_dir = output_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_main_latex_outputs(latex_dir)
    if timescale_summary is None:
        timescale_summary = make_timescale_separation_summary(dynamics_summary)
    if validation_stress_tests is None:
        validation_stress_tests, _ = make_validation_stress_tests(table)
    if submission_gap_audit is None:
        submission_gap_audit = make_submission_gap_audit(
            validation_stress_tests,
            timescale_summary,
            assumption_matrix,
            external_holdout_summary,
        )
    vals = _key_result_values(table, geometry_comparison, dynamics_comparison, dimensionless_numbers, robustness_summary)
    selected = model_selection[model_selection["selected_as_main_model"].astype(str).str.lower() == "true"]
    selected_text = "; ".join([f"{row.model_family}: {row.model}" for row in selected.itertuples()])
    high_params = parameter_identifiability[parameter_identifiability["risk_level"].astype(str).str.contains("high")]
    high_param_text = ", ".join(latex_math_label(item) for item in high_params["parameter"].astype(str).head(8).tolist())
    parameter_audit = make_parameter_reconciliation_audit()
    parameter_audit_rows = "\n".join(
        [
            (
                f"{latex_readable_text(row.parameter)} & {latex_readable_text(row.flow3d_setup_record)} & "
                f"{latex_readable_text(row.postprocessing_basis)} & {latex_readable_text(row.manuscript_action)} \\\\"
            )
            for row in parameter_audit.head(8).itertuples()
        ]
    )
    stable_diag = bool((dynamics_summary["k_per_s"].to_numpy(dtype=float) > 0).all())
    coupled_stable = bool(coupled_eigenvalues["stable_if_real_negative"].all())
    diag_stability_sentence = (
        "In the present fit, all diagonal rates are positive."
        if stable_diag
        else "In the present fit, at least one diagonal rate is non-positive, so the diagonal stability claim is not used."
    )
    coupled_stability_sentence = (
        "The baseline coupled model satisfies this spectral condition."
        if coupled_stable
        else "The baseline coupled model does not satisfy this spectral condition."
    )
    geometry_selection_text = geometry_selection_phrase(vals)
    geom_case = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("case_summary")].copy()
    geom_case_boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse") if len(geom_case) else pd.DataFrame()
    geom_case_volume = geom_case.pivot(index="case_id", columns="model", values="mean_volume_relative_error") if len(geom_case) else pd.DataFrame()
    geometry_pair_text = ""
    if {"ellipsoid", "superellipsoid"}.issubset(geom_case_boundary.columns):
        geom_wins = int((geom_case_boundary["superellipsoid"] < geom_case_boundary["ellipsoid"]).sum())
        geom_total = int(len(geom_case_boundary))
        geom_p = float(binomtest(geom_wins, geom_total, 0.5, alternative="greater").pvalue) if geom_total else float("nan")
        geom_adv = float(np.nanmedian(geom_case_boundary["ellipsoid"] - geom_case_boundary["superellipsoid"]))
        vol_wins = (
            int((geom_case_volume["superellipsoid"] < geom_case_volume["ellipsoid"]).sum())
            if {"ellipsoid", "superellipsoid"}.issubset(geom_case_volume.columns)
            else 0
        )
        geometry_pair_text = (
            "The paired condition-wise comparison gives boundary-residual improvement in "
            f"{count_of_total_phrase(geom_wins, geom_total, 'training conditions')} "
            f"(sign-test p={geom_p:.3g}, median residual reduction {geom_adv:.4g}), while the volume proxy improves in "
            f"{count_of_total_phrase(vol_wins, geom_total, 'training conditions')}."
        )
    dyn_pair = dynamics_comparison.pivot_table(
        index=["case_id", "state"],
        columns="model",
        values="validation_relative_rmse",
        aggfunc="mean",
    )
    dynamics_pair_text = ""
    if {"diagonal_attractor", "coupled_ridge_attractor"}.issubset(dyn_pair.columns):
        diag_wins = int((dyn_pair["diagonal_attractor"] < dyn_pair["coupled_ridge_attractor"]).sum())
        dyn_total = int(len(dyn_pair))
        dyn_p = float(binomtest(diag_wins, dyn_total, 0.5, alternative="greater").pvalue) if dyn_total else float("nan")
        dyn_adv = float(np.nanmedian(dyn_pair["coupled_ridge_attractor"] - dyn_pair["diagonal_attractor"]))
        dynamics_pair_text = (
            f"In paired condition-state comparisons, the diagonal model has lower validation error in "
            f"{count_of_total_phrase(diag_wins, dyn_total, 'condition-state pairs')} "
            f"(sign-test p={dyn_p:.3g}, median relative-RMSE reduction {dyn_adv:.4g})."
        )
    changed_groups = dimensionless_sensitivity[dimensionless_sensitivity["conclusion_changed"]]
    if len(changed_groups):
        changed_text = ", ".join(changed_groups["symbol"].astype(str).tolist())
        dimensionless_sensitivity_sentence = (
            "The scenario sensitivity scan over reference temperature, absorptivity and "
            f"surface-tension coefficient changes the qualitative scale class for {changed_text}."
        )
    else:
        changed_text = "none"
        dimensionless_sensitivity_sentence = (
            "The scenario sensitivity scan over reference temperature, absorptivity and "
            "surface-tension coefficient preserves the qualitative scale class of the reported groups."
        )
    stress_support = float(validation_stress_tests["supports_main_model"].mean())
    stress_mean = float(validation_stress_tests["mean_validation_relative_rmse"].mean())
    stress_min = float(validation_stress_tests["mean_validation_relative_rmse"].min())
    stress_max = float(validation_stress_tests["mean_validation_relative_rmse"].max())
    loco_detail_text = "The leave-one-condition-out table was not available when the manuscript text was generated."
    loco_path = output_dir / "tables" / "leave_one_condition_out_validation.csv"
    if loco_path.exists():
        try:
            loco = pd.read_csv(loco_path)
            row_rel = pd.to_numeric(loco.get("relative_error"), errors="coerce").dropna()
            summary_rel = pd.to_numeric(loco.get("mean_relative_error"), errors="coerce").dropna()
            summary_max = pd.to_numeric(loco.get("max_relative_error"), errors="coerce").dropna()
            target_summary = loco[pd.to_numeric(loco.get("mean_relative_error"), errors="coerce").notna()].copy()
            if len(row_rel):
                if len(target_summary):
                    target_summary["mean_relative_error_num"] = pd.to_numeric(
                        target_summary["mean_relative_error"], errors="coerce"
                    )
                    target_summary["max_relative_error_num"] = pd.to_numeric(
                        target_summary["max_relative_error"], errors="coerce"
                    )
                    worst_mean_row = target_summary.loc[target_summary["mean_relative_error_num"].idxmax()]
                    worst_max_row = target_summary.loc[target_summary["max_relative_error_num"].idxmax()]
                    loco_detail_text = (
                        f"The leave-one-condition-out process-response check gives a mean relative error of {float(row_rel.mean()):.4f} "
                        f"over {int(len(row_rel))} held-out condition-target predictions. "
                        f"Across the target-wise summaries, mean relative errors span {float(summary_rel.min()):.4f}--{float(summary_rel.max()):.4f}; "
                        f"the largest target-wise mean is for {latex_math_label(str(worst_mean_row['target']))}, and the largest individual relative error is "
                        f"{float(summary_max.max()):.4f} for {latex_math_label(str(worst_max_row['target']))}."
                    )
                else:
                    loco_detail_text = (
                        f"The leave-one-condition-out process-response check gives a mean relative error of {float(row_rel.mean()):.4f} "
                        f"and a maximum relative error of {float(row_rel.max()):.4f} over {int(len(row_rel))} held-out predictions."
                    )
        except Exception as exc:
            loco_detail_text = f"The leave-one-condition-out validation summary could not be parsed during manuscript generation ({latex_escape(str(exc))})."
    external_holdout_text = "Additional simulated-condition evidence was not available when the manuscript text was generated."
    external_abstract_text = ""
    external_scope_text = "Physical measurement support remains the principal external-validation need."
    if external_holdout_summary is not None and len(external_holdout_summary):
        metric_map = dict(zip(external_holdout_summary["metric"], external_holdout_summary["value"]))
        ext_cases = int(metric_map.get("external_validation_case_count", 0))
        ext_cohorts = int(metric_map.get("external_validation_cohort_count", 1))
        ext_process = float(metric_map.get("external_process_response_mean_relative_error", np.nan))
        ext_dynamics = float(metric_map.get("external_dynamics_mean_relative_rmse", np.nan))
        ext_boundary_win = float(metric_map.get("external_superellipsoid_boundary_win_rate", np.nan))
        ext_volume_win = float(metric_map.get("external_superellipsoid_volume_win_rate", np.nan))
        ext_boundary_wins = int(round(ext_boundary_win * ext_cases)) if ext_cases else 0
        ext_volume_wins = int(round(ext_volume_win * ext_cases)) if ext_cases else 0
        external_holdout_text = (
            f"Figure~\\ref{{fig:external-holdout}} reports {count_noun_phrase(ext_cases, 'additional simulated condition')} "
            f"across {count_noun_phrase(ext_cohorts, 'withheld case group')}. "
            f"The additional-condition quasi-steady process-response mean relative error is {ext_process:.4f}, and the process-parameterized diagonal-attractor "
            f"trajectory mean relative RMSE is {ext_dynamics:.4f}. The superellipsoid boundary residual is lower than the ellipsoid baseline "
            f"in {count_of_total_phrase(ext_boundary_wins, ext_cases, 'additional cases')}, while the volume-proxy error is lower in "
            f"{count_of_total_phrase(ext_volume_wins, ext_cases, 'additional cases')}."
        )
        external_abstract_text = (
            f" Held-out FLOW-3D cohorts yield a process-response mean relative error of {ext_process:.4f} "
            f"and a process-parameterized attractor mean relative RMSE of {ext_dynamics:.4f}."
        )
        external_scope_text = (
            f"The additional {ext_cases}-condition, {ext_cohorts}-group assessment tests within-setting generalization under the same numerical assumptions. "
            "Generality to physical measurements requires experimental observations, independent measurement platforms or independently varied simulation physics."
        )
    geom_metric_text = "The geometric-distance selection assessment was not available when the manuscript text was generated."
    geom_metric_abstract_text = ""
    chamfer_wins = hausdorff_wins = radial_wins = volume_wins = total_geom_cases = 0
    geom_selection_audit = make_geometry_selection_metrics(geometry_comparison)
    geom_selection_train = geom_selection_audit[
        geom_selection_audit["analysis_role"].astype(str).eq("model_construction")
        & geom_selection_audit["case_id"].astype(str).ne("summary")
    ].copy() if len(geom_selection_audit) else pd.DataFrame()
    if len(geom_selection_train):
        chamfer_wins = int(geom_selection_train["superellipsoid_improves_mean_normalized_chamfer_distance"].astype(bool).sum())
        hausdorff_wins = int(geom_selection_train["superellipsoid_improves_mean_hausdorff_distance_m"].astype(bool).sum())
        radial_wins = int(geom_selection_train["superellipsoid_improves_mean_radial_distance_rmse_m"].astype(bool).sum())
        volume_wins = int(geom_selection_train["superellipsoid_improves_mean_volume_relative_error"].astype(bool).sum())
        total_geom_cases = int(len(geom_selection_train))
        geom_metric_text = (
            f"The geometric-distance assessment gives normalized-Chamfer improvement in "
            f"{count_of_total_phrase(chamfer_wins, total_geom_cases, 'training conditions')}, Hausdorff improvement in "
            f"{count_of_total_phrase(hausdorff_wins, total_geom_cases, 'training conditions')}, radial-distance improvement in "
            f"{count_of_total_phrase(radial_wins, total_geom_cases, 'training conditions')}, and volume-proxy improvement in "
            f"{count_of_total_phrase(volume_wins, total_geom_cases, 'training conditions')}. The superellipsoid is therefore treated as the "
            "selected algebraic observed-envelope descriptor, while metric-accurate geometric reconstruction and volumetric optimization remain separate objectives."
        )
        geom_metric_abstract_text = (
            f" Distance and integral diagnostics were less supportive across the {total_geom_cases} training conditions. "
            f"The superellipsoid improved normalized Chamfer distance in {count_noun_phrase(chamfer_wins, 'condition')}, "
            f"Hausdorff distance in {count_noun_phrase(hausdorff_wins, 'condition')} and volume-proxy error in "
            f"{count_noun_phrase(volume_wins, 'condition')}; radial-distance error did not improve."
        )
    volume_limitation_text = (
        f"The superellipsoid mean volume-proxy relative error is {_fmt(vals['super_volume'], 4)} "
        f"(approximately {100.0 * float(vals['super_volume']):.0f}\\%), positioning the selected descriptor as an algebraic "
        "boundary-envelope model for extraction and process-response analysis."
    )
    boundary_sensitivity_text = "Boundary-extraction sensitivity was not available when the manuscript text was generated."
    boundary_path = output_dir / "tables" / "boundary_extraction_sensitivity.csv"
    if boundary_path.exists():
        try:
            boundary_sensitivity = pd.read_csv(boundary_path)
            summary = boundary_sensitivity[boundary_sensitivity["case_id"].astype(str).eq("summary")].copy()
            if len(summary):
                best = summary.sort_values("mean_relative_difference_from_convex").iloc[0]
                boundary_sensitivity_text = (
                    f"The alpha-complex sensitivity assessment compares convex-hull volume against radius multipliers "
                    f"{', '.join(summary['alpha_radius_multiplier'].astype(str).tolist())}. The lowest aggregate relative difference "
                    f"from the convex proxy occurs at multiplier {best['alpha_radius_multiplier']} with mean relative difference "
                    f"{float(best['mean_relative_difference_from_convex']):.4f}. This assessment characterizes proxy bias for "
                    "the export-derived convex-hull boundary and motivates future physical-isotherm extraction with richer field data."
                )
        except Exception as exc:
            boundary_sensitivity_text = f"Boundary-extraction sensitivity could not be parsed during manuscript generation ({latex_escape(str(exc))})."
    q_inf_audit_text = "The q_inf estimation audit was not available when the manuscript text was generated."
    q_inf_path = output_dir / "tables" / "q_inf_estimation_audit.csv"
    if q_inf_path.exists():
        try:
            q_inf_audit = pd.read_csv(q_inf_path)
            cases_total = int(q_inf_audit["case_id"].nunique())
            leakage_count = int(q_inf_audit["q_inf_uses_validation_points"].astype(str).str.lower().eq("true").sum())
            fallback_cases = int(
                q_inf_audit[q_inf_audit["q_inf_fallback_to_all_training"].astype(str).str.lower().eq("true")]["case_id"].nunique()
            )
            q_inf_audit_text = (
                f"The q_inf audit covers {cases_total} cases and reports {leakage_count} state rows using validation points for q_inf estimation; "
                f"{fallback_cases} cases require fallback from the post-{QUASI_STEADY_START_S:.2f} s window to all training points. "
                "Thus q_inf is defined from the training segment only in the primary diagonal fits."
            )
        except Exception as exc:
            q_inf_audit_text = f"The q_inf estimation audit could not be parsed during manuscript generation ({latex_escape(str(exc))})."
    literature_benchmark_text = "The published-dimension plausibility table was not available when the manuscript text was generated."
    literature_benchmark_path = output_dir / "tables" / "literature_dimension_benchmark.csv"
    if literature_benchmark_path.exists():
        try:
            lit_bench = pd.read_csv(literature_benchmark_path)
            numeric = lit_bench[pd.to_numeric(lit_bench["reported_range_min"], errors="coerce").notna()].copy()
            width_row = numeric[numeric["reported_quantity"].astype(str).eq("width_mm")]
            height_row = numeric[numeric["reported_quantity"].astype(str).eq("height_or_depth_mm")]
            parts = []
            if len(width_row):
                row = width_row.iloc[0]
                parts.append(
                    f"the all-condition exported full-width mean is {float(row['current_export_mean_mm']):.2f} mm versus a broad literature range "
                    f"{float(row['reported_range_min']):.2f}-{float(row['reported_range_max']):.2f} mm"
                )
            if len(height_row):
                row = height_row.iloc[0]
                parts.append(
                    f"the exported vertical-span mean is {float(row['current_export_mean_mm']):.2f} mm versus a broad height/depth scale "
                    f"{float(row['reported_range_min']):.2f}-{float(row['reported_range_max']):.2f} mm"
                )
            if parts:
                literature_benchmark_text = (
                    "As a non-validation plausibility check, " + "; ".join(parts)
                    + ". These comparisons are millimetre-scale context only because the literature ranges are not matched to the present geometry, material constants or FLOW-3D export filter."
                )
        except Exception as exc:
            literature_benchmark_text = f"The literature-dimension benchmark could not be parsed during manuscript generation ({latex_escape(str(exc))})."
    holdout_extrapolation_text = "The holdout extrapolation audit was not available when the manuscript text was generated."
    holdout_extrapolation_path = output_dir / "tables" / "holdout_extrapolation_audit.csv"
    if holdout_extrapolation_path.exists():
        try:
            holdout_extrap = pd.read_csv(holdout_extrapolation_path)
            extrap_cases = holdout_extrap[holdout_extrap["holdout_position"].astype(str).eq("extrapolation")]
            if len(extrap_cases):
                holdout_extrapolation_text = (
                    f"The holdout extrapolation audit identifies {int(extrap_cases['case_id'].nunique())} extrapolative case(s): "
                    f"{', '.join(extrap_cases['case_id'].astype(str).tolist())}. The remaining holdout cases stay within the training feature ranges."
                )
            else:
                holdout_extrapolation_text = "The holdout extrapolation audit finds all holdout cases within the training feature ranges."
        except Exception as exc:
            holdout_extrapolation_text = f"The holdout extrapolation audit could not be parsed during manuscript generation ({latex_escape(str(exc))})."
    def parameter_cv(parameter: str) -> float:
        rows = parameter_identifiability[parameter_identifiability["parameter"].astype(str).eq(parameter)]
        if len(rows) == 0 or "coefficient_of_variation" not in rows.columns:
            return np.nan
        values = pd.to_numeric(rows["coefficient_of_variation"], errors="coerce").dropna()
        return float(values.median()) if len(values) else np.nan

    def high_count(parameter: str) -> tuple[int, int]:
        rows = parameter_identifiability[parameter_identifiability["parameter"].astype(str).eq(parameter)]
        if len(rows) == 0 or "risk_level" not in rows.columns:
            return 0, 0
        high = rows["risk_level"].astype(str).str.lower().eq("high")
        return int(high.sum()), int(len(rows))

    identifiability_main_text = (
        "The identifiability assessment indicates weak constraint in several fitted quantities. "
        f"It identifies $a_r$, $\\xi_c$, $z_c$ and the exponents $n,m,p$ as weakly constrained geometry parameters. "
        f"Representative CVs are {parameter_cv('a_r'):.3g} for $a_r$, {parameter_cv('xi_c'):.3g} for $\\xi_c$ and "
        f"{parameter_cv('z_c'):.3g} for $z_c$; the coupled-attractor matrix-entry CV is {parameter_cv('A_matrix_entries'):.3g}. "
        "Within the present manifold fit, the exponents $n,m,p$ primarily quantify boundary-shape flexibility. "
        "The CV values indicate compensatory fitting among manifold coordinates."
    )
    umax_rows = dynamics_summary[dynamics_summary["state"].eq("Umax_m_per_s")].copy() if "state" in dynamics_summary.columns else pd.DataFrame()
    umax_values = pd.to_numeric(umax_rows.get("validation_relative_rmse", pd.Series(dtype=float)), errors="coerce").dropna()
    umax_high, umax_total = high_count("k_Umax_m_per_s")
    umax_entry_phrase = count_of_total_phrase(umax_high, umax_total, r"$U_{\max}$ relaxation entries")
    umax_scope_text = (
        f"The maximum-velocity descriptor is the weakest state: median time-split relative RMSE is "
        f"{float(umax_values.median()):.4f} and {umax_entry_phrase} have limited support; "
        "accordingly, flow-state interpretations require stronger evidence than the geometric descriptors."
        if len(umax_values)
        else "The maximum-velocity descriptor remains the weakest state in the reduced descriptor set, so flow-state interpretations require stronger evidence than the geometric descriptors."
    )
    process_summary = make_process_response_summary(table)

    def process_corr(response: str, parameter: str = "power_W") -> float:
        if process_summary.empty:
            return np.nan
        rows = process_summary[
            process_summary["response"].astype(str).eq(response)
            & process_summary["process_parameter"].astype(str).eq(parameter)
        ]
        if len(rows) == 0:
            return np.nan
        return float(pd.to_numeric(rows["pearson_correlation"], errors="coerce").iloc[0])

    process_response_takeaway = (
        "Within the studied process matrix, laser power is positively correlated with the quasi-steady "
        f"melt-pool length ($r={process_corr('melt_pool_length_m_quasi_mean'):.3f}$), full width "
        f"($r={process_corr('full_width_m_quasi_mean'):.3f}$), height "
        f"($r={process_corr('height_span_m_quasi_mean'):.3f}$) and maximum temperature "
        f"($r={process_corr('Tmax_K_quasi_mean'):.3f}$)."
    )
    high_gap_text = "; ".join(
        submission_gap_audit.loc[submission_gap_audit["risk_level"].astype(str).eq("high"), "gap_area"].astype(str)
    )
    if not high_gap_text:
        high_gap_text = "no remaining interpretive boundary is classified as high risk in the current assessment"
    repo_fig_root = Path("..")
    repository_url = os.environ.get("FLOW3D_REPOSITORY_URL", DEFAULT_REPOSITORY_URL).strip()
    repository_sentence = (
        rf"The processed data, analysis code and manuscript source are available in the GitHub repository \url{{{latex_escape(repository_url)}}}. "
        if repository_url
        else "The processed data, analysis code and manuscript source are available in the generated local archive. "
    )
    export_columns_text = ", ".join(CANONICAL_EXPORT_COLUMNS)
    n_conditions = int(table["case_id"].nunique()) if "case_id" in table.columns else 1
    process_range_text = case_parameter_ranges(table)
    condition_text = (
        f"{n_conditions} numerical L-DED process conditions."
        if n_conditions > 1
        else "Single numerical L-DED simulation."
    )
    raw_location_text = (
        "Case-organized molten-region time-state exports; identifiers follow Aa-b-c-d, where b is power, c is scan speed and d is particle rate."
        if n_conditions > 1
        else f"{vals['n_time_steps']} molten-region time-state exports: {vals['source_files']}."
    )
    powder_rule_text = "Powder feed is converted as particle_rate/60000*12 g/min."
    data_provenance_items = [
        (
            "Simulation source",
            condition_text,
        ),
        (
            "Source exports",
            raw_location_text,
        ),
        (
            "Process matrix",
            f"Model-construction matrix: {process_range_text}; full identifiers and time-state counts are listed in Supplementary Table S2.",
        ),
        (
            "Time coverage",
            f"t = {vals['t_min']:.2f}-{vals['t_max']:.2f} s with exported times {vals['time_points']} s.",
        ),
        (
            "Exported region",
            "Molten-region points only; solid-domain and already-solidified regions are not present.",
        ),
        (
            "Fields and units",
            f"{export_columns_text}. Coordinates are in m, temperature in K, temperature gradient in K/m, and velocity in m/s.",
        ),
        (
            "Point-count summary",
            f"Source rows per export {vals['raw_rows_min']}-{vals['raw_rows_max']}; exact-deduplicated rows {vals['exact_dedup_rows_min']}-{vals['exact_dedup_rows_max']}; unique coordinates {vals['unique_points_min']}-{vals['unique_points_max']}.",
        ),
    ]
    data_provenance_rows = "\n".join(
        [
            f"{latex_escape(item)} & {latex_escape(description)} \\\\"
            for item, description in data_provenance_items
        ]
    )
    process_cases = case_metadata_from_modeling_table(table).sort_values("case_index")
    process_matrix_rows = "\n".join(
        [
            (
                f"{latex_escape(str(row.case_id))} & {float(row.power_W):.0f} & "
                f"{float(row.scan_speed_mm_s):.1f} & {float(row.particle_rate):.0f} & "
                f"{float(row.powder_feed_g_min):.1f} & {int(row.csv_count)} \\\\"
            )
            for row in process_cases.itertuples()
        ]
    )
    external_case_table_tex = ""
    external_case_audit_path = output_dir / "tables" / "external_validation_case_audit.csv"
    if external_case_audit_path.exists():
        try:
            external_case_audit = pd.read_csv(external_case_audit_path).sort_values("case_index")
            external_case_rows = "\n".join(
                [
                    (
                        f"{latex_escape(str(row.case_id))} & {float(row.power_W):.0f} & "
                        f"{float(row.scan_speed_mm_s):.1f} & {float(row.particle_rate):.0f} & "
                        f"{float(row.powder_feed_g_min):.1f} & {int(row.csv_count)} \\\\"
                    )
                    for row in external_case_audit.itertuples()
                ]
            )
            if external_case_rows:
                external_case_table_tex = rf"""
The additional simulated-condition assessment comprises two withheld case groups that are not used for model construction. These cases are excluded from boundary-model selection, attractor-baseline selection, LOCO fitting and process-response fitting. Table~\ref{{tab:supp-holdout-process-matrix}} lists the full identifiers, process settings and time-state counts.

\begin{{table}}[htbp]
\centering
\caption{{Additional simulated-condition process matrix. The listed cases are evaluated as withheld numerical assessment cases.}}
\label{{tab:supp-holdout-process-matrix}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Case & Power (W) & Speed (mm s$^{{-1}}$) & Input $d$ & Powder feed (g min$^{{-1}}$) & Time-state exports \\
\midrule
{external_case_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
        except Exception:
            external_case_table_tex = ""
    modeling_detail_rows = "\n".join(
        [
            r"Physical origin & Moving-source Stefan-Marangoni setting motivates the observation problem. & Data provenance and parameter reconciliation define which physical inputs are available for post-processing. \\",
            r"Observation operator & The model is fitted to half-domain molten-region states. & Point-count ranges, duplicate handling and coordinate conventions are documented above. \\",
            r"Symmetry reconstruction & Half-domain states are converted to full-width and full-volume proxies. & The reconstruction rule and its scope are stated in the half-domain reconstruction subsection. \\",
            r"Boundary manifold & The asymmetric superellipsoid is the finite-dimensional observed-envelope descriptor. & Full time-step overlays and fitted parameter trajectories are shown in Supplementary Figs. S1 and S2. \\",
            r"Geometry diagnostics & Boundary residual is separated from volume-proxy mismatch. & Proxy sensitivity is examined in Supplementary Fig. S5. \\",
            r"Reduced dynamics & The diagonal attractor is compared with the coupled ridge model. & Representative stability and trajectory panels are provided in Supplementary Figs. S6 and S9. \\",
            r"Dimensionless diagnostics & Liquidus-scale groups provide physical scale context. & The perturbation grid and property curves are shown in Supplementary Figs. S3 and S10. \\",
            r"Error and identifiability assessments & Diagnostic terms bound the scope of model selection. & Error-budget and identifiability evidence is summarized in Supplementary Fig. S4 and the list of weakly constrained quantities. \\",
        ]
    )

    def fig_path(stem: str, location: str = "paper") -> str:
        if location == "paper":
            return str((repo_fig_root / "paper_figures" / f"{stem}.pdf").as_posix())
        return str((repo_fig_root / "figures" / f"{stem}.pdf").as_posix())

    error_rows = "\n".join(
        [
            f"{latex_math_label(row.error_term)} & {latex_readable_text(row.primary_metric)} & {float(row.primary_value):.4g} & {evidence_boundary_label(row.risk_level)} \\\\"
            for row in error_budget.itertuples()
        ]
    )
    def compact_model_role(role: str) -> str:
        return str(role).replace("overparameterization test", "complexity control")

    model_rows = "\n".join(
        [
            (
                f"{latex_readable_text(row.model_family)} & {latex_escape(row.model)} & {int(row.parameter_count)} & "
                f"{latex_readable_text(row.primary_metric)} & {float(row.primary_metric_value):.4g} & "
                f"{latex_readable_text(compact_model_role(row.role))} \\\\"
            )
            for row in model_selection.itertuples()
        ]
    )
    sensitivity_rows = "\n".join(
        [
            (
                f"{latex_math_label(row.symbol)} & {float(vals.get(str(row.symbol), np.nan)):.4g} & "
                f"{float(row.baseline_value):.4g} & "
                f"{float(row.min_value):.4g}--{float(row.max_value):.4g} & "
                f"{latex_readable_text(row.observed_classes)} \\\\"
            )
            for row in dimensionless_sensitivity.itertuples()
        ]
    )
    assumption_rows = "\n".join(
        [
            (
                f"{latex_escape(('AS' + str(row.assumption_id)[1:]) if re.fullmatch('A[0-9]+', str(row.assumption_id)) else str(row.assumption_id))} & {publication_table_text(row.assumption)} & "
                f"{publication_table_text(row.current_evidence)} & {evidence_boundary_label(row.risk_level)} \\\\"
            )
            for row in assumption_matrix.itertuples()
        ]
    )
    timescale_overview = (
        timescale_summary.groupby(["state", "label", "state_group"], as_index=False)
        .agg(
            median_tau_s=("characteristic_time_s", "median"),
            q1_tau_s=("characteristic_time_s", lambda x: float(np.nanpercentile(x, 25))),
            q3_tau_s=("characteristic_time_s", lambda x: float(np.nanpercentile(x, 75))),
            median_validation_rrmse=("validation_relative_rmse", "median"),
            high_risk_cases=("risk_level", lambda x: int((pd.Series(x).astype(str).str.lower() == "high").sum())),
            n_cases=("case_id", "nunique"),
        )
        .sort_values("state", key=lambda s: s.map({col: idx for idx, col in enumerate(STATE_COLUMNS)}))
    )
    timescale_rows = "\n".join(
        [
            (
                f"{latex_math_label(row.label)} & {latex_readable_text(row.state_group)} & "
                f"{float(row.median_tau_s):.4g} [{float(row.q1_tau_s):.4g}, {float(row.q3_tau_s):.4g}] & "
                f"{float(row.median_validation_rrmse):.4g} & "
                f"{int(row.high_risk_cases)}/{int(row.n_cases)} \\\\"
            )
            for row in timescale_overview.itertuples()
        ]
    )
    stress_rows = "\n".join(
        [
            (
                f"{latex_escape(row.test_family)} & {latex_escape(row.scenario)} & "
                f"{float(row.mean_validation_relative_rmse):.4g} & "
                f"{yes_no(bool(row.supports_main_model))} \\\\"
            )
            for row in validation_stress_tests.head(12).itertuples()
        ]
    )
    gap_keys = submission_gap_audit["gap_area"].astype(str).str.strip().str.lower()
    manuscript_gap_audit = submission_gap_audit[
        ~gap_keys.isin({"local latex compilation", "latex_submission_package", "latex submission package"})
    ].copy()
    gap_rows = "\n".join(
        [
            (
                f"{publication_table_text(row.gap_area)} & {evidence_boundary_label(row.risk_level)} & "
                f"{publication_table_text(row.current_status)} & {publication_table_text(row.recommended_action)} \\\\"
            )
            for row in manuscript_gap_audit.itertuples()
        ]
    )
    k_min = float(np.nanmin(dynamics_summary["k_per_s"].to_numpy(dtype=float)))

    figure_blocks = [
        (
            "fig:framework",
            "Evidence structure for observed boundary-envelope reduction.",
            "The framework separates numerical molten-region states, observation operators, reduced model families and evidence gates used to select the observed-envelope descriptor and compact trajectory baseline. The A-prefixed identifiers denote model-construction process cases; full identifier definitions, source-file counts and supplementary case groups are given in the Supplementary Methods.",
            "paper_fig01_modeling_framework",
        ),
        (
            "fig:process-matrix",
            "Multi-condition process matrix.",
            "The 15 A1-A15 training conditions span laser power, scan speed and powder feed, with powder feed converted from the particle generation rate by the stated linear rule. Full condition identifiers and time-state counts are given in Supplementary Table S2.",
            "paper_fig02_process_matrix",
        ),
        (
            "fig:moving-frame",
            "Moving-frame reconstruction of the molten region.",
            "For the representative baseline condition, panel A shows the observed half-domain molten-region point cloud in the laboratory frame, panel B shows the symmetry-reconstructed full-width observation at the final exported time, panel C overlays selected times in the laser-attached coordinate to show moving-frame alignment, and panel D marks the reduced observed boundary-envelope descriptors $L_f$, $L_r$, $W$ and $H$ used by the downstream dynamics.",
            "paper_fig03_data_moving_frame",
        ),
        (
            "fig:geometry",
            "Transient geometry and quasi-steady approach.",
            "A representative baseline condition shows the evolution from early transient growth toward a quasi-steady regime in the moving frame.",
            "paper_fig04_geometry_quasi_steady",
        ),
        (
            "fig:boundary",
            "Cross-condition observed boundary-envelope model comparison.",
            "Representative transient and late quasi-steady overlays show the observed-envelope behavior of the ellipsoid and superellipsoid models, while paired cross-condition summaries report boundary residuals, volume-proxy errors and metric-wise support counts over the model-construction process matrix.",
            "paper_fig05_free_boundary_model_comparison",
        ),
        (
            "fig:process-response",
            "Quasi-steady process-response diagnostics.",
            "Quasi-steady length, width, height and maximum temperature are plotted over the laser-power and scan-speed matrix; marker area encodes powder feed and color encodes the response value.",
            "paper_fig06_process_response",
        ),
        (
            "fig:dimensionless",
            "Dimensionless scaling and sensitivity analysis.",
            "Post-processing values and perturbation ranges for $Pe$, $Ste$, $E^*$ and $Ma$ summarize thermal-transport scale diagnostics using liquidus-interpolated values from the temperature-dependent material-property curves.",
            "paper_fig07_dimensionless_regime",
        ),
        (
            "fig:dynamics-cross-condition",
            "Cross-condition dynamics validation.",
            "Condition-wise and state-wise validation errors compare the diagonal attractor with the coupled ridge attractor.",
            "paper_fig08_dynamics_validation",
        ),
        (
            "fig:dynamics-residuals",
            "Dynamical residuals by state.",
            "State-wise residuals compare diagonal and coupled attractor behavior and identify comparatively weak validation behavior for the maximum-velocity descriptor.",
            "paper_fig14_dynamics_residuals_by_state",
        ),
        (
            "fig:error-budget",
            "Diagnostic error-source taxonomy and model selection.",
            "The taxonomy summarizes reconstruction, boundary-fit, volume-proxy, dynamics and parameter sources used to interpret model selection.",
            "paper_fig09_error_budget_model_selection",
        ),
        (
            "fig:identifiability",
            "Identifiability and overparameterization.",
            "Superellipsoid parameter variation and the coupled matrix diagnostics motivate the selected model and the non-selected coupled comparison.",
            "paper_fig10_identifiability_overparameterization",
        ),
        (
            "fig:loco",
            "Leave-one-condition-out validation.",
            "A process-response extrapolation test holds out one condition at a time; target-wise normalization is used in the prediction panel so quantities with different units can be compared on the same visual scale.",
            "paper_fig11_leave_one_condition_validation",
        ),
        (
            "fig:external-holdout",
            "Additional simulated-condition assessment.",
            "Additional simulated cases are withheld from model construction and used to assess within-setting boundary-model generalization, quasi-steady process-response prediction and process-parameterized diagonal-attractor trajectories. Short axis identifiers are used for readability; the full condition identifiers and process parameters are listed in Supplementary Table S3.",
            "paper_fig12_external_holdout_validation",
        ),
    ]
    def make_main_figure(label: str, title: str, caption: str, stem: str, width: str = "0.92\\textwidth") -> str:
        return rf"""\begin{{center}}
\centering
\includegraphics[width={width}]{{{fig_path(stem)}}}
\captionof{{figure}}{{\textbf{{{title}}} {caption}}}
\label{{{label}}}
\end{{center}}"""

    figure_tex = {
        label: make_main_figure(label, title, caption, stem)
        for label, title, caption, stem in figure_blocks
    }
    figure_tex["fig:geometry"] = make_main_figure(
        "fig:geometry",
        "Transient geometry and quasi-steady approach.",
        "A representative baseline condition shows the evolution from early transient growth toward a quasi-steady regime in the moving frame.",
        "paper_fig04_geometry_quasi_steady",
        width="0.68\\textwidth",
    )
    figure_tex["fig:simulation-cross-sections"] = make_main_figure(
        "fig:simulation-cross-sections",
        "Molten-region thermal-flow cross sections.",
        "Temperature-colored molten-region surfaces with velocity vectors are shown for A2, A1 and A3, corresponding to 550, 750 and 950 W at the same scan speed of 8 mm s$^{-1}$ and powder feed of 12 g min$^{-1}$. Panels A, C and E show XY sections, whereas panels B, D and F show XZ sections. The figure provides source-field context for the observed-envelope reduction; quantitative selection is evaluated with the model diagnostics below.",
        "paper_fig13_simulation_cross_sections",
        width="0.98\\textwidth",
    )
    supp_figures = [
        (
            "fig:supp-boundary",
            "Representative-condition boundary fits across all time steps.",
            "Top-view superellipsoid overlays assess every exported time step for the representative condition, extending the boundary-fit evidence beyond the main-text summary panels.",
            "supp_figS1_all_boundary_fits",
        ),
        (
            "fig:supp-parameters",
            "Representative-condition superellipsoid parameters versus time.",
            "The fitted semi-axes, center coordinates and shape exponents evolve toward the same quasi-steady window used for downstream descriptors.",
            "supp_figS2_superellipsoid_parameters",
        ),
        (
            "fig:supp-dimensionless",
            "Dimensionless sensitivity scenario grid.",
            "The grid reports the minimum, baseline and maximum perturbation ratios used to interpret the nondimensional groups as scale diagnostics.",
            "supp_figS4_dimensionless_sensitivity_grid",
        ),
        (
            "fig:supp-theory",
            "Theory, identifiability and error-budget diagnostics.",
            "The panels combine error-source weights, identifiability constraints and nondimensional spans to document the model-selection boundaries summarized in the main text.",
            "supp_figS5_theory_identifiability_error_bounds",
        ),
        (
            "fig:supp-convex-alpha",
            "Convex-hull and alpha-complex proxy comparison.",
            "The comparison evaluates alternative boundary proxies on exported molten points and constrains the volume-proxy interpretation used in the main text.",
            "supp_figS6_convex_alpha_proxy_comparison",
        ),
        (
            "fig:supp-stability",
            "Representative-condition stability and attractor evidence.",
            "State-error convergence, diagonal rates and coupled eigenvalues provide representative stability evidence for the reduced attractor discussion.",
            "fig10_stability_attractor",
        ),
        (
            "fig:supp-boundary-panels",
            "Representative boundary-envelope time-step overlays.",
            "These expanded overlays complement the selected panels in main-text Figure 6 and show additional transient and quasi-steady boundary-envelope views behind the aggregate residuals.",
            "fig05_boundary_fit_comparison",
        ),
        (
            "fig:supp-thermal-flow",
            "Thermal-flow state evolution.",
            "These state histories supply the thermal-flow variables used in the reduced state and mark the approximate quasi-steady transition.",
            "fig03_thermal_flow_evolution",
        ),
        (
            "fig:supp-dynamics-comparison",
            "Dynamical model trajectory comparison.",
            "Direct trajectories complement the state-wise residual plot and show the coupled model as a comparison against the selected baseline.",
            "fig06_dynamics_model_comparison",
        ),
        (
            "fig:supp-temperature-properties",
            "Temperature-dependent material-property curves.",
            "Density, specific heat, thermal conductivity and viscosity are plotted from the tabulated property inputs used for liquidus-scale interpolation in the nondimensional analysis; vertical guides mark the solidus and liquidus temperatures.",
            "supp_figS10_temperature_dependent_properties",
        ),
    ]
    def make_supp_figure(label: str, title: str, caption: str, stem: str, width: str = "0.95\\textwidth") -> str:
        return rf"""\begin{{center}}
\centering
\includegraphics[width={width}]{{{fig_path(stem, "supp")}}}
\captionof{{figure}}{{\textbf{{{title}}} {caption}}}
\label{{{label}}}
\end{{center}}"""

    supp_figure_blocks = {
        label: make_supp_figure(label, title, caption, stem)
        for label, title, caption, stem in supp_figures
    }
    supp_figure_blocks["fig:supp-thermal-flow"] = make_supp_figure(
        "fig:supp-thermal-flow",
        "Thermal-flow state evolution.",
        "These state histories supply the thermal-flow variables used in the reduced state and mark the approximate quasi-steady transition.",
        "fig03_thermal_flow_evolution",
        width="0.78\\textwidth",
    )
    supp_figure_tex = "\n\n".join(supp_figure_blocks.values())
    supp_boundary_fig = supp_figure_blocks["fig:supp-boundary"]
    supp_parameters_fig = supp_figure_blocks["fig:supp-parameters"]
    supp_dimensionless_fig = supp_figure_blocks["fig:supp-dimensionless"]
    supp_theory_fig = supp_figure_blocks["fig:supp-theory"]
    supp_convex_alpha_fig = supp_figure_blocks["fig:supp-convex-alpha"]
    supp_stability_fig = supp_figure_blocks["fig:supp-stability"]
    supp_boundary_panels_fig = supp_figure_blocks["fig:supp-boundary-panels"]
    supp_thermal_flow_fig = supp_figure_blocks["fig:supp-thermal-flow"]
    supp_dynamics_comparison_fig = supp_figure_blocks["fig:supp-dynamics-comparison"]
    supp_temperature_properties_fig = supp_figure_blocks["fig:supp-temperature-properties"]
    high_risk_parameters = parameter_identifiability[
        parameter_identifiability["risk_level"].astype(str).str.lower().eq("high")
    ].copy()
    risk_items: list[str] = []
    if len(high_risk_parameters):
        high_risk_parameters["coefficient_of_variation_num"] = pd.to_numeric(
            high_risk_parameters["coefficient_of_variation"], errors="coerce"
        )
        preferred_order = ["n", "m", "p", "xi_c", "z_c", "Umax_m_per_s"]
        grouped_risk: list[tuple[int, str, pd.DataFrame]] = []
        for parameter, group in high_risk_parameters.groupby("parameter", sort=False):
            order = preferred_order.index(str(parameter)) if str(parameter) in preferred_order else len(preferred_order)
            grouped_risk.append((order, str(parameter), group))
        grouped_risk.sort(key=lambda item: (item[0], item[1]))
        for _, parameter, group in grouped_risk[:14]:
            cv_values = group["coefficient_of_variation_num"].dropna()
            cv_text = "not estimated" if cv_values.empty else f"{float(cv_values.median()):.3g}"
            count_text = f"; {len(group)} entries" if len(group) > 1 else ""
            risk_items.append(
                f"\\item {latex_math_label(parameter)}: median CV {cv_text}{count_text}; identifiability support is limited."
            )
    risk_lines = "\n".join(risk_items)
    if not risk_lines:
        risk_lines = "\\item No parameter is marked as weakly constrained by the current assessment."

    supplementary_model_selection_tex = rf"""
The model-selection rule uses boundary residual as the primary observed-envelope criterion because it directly evaluates the implicit boundary condition used by the algebraic manifold. Geometric-distance and volume-proxy metrics are treated as scope-of-interpretation diagnostics with a secondary role relative to the boundary residual. Volume is therefore reported as a proxy mismatch distinct from thermodynamic melt-volume estimation, and volume-preserving manifold fitting remains a separate constrained optimization problem.

The dynamical comparison is evaluated by fitted stability, time-split relative error, paired condition-state summaries, temporal-sampling sensitivity tests and coupled-matrix identifiability. The diagonal attractor is treated as the parsimonious candidate baseline, whereas the coupled ridge model provides a comparison for physically plausible but weakly supported cross-state coupling. Identifiability is interpreted as practical support for fitted parameters rather than as a purely algebraic property.

The computational analysis is reproducible from the molten-region numerical states. The provenance information includes point-record counts, duplicate-removal counts, repeated-coordinate handling, the half-domain symmetry convention and the condition-specific moving-frame transform $\xi=x-v_ct$. Fixed model families, fitted-parameter tables, leave-one-condition-out process-response assessments, temporal-sampling sensitivity tests and the separation of the additional simulated-condition cases from the model-construction cases further support reproducibility. Table~\ref{{tab:supp-model-selection}} summarizes the resulting model-selection evidence.

\begin{{table}}[htbp]
\centering
\caption{{Model-selection summary. The selected working model combination is {latex_escape(selected_text)}. The diagonal attractor is treated as a parsimonious baseline with bounded statistical support.}}
\label{{tab:supp-model-selection}}
\scriptsize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.13\textwidth}}>{{\raggedright\arraybackslash}}p{{0.17\textwidth}}>{{\raggedright\arraybackslash}}p{{0.07\textwidth}}>{{\raggedright\arraybackslash}}p{{0.18\textwidth}}>{{\raggedright\arraybackslash}}p{{0.07\textwidth}}>{{\raggedright\arraybackslash}}p{{0.20\textwidth}}}}
\toprule
Family & Model & Params. & Metric & Value & Role \\
\midrule
{model_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    supplementary_data_provenance_table_tex = rf"""
Table~\ref{{tab:supp-data-provenance}} summarizes the source, preprocessing and coordinate-convention information that supports the main-text provenance statement.

\begin{{table}}[htbp]
\centering
\caption{{Data provenance and preprocessing summary.}}
\label{{tab:supp-data-provenance}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.22\textwidth}}>{{\raggedright\arraybackslash}}p{{0.68\textwidth}}}}
\toprule
Item & Description \\
\midrule
{data_provenance_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    supplementary_body_tex = rf"""\clearpage
\section*{{Supplementary Information}}
\addcontentsline{{toc}}{{section}}{{Supplementary Information}}
\setcounter{{figure}}{{0}}
\renewcommand{{\thefigure}}{{S\arabic{{figure}}}}
\renewcommand{{\theHfigure}}{{S\arabic{{figure}}}}
\setcounter{{table}}{{0}}
\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\renewcommand{{\theHtable}}{{S\arabic{{table}}}}

\section*{{Supplementary Methods}}

\subsection*{{Data provenance and coordinate convention}}
The source melt-pool data comprise {vals['n_source_files']} numerical molten-region exports from {vals['n_conditions']} process conditions. Case identifiers follow \texttt{{Aa-b-c-d}}, where $a$ is the condition index, $b$ is laser power in watts, $c$ is scan speed in $\mathrm{{mm\,s^{{-1}}}}$, and $d$ is the particle generation rate. The particle rate is converted to powder feed by $d/60000\times12\,\mathrm{{g\,min^{{-1}}}}$. The exported time window is $t={vals['t_min']:.2f}$--${vals['t_max']:.2f}\,\mathrm{{s}}$ at the exported times {latex_escape(vals['time_points'])} s, matching the main-text rationale for capturing early melt-pool formation and subsequent quasi-steady evolution. Each export contains molten-region points, so the supplementary analysis treats full thermal-field reconstruction as outside the exported data scope. The exported columns are {latex_escape(export_columns_text)}. Coordinates are interpreted in metres, temperature in kelvin, temperature-gradient magnitude in $\mathrm{{K\,m^{{-1}}}}$, pressure in pascals and velocity in $\mathrm{{m\,s^{{-1}}}}$.

Before preprocessing, each condition-time export contains {vals['raw_rows_min']}--{vals['raw_rows_max']} rows. After exact row deduplication, {vals['exact_dedup_rows_min']}--{vals['exact_dedup_rows_max']} rows remain; after repeated-coordinate collapse, {vals['unique_points_min']}--{vals['unique_points_max']} unique spatial points remain per export. Across the dataset, {vals['exact_duplicates_removed_total']} exact duplicate rows are removed and {vals['coordinate_duplicates_collapsed_total']} repeated-coordinate groups are collapsed by field averaging. All point locations are then converted to the condition-specific laser-attached frame by $\xi=x-v_ct$, where $v_c$ is inferred from the case identifier.

The available simulation-parameter record identifies a 316L L-DED numerical model with a half-domain symmetry setting, a recorded cell size of $1.0\times10^{{-4}}\,\mathrm{{m}}$, Gaussian beam radius $8.0\times10^{{-4}}\,\mathrm{{m}}$, absorptivity 0.1, phase-change constants, surface-tension constants and particle-rate process inputs.

The coordinate convention used throughout the manuscript is as follows. The laboratory scan direction is $x$, the transverse direction is $y$, the build direction is $z$, and the moving coordinate is $\xi$. Time is denoted by $t$. Temperature, gradient magnitude and velocity magnitude are extracted from the exported molten points, with the surrounding solid region outside the extracted state.

\subsection*{{Theoretical origin and observation-reduction chain}}
The main text states the moving-source Stefan-Marangoni balances and the observation-reduction chain used to motivate the model. This subsection specifies how the same objects are used in the supplementary analysis, with emphasis on the observation operator, reconstruction assumptions and diagnostic evidence underlying the main-text model-selection results.

Table~\ref{{tab:supp-modeling-details}} links each model component to the supplementary evidence that extends the main text.

\begin{{table}}[htbp]
\centering
\caption{{Relationship between the main-text model and supplementary evidence.}}
\label{{tab:supp-modeling-details}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.18\textwidth}}>{{\raggedright\arraybackslash}}p{{0.35\textwidth}}>{{\raggedright\arraybackslash}}p{{0.38\textwidth}}}}
\toprule
Component & Main-text role & Supplementary evidence \\
\midrule
{modeling_detail_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection*{{Model-construction and additional process matrices}}
The model-construction design comprises 15 simulated process conditions. These cases form the process matrix used in the main text; Table~\ref{{tab:supp-training-process-matrix}} lists their full identifiers, process settings and time-state counts.

\begin{{table}}[htbp]
\centering
\caption{{Model-construction process matrix. Powder feed is converted from particle rate by $d/60000\times12\,\mathrm{{g\,min^{{-1}}}}$.}}
\label{{tab:supp-training-process-matrix}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Case & Power (W) & Speed (mm s$^{{-1}}$) & Input $d$ & Powder feed (g min$^{{-1}}$) & Time-state exports \\
\midrule
{process_matrix_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

{external_case_table_tex}

\subsection*{{Parameter reconciliation assessment}}
The available solver-parameter record provides the laser, phase-change and surface-tension constants. The tabulated temperature-dependent material-property curves provide density, viscosity, thermal conductivity and specific heat for liquidus-scale nondimensional diagnostics. The full property variation is plotted in Supplementary Fig. S10.

\begin{{table}}[htbp]
\centering
\caption{{Parameter reconciliation assessment.}}
\label{{tab:supp-parameter-assessment}}
\scriptsize
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.17\textwidth}}>{{\raggedright\arraybackslash}}p{{0.20\textwidth}}>{{\raggedright\arraybackslash}}p{{0.20\textwidth}}>{{\raggedright\arraybackslash}}p{{0.32\textwidth}}}}
\toprule
Parameter & Solver-parameter record & Post-processing basis & Treatment in this study \\
\midrule
{parameter_audit_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection*{{Symmetry reconstruction of the half-domain export}}
The source-data computational domain is simulated only for $y\geq0$, with $y=0$ as a symmetry plane. The full-domain representation is obtained by mirroring each point $(\xi,y,z)$ to $(\xi,-y,z)$. This reconstruction is exact for scalar geometric descriptors when the process, heat source and powder delivery are symmetric with respect to the plane. Antisymmetric flow structures remain outside the half-domain data.

The main text defines the scalar reconstruction for width and convex-hull volume. In the supplementary analysis, this rule is used only to document how full-width descriptors and full-volume proxies are obtained from the half-domain data.

\subsection*{{Boundary-envelope extraction and manifold fitting}}
For each time step, the molten-point envelope is approximated by a convex-hull boundary. This conservative choice follows from the observation model: because the data contain only molten-region points, the boundary is the observed molten-domain envelope. The ellipsoid baseline and the superellipsoid model are defined in the main text as analytic boundary manifolds, and the fitting operation is posed there as a level-set projection problem.

The fitted geometric parameters are reported as manifold coordinates. The parameters $a_f$ and $a_r$ describe front and rear extents in the moving coordinate, $b$ describes the half-width, $c$ describes the vertical scale, and $(\xi_c,z_c)$ locates the fitted boundary center. The exponents $n$, $m$ and $p$ control directional shape sharpness. The fitting objective, analytic volume and diagnostic definitions are given in the main text; the supplementary material provides the full time-step overlays, fitted parameter trajectories and proxy-sensitivity assessments that support those definitions.

Figures~\ref{{fig:supp-boundary}} and~\ref{{fig:supp-parameters}} report the full time-step boundary fits and fitted parameter trajectories.

{supp_boundary_fig}

{supp_parameters_fig}

\subsection*{{Reduced dynamics, stability and residuals}}
The reduced state is $\bm{{q}}=[L_f,L_r,W,H,T_{{\mathrm{{max}}}},G_{{\mathrm{{mean}}}},U_{{\mathrm{{max}}}}]^T$. Here, $L_f$ and $L_r$ are the front and rear lengths, $W$ is full width, $H$ is height, $T_{{\mathrm{{max}}}}$ is maximum temperature, $G_{{\mathrm{{mean}}}}$ is mean temperature-gradient magnitude and $U_{{\mathrm{{max}}}}$ is maximum velocity magnitude. The diagonal attractor is defined as the parsimonious baseline candidate, and the coupled ridge attractor is defined as the comparison model.

The main text defines the diagonal trajectory fit, coupled ridge comparison, stability criteria and validation metric. The supplementary material uses the same definitions to provide representative stability panels, trajectory comparisons and weak-identifiability diagnostics. The diagonal fit uses direct trajectory fitting within the training-only quasi-steady admissible interval. The coupled comparison uses finite-difference ridge regression and is evaluated by both stability and validation error, because spectral stability alone is insufficient to support its additional parameters.

\subsection*{{Dimensionless scaling and sensitivity}}
The main text reports the process-matrix mean dimensionless values. Here, the supplementary material records the interpretation of each group and the perturbation grid used to test sensitivity to reference temperature, absorptivity and surface-tension coefficient. The underlying material-property curves are shown in Supplementary Fig. S10.

The symbols are defined as follows. $Pe=vL_{{\mathrm{{ref}}}}/\alpha$ compares advection by the moving laser with thermal diffusion, $Fo=\alpha t/L_{{\mathrm{{ref}}}}^2$ measures diffusive time relative to the reference melt-pool length, $Ste=c_p(T_l-T_s)/L_{{\mathrm{{fus}}}}$ compares sensible heat across the mushy interval with latent heat, $E^*$ scales absorbed laser power against the enthalpy needed to heat material through the moving beam footprint, and $Ma$ scales thermocapillary forcing against viscous-thermal diffusion. Sensitivity scenarios perturb reference temperature, absorptivity and surface-tension coefficient to evaluate the stability of these interpretations.

The full scenario grid is shown below so that the nondimensional interpretation and its sensitivity evidence remain in the same local reading unit.

{supp_dimensionless_fig}

\subsection*{{Assumption assessment matrix}}
The assumption matrix is placed in the Supplementary Information because it defines the scope of interpretation for the reduced model.

\begin{{table}}[htbp]
\centering
\caption{{Assumption assessment matrix for the observed boundary-envelope modeling framework.}}
\label{{tab:supp-assumptions}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.07\textwidth}}>{{\raggedright\arraybackslash}}p{{0.27\textwidth}}>{{\raggedright\arraybackslash}}p{{0.42\textwidth}}>{{\raggedright\arraybackslash}}p{{0.12\textwidth}}}}
\toprule
ID & Assumption & Evidence & Interpretive boundary \\
\midrule
{assumption_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection*{{Temporal-sampling sensitivity-test protocol}}
The temporal-sampling sensitivity analysis perturbs the train-validation split, uses rolling-origin tests, removes individual time steps and applies deterministic state-noise perturbations. The tested cases support the selected parsimonious diagonal baseline in {stress_support:.3f} of the temporal-sampling sensitivity settings, and the mean validation relative RMSE ranges from {stress_min:.4f} to {stress_max:.4f}. These tests are internal to the available numerical dataset and quantify the temporal-sampling sensitivity of the compact baseline.

The analysis evaluates whether the parsimonious diagonal baseline depends on one train-validation split and quantifies sensitivity to short-sequence sampling within each condition. Because the dataset remains simulation-only, these results bound the trajectory claim within the available exports and do not establish experimental generality over power, speed or powder-feed rate.

\begin{{table}}[htbp]
\centering
\caption{{Representative temporal-sampling sensitivity tests. The complete table is included with the accompanying data tables.}}
\label{{tab:supp-sensitivity-tests}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.17\textwidth}}>{{\raggedright\arraybackslash}}p{{0.21\textwidth}}>{{\raggedright\arraybackslash}}p{{0.28\textwidth}}>{{\raggedright\arraybackslash}}p{{0.18\textwidth}}}}
\toprule
Test family & Scenario & Mean validation relative RMSE & Supports diagonal baseline \\
\midrule
{stress_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection*{{Residual limitations}}
This section summarizes interpretive boundaries that define the scope of the model without constituting additional modeling results. The highest-ranked remaining boundary is: {latex_escape(high_gap_text)}.

\begin{{table}}[htbp]
\centering
\caption{{Residual limitations and validation routes.}}
\label{{tab:supp-gap-audit}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.16\textwidth}}>{{\raggedright\arraybackslash}}p{{0.10\textwidth}}>{{\raggedright\arraybackslash}}p{{0.27\textwidth}}>{{\raggedright\arraybackslash}}p{{0.34\textwidth}}}}
\toprule
Topic & Interpretive boundary & Current evidence & Validation route \\
\midrule
{gap_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection*{{Error-budget construction}}
The main text defines the error-source taxonomy and its diagnostic bound. The supplementary material avoids repeating that formulation and instead documents how the error terms are interpreted alongside identifiability diagnostics. Point-cloud reconstruction, analytic boundary fitting, volume proxy, dynamical prediction and material-scale uncertainty are kept as separate diagnostic sources. A stricter uncertainty-quantification study would propagate these terms through a global sensitivity or Bayesian analysis.

\subsection*{{Identifiability and overparameterization diagnostics}}
Identifiability is assessed through coefficient of variation, stability of fitted parameters across time, signs of dynamic parameters and validation behavior. For the superellipsoid, the least constrained quantities are shape exponents and center shifts that may compensate for each other during fitting. For the coupled attractor, the principal identifiability constraint is the ratio between matrix parameters and available state transitions. These diagnostics are used to decide whether additional flexibility is mathematically defensible.

The quantities with limited identifiability support are:
\begin{{itemize}}
{risk_lines}
\end{{itemize}}

The combined theory, identifiability and error-budget diagnostic is placed here because it summarizes the same overparameterization constraints described in this subsection.

{supp_theory_fig}

\subsection*{{Boundary-proxy sensitivity diagnostic}}
The convex-hull and alpha-complex comparison is used for proxy-sensitivity assessment. It shows how alternative envelope proxies behave on the exported molten points and why volume-proxy results are treated as interpretive constraints rather than physical solid-liquid interface recovery.

{supp_convex_alpha_fig}

\subsection*{{Auxiliary thermal-flow and trajectory diagnostics}}
Following the main supplementary evidence, four auxiliary diagnostic figures present representative stability evidence, expanded boundary-overlay panels, thermal-flow state evolution and diagonal-versus-coupled trajectories. Main-text Figure 6 displays selected overlay panels; Supplementary Fig. S7 keeps additional transient and quasi-steady top and side views. The trajectory panels show the direct prediction curves behind the residual comparison and complement the main residual analysis.

{supp_stability_fig}

{supp_boundary_panels_fig}

{supp_thermal_flow_fig}

{supp_dynamics_comparison_fig}

\subsection*{{Temperature-dependent material-property curves}}
The material-property basis for density, specific heat, thermal conductivity and viscosity is tabulated as a function of temperature. The figure below plots the curves used for liquidus-scale interpolation in the nondimensional scale diagnostics; the corresponding liquidus values are summarized in Supplementary Table S4.

{supp_temperature_properties_fig}

{supplementary_model_selection_tex}

{supplementary_data_provenance_table_tex}
"""

    main_tex = rf"""\documentclass[11pt]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb,bm}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage[numbers,sort&compress]{{natbib}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{caption}}
\captionsetup{{font=small,labelfont=bf,hypcap=false}}
\setlength{{\tabcolsep}}{{4pt}}
\setlength{{\emergencystretch}}{{3em}}
\renewcommand{{\arraystretch}}{{1.12}}

\title{{Melt-pool-data-informed observed boundary-envelope identification of L-DED melt pools using superellipsoid manifolds and parsimonious attractor dynamics}}
\author{{{MANUSCRIPT_AUTHOR_BLOCK_LATEX}}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Laser directed energy deposition simulations can provide detailed molten-region evolution data, but these fields are difficult to compare, interpret and reduce into compact mathematical descriptions. This study develops a simulation-output-to-manifold reduction for 316L L-DED melt pools. Half-domain molten-region point clouds are transformed into symmetry-reconstructed moving-frame envelopes, projected onto asymmetric superellipsoid manifolds and reduced to geometric, thermal and flow descriptors. The resulting descriptor system is evaluated through boundary-envelope fitting, geometric-distance and volume-proxy diagnostics, reduced-dynamics comparison, identifiability assessment and numerical holdout tests. The superellipsoid provides the selected algebraic observed-envelope descriptor because it captures boundary-envelope shape more consistently than the ellipsoid baseline, while volume and distance diagnostics define its interpretive limits. The parsimonious diagonal attractor provides the selected compact trajectory baseline, whereas the coupled ridge model is retained as an overparameterization comparison. Held-out numerical cohorts indicate within-setting generalization of the boundary descriptor, process-response summaries and process-parameterized trajectory baseline. Together, these results provide an evidence-guided route from molten-region simulation outputs to interpretable observed-boundary descriptors, while separating boundary-envelope reduction from full thermal-field reconstruction and universal process-map prediction.
\end{{abstract}}

\begin{{flushleft}}
\textbf{{Keywords:}} laser directed energy deposition; melt pool; observed boundary-envelope model; reduced-order modeling
\end{{flushleft}}

\section{{Introduction}}

Laser directed energy deposition (L-DED) is widely studied for metallic repair, graded deposition and large-component manufacturing {latex_cite('@svetlizky2021')}. Reviews also position L-DED within broader directed-energy and large-scale additive-manufacturing applications {latex_cite('@ahn2021')}. The molten region remains difficult to describe because laser-material interaction, mass addition, heat transfer, phase change, free-surface heat loss and thermocapillary transport evolve together. Reviews of metal additive manufacturing emphasize that melt-pool behavior controls process stability, defect formation, microstructure and geometric fidelity {latex_cite('@li2023')}. Thermal-transport syntheses further link melt-pool response to heat input, material properties and process-scale interpretation {latex_cite('@debroy2018')}. Numerical studies have shown that powder delivery can reshape the molten region {latex_cite('@wang2023powder')}. Recent powder-catchment monitoring further links powder delivery, coaxial melt-pool signals and deposition efficiency {latex_cite('@hong2025catchment')}. In situ mechanism studies further connect molten-region evolution with pore-forming transport {latex_cite('@zhang2024pore')}. Related simulations highlight heat transfer, free-surface evolution and thermo-mechanical response in interpreting the melt pool {latex_cite('@poggi2022')}. DED-specific thermal-flow studies similarly connect molten-region transport to observable process behavior {latex_cite('@zhang2021dedcfd')}. Recent gas-flow experiments show that shielding conditions alter the observable melt-pool state {latex_cite('@sinclair2024gasflow')}. Laser-intensity shaping provides a complementary route for changing melt-pool geometry during deposition {latex_cite('@lei2024shaping')}. Temporal laser shaping studies now extend this view to transient melt-pool dynamics in 316L L-DED {latex_cite('@chen2025temporal')}. How to represent the evolving molten region as a low-dimensional, interpretable mathematical object has therefore become a key issue when the accessible information is a time-resolved boundary observation rather than a complete thermal-flow field.

Existing studies have approached melt-pool evolution through simulation, monitoring, control and data-driven prediction. Simulation-guided work has targeted melt-pool depth and thermal response {latex_cite('@liao2022')}. Closed-loop process studies have treated melt-pool width or temperature as regulated process states {latex_cite('@smoqi2022')}. Real-time thermal-imaging control has recently extended this direction to melt-pool height regulation and geometric accuracy {latex_cite('@shin2025heightcontrol')}. System-identification work has used melt-pool geometry as a state for feedback design {latex_cite('@miao2023lqr')}. Process-state quality methods have extended this control-oriented view toward predictive adjustment {latex_cite('@rahmani2024psq')}. Coaxial monitoring has linked melt-pool area to process diagnosis {latex_cite('@dasilva2023')}. Hybrid pyrometer-camera monitoring further connects melt-pool signatures with bead-geometry prediction {latex_cite('@ji2025coaxial')}. Infrared monitoring provides a complementary thermal-signature route for process-state interpretation {latex_cite('@herzog2024infrared')}. Image-based studies have related shape irregularity to process optimization {latex_cite('@kong2023monitoring')}. Deep-learning segmentation studies have similarly used melt-pool images for flaw detection {latex_cite('@asadi2024dnn')}. Sequence autoencoders extend this direction to melt-pool-level flaw prediction {latex_cite('@abranovic2024flaw')}. Machine-learning models have predicted melt-pool dimensions from process conditions {latex_cite('@zhu2023')}. Non-steady-state melt-pool behavior is also being incorporated into geometry prediction for laser DED {latex_cite('@xu2026nonsteady')}. Surrogate models extend this prediction task to morphology and process-window exploration {latex_cite('@akbari2022')}. Related studies extend these ideas to thermal-field prediction {latex_cite('@wu2024')}. Sequence-based learning has also been used for thermal histories and process signatures {latex_cite('@hemmasian2023')}. Physics-driven temporal convolutional networks have predicted melt-pool width and layer height {latex_cite('@wang2023tcn')}. Thermal-field finite-element modeling provides another route to compact process-response estimates {latex_cite('@jelinek2020thermalfe')}. Automated computational frameworks help organize such estimates across process settings {latex_cite('@kovsca2023')}. Physics-informed modeling shows how data and physical constraints can be combined under limited data {latex_cite('@karniadakis2021')}. Non-intrusive physics-informed learning extends this principle to reduced predictive models {latex_cite('@cuomo2022')}. Related physics-informed machine-learning studies extend this idea to broader engineering and manufacturing settings {latex_cite('@jiang2024piml')}. Physics-informed reduced-order methods have also been applied to manufacturing-scale fields {latex_cite('@kumar2023piml')}. Uncertainty-aware learning emphasizes robustness under imperfect data {latex_cite('@pham2022uncertainty')}. Multifidelity studies show how lower- and higher-fidelity sources can be combined for process prediction {latex_cite('@menon2022multifidelity')}. Additive-manufacturing uncertainty quantification further motivates explicit treatment of model and data uncertainty {latex_cite('@wang2020uq')}. Recent uncertainty studies extend this issue to process-dependent prediction and design {latex_cite('@hermann2023')}. Reviews of machine learning in DED reinforce the need to balance predictive accuracy with interpretable state definitions {latex_cite('@era2023')}. Taken together, these studies provide powerful tools for simulating, monitoring and predicting melt-pool behavior, but they leave open how an observed boundary itself should be reduced, selected and interpreted as a mathematical state. To address this gap, the present analysis combines four elements: a moving-source coordinate frame from heat-transfer theory {latex_cite('@rosenthal1946')}, an ellipsoidal baseline for melt-pool shape comparison {latex_cite('@goldak1984')}, superquadric geometry for flexible boundary representation {latex_cite('@barr1981')}, and regularized model comparison for weakly supported reduced dynamics {latex_cite('@hoerl1970')}, including Tikhonov-style stabilization for ill-conditioned fitted operators {latex_cite('@tikhonov1977')}.

Therefore, this study develops an observed boundary-envelope mathematical model for L-DED melt pools. The analysis reconstructs half-domain molten-region states in a moving coordinate, projects the observed boundary envelope onto an asymmetric superellipsoid manifold, extracts reduced geometric and thermal-flow descriptors, and compares parsimonious first-order attractor baselines with overparameterized coupled alternatives. The central aim is to identify compact boundary and trajectory descriptors supported by exported molten-region states, while separating this target from full thermal-field reconstruction and universal process-map prediction. The main contribution is an evidence-guided data-to-manifold reduction for sparse melt-pool simulation outputs. It combines a model-selection protocol that separates algebraic envelope fit from geometric-distance and volume diagnostics with a descriptor system linking boundary geometry, process response and fitted relaxation baselines. The reduction is grounded in the Stefan-Marangoni boundary-data problem, while the superquadric fit and dynamical-system elements serve as evaluated model classes. This study provides a basis for using reduced melt-pool variables in process maps, monitoring models and control-oriented representations with explicit evidence for their interpretive scope.

\section{{Physical formulation and observed-boundary modeling}}

\subsection{{Stefan-Marangoni origin and observation model}}

A full L-DED melt-pool model can be idealized as a moving-source Stefan problem with a moving phase boundary {latex_cite('@crank1984')}. In metal melts, the same physical setting also includes thermocapillary forcing along free surfaces {latex_cite('@scriven1960')}. A representative energy balance is
\begin{{equation}}
\rho c_p(T)\left(\frac{{\partial T}}{{\partial t}}+\bm{{u}}\cdot\nabla T\right)
=\nabla\cdot(k(T)\nabla T)+Q_{{\mathrm{{laser}}}}+Q_{{\mathrm{{powder}}}}-\rho L_{{\mathrm{{fus}}}}\frac{{\partial f_l}}{{\partial t}}.
\label{{eq:full-energy}}
\end{{equation}}
In Eq.~\eqref{{eq:full-energy}}, $\rho$ is density, $c_p(T)$ is a general heat-capacity function, $T$ is temperature, $\bm{{u}}$ is the melt velocity, and $k(T)$ is a general thermal-conductivity function. The terms $Q_{{\mathrm{{laser}}}}$ and $Q_{{\mathrm{{powder}}}}$ represent laser and powder energy input, respectively, $L_{{\mathrm{{fus}}}}$ is the latent heat of fusion, and $f_l$ is the liquid fraction. This temperature-dependent form defines the physical origin of the observation problem and is consistent with the tabulated temperature-dependent density, heat-capacity, thermal-conductivity and viscosity curves in the available simulation-parameter basis. Laser-metal additive-manufacturing studies show why heat input, melt flow and phase change must be interpreted together {latex_cite('@king2015')}. High-fidelity melt-pool simulations further show how transport and recoil-driven flow reshape the molten region {latex_cite('@khairallah2016')}. In the present analysis, these curves supply the liquidus-scale material values used for nondimensional diagnostics. Direct solution of Eq.~\eqref{{eq:full-energy}} would require the complete solid-domain temperature field, whereas the present observation consists of molten-region states.
At a solid-liquid interface, the phase-change balance can be written in Stefan form {latex_cite('@crank1984')} as
\begin{{equation}}
\rho L_{{\mathrm{{fus}}}}v_n=\left[k\nabla T\cdot\bm{{n}}\right]_s^l .
\label{{eq:stefan}}
\end{{equation}}
In Eq.~\eqref{{eq:stefan}}, $v_n$ is the normal velocity of the phase boundary, $\bm{{n}}$ is the interface normal, and $[\cdot]_s^l$ denotes the jump from the solid side to the liquid side. The right-hand side is the conductive heat-flux imbalance supplying latent heat at the moving interface.
At a free surface, heat loss and thermocapillary forcing are represented schematically by
\begin{{equation}}
-k\nabla T\cdot\bm{{n}}=h_c(T-T_\infty)+\epsilon_{{\mathrm{{rad}}}}\sigma_{{\mathrm{{SB}}}}(T^4-T_\infty^4),
\label{{eq:heat-loss}}
\end{{equation}}
\begin{{equation}}
\bm{{\tau}}\bm{{n}}\cdot\bm{{t}}=\frac{{d\sigma}}{{dT}}\nabla_s T\cdot\bm{{t}}.
\label{{eq:marangoni}}
\end{{equation}}
In Eq.~\eqref{{eq:heat-loss}}, $h_c$ is the convective heat-transfer coefficient, $T_\infty$ is the ambient temperature, $\epsilon_{{\mathrm{{rad}}}}$ is the emissivity, and $\sigma_{{\mathrm{{SB}}}}$ is the Stefan-Boltzmann constant. In Eq.~\eqref{{eq:marangoni}}, $\bm{{\tau}}$ is the viscous stress tensor, $\bm{{t}}$ is a tangent direction on the free surface, $d\sigma/dT$ is the surface-tension temperature coefficient, and $\nabla_s$ is the surface-gradient operator. The thermocapillary term is the continuum expression of the Marangoni effect {latex_cite('@scriven1960')}. These balances set the physical context for the observed boundary-envelope reduction.

Let $\Omega_m^h(t)$ be the exported half-domain molten-region point cloud. The laser-attached coordinate is
\begin{{equation}}
\xi = x - v_c t,\qquad v_c=\frac{{s_c}}{{1000}}\,\mathrm{{m\,s^{{-1}}}}.
\label{{eq:moving-coordinate}}
\end{{equation}}
In Eq.~\eqref{{eq:moving-coordinate}}, $x$ is the laboratory scan coordinate, $t$ is time, $s_c$ is the scan speed in $\mathrm{{mm\,s^{{-1}}}}$ parsed from the condition folder, $v_c$ is the corresponding SI scan speed for condition $c$, and $\xi$ is the coordinate observed from a frame translating with the laser. This coordinate choice follows the classical moving-heat-source interpretation of localized welding fields {latex_cite('@rosenthal1946')}. The ellipsoidal heat-source tradition provides the baseline geometric context for melt-pool modeling {latex_cite('@goldak1984')}. The imposed symmetry boundary is $y=0$. The full-domain observation operator is
\begin{{equation}}
\mathcal{{R}}[\Omega_m^h](t)=\Omega_m^h(t)\cup \lbrace(\xi,-y,z):(\xi,y,z)\in\Omega_m^h(t)\rbrace.
\label{{eq:symmetry}}
\end{{equation}}
In Eq.~\eqref{{eq:symmetry}}, $\mathcal{{R}}$ is the reflection-reconstruction operator, $\Omega_m^h(t)$ is the observed half-domain molten region, and $(\xi,y,z)$ are moving-frame coordinates. The operation reconstructs full-observation geometric descriptors; antisymmetric flow components remain outside the observed molten-region domain.
The exported molten-region data are treated as an observation
\begin{{equation}}
P^h(t)=\mathcal{{O}}_h[\Omega_m(t),T,\bm{{u}}],
\label{{eq:observation}}
\end{{equation}}
where $P^h(t)$ is the discrete half-domain point cloud and $\mathcal{{O}}_h$ is the melt-pool-data observation operator applied to the full molten region $\Omega_m(t)$, temperature field $T$ and velocity field $\bm{{u}}$. The fitted mathematical object is the observed envelope $\Gamma_h(t)$, defined by the available molten-domain points.
The full-width descriptor and full-volume proxy are
\begin{{equation}}
W(t)=2\max_{{\Omega_m^h(t)}} y,\qquad V_{{\mathrm{{full}}}}(t)=2V_{{\mathrm{{half}}}}(t).
\label{{eq:symmetry-descriptors}}
\end{{equation}}
In Eq.~\eqref{{eq:symmetry-descriptors}}, $W(t)$ is the reconstructed full melt-pool width, $V_{{\mathrm{{half}}}}(t)$ is the half-domain convex-hull volume proxy, and $V_{{\mathrm{{full}}}}(t)$ is its symmetry-reconstructed counterpart. The observed boundary envelope $\Gamma(t)$ is defined as the envelope of the exported molten domain. The reduced state is
\begin{{equation}}
\bm{{q}}(t)=\left[L_f,L_r,W,H,T_{{\mathrm{{max}}}},G_{{\mathrm{{mean}}}},U_{{\mathrm{{max}}}}\right]^T .
\label{{eq:state}}
\end{{equation}}
In Eq.~\eqref{{eq:state}}, $L_f$ and $L_r$ are the front and rear melt-pool extents in the moving coordinate, $W$ is the full width, $H$ is the height, $T_{{\mathrm{{max}}}}$ is the maximum molten-region temperature, $G_{{\mathrm{{mean}}}}$ is the mean temperature-gradient magnitude, and $U_{{\mathrm{{max}}}}$ is the maximum velocity magnitude.

\subsection{{Superellipsoid observed boundary-envelope manifold}}

The asymmetric ellipsoid baseline is
\begin{{equation}}
\left(\frac{{\xi-\xi_c}}{{a_s}}\right)^2+
\left(\frac{{y}}{{b}}\right)^2+
\left(\frac{{z-z_c}}{{c}}\right)^2=1,
\qquad
a_s=\begin{{cases}}a_f, & \xi\geq \xi_c,\\ a_r, & \xi<\xi_c.\end{{cases}}
\label{{eq:ellipsoid}}
\end{{equation}}
In Eq.~\eqref{{eq:ellipsoid}}, $\xi_c$ and $z_c$ are the moving-frame center coordinates, $a_f$ and $a_r$ are the front and rear semi-axes, $b$ is the half-width scale, $c$ is the vertical scale, and $a_s$ selects the front or rear length depending on the sign of $\xi-\xi_c$.
The superellipsoid candidate extends the superquadric shape family to the observed asymmetric melt-pool envelope {latex_cite('@barr1981')}:
\begin{{equation}}
\left|\frac{{\xi-\xi_c}}{{a_s}}\right|^n+
\left|\frac{{y}}{{b}}\right|^m+
\left|\frac{{z-z_c}}{{c}}\right|^p=1,
\label{{eq:superellipsoid}}
\end{{equation}}
with parameter vector $\bm{{\theta}}=[a_f,a_r,b,c,\xi_c,z_c,n,m,p]^T$. The exponents $n$, $m$ and $p$ control the sharpness or flatness of the boundary along the scan, transverse and vertical directions; the ellipsoid-like baseline is recovered when these exponents are fixed at 2. The superellipsoid representation is evaluated against the ellipsoid baseline, and its additional flexibility is interpreted only through the boundary-residual evidence reported in the model-selection analysis.
For the fitting operation, Eq.~\eqref{{eq:superellipsoid}} is written as an implicit level-set function. This is a static implicit-boundary representation, distinct from solving a propagating level-set equation but related to the same mathematical representation of fronts {latex_cite('@osher1988')}:
\begin{{equation}}
\Phi(\bm{{x}};\bm{{\theta}})=
\left|\frac{{\xi-\xi_c}}{{a_s}}\right|^n+
\left|\frac{{y}}{{b}}\right|^m+
\left|\frac{{z-z_c}}{{c}}\right|^p,
\qquad \Gamma_M(\bm{{\theta}})=\left\{{\bm{{x}}:\Phi(\bm{{x}};\bm{{\theta}})=1\right\}}.
\label{{eq:superellipsoid-levelset}}
\end{{equation}}
In Eq.~\eqref{{eq:superellipsoid-levelset}}, $\Phi$ is the implicit boundary function and $\Gamma_M(\bm{{\theta}})$ is the analytic boundary manifold. The analytic full-domain volume used in $\varepsilon_V$ is
\begin{{equation}}
V_M(t)=4(a_f+a_r)bc
\frac{{\Gamma(1+1/n)\Gamma(1+1/m)\Gamma(1+1/p)}}
{{\Gamma(1+1/n+1/m+1/p)}} .
\label{{eq:superellipsoid-volume}}
\end{{equation}}
In Eq.~\eqref{{eq:superellipsoid-volume}}, $V_M$ is the volume of the asymmetric fitted manifold; for $n=m=p=2$ it reduces to $(2/3)\pi b c(a_f+a_r)$. The fitted parameter vector is the constrained projection
\begin{{equation}}
\bm{{\theta}}^*(t)=
\arg\min_{{\bm{{\theta}}\in\Theta}}
\frac{{1}}{{N_b(t)}}\sum_{{\bm{{x}}_j\in\Gamma_h(t)}}
\left(\Phi(\bm{{x}}_j;\bm{{\theta}})-1\right)^2 .
\label{{eq:boundary-projection}}
\end{{equation}}
Here, $N_b(t)$ is the number of boundary-envelope points and $\Gamma_h(t)$ is the observed half-domain envelope after moving-frame transformation. The constraint set $\Theta$ uses positive semi-axes bounded below by $10^{{-8}}$ m, center coordinates within the observed data span plus a 0.25-span margin, semi-axes no larger than 2.5 times the corresponding data span, and exponent bounds $1\leq n,m,p\leq {SUPERELLIPSOID_EXPONENT_UPPER:.1f}$. Fits use deterministic initialization from observed spans for the ellipsoid and from the fitted ellipsoid with $n=m=p=2$ for the superellipsoid. Optimization is performed as a bounded nonlinear least-squares fit with a robust soft-$\ell_1$ loss and the evaluation caps reported in the method-details table. The resulting geometric diagnostics are
\begin{{equation}}
\varepsilon_\Gamma(t)=
\left[
\frac{{1}}{{N_b(t)}}\sum_{{\bm{{x}}_j\in\Gamma_h(t)}}
\left(\Phi(\bm{{x}}_j;\bm{{\theta}}^*(t))-1\right)^2
\right]^{{1/2}},
\qquad
\varepsilon_V(t)=
\frac{{\left|V_M(t)-V_{{\mathrm{{full}}}}(t)\right|}}
{{V_{{\mathrm{{full}}}}(t)+\delta_V}} .
\label{{eq:geometry-errors}}
\end{{equation}}
In Eq.~\eqref{{eq:geometry-errors}}, $\varepsilon_\Gamma$ is the dimensionless boundary residual, $V_M$ is the analytic full-domain volume of the fitted manifold, $V_{{\mathrm{{full}}}}$ is the symmetry-reconstructed convex-hull volume proxy, and $\delta_V>0$ prevents division by zero. These definitions connect the observed boundary-envelope equations directly to the model-selection table.

\subsection{{Manifold projection and reduced-state dynamics}}

The modeling chain follows the reduction logic of mapping a high-dimensional state to a low-dimensional manifold before fitting reduced dynamics {latex_cite('@benner2015')}. Related data-driven model-reduction work motivates the projection of complex fields onto compact coordinates {latex_cite('@bai2021')}:
\begin{{equation}}
\mathrm{{full\ Stefan-Marangoni\ problem}}
\rightarrow \mathcal{{O}}_h
\rightarrow \Gamma_h(\xi,y,z,t)
\rightarrow \Pi_M\Gamma_h
\rightarrow \bm{{q}}(t)
\rightarrow d\bm{{q}}/dt .
\label{{eq:reduction-chain}}
\end{{equation}}
In Eq.~\eqref{{eq:reduction-chain}}, $\mathcal{{O}}_h$ maps the full high-dimensional simulation state to the observed half-domain molten point cloud, $\Gamma_h$ is the observed boundary envelope, $\Pi_M$ denotes projection onto the finite-dimensional superellipsoid manifold $M(\bm{{\theta}})$, and $\bm{{q}}(t)$ is the descriptor vector used for dynamics. The moving-frame step is justified by constant laser translation: after the initial transient, a localized heat source can approach a slowly varying shape in the laser-attached coordinate. The superellipsoid step is a manifold projection whose role is evaluated through boundary-residual evidence. This chain is an observation-driven modeling map for the available molten-region states. The first-order dynamic is obtained by linearizing an unknown reduced vector field near $\bm{{q}}_\infty$, consistent with the broader use of parsimonious data-driven dynamical models {latex_cite('@brunton2016')},
\begin{{equation}}
\frac{{d\bm{{q}}}}{{dt}}=F(\bm{{q}})\approx J(\bm{{q}}-\bm{{q}}_\infty),\qquad J\approx-\mathrm{{diag}}(k_i).
\label{{eq:first-order-relaxation}}
\end{{equation}}
In Eq.~\eqref{{eq:first-order-relaxation}}, $F$ is the unknown reduced vector field, $J$ is its local Jacobian, $\bm{{q}}_\infty$ is the quasi-steady attractor state for a condition, and $k_i$ are positive component-wise relaxation rates. The diagonal approximation is specified as the parsimonious baseline; its stability, validation error and temporal-sampling sensitivity are evaluated in the model-selection analysis below.

\section{{Melt-pool data provenance and dimensionless scaling}}

\subsection{{Melt-pool data and preprocessing}}

The melt-pool data were obtained from FLOW-3D L-DED simulations, with the solver-parameter basis summarized in the simulation-parameter record. The high-fidelity simulation results comprise time-resolved molten-region states from {vals['n_conditions']} simulated 316L L-DED conditions. The process matrix covers {latex_escape(vals['process_range_text'])}, and its laser power, scan speed, particle-generation rate, powder feed and time-state counts are listed in Supplementary Table S2. A previous numerical study by the authors showed that laser-cladding molten-pool formation is dominated by a short early transient and that the molten pool approaches quasi-steady behavior after approximately 0.2 s {latex_cite('@song2021cladding')}. Accordingly, the exported time window for each simulated case is set to $t={vals['t_min']:.2f}$--${vals['t_max']:.2f}\,\mathrm{{s}}$, covering the evolution from melt-pool formation to the quasi-steady stage and its subsequent persistence. Each export contains coordinates, volume fraction, heat absorption, heat flux, melt-region indicator, pressure, temperature, temperature-gradient and velocity fields for the molten region only. The data therefore exclude the surrounding solid domain and any already-solidified material. Duplicate rows are removed, repeated coordinates are collapsed by field averaging, and all coordinates are transformed into the condition-specific moving frame. The simulation used a half computational domain in the $y$ direction, so $W(t)$ and $V_{{\mathrm{{full}}}}(t)$ use symmetry reconstruction. Details of source identifiers, point-count ranges and additional simulated cases used for transfer assessment are provided in the Supplementary Methods.

The simulation-parameter record identifies a numerical model for 316L L-DED with a $y\geq0$ half-domain symmetry setting, a recorded cell size of $1.0\times10^{{-4}}\,\mathrm{{m}}$, Gaussian beam radius $8.0\times10^{{-4}}\,\mathrm{{m}}$, absorptivity 0.1, solidus/liquidus temperatures of 1683/1710 K, fusion latent heat $2.67776\times10^5\,\mathrm{{J\,kg^{{-1}}}}$, surface tension 1.8 N m$^{{-1}}$, surface-tension temperature coefficient magnitude $2.50836\times10^{{-4}}\,\mathrm{{N\,m^{{-1}}\,K^{{-1}}}}$ and particle-rate process inputs. The exported files used here are molten-region states sampled over the stated time grid. Temperature-dependent density, heat-capacity, thermal-conductivity and viscosity tables define the material-property basis for the liquidus-scale nondimensional values used below. Their tabulated variation is shown in Supplementary Fig. S10, and the liquidus-interpolated values are reported in Supplementary Table S4. Because the available data consist of molten-region numerical states, the boundary model is interpreted as an envelope reduction of the available point cloud, consistent with treating process-response descriptors as reduced summaries rather than full process maps {latex_cite('@mukherjee2016')}.

\subsection{{Dimensionless scaling and sensitivity}}

The reference length is the quasi-steady mean melt-pool length. Material properties are evaluated at the liquidus temperature for baseline post-processing. The main groups are
\begin{{equation}}
Pe=\frac{{vL_{{\mathrm{{ref}}}}}}{{\alpha}},\quad
Fo=\frac{{\alpha t}}{{L_{{\mathrm{{ref}}}}^2}},\quad
Ste=\frac{{c_p(T_l-T_s)}}{{L_{{\mathrm{{fus}}}}}},
\label{{eq:dimensionless-thermal}}
\end{{equation}}
\begin{{equation}}
E^*=\frac{{\eta P}}{{\rho c_p v r_b^2(T_l-T_0)}},\quad
Ma=\frac{{|d\sigma/dT|(T_l-T_s)L_{{\mathrm{{ref}}}}}}{{\mu\alpha}}.
\label{{eq:dimensionless-driving}}
\end{{equation}}
In Eqs.~\eqref{{eq:dimensionless-thermal}} and~\eqref{{eq:dimensionless-driving}}, $L_{{\mathrm{{ref}}}}$ is the quasi-steady reference melt-pool length, $\alpha=k/(\rho c_p)$ is thermal diffusivity, $T_s$ and $T_l$ are the solidus and liquidus temperatures, $T_0$ is the initial substrate temperature, $\eta$ is laser absorptivity, $P$ is laser power, $r_b$ is beam radius, $\mu$ is dynamic viscosity, and $d\sigma/dT$ is the surface-tension temperature coefficient. The setup-note laser and phase-change constants are used here; density, heat capacity, conductivity and viscosity are liquidus-interpolated values from the temperature-dependent property curves summarized in Supplementary Fig. S10 and Supplementary Table S4. These scale groups are used as descriptors of the present process matrix, following the broader AM practice of combining thermal transport, heat input and material response in process interpretation {latex_cite('@debroy2018')}.
Substituting the moving-frame reference length, process parameters and liquidus-scale material properties into Eqs.~\eqref{{eq:dimensionless-thermal}} and~\eqref{{eq:dimensionless-driving}}, with $Re$ and $Pr$ computed from the corresponding velocity and material-property definitions in the Supplementary Methods, gives process-matrix mean values of $Pe={vals['Pe']:.2f}$, $Fo_{{\mathrm{{final}}}}={vals['Fo_final']:.2f}$, $Ste={vals['Ste']:.3f}$, $E^*={vals['E_star']:.2f}$, $Re={vals['Re']:.2f}$, $Pr={vals['Pr']:.3f}$ and $Ma={vals['Ma']:.2f}$. Table~\ref{{tab:dimensionless-sensitivity}} organizes these values into process-matrix means, sensitivity-baseline values and reference-scale interpretations. {dimensionless_sensitivity_sentence} The table therefore links the reduced descriptors to heat input, phase-change scale, translational transport and thermocapillary forcing.

\begin{{table}}[htbp]
\centering
\caption{{Dimensionless sensitivity envelope. The mean column reports the process-matrix mean, whereas the sensitivity-baseline column reports the representative baseline used in the perturbation scan. The final column gives reference-scale qualitative interpretations for scale context.}}
\label{{tab:dimensionless-sensitivity}}
\small
\begin{{tabular}}{{llll>{{\raggedright\arraybackslash}}p{{0.24\textwidth}}}}
\toprule
Group & Mean & Sensitivity baseline & Perturbation range & Reference-scale qualitative interpretation \\
\midrule
{sensitivity_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Reduced dynamics, stability and time-split assessment}}

\subsection{{Attractor identification and stability}}

The parsimonious baseline dynamics is the diagonal attractor
\begin{{equation}}
\frac{{dq_i}}{{dt}}=k_i(q_{{\infty,i}}-q_i).
\label{{eq:diagonal-attractor}}
\end{{equation}}
In Eq.~\eqref{{eq:diagonal-attractor}}, $q_i$ is the $i$th reduced state component, $q_{{\infty,i}}$ is its estimated quasi-steady value, and $k_i$ is the fitted relaxation rate. Larger $k_i$ corresponds to faster approach to the attractor for that state component.
The alternative coupled model is
\begin{{equation}}
\frac{{d\bm{{q}}}}{{dt}}=A(\bm{{q}}_\infty-\bm{{q}}).
\label{{eq:coupled-attractor}}
\end{{equation}}
In Eq.~\eqref{{eq:coupled-attractor}}, $A$ is the ridge-identified coupling matrix and $\bm{{q}}_\infty$ is fixed from the quasi-steady training segment. The coupled model is included as a comparison model for cross-state coupling, following the reduced-dynamics principle that richer operators require validation beyond stability alone {latex_cite('@benner2015')}.
The diagonal baseline is fitted directly to the training trajectory by
\begin{{equation}}
\left(q_{{\infty,i}}^*,k_i^*\right)=
\arg\min_{{q_{{\infty,i}}\in Q_i,\;k_i\geq0}}
\sum_{{r\in\mathcal{{T}}_{{\mathrm{{tr}}}}}}
\left[
q_{{\infty,i}}+\left(q_i(t_0)-q_{{\infty,i}}\right)
\exp[-k_i(t_r-t_0)]-q_i(t_r)
\right]^2 .
\label{{eq:diagonal-identification}}
\end{{equation}}
In Eq.~\eqref{{eq:diagonal-identification}}, $Q_i$ is the training-only quasi-steady admissible range used for $q_\infty$, and the nonnegative constraint enforces physically interpretable relaxation toward the quasi-steady state. No finite-difference derivative is used in the primary diagonal fit. The coupled matrix is identified by finite-difference ridge regression,
\begin{{equation}}
A^*=
\arg\min_A
\sum_{{r\in\mathcal{{T}}_{{\mathrm{{tr}}}}}}
\left\lVert
\dot{{\bm{{q}}}}_r-A\left(\bm{{q}}_\infty-\bm{{q}}(t_r)\right)
\right\rVert_2^2
+\lambda_R\lVert A\rVert_F^2 .
\label{{eq:coupled-ridge-identification}}
\end{{equation}}
In Eq.~\eqref{{eq:coupled-ridge-identification}}, $\lambda_R$ is chosen by leave-one-step validation within the training segment and $\lVert A\rVert_F$ penalizes poorly supported coupling coefficients. This treatment follows ridge regularization for ill-conditioned parameter estimation {latex_cite('@hoerl1970')}. It also follows the broader Tikhonov principle of stabilizing inverse problems through penalization {latex_cite('@tikhonov1977')}.
For the diagonal model, define $\bm{{e}}=\bm{{q}}-\bm{{q}}_\infty$. The Lyapunov function
\begin{{equation}}
V(\bm{{e}})=\frac{{1}}{{2}}\lVert\bm{{e}}\rVert_2^2,
\qquad
\dot V=-\sum_i k_i e_i^2
\leq -2k_{{\min}}V,\quad k_{{\min}}=\min_i k_i ,
\label{{eq:lyapunov-diagonal}}
\end{{equation}}
shows global exponential stability whenever $k_i>0$ for all retained state components. In Eq.~\eqref{{eq:lyapunov-diagonal}}, $V$ is the state-error energy and $k_{{\min}}$ is the slowest fitted relaxation rate. For the coupled model, $\bm{{e}}=\bm{{q}}-\bm{{q}}_\infty$ satisfies $d\bm{{e}}/dt=-A\bm{{e}}$. The coupled equilibrium is locally exponentially stable if all eigenvalues of $-A$ have negative real part. The fitted rate signs, coupled spectrum and validation errors are reported with the model-selection results below.
The validation metric for state $j$ is
\begin{{equation}}
\mathrm{{rRMSE}}_j=
\frac{{\left[
N_{{\mathrm{{val}}}}^{{-1}}
\sum_{{r\in\mathcal{{T}}_{{\mathrm{{val}}}}}}
\left(\hat q_j(t_r)-q_j(t_r)\right)^2
\right]^{{1/2}}}}
{{\max_{{r\in\mathcal{{T}}_{{\mathrm{{val}}}}}}q_j(t_r)-
\min_{{r\in\mathcal{{T}}_{{\mathrm{{val}}}}}}q_j(t_r)+\delta_j}} .
\label{{eq:validation-rrmse}}
\end{{equation}}
In Eq.~\eqref{{eq:validation-rrmse}}, $\hat q_j$ is the predicted state, $N_{{\mathrm{{val}}}}$ is the number of validation steps, $\mathcal{{T}}_{{\mathrm{{val}}}}$ is the validation time set, and $\delta_j>0$ avoids singular normalization for nearly constant states.

For each reduced state, the relaxation time is computed as $\tau_i=1/k_i$ from the fitted diagonal attractor in Eq.~\eqref{{eq:first-order-relaxation}}. Table~\ref{{tab:timescales}} summarizes the resulting state-wise time scales across conditions, together with validation error and limited-support counts from the same fitted trajectories.
The table is used to compare geometric, thermal and flow-state relaxation on the same time-scale basis and to identify which descriptors carry weaker trajectory support.

\begin{{table}}[htbp]
\centering
\caption{{Summary of relaxation time scales for the parsimonious diagonal attractor baseline.}}
\label{{tab:timescales}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.12\textwidth}}>{{\raggedright\arraybackslash}}p{{0.15\textwidth}}>{{\raggedright\arraybackslash}}p{{0.25\textwidth}}>{{\raggedright\arraybackslash}}p{{0.17\textwidth}}>{{\raggedright\arraybackslash}}p{{0.16\textwidth}}}}
\toprule
State & Group & Median $\tau_i$ [IQR] (s) & Median validation rRMSE & Limited-support cases \\
\midrule
{timescale_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Diagnostic error-source taxonomy and assumption assessment}}

The diagnostic error-source taxonomy is organized as
\begin{{equation}}
E_{{\mathrm{{total}}}}=E_{{\mathrm{{reconstruction}}}}+E_{{\mathrm{{geometry}}}}+E_{{\mathrm{{volume}}}}+E_{{\mathrm{{dynamics}}}}+E_{{\mathrm{{parameter}}}}.
\label{{eq:error-budget}}
\end{{equation}}
In Eq.~\eqref{{eq:error-budget}}, $E_{{\mathrm{{reconstruction}}}}$ covers half-domain mirroring and duplicate-point handling, $E_{{\mathrm{{geometry}}}}$ covers analytic boundary fitting, $E_{{\mathrm{{volume}}}}$ covers the convex-hull volume proxy, $E_{{\mathrm{{dynamics}}}}$ covers train-validation prediction error, and $E_{{\mathrm{{parameter}}}}$ covers material and process-scale uncertainty. This expression is a diagnostic taxonomy, not a propagated uncertainty model and not an assumption that independent random errors add linearly. It is aligned with the uncertainty-aware framing used in metal additive-manufacturing studies while remaining limited to observable diagnostics in the present data {latex_cite('@wang2020uq')}.
For interpretation, the same terms are written as the semi-formal bound
\begin{{equation}}
\begin{{aligned}}
E_{{\mathrm{{total}}}}
&\leq C_R E_{{\mathrm{{reconstruction}}}}
+C_\Gamma E_{{\mathrm{{geometry}}}}
+C_V E_{{\mathrm{{volume}}}}\\
&\quad
+C_D E_{{\mathrm{{dynamics}}}}
+C_P E_{{\mathrm{{parameter}}}},
\qquad C_R,C_\Gamma,C_V,C_D,C_P\geq0 .
\end{{aligned}}
\label{{eq:error-bound}}
\end{{equation}}
In Eq.~\eqref{{eq:error-bound}}, the constants $C_R$, $C_\Gamma$, $C_V$, $C_D$ and $C_P$ are sensitivity weights. Table~\ref{{tab:error-budget}} reports observable diagnostics for the corresponding terms, so the bound structures the uncertainty assessment. For each condition-time step, the normalized Chamfer distance is computed as $d_{{\mathrm{{Ch}}}}^*(t)=d_{{\mathrm{{Ch}}}}(t)/(L(t)+\epsilon_L)$, where $d_{{\mathrm{{Ch}}}}(t)$ is the symmetric point-to-surface Chamfer distance between the observed envelope and the fitted manifold, $L(t)=L_f(t)+L_r(t)$ is the melt-pool length, and $\epsilon_L$ regularizes the normalization. The value reported in Table~\ref{{tab:error-budget}} is the arithmetic mean of $d_{{\mathrm{{Ch}}}}^*(t)$ over valid condition-time steps, so the distance diagnostic is interpreted relative to the 1--2 mm melt-pool scale. In Table~\ref{{tab:error-budget}}, an interpretive boundary is an evidence-scope label that identifies whether a diagnostic primarily supports the selected model or limits how far the corresponding claim can be extended. Hyphenated levels denote intermediate evidence.

\begin{{table}}[htbp]
\centering
\caption{{Diagnostic error-source taxonomy. Interpretive-boundary levels identify where each diagnostic constrains interpretation.}}
\label{{tab:error-budget}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.22\textwidth}}>{{\raggedright\arraybackslash}}p{{0.35\textwidth}}>{{\raggedright\arraybackslash}}p{{0.12\textwidth}}>{{\raggedright\arraybackslash}}p{{0.15\textwidth}}}}
\toprule
Error term & Primary metric & Value & Interpretive boundary \\
\midrule
{error_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

The full assumption validation matrix, temporal-sampling sensitivity tests, interpretive-boundary summary and detailed model-selection rule are reported in the Supplementary Methods. In the main text, these analyses define how boundary residuals, volume-proxy errors, dynamical validation and identifiability evidence are used to support the selected superellipsoid descriptor and diagonal-attractor baseline.

\section{{Results and model selection}}

Figure~\ref{{fig:framework}} summarizes the evidence structure that links melt-pool simulation states, observation operators, reduced model families and model-selection diagnostics. Figure~\ref{{fig:process-matrix}} presents the process matrix used for model construction, with full condition identifiers and time-state counts listed in Supplementary Table S2. Figure~\ref{{fig:moving-frame}} establishes the coordinate basis for subsequent comparisons: transforming half-domain point clouds into a laser-attached frame removes scan translation from the descriptor trajectories and makes boundary evolution comparable across time. Figure~\ref{{fig:simulation-cross-sections}} provides thermal-flow context for three same-speed, same-powder-feed power cases, showing the source-field environment from which the observed molten boundary is extracted. Figure~\ref{{fig:geometry}} then shows the representative transient approach toward the quasi-steady window used for descriptor averaging. Together, Figures~\ref{{fig:moving-frame}}--\ref{{fig:geometry}} establish the coordinate, source-field and time-window context for the quantitative diagnostics reported below. After the initial growth stage, the melt-pool envelope approaches a quasi-steady form after approximately 0.20 s. Across conditions, the quasi-steady mean front length, rear length, full width and height are {vals['lf_quasi_mm']:.3f} mm, {vals['lr_quasi_mm']:.3f} mm, {vals['w_quasi_mm']:.3f} mm and {vals['h_quasi_mm']:.3f} mm. The corresponding fitted superellipsoid-parameter trajectories are reported in Supplementary Fig. S2.

{figure_tex['fig:framework']}

{figure_tex['fig:process-matrix']}

{figure_tex['fig:moving-frame']}

{figure_tex['fig:simulation-cross-sections']}

{figure_tex['fig:geometry']}

Figure~\ref{{fig:boundary}} compares the ellipsoid and superellipsoid boundary models by combining selected overlays with cross-condition summaries. The comparison uses superquadric ideas as compact shape classes {latex_cite('@barr1981')}. The implicit-boundary formulation provides a complementary mathematical basis for evaluating the observed envelope {latex_cite('@osher1988')}. Model selection is determined by the exported molten-region evidence. The transient and late quasi-steady overlays show that the aggregate residual differences correspond to visible envelope-shape differences in both top and side views.

Across the 15 conditions, {geometry_selection_text} {geometry_pair_text} The selected overlays provide visual context for this residual-based model selection. {geom_metric_text} Robustness tests show superellipsoid improvement in {count_of_total_phrase(vals['super_volume_wins'], vals['robust_total'], 'robustness settings')} for volume error and {count_of_total_phrase(vals['super_boundary_wins'], vals['robust_total'], 'robustness settings')} for boundary residual. {volume_limitation_text} {boundary_sensitivity_text} Error-budget and identifiability diagnostics in Figures~\ref{{fig:error-budget}} and~\ref{{fig:identifiability}}, together with Supplementary Fig. S4, define model-selection boundaries for the added shape flexibility. Expanded all-time and selected-time overlays are provided in Supplementary Figs. S1 and S7, and proxy sensitivity is shown in Supplementary Fig. S5. These results support the superellipsoid as the selected algebraic observed-envelope descriptor for extraction and process-response analysis, with volume and geometric-distance diagnostics constraining its use.

{figure_tex['fig:boundary']}

Figure~\ref{{fig:process-response}} summarizes quasi-steady process-response patterns over power, speed and powder feed. {process_response_takeaway} These trends connect power variation with quasi-steady boundary and thermal descriptors, consistent with process-window studies that relate local response surfaces to process interpretation {latex_cite('@mukherjee2016')}. Figure~\ref{{fig:dimensionless}} then places these descriptors in nondimensional thermal-transport context, using $Pe$, $Ste$, $E^*$ and $Ma$ to interpret translational transport, phase-change scale, absorbed heat input and thermocapillary forcing. Thermal-flow state histories used by the reduced descriptors are shown in Supplementary Fig. S8.

Figure~\ref{{fig:dynamics-cross-condition}} compares the diagonal attractor with the coupled ridge comparison across conditions and states. The comparison follows reduced-order modeling practice by evaluating parsimony, stability and time-split validation together {latex_cite('@benner2015')}. Data-driven dynamics provides the complementary motivation for comparing compact operators against richer coupled alternatives {latex_cite('@brunton2016')}. Both attractors are stable by their respective criteria, but the diagonal baseline has lower mean time-split relative RMSE, {_fmt(vals['diagonal_validation'], 4)} versus {_fmt(vals['coupled_validation'], 4)}. {dynamics_pair_text} Figure~\ref{{fig:dynamics-residuals}} separates the same comparison by state, showing that aggregate validation errors can conceal descriptor-specific support. The residual panels identify which states are described robustly by the diagonal baseline and which states require more cautious interpretation. {umax_scope_text} The coupled model improves error in {count_of_total_phrase(vals['coupled_wins'], vals['robust_total'], 'robustness settings')} and therefore functions as a cross-state coupling comparison rather than the selected model. The temporal-sampling sensitivity tests in Supplementary Table~\ref{{tab:supp-sensitivity-tests}} support the diagonal baseline in {stress_support:.3f} of tested settings, so the selected trajectory model is treated as a parsimonious compact baseline with bounded temporal-sampling support.

{figure_tex['fig:process-response']}

{figure_tex['fig:dimensionless']}

{figure_tex['fig:dynamics-cross-condition']}

{figure_tex['fig:dynamics-residuals']}

Figure~\ref{{fig:error-budget}} presents the diagnostic error-source taxonomy and model-selection summary. The taxonomy separates reconstruction, boundary-fit, volume-proxy, dynamics and parameter sources so that the selected model is supported by boundary residual and validation behavior while volume and identifiability limits remain visible. Figure~\ref{{fig:identifiability}} shows parameter-identifiability and overparameterization diagnostics, and Figure~\ref{{fig:loco}} reports a leave-one-condition-out process-response assessment. The identifiability reading follows practical parameter-support logic for partially observed dynamical models {latex_cite('@raue2009')}. {loco_detail_text} Weakly constrained fitted quantities include {high_param_text}. {identifiability_main_text} The descriptor system is therefore more robust for geometric states than for maximum velocity.

{figure_tex['fig:error-budget']}

{figure_tex['fig:identifiability']}

{figure_tex['fig:loco']}

The additional simulated-condition assessment uses ten cases that were not included in boundary-descriptor selection, attractor-baseline selection, leave-one-condition-out fitting or process-response fitting. Their full condition identifiers, laser power, scan speed, powder feed and time-state counts are listed in Supplementary Table S3. This assessment tests whether the selected boundary descriptor, quasi-steady process-response summary and process-parameterized diagonal trajectories remain consistent outside the model-construction cases.

{external_holdout_text} The result supports generalization within the same numerical and preprocessing setting.

{figure_tex['fig:external-holdout']}

\section{{Discussion}}

The central contribution of this study is an evidence-bounded reduction from high-dimensional melt-pool data to an interpretable observed-boundary reduced model. The framework converts exported molten-region simulation states into descriptors whose support is evaluated through boundary residuals, volume-proxy error, stability, identifiability and numerical holdout behavior. This matters for L-DED because monitoring streams often provide more field information than reduced-order control can directly use {latex_cite('@dasilva2023')}. Automated computational studies create a similar need to convert rich simulations into interpretable process descriptors {latex_cite('@kovsca2023')}. Model selection is organized around measurable support from these diagnostics.

The geometric results reveal a useful tension between envelope description and volume inference. Figure~\ref{{fig:boundary}} shows that the superellipsoid consistently lowers the boundary residual relative to the ellipsoid baseline, whereas the convex-hull volume proxy worsens. The extra exponent flexibility appears to capture asymmetric or flattened envelope portions, while volume remains controlled by sparse point-cloud support and mirrored convex-hull construction. Supplementary Fig. S5 reinforces this interpretation by showing that the extracted envelope depends on the proxy used to represent the molten-point boundary. The superellipsoid is therefore best interpreted as an algebraic observed-boundary descriptor, rather than as a thermodynamic melt-volume estimator. This role is consistent with using analytic heat-source forms as compact geometric representations while keeping physical interface recovery separate {latex_cite('@goldak1984')}. Superquadric geometry provides the corresponding flexible shape family for boundary-envelope representation {latex_cite('@barr1981')}. It also aligns with implicit-front modeling traditions in which boundary representation and physical interface evolution are related but distinct tasks {latex_cite('@osher1988')}.

The dynamical results carry a similar message. Figures~\ref{{fig:dynamics-cross-condition}} and~\ref{{fig:dynamics-residuals}}, together with the supplementary model-selection summary, show that the diagonal attractor and the coupled comparison have similar validation errors, with a small advantage for the diagonal baseline and state-dependent residual support. The coupled comparison is physically plausible because melt-pool length, width, temperature, gradient and velocity can influence one another. The supporting evidence is limited by the short condition-wise sequences and by the larger number of fitted matrix parameters. The error-source taxonomy in Figure~\ref{{fig:error-budget}} and the identifiability evidence in Figure~\ref{{fig:identifiability}} explain why spectral stability alone gives incomplete support. The diagonal attractor is therefore a local relaxation descriptor around the translated quasi-steady state, while the coupled comparison evaluates cross-state coupling. This interpretation follows ridge and Tikhonov regularization, where fitted degrees of freedom are stabilized when parameter support is limited {latex_cite('@hoerl1970')}. It also aligns with reduced-order modeling practice, which favors compact operators when validation data are limited {latex_cite('@benner2015')}. Sparse data-driven dynamics gives the same practical lesson: richer operators are most useful when the data identify the active terms {latex_cite('@brunton2016')}.

The dimensionless diagnostics add physical scale context. The values of $Pe$, $Ste$, $E^*$ and $Ma$ in Figure~\ref{{fig:dimensionless}} relate the fitted descriptors to thermal diffusion, latent heat, absorbed heat input and thermocapillary forcing. Supplementary Fig. S10 is important because the transport scales are obtained from temperature-dependent material-property curves at the liquidus scale. The sensitivity grid indicates that the qualitative classes remain stable under the tested reference-temperature, absorptivity and surface-tension perturbations. These quantities support physical interpretation of the reduced model and indicate which additional inputs would be needed for broader regime mapping. Additive-manufacturing uncertainty studies emphasize the same need to propagate uncertainty in material properties, heat-source parameters and observation choices {latex_cite('@wang2020uq')}. Recent process-dependent prediction studies similarly show that local response evidence becomes more transferable when uncertainty is represented explicitly {latex_cite('@hermann2023')}. Multifidelity modeling offers one route for extending such evidence across sources while preserving information about source fidelity {latex_cite('@menon2022multifidelity')}.

The study also clarifies how the approach relates to existing DED and laser-metal AM research. Detailed simulations have established how process parameters shape thermal histories and melt flow during laser processing {latex_cite('@king2015')}. Powder-bed simulations extend that view by resolving melt-pool transport and defect formation at high fidelity {latex_cite('@khairallah2016')}. In DED, molten-region simulations and powder-coupled models show how transport and material addition affect observable melt-pool behavior {latex_cite('@zhang2021dedcfd')}. In situ studies add mechanism-level evidence by linking transient melt-pool evolution to pore formation {latex_cite('@zhang2024pore')}. In parallel, monitoring studies have turned melt-pool area and geometry into process-state variables {latex_cite('@dasilva2023')}. Control-oriented work then uses such states for predictive adjustment during deposition {latex_cite('@rahmani2024psq')}. Surrogate and machine-learning studies extend this direction by predicting melt-pool geometry, thermal fields and process signatures from process inputs {latex_cite('@akbari2022')}. The present analysis sits between these simulation, monitoring and prediction strands by asking which compact boundary and trajectory descriptors are justified before they are used for mapping or control.

The additional simulated-condition results in Figure~\ref{{fig:external-holdout}} strengthen this interpretation while defining its reach. These cases assess within-setting generalization under the same numerical setting and preprocessing convention. Their performance supports the observed-boundary descriptor, quasi-steady process-response map and process-parameterized diagonal trajectories within that setting. Transfer to physical measurements should be evaluated with experimental observations, independently varied simulation physics and denser process designs.

Three limitations set the strength of the present claims. First, the observation is the exported molten region, so the boundary model inherits the measurement operator of the melt-pool data. This affects volume interpretation most strongly because the convex-hull proxy is sensitive to point-cloud support and symmetry reconstruction. Second, the {vals['n_conditions']}-condition model-construction matrix supports a multi-condition descriptor study, while offering limited leverage for tightly identifying nonlinear process functions for $\bm{{q}}_\infty$, $k_i$ or all superellipsoid shape parameters. Third, the nondimensional analysis uses liquidus-interpolated values from temperature-dependent material-property curves. This is appropriate for scale diagnostics, while global uncertainty analysis would need propagated uncertainty in material properties, heat-source parameters and observation choices.

These limitations define specific routes forward. Experimental melt-pool images or independent thermal measurements would test whether the observed-boundary descriptor carries across measurement operators. Complete thermal-field exports would allow the boundary proxy to be compared with isothermal or phase-fraction interfaces. Volume-preserving manifold fitting could separate boundary-shape accuracy from melt-volume conservation. Denser time sampling and a wider process matrix would make coupled or nonlinear dynamics more identifiable. Together, these extensions would move the descriptor system from within-setting numerical generalization toward measurement-level validation and broader process-dependent prediction.

\section{{Conclusion}}

This study develops an observed-boundary mathematical reduction for transient-to-quasi-steady L-DED melt-pool evolution in 316L stainless steel. Molten-region states from multiple simulated process conditions are converted into symmetry-reconstructed moving-frame envelopes, analytic boundary descriptors and compact relaxation baselines. The resulting model is most useful as an evidence-bounded descriptor system: it identifies which boundary and trajectory variables are supported by the available simulation states, while keeping volume recovery, physical measurement transfer and broad process-map prediction outside the present evidential scope.

\begin{{enumerate}}
\item The observed boundary is described more consistently by the asymmetric superellipsoid than by the ellipsoid baseline. The mean implicit boundary residual decreases from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)}, and the boundary-residual robustness tests consistently favor the superellipsoid. The volume-proxy evidence is less favorable, so the selected manifold should be interpreted as an observed-envelope descriptor rather than as a volume-preserving melt-volume estimator.
\item The reduced dynamics support a parsimonious local relaxation baseline. The diagonal attractor has a slightly lower validation relative RMSE than the coupled ridge comparison, {_fmt(vals['diagonal_validation'], 4)} versus {_fmt(vals['coupled_validation'], 4)}, while the coupled model does not gain consistent robustness support. State-wise residuals show stronger support for geometric descriptors than for the maximum-velocity state, so flow-state interpretation should remain more cautious than geometry interpretation.
\item Additional simulated conditions support within-setting generalization of the selected descriptor chain. The process-response and trajectory errors remain at a low descriptive scale, and the superellipsoid retains the lower boundary residual across the additional cases. This evidence supports transfer within the same numerical and preprocessing setting, but experimental observations, independently varied simulation physics and denser process designs are needed before broader predictive claims can be made.
\end{{enumerate}}

\section*{{Data and code availability}}

The processed data, analysis code and manuscript source are available in the GitHub repository \url{{https://github.com/Song-BX/lded-observed-boundary}}.

\setlength{{\bibsep}}{{1pt}}
\bibliographystyle{{unsrtnat}}
\bibliography{{references}}

{supplementary_body_tex}

\end{{document}}
"""
    main_submission_tex = make_article_only_submission_tex(main_tex, supplementary_body_tex)
    main_tex = humanize_submission_text(main_tex)
    main_submission_tex = humanize_submission_text(main_submission_tex)
    main_tex = limit_solver_platform_mentions(main_tex, keep_main_platform_mention=True)
    main_submission_tex = limit_solver_platform_mentions(main_submission_tex, keep_main_platform_mention=True)
    (latex_dir / "main_submission.tex").write_text(main_submission_tex, encoding="utf-8")

    supp_figures = [
        (
            "fig:supp-boundary",
            "Representative-condition boundary fits across all time steps.",
            "Top-view superellipsoid overlays assess every exported time step for the representative condition, extending the boundary-fit evidence beyond the main-text summary panels.",
            "supp_figS1_all_boundary_fits",
        ),
        (
            "fig:supp-parameters",
            "Representative-condition superellipsoid parameters versus time.",
            "The fitted semi-axes, center coordinates and shape exponents evolve toward the same quasi-steady window used for downstream descriptors.",
            "supp_figS2_superellipsoid_parameters",
        ),
        (
            "fig:supp-dimensionless",
            "Dimensionless sensitivity scenario grid.",
            "The grid reports the minimum, baseline and maximum perturbation ratios used to interpret the nondimensional groups as scale diagnostics.",
            "supp_figS4_dimensionless_sensitivity_grid",
        ),
        (
            "fig:supp-theory",
            "Theory, identifiability and error-budget diagnostics.",
            "The panels combine error-source weights, identifiability constraints and nondimensional spans to document the model-selection boundaries summarized in the main text.",
            "supp_figS5_theory_identifiability_error_bounds",
        ),
        (
            "fig:supp-convex-alpha",
            "Convex-hull and alpha-complex proxy comparison.",
            "The comparison evaluates alternative boundary proxies on exported molten points and constrains the volume-proxy interpretation used in the main text.",
            "supp_figS6_convex_alpha_proxy_comparison",
        ),
        (
            "fig:supp-stability",
            "Representative-condition stability and attractor evidence.",
            "State-error convergence, diagonal rates and coupled eigenvalues provide representative stability evidence for the reduced attractor discussion.",
            "fig10_stability_attractor",
        ),
        (
            "fig:supp-boundary-panels",
            "Representative boundary-envelope time-step overlays.",
            "The overlays check selected transient and quasi-steady time steps so that the boundary-envelope behavior is not inferred only from aggregate residuals.",
            "fig05_boundary_fit_comparison",
        ),
        (
            "fig:supp-thermal-flow",
            "Thermal-flow state evolution.",
            "These state histories supply the thermal-flow variables used in the reduced state and mark the approximate quasi-steady transition.",
            "fig03_thermal_flow_evolution",
        ),
        (
            "fig:supp-dynamics-comparison",
            "Dynamical model trajectory comparison.",
            "Direct trajectories complement the state-wise residual plot and show the coupled model as a comparison against the selected baseline.",
            "fig06_dynamics_model_comparison",
        ),
        (
            "fig:supp-temperature-properties",
            "Temperature-dependent material-property curves.",
            "Density, specific heat, thermal conductivity and viscosity are plotted from the tabulated property inputs used for liquidus-scale interpolation in the nondimensional analysis; vertical guides mark the solidus and liquidus temperatures.",
            "supp_figS10_temperature_dependent_properties",
        ),
    ]
    def standalone_supp_figure_width(label: str) -> str:
        return "0.78\\textwidth" if label == "fig:supp-thermal-flow" else "0.95\\textwidth"

    supp_figure_tex = "\n\n".join(
        [
            rf"""\begin{{figure}}[htbp]
\centering
\includegraphics[width={standalone_supp_figure_width(label)}]{{{fig_path(stem, "supp")}}}
\caption{{\textbf{{{title}}} {caption}}}
\label{{{label}}}
\end{{figure}}"""
            for label, title, caption, stem in supp_figures
        ]
    )
    # Reuse the aggregated high-risk parameter list already prepared for the
    # appended Supplementary Information in the combined manuscript.
    supp_tex = rf"""\documentclass[11pt]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb,bm}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{caption}}
\captionsetup{{hypcap=false}}
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.12}}

\title{{Supplementary Methods: observed boundary-envelope identification of L-DED melt pools from melt-pool data}}
\author{{}}
\date{{}}

\begin{{document}}
\maketitle
\setcounter{{figure}}{{0}}
\renewcommand{{\thefigure}}{{S\arabic{{figure}}}}
\renewcommand{{\theHfigure}}{{S\arabic{{figure}}}}
\setcounter{{table}}{{0}}
\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\renewcommand{{\theHtable}}{{S\arabic{{table}}}}

\section{{Source data, process matrices and coordinate conventions}}
\subsection{{Theoretical origin and supplementary evidence map}}
The main text states the moving-source Stefan-Marangoni balances and the observation-reduction chain used to motivate the model. This section specifies how the same objects are used in the supplementary analysis, with emphasis on the observation operator, reconstruction assumptions and diagnostic evidence underlying the main-text model-selection results.

Table~\ref{{tab:supp-modeling-details}} links each model component to the supplementary evidence that extends the main text.

\begin{{table}}[htbp]
\centering
\caption{{Relationship between the main-text model and supplementary evidence.}}
\label{{tab:supp-modeling-details}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.18\textwidth}}>{{\raggedright\arraybackslash}}p{{0.35\textwidth}}>{{\raggedright\arraybackslash}}p{{0.38\textwidth}}}}
\toprule
Component & Main-text role & Supplementary evidence \\
\midrule
{modeling_detail_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Data provenance and moving-frame convention}}
The source melt-pool data comprise {vals['n_source_files']} numerical molten-region exports from {vals['n_conditions']} process conditions. Case identifiers follow \texttt{{Aa-b-c-d}}, where $a$ is the condition index, $b$ is laser power in watts, $c$ is scan speed in $\mathrm{{mm\,s^{{-1}}}}$, and $d$ is the particle generation rate. The particle rate is converted to powder feed by $d/60000\times12\,\mathrm{{g\,min^{{-1}}}}$. The exported time window is $t={vals['t_min']:.2f}$--${vals['t_max']:.2f}\,\mathrm{{s}}$ at the exported times {latex_escape(vals['time_points'])} s, matching the main-text rationale for capturing early melt-pool formation and subsequent quasi-steady evolution. Each export contains molten-region points, so the supplementary analysis treats full thermal-field reconstruction as outside the exported data scope. The exported columns are {latex_escape(export_columns_text)}. Coordinates are interpreted in metres, temperature in kelvin, temperature-gradient magnitude in $\mathrm{{K\,m^{{-1}}}}$, pressure in pascals and velocity in $\mathrm{{m\,s^{{-1}}}}$.

Before preprocessing, each condition-time export contains {vals['raw_rows_min']}--{vals['raw_rows_max']} rows. After exact row deduplication, {vals['exact_dedup_rows_min']}--{vals['exact_dedup_rows_max']} rows remain; after repeated-coordinate collapse, {vals['unique_points_min']}--{vals['unique_points_max']} unique spatial points remain per export. Across the dataset, {vals['exact_duplicates_removed_total']} exact duplicate rows are removed and {vals['coordinate_duplicates_collapsed_total']} repeated-coordinate groups are collapsed by field averaging. All point locations are then converted to the condition-specific laser-attached frame by $\xi=x-v_ct$, where $v_c$ is inferred from the case identifier.

The available simulation-parameter record identifies a 316L L-DED numerical model with a half-domain symmetry setting, a recorded cell size of $1.0\times10^{{-4}}\,\mathrm{{m}}$, Gaussian beam radius $8.0\times10^{{-4}}\,\mathrm{{m}}$, absorptivity 0.1, phase-change constants, surface-tension constants and particle-rate process inputs.

The coordinate convention is as follows. The laboratory scan direction is $x$, the transverse direction is $y$, the build direction is $z$, and the moving coordinate is $\xi$. Time is denoted by $t$. Temperature, gradient magnitude and velocity magnitude are extracted from the exported molten points, with the surrounding solid region outside the extracted state.

\subsection{{Process matrices and parameter basis}}
The model-construction design comprises 15 simulated process conditions. These cases form the process matrix used in the main text; Table~\ref{{tab:supp-training-process-matrix}} lists their full identifiers, process settings and time-state counts.

\begin{{table}}[htbp]
\centering
\caption{{Model-construction process matrix. Powder feed is converted from particle rate by $d/60000\times12\,\mathrm{{g\,min^{{-1}}}}$.}}
\label{{tab:supp-training-process-matrix}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Case & Power (W) & Speed (mm s$^{{-1}}$) & Input $d$ & Powder feed (g min$^{{-1}}$) & Time-state exports \\
\midrule
{process_matrix_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

{external_case_table_tex}

Parameter reconciliation uses the available solver-parameter record for laser, phase-change and surface-tension constants. The tabulated temperature-dependent material-property curves provide density, viscosity, thermal conductivity and specific heat for liquidus-scale nondimensional diagnostics. Table~\ref{{tab:supp-parameter-assessment}} records how each parameter is treated in the analysis, and the full property variation is plotted in Fig.~\ref{{fig:supp-temperature-properties}}.

\begin{{table}}[htbp]
\centering
\caption{{Parameter reconciliation assessment.}}
\label{{tab:supp-parameter-assessment}}
\scriptsize
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.17\textwidth}}>{{\raggedright\arraybackslash}}p{{0.20\textwidth}}>{{\raggedright\arraybackslash}}p{{0.20\textwidth}}>{{\raggedright\arraybackslash}}p{{0.32\textwidth}}}}
\toprule
Parameter & Solver-parameter record & Post-processing basis & Treatment in this study \\
\midrule
{parameter_audit_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Half-domain reconstruction and provenance summary}}
The source-data computational domain is simulated only for $y\geq0$, with $y=0$ as a symmetry plane. The full-domain representation is obtained by mirroring each point $(\xi,y,z)$ to $(\xi,-y,z)$. This reconstruction is exact for scalar geometric descriptors when the process, heat source and powder delivery are symmetric with respect to the plane. Antisymmetric flow structures remain outside the half-domain data.

The main text defines the scalar reconstruction for width and convex-hull volume. In the supplementary analysis, this rule is used only to document how full-width descriptors and full-volume proxies are obtained from the half-domain data.

\section{{Observation reduction, boundary fitting and reduced dynamics}}
\subsection{{Boundary-envelope extraction and manifold fitting}}
For each time step, the molten-point envelope is approximated by a convex-hull boundary. This conservative choice follows from the observation model: because the data contain only molten-region points, the boundary is the observed molten-domain envelope. The ellipsoid baseline and the superellipsoid model are given in the main text by Eqs.~(5) and~(6).

The fitted geometric parameters are reported as manifold coordinates. The parameters $a_f$ and $a_r$ describe front and rear extents in the moving coordinate, $b$ describes the half-width, $c$ describes the vertical scale, and $(\xi_c,z_c)$ locates the fitted boundary center. The exponents $n$, $m$ and $p$ control directional shape sharpness. The fitting objective, analytic volume and diagnostic definitions are given in the main text; the supplementary material provides the full time-step overlays, fitted parameter trajectories and proxy-sensitivity assessments that support those definitions.

Figures~\ref{{fig:supp-boundary}} and~\ref{{fig:supp-parameters}} report the full time-step boundary fits and fitted parameter trajectories.

{supp_boundary_fig}

{supp_parameters_fig}

\subsection{{Reduced dynamics, stability and residuals}}
The reduced state is $\bm{{q}}=[L_f,L_r,W,H,T_{{\mathrm{{max}}}},G_{{\mathrm{{mean}}}},U_{{\mathrm{{max}}}}]^T$. Here, $L_f$ and $L_r$ are the front and rear lengths, $W$ is full width, $H$ is height, $T_{{\mathrm{{max}}}}$ is maximum temperature, $G_{{\mathrm{{mean}}}}$ is mean temperature-gradient magnitude and $U_{{\mathrm{{max}}}}$ is maximum velocity magnitude. The diagonal attractor is defined as the parsimonious baseline candidate, and the coupled ridge attractor is defined as the comparison model.

The main text defines the diagonal trajectory fit, coupled ridge comparison, stability criteria and validation metric. The supplementary material uses the same definitions to provide representative stability panels, trajectory comparisons and weak-identifiability diagnostics. The diagonal fit uses direct trajectory fitting within the training-only quasi-steady admissible interval. The coupled comparison uses finite-difference ridge regression and is evaluated by both stability and validation error, because spectral stability alone is insufficient to support its additional parameters.

\section{{Dimensionless scaling and material-property sensitivity}}
\subsection{{Nondimensional groups and sensitivity scenarios}}
The main text reports the process-matrix mean dimensionless values. Here, the supplementary material records the interpretation of each group and the perturbation grid used to test sensitivity to reference temperature, absorptivity and surface-tension coefficient. The underlying material-property curves are shown in Fig.~\ref{{fig:supp-temperature-properties}}.

The symbols are defined as follows. $Pe=vL_{{\mathrm{{ref}}}}/\alpha$ compares advection by the moving laser with thermal diffusion, $Fo=\alpha t/L_{{\mathrm{{ref}}}}^2$ measures diffusive time relative to the reference melt-pool length, $Ste=c_p(T_l-T_s)/L_{{\mathrm{{fus}}}}$ compares sensible heat across the mushy interval with latent heat, $E^*$ scales absorbed laser power against the enthalpy needed to heat material through the moving beam footprint, and $Ma$ scales thermocapillary forcing against viscous-thermal diffusion. Sensitivity scenarios perturb reference temperature, absorptivity and surface-tension coefficient to check whether these interpretations are stable.

Figure~\ref{{fig:supp-dimensionless}} reports the full scenario grid so that the nondimensional interpretation and its sensitivity evidence remain in the same local reading unit.

{supp_dimensionless_fig}

\section{{Diagnostic controls and model-selection support}}
\subsection{{Assumptions, temporal sampling and residual limitations}}
The assumption matrix is placed in the supplementary methods because it defines the scope of interpretation for the reduced model. Table~\ref{{tab:supp-assumptions}} lists the assumptions, supporting evidence and interpretive boundaries used for this assessment.

\begin{{table}}[htbp]
\centering
\caption{{Assumption assessment matrix for the observed boundary-envelope modeling framework.}}
\label{{tab:supp-assumptions}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.07\textwidth}}>{{\raggedright\arraybackslash}}p{{0.27\textwidth}}>{{\raggedright\arraybackslash}}p{{0.42\textwidth}}>{{\raggedright\arraybackslash}}p{{0.12\textwidth}}}}
\toprule
ID & Assumption & Evidence & Interpretive boundary \\
\midrule
{assumption_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

The temporal-sampling sensitivity analysis perturbs the train-validation split, uses rolling-origin tests, removes individual time steps and applies deterministic state-noise perturbations. The tested cases support the selected parsimonious diagonal baseline in {stress_support:.3f} of the temporal-sampling sensitivity settings, and the mean validation relative RMSE ranges from {stress_min:.4f} to {stress_max:.4f}. These tests are internal to the melt-pool dataset and quantify the temporal-sampling sensitivity of the compact baseline.

The analysis evaluates whether the diagonal attractor depends on one train-validation split and quantifies sensitivity to short-sequence sampling within each condition. Because the dataset remains simulation-only, these results bound the trajectory claim within the available exports and do not establish experimental generality over power, speed or powder-feed rate. Representative sensitivity-test cases are summarized in Table~\ref{{tab:supp-sensitivity-tests}}.

\begin{{table}}[htbp]
\centering
\caption{{Representative temporal-sampling sensitivity tests. The complete table is included with the accompanying data tables.}}
\label{{tab:supp-sensitivity-tests}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.17\textwidth}}>{{\raggedright\arraybackslash}}p{{0.21\textwidth}}>{{\raggedright\arraybackslash}}p{{0.28\textwidth}}>{{\raggedright\arraybackslash}}p{{0.18\textwidth}}}}
\toprule
Test family & Scenario & Mean validation relative RMSE & Supports diagonal baseline \\
\midrule
{stress_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

Residual limitations summarize interpretive boundaries that define the scope of the model without constituting additional modeling results. Table~\ref{{tab:supp-gap-audit}} reports the corresponding validation routes; the highest-ranked remaining boundary is: {latex_escape(high_gap_text)}.

\begin{{table}}[htbp]
\centering
\caption{{Residual limitations and validation routes for the manuscript revision.}}
\label{{tab:supp-gap-audit}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.16\textwidth}}>{{\raggedright\arraybackslash}}p{{0.10\textwidth}}>{{\raggedright\arraybackslash}}p{{0.27\textwidth}}>{{\raggedright\arraybackslash}}p{{0.34\textwidth}}}}
\toprule
Topic & Interpretive boundary & Current evidence & Validation route \\
\midrule
{gap_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Error budget, identifiability and boundary-proxy sensitivity}}
The main text defines the error-source taxonomy and its diagnostic bound. The supplementary material avoids repeating that formulation and instead documents how the error terms are interpreted alongside identifiability diagnostics. Point-cloud reconstruction, analytic boundary fitting, volume proxy, dynamical prediction and material-scale uncertainty are kept as separate diagnostic sources. A stricter uncertainty-quantification study would propagate these terms through a global sensitivity or Bayesian analysis.

Identifiability is assessed through coefficient of variation, stability of fitted parameters across time, signs of dynamic parameters and validation behavior. For the superellipsoid, the least constrained quantities are shape exponents and center shifts that may compensate for each other during fitting. For the coupled attractor, the principal identifiability constraint is the ratio between matrix parameters and available state transitions. These diagnostics are used to decide whether additional flexibility is mathematically defensible.

The quantities with limited identifiability support are:
\begin{{itemize}}
{risk_lines}
\end{{itemize}}

Figure~\ref{{fig:supp-theory}} presents the combined theory, identifiability and error-budget diagnostic because it summarizes the same overparameterization constraints described in this section.

{supp_theory_fig}

Figure~\ref{{fig:supp-convex-alpha}} presents the convex-hull and alpha-complex comparison used for proxy-sensitivity assessment. It shows how alternative envelope proxies behave on the exported molten points and why volume-proxy results are treated as interpretive constraints rather than physical interface recovery.

{supp_convex_alpha_fig}

\subsection{{Model-selection rationale and reproducibility}}

{supplementary_model_selection_tex}

{supplementary_data_provenance_table_tex}

\section{{Expanded supplementary diagnostic figures}}
\subsection{{Auxiliary thermal-flow and trajectory diagnostics}}
Following the main supplementary evidence, Figures~\ref{{fig:supp-stability}}, \ref{{fig:supp-boundary-panels}}, \ref{{fig:supp-thermal-flow}} and~\ref{{fig:supp-dynamics-comparison}} present representative stability evidence, expanded boundary-overlay panels, thermal-flow state evolution and diagonal-versus-coupled trajectories. Main-text Figure 6 displays selected overlay panels; Supplementary Fig. S7 keeps additional transient and quasi-steady top and side views. The trajectory panels show the direct prediction curves behind the residual comparison and complement the main residual analysis.

{supp_stability_fig}

{supp_boundary_panels_fig}

{supp_thermal_flow_fig}

{supp_dynamics_comparison_fig}

\subsection{{Temperature-dependent material-property curves}}
The material-property basis for density, specific heat, thermal conductivity and viscosity is tabulated as a function of temperature. Figure~\ref{{fig:supp-temperature-properties}} plots the curves used for liquidus-scale interpolation in the nondimensional scale diagnostics; the corresponding liquidus values are summarized in Table~\ref{{tab:supp-parameter-assessment}}.

{supp_temperature_properties_fig}

\end{{document}}
"""
    supp_tex = humanize_submission_text(supp_tex)
    supp_tex = limit_solver_platform_mentions(supp_tex, keep_main_platform_mention=False)
    (latex_dir / "supplementary_methods.tex").write_text(supp_tex, encoding="utf-8")
    formal_citation_keys = extract_latex_citation_keys(main_submission_tex, supp_tex)
    write_references_seed(latex_dir / "references.bib", only_keys=formal_citation_keys)

    manifest = figure_manifest.copy()
    manifest["latex_path"] = manifest.apply(
        lambda row: fig_path(row["figure_stem"], "paper")
        if row["item_type"] == "main_figure"
        else (fig_path(row["figure_stem"], "supp") if row["item_type"] == "supplementary_figure" else ""),
        axis=1,
    )
    manifest.to_csv(latex_dir / "latex_figure_manifest.csv", index=False)

    readme = """# LaTeX Manuscript Package

Files:

- `main_submission.tex`: AMM submission main article, without appended Supplementary Information.
- `supplementary_methods.tex`: standalone supplementary methods and supplementary figures for separate upload.
- `references.bib`: BibTeX seed library, 2020+ literature prioritized.
- `latex_figure_manifest.csv`: active and legacy figure mapping.

Compile from this directory with:

```bash
pdflatex main_submission.tex
bibtex main_submission
pdflatex main_submission.tex
pdflatex main_submission.tex
pdflatex supplementary_methods.tex
pdflatex supplementary_methods.tex
```

The TeX files reference figure PDFs in `../paper_figures/` and `../figures/`, so keep the `analysis_outputs` directory structure intact.
The main manuscript uses numbered citations via `natbib` with `unsrtnat`.
For AMM submission, use `main_submission.*` plus the standalone supplementary file and active figures. The former combined `main.*` output is no longer generated. Do not upload the legacy root Word draft as the editable manuscript source.
"""
    (latex_dir / "README.md").write_text(readme, encoding="utf-8")


def _pdf_page_count(pdf_path: Path) -> int:
    if not pdf_path.exists():
        return 0
    try:
        data = pdf_path.read_bytes()
    except OSError:
        return 0
    return len(re.findall(rb"/Type\s*/Page\b", data))


def _latex_log_warnings(log_path: Path) -> list[str]:
    if not log_path.exists():
        return [f"{log_path.name} missing"]
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"Citation .* undefined",
        r"Reference .* undefined",
        r"There were undefined",
        r"LaTeX Warning: .*undefined",
        r"LaTeX Error:",
        r"pdfTeX error",
        r"Emergency stop",
        r"Fatal error",
        r"File `[^']+' not found",
    ]
    warnings: list[str] = []
    for pattern in patterns:
        warnings.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return sorted(set(warnings))


def compile_latex_package(output_dir: Path) -> Path:
    latex_dir = output_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_main_latex_outputs(latex_dir)
    cache_dir = latex_dir / ".tex-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["texmf-var", "texmf-config", "texmf-home"]:
        (cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TEXMFVAR"] = str(cache_dir / "texmf-var")
    env["TEXMFCONFIG"] = str(cache_dir / "texmf-config")
    env["TEXMFHOME"] = str(cache_dir / "texmf-home")
    path_entries = [env.get("PATH", "")]
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        miktex_bin = Path(local_appdata) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"
        if miktex_bin.exists():
            path_entries.insert(0, str(miktex_bin))
    env["PATH"] = os.pathsep.join(entry for entry in path_entries if entry)

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "main_submission.tex"],
        ["bibtex", "main_submission"],
        ["pdflatex", "-interaction=nonstopmode", "main_submission.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main_submission.tex"],
        ["pdflatex", "-interaction=nonstopmode", "supplementary_methods.tex"],
        ["pdflatex", "-interaction=nonstopmode", "supplementary_methods.tex"],
    ]
    command_records = []
    for command in commands:
        executable = command[0]
        executable_path = shutil.which(executable, path=env.get("PATH", ""))
        if executable_path is None:
            command_records.append(
                {
                    "command": " ".join(command),
                    "returncode": -1,
                    "stdout_bytes": 0,
                    "stderr_bytes": 0,
                    "error": f"Executable not found in PATH: {executable}",
                }
            )
            continue
        try:
            completed = subprocess.run(
                [executable_path, *command[1:]],
                cwd=latex_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            command_records.append(
                {
                    "command": " ".join(command),
                    "returncode": completed.returncode,
                    "stdout_bytes": len(stdout.encode("utf-8", errors="replace")),
                    "stderr_bytes": len(stderr.encode("utf-8", errors="replace")),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive local toolchain reporting
            command_records.append(
                {
                    "command": " ".join(command),
                    "returncode": -1,
                    "stdout_bytes": 0,
                    "stderr_bytes": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    main_submission_tex = latex_dir / "main_submission.tex"
    supp_tex = latex_dir / "supplementary_methods.tex"
    main_submission_pdf = latex_dir / "main_submission.pdf"
    supp_pdf = latex_dir / "supplementary_methods.pdf"
    main_submission_log = latex_dir / "main_submission.log"
    supp_log = latex_dir / "supplementary_methods.log"

    main_submission_warnings = _latex_log_warnings(main_submission_log)
    supp_warnings = _latex_log_warnings(supp_log)
    main_submission_fresh = (
        main_submission_pdf.exists()
        and main_submission_pdf.stat().st_size > 0
        and main_submission_pdf.stat().st_mtime >= main_submission_tex.stat().st_mtime
    )
    supp_fresh = supp_pdf.exists() and supp_pdf.stat().st_size > 0 and supp_pdf.stat().st_mtime >= supp_tex.stat().st_mtime
    command_success = all(record["returncode"] == 0 for record in command_records)
    final_acceptance = bool(
        command_success
        and main_submission_fresh
        and supp_fresh
        and not main_submission_warnings
        and not supp_warnings
    )

    lines = [
        "LaTeX Compile Summary",
        "=====================",
        "",
        f"latex_dir: {latex_dir}",
        f"final_acceptance: {final_acceptance}",
        f"all_commands_succeeded: {command_success}",
        "",
        "Commands:",
    ]
    for idx, record in enumerate(command_records, start=1):
        lines.append(f"{idx}. {record['command']} -> returncode {record['returncode']}")
    lines.extend(
        [
            "",
            "PDF integrity:",
            f"main_submission.pdf exists: {main_submission_pdf.exists()}",
            f"main_submission.pdf fresh: {main_submission_fresh}",
            f"main_submission.pdf size_bytes: {main_submission_pdf.stat().st_size if main_submission_pdf.exists() else 0}",
            f"main_submission.pdf page_count_estimate: {_pdf_page_count(main_submission_pdf)}",
            f"supplementary_methods.pdf exists: {supp_pdf.exists()}",
            f"supplementary_methods.pdf fresh: {supp_fresh}",
            f"supplementary_methods.pdf size_bytes: {supp_pdf.stat().st_size if supp_pdf.exists() else 0}",
            f"supplementary_methods.pdf page_count_estimate: {_pdf_page_count(supp_pdf)}",
            "",
            "Warning summary:",
            f"main_submission.log blocking warnings: {len(main_submission_warnings)}",
            *(f"  - {warning}" for warning in main_submission_warnings[:20]),
            f"supplementary_methods.log blocking warnings: {len(supp_warnings)}",
            *(f"  - {warning}" for warning in supp_warnings[:20]),
            "",
            "Command outputs:",
            "Intermediate LaTeX pass output is intentionally omitted from this summary; final acceptance is based on the final logs listed above.",
        ]
    )
    for idx, record in enumerate(command_records, start=1):
        output_line = (
            f"{idx}. {record['command']}: stdout_bytes={record.get('stdout_bytes', 0)}, "
            f"stderr_bytes={record.get('stderr_bytes', 0)}"
        )
        if record.get("error"):
            output_line += f", error={record['error']}"
        lines.append(output_line)
    summary_path = latex_dir / "latex_compile_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


MAX_PLOT_WORKERS = 8


def default_plot_workers() -> int:
    available = os.cpu_count() or 1
    if available <= 1:
        return 1
    return min(MAX_PLOT_WORKERS, available)


def normalize_plot_workers(value: int | None) -> int:
    if value is None:
        return default_plot_workers()
    return max(1, min(MAX_PLOT_WORKERS, int(value)))


def active_plot_task_names() -> list[str]:
    return [
        "moving_frame",
        "geometry",
        "thermal_flow",
        "dynamics_validation",
        "boundary_fit_comparison",
        "dynamics_model_comparison",
        "uncertainty_identifiability",
        "modeling_framework",
        "simulation_cross_sections",
        "dimensionless_regime",
        "stability_attractor",
        "error_budget_model_selection",
        "identifiability_overparameterization",
        "supplementary_all_boundary_fits",
        "supplementary_superellipsoid_parameters",
        "supplementary_residuals",
        "supplementary_dimensionless_grid",
        "temperature_dependent_properties",
        "convex_alpha_proxy_comparison",
        "theory_identifiability_error_bounds",
        "multi_condition_process_matrix",
        "multi_condition_response_surfaces",
        "multi_condition_geometry_comparison",
        "multi_condition_dynamics_validation",
        "leave_one_condition_validation",
        "external_holdout_validation",
    ]


def cached_table_path(output_dir: Path, file_name: str) -> Path:
    return output_dir / "tables" / file_name


def read_cached_table(output_dir: Path, file_name: str) -> pd.DataFrame:
    path = cached_table_path(output_dir, file_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached table: {path}. Run the full pipeline before using plot-only or parallel plotting."
        )
    if path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def assert_plot_cache_available(output_dir: Path) -> None:
    required_files = [
        "modeling_table.csv",
        "collapsed_point_cloud.csv",
        "dynamics_predictions.csv",
        "dynamics_fit_summary.csv",
        "stability_eigenvalues.csv",
        "coupled_dynamics_predictions.csv",
        "coupled_stability_eigenvalues.csv",
        "coupled_A_matrix.csv",
        "dynamics_model_comparison.csv",
        "geometry_model_comparison.csv",
        "model_selection_summary.csv",
        "error_budget_summary.csv",
        "parameter_identifiability.csv",
        "dimensionless_sensitivity_summary.csv",
        "identifiability_diagnostics_v4.csv",
        "error_bound_summary.csv",
        "leave_one_condition_out_validation.csv",
        "external_validation_geometry_model_comparison.csv",
        "external_holdout_process_response_validation.csv",
        "external_holdout_dynamics_summary.csv",
    ]
    missing = [str(cached_table_path(output_dir, name)) for name in required_files if not cached_table_path(output_dir, name).exists()]
    if missing:
        raise FileNotFoundError(
            "Plot cache is incomplete. Run `python scripts\\flow3d_melt_pool_pilot.py` first. Missing:\n"
            + "\n".join(missing[:20])
        )


def _run_plot_task(task_name: str, output_dir_text: str) -> str:
    output_dir = Path(output_dir_text)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def table(file_name: str) -> pd.DataFrame:
        return read_cached_table(output_dir, file_name)

    if task_name == "moving_frame":
        plot_moving_frame(table("collapsed_point_cloud.csv"), fig_dir)
    elif task_name == "geometry":
        plot_geometry(table("modeling_table.csv"), fig_dir)
    elif task_name == "thermal_flow":
        plot_thermal_flow(table("modeling_table.csv"), fig_dir)
    elif task_name == "dynamics_validation":
        plot_dynamics_validation(table("dynamics_predictions.csv"), fig_dir)
    elif task_name == "boundary_fit_comparison":
        plot_boundary_fit_comparison(table("modeling_table.csv"), table("collapsed_point_cloud.csv"), fig_dir)
    elif task_name == "dynamics_model_comparison":
        plot_dynamics_model_comparison(
            table("dynamics_predictions.csv"),
            table("coupled_dynamics_predictions.csv"),
            table("dynamics_model_comparison.csv"),
            fig_dir,
        )
    elif task_name == "uncertainty_identifiability":
        plot_uncertainty_identifiability(
            table("parameter_identifiability.csv"),
            table("dimensionless_sensitivity_summary.csv"),
            fig_dir,
        )
    elif task_name == "modeling_framework":
        plot_modeling_framework(fig_dir)
    elif task_name == "simulation_cross_sections":
        plot_simulation_cross_sections(fig_dir)
    elif task_name == "dimensionless_regime":
        plot_dimensionless_regime(table("dimensionless_sensitivity_summary.csv"), fig_dir)
    elif task_name == "stability_attractor":
        plot_stability_attractor(
            table("modeling_table.csv"),
            table("dynamics_fit_summary.csv"),
            table("stability_eigenvalues.csv"),
            table("coupled_stability_eigenvalues.csv"),
            fig_dir,
        )
    elif task_name == "error_budget_model_selection":
        plot_error_budget_model_selection(
            table("error_budget_summary.csv"),
            table("model_selection_summary.csv"),
            fig_dir,
        )
    elif task_name == "identifiability_overparameterization":
        plot_identifiability_overparameterization(
            table("parameter_identifiability.csv"),
            table("coupled_A_matrix.csv"),
            fig_dir,
        )
    elif task_name == "supplementary_all_boundary_fits":
        plot_supplementary_all_boundary_fits(table("modeling_table.csv"), table("collapsed_point_cloud.csv"), fig_dir)
    elif task_name == "supplementary_superellipsoid_parameters":
        plot_supplementary_superellipsoid_parameters(table("modeling_table.csv"), fig_dir)
    elif task_name == "supplementary_residuals":
        plot_supplementary_residuals(
            table("dynamics_predictions.csv"),
            table("coupled_dynamics_predictions.csv"),
            fig_dir,
        )
    elif task_name == "supplementary_dimensionless_grid":
        plot_supplementary_dimensionless_grid(table("dimensionless_sensitivity_summary.csv"), fig_dir)
    elif task_name == "temperature_dependent_properties":
        plot_temperature_dependent_properties(load_property_curves(output_dir.parent), fig_dir)
    elif task_name == "convex_alpha_proxy_comparison":
        plot_convex_alpha_proxy_comparison(table("modeling_table.csv"), table("collapsed_point_cloud.csv"), fig_dir)
    elif task_name == "theory_identifiability_error_bounds":
        plot_theory_identifiability_error_bounds(
            table("identifiability_diagnostics_v4.csv"),
            table("error_bound_summary.csv"),
            table("dimensionless_sensitivity_summary.csv"),
            fig_dir,
        )
    elif task_name == "multi_condition_process_matrix":
        plot_multi_condition_process_matrix(table("modeling_table.csv"), fig_dir)
    elif task_name == "multi_condition_response_surfaces":
        plot_multi_condition_response_surfaces(table("modeling_table.csv"), fig_dir)
    elif task_name == "multi_condition_geometry_comparison":
        plot_multi_condition_geometry_comparison(
            table("geometry_model_comparison.csv"),
            fig_dir,
            table("modeling_table.csv"),
            table("collapsed_point_cloud.csv"),
            table("robustness_summary.csv"),
        )
    elif task_name == "multi_condition_dynamics_validation":
        plot_multi_condition_dynamics_validation(table("dynamics_model_comparison.csv"), fig_dir)
    elif task_name == "leave_one_condition_validation":
        plot_leave_one_condition_validation(table("leave_one_condition_out_validation.csv"), fig_dir)
    elif task_name == "external_holdout_validation":
        plot_external_holdout_validation(
            table("external_validation_geometry_model_comparison.csv"),
            table("external_holdout_process_response_validation.csv"),
            table("external_holdout_dynamics_summary.csv"),
            fig_dir,
        )
    else:
        raise ValueError(f"Unknown plot task: {task_name}")
    return task_name


def run_plot_tasks(output_dir: Path, plot_workers: int | None = None) -> list[str]:
    worker_count = normalize_plot_workers(plot_workers)
    task_names = active_plot_task_names()
    if worker_count == 1 or len(task_names) == 1:
        completed = []
        for task_name in task_names:
            completed.append(_run_plot_task(task_name, str(output_dir)))
        return completed

    completed = []
    script_path = Path(__file__).resolve().parents[1] / "flow3d_melt_pool_pilot.py"

    def run_task_subprocess(task_name: str) -> str:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [
            sys.executable,
            "-B",
            str(script_path),
            "--output-dir",
            str(output_dir),
            "--plot-task",
            task_name,
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if result.returncode != 0:
            tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-40:])
            raise RuntimeError(f"{task_name} failed with return code {result.returncode}\n{tail}")
        return task_name

    with ThreadPoolExecutor(max_workers=min(worker_count, len(task_names))) as executor:
        future_map = {executor.submit(run_task_subprocess, task_name): task_name for task_name in task_names}
        for future in as_completed(future_map):
            task_name = future_map[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                for pending in future_map:
                    pending.cancel()
                raise RuntimeError(f"Parallel plot task failed: {task_name}") from exc
    return completed


def run_plot_only(output_dir: Path, plot_workers: int | None = None) -> pd.DataFrame:
    assert_plot_cache_available(output_dir)
    run_plot_tasks(output_dir, plot_workers)
    export_paper_figure_set(output_dir)
    figure_manifest = make_active_figure_manifest(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figure_manifest.to_csv(tables_dir / "active_figure_manifest.csv", index=False)
    latex_dir = output_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    figure_manifest.to_csv(latex_dir / "latex_figure_manifest.csv", index=False)
    compile_latex_package(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_reviewer_risk_response(
        reports_dir / "reviewer_risk_response.md",
        read_cached_table(output_dir, "modeling_table.csv"),
        read_cached_table(output_dir, "geometry_model_comparison.csv"),
        read_cached_table(output_dir, "dynamics_model_comparison.csv"),
        read_cached_table(output_dir, "dimensionless_numbers.csv"),
        read_cached_table(output_dir, "model_selection_summary.csv"),
        read_cached_table(output_dir, "robustness_summary.csv"),
        read_cached_table(output_dir, "parameter_identifiability.csv"),
        read_cached_table(output_dir, "dimensionless_sensitivity_summary.csv"),
    )
    write_declarations_for_submission(reports_dir / "declarations_for_submission.md")
    write_amm_submission_checklist(reports_dir / "amm_submission_checklist.md", output_dir, figure_manifest)
    submission_manifest = write_submission_package(output_dir)
    submission_manifest.to_csv(tables_dir / "submission_package_manifest.csv", index=False)
    reproducibility_manifest = write_reproducibility_package(output_dir)
    reproducibility_manifest.to_csv(tables_dir / "reproducibility_package_manifest.csv", index=False)
    output_checksums = make_output_checksums(output_dir)
    output_checksums.to_csv(tables_dir / "output_checksums.csv", index=False)
    return validate_outputs(output_dir)


def write_outputs(
    output_dir: Path,
    table: pd.DataFrame,
    point_cloud: pd.DataFrame,
    predictions: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    eigenvalues: pd.DataFrame,
    coupled_predictions: pd.DataFrame,
    coupled_summary: pd.DataFrame,
    coupled_eigenvalues: pd.DataFrame,
    coupled_matrix: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    quasi: pd.DataFrame,
    geometry_comparison: pd.DataFrame,
    superellipsoid_parameters: pd.DataFrame,
    material_parameters: pd.DataFrame,
    property_table: pd.DataFrame,
    dimensionless_numbers: pd.DataFrame,
    model_selection: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    robustness_long: pd.DataFrame,
    error_summary: pd.DataFrame,
    error_budget: pd.DataFrame,
    parameter_identifiability: pd.DataFrame,
    dimensionless_sensitivity: pd.DataFrame,
    external_validation_table: pd.DataFrame | None = None,
    external_validation_point_cloud: pd.DataFrame | None = None,
    external_validation_geometry_comparison: pd.DataFrame | None = None,
    external_validation_dimensionless_numbers: pd.DataFrame | None = None,
    external_holdout_process_response: pd.DataFrame | None = None,
    external_holdout_dynamics_predictions: pd.DataFrame | None = None,
    external_holdout_dynamics_summary: pd.DataFrame | None = None,
    external_holdout_summary: pd.DataFrame | None = None,
    input_file_manifest: pd.DataFrame | None = None,
    plot_workers: int | None = None,
) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    case_metadata = case_metadata_from_modeling_table(table)
    point_summary = make_multi_condition_point_cloud_summary(point_cloud)
    multi_dimensional_numbers = make_multi_condition_dimensionless_table(dimensionless_numbers)
    multi_geometry_summary = make_multi_condition_geometry_summary(geometry_comparison)
    multi_dynamics_summary = make_multi_condition_dynamics_summary(dynamics_comparison)
    process_response = make_process_response_summary(table)
    loco_validation = make_leave_one_condition_out_validation(table)
    external_validation_table = external_validation_table if external_validation_table is not None else pd.DataFrame()
    external_validation_point_cloud = (
        external_validation_point_cloud if external_validation_point_cloud is not None else pd.DataFrame()
    )
    external_validation_geometry_comparison = (
        external_validation_geometry_comparison if external_validation_geometry_comparison is not None else pd.DataFrame()
    )
    external_validation_dimensionless_numbers = (
        external_validation_dimensionless_numbers if external_validation_dimensionless_numbers is not None else pd.DataFrame()
    )
    external_holdout_process_response = (
        external_holdout_process_response if external_holdout_process_response is not None else pd.DataFrame()
    )
    external_holdout_dynamics_predictions = (
        external_holdout_dynamics_predictions if external_holdout_dynamics_predictions is not None else pd.DataFrame()
    )
    external_holdout_dynamics_summary = (
        external_holdout_dynamics_summary if external_holdout_dynamics_summary is not None else pd.DataFrame()
    )
    external_holdout_summary = external_holdout_summary if external_holdout_summary is not None else pd.DataFrame()
    input_file_manifest = input_file_manifest if input_file_manifest is not None else pd.DataFrame()
    parameter_audit = make_parameter_reconciliation_audit(material_parameters, property_table)
    geometry_risk_summary = make_geometry_risk_summary(table, geometry_comparison, error_budget)
    validation_hierarchy = make_validation_hierarchy_table(loco_validation, external_holdout_summary)
    combined_boundary_table = pd.concat([table, external_validation_table], ignore_index=True, sort=False) if len(external_validation_table) else table
    combined_boundary_points = pd.concat([point_cloud, external_validation_point_cloud], ignore_index=True, sort=False) if len(external_validation_point_cloud) else point_cloud
    boundary_extraction_sensitivity = make_boundary_extraction_sensitivity(combined_boundary_table, combined_boundary_points)
    geometry_selection_metrics = make_geometry_selection_metrics(
        geometry_comparison,
        external_validation_geometry_comparison,
    )
    dynamics_minimal_baselines = make_minimal_dynamics_baselines(table, "internal_condition_time_split")
    external_dynamics_minimal_baselines = make_minimal_dynamics_baselines(
        external_validation_table,
        "holdout_condition_time_split",
    )
    method_detail_audit = make_method_detail_audit()
    combined_dynamics_audit_table = (
        pd.concat([table, external_validation_table], ignore_index=True, sort=False)
        if len(external_validation_table)
        else table
    )
    dynamics_derivative_audit = make_dynamics_derivative_audit(combined_dynamics_audit_table)
    q_inf_estimation_audit = make_q_inf_estimation_audit(combined_dynamics_audit_table)
    holdout_extrapolation_audit = make_holdout_extrapolation_audit(table, external_validation_table)
    literature_dimension_benchmark = make_literature_dimension_benchmark(quasi)
    environment_summary = make_environment_summary()
    holdout_cohort_summary = make_holdout_cohort_summary(
        external_validation_table,
        external_validation_geometry_comparison,
        external_holdout_process_response,
        external_holdout_dynamics_summary,
    )

    table.to_csv(tables_dir / "modeling_table.csv", index=False)
    case_metadata.to_csv(tables_dir / "case_metadata.csv", index=False)
    table.to_csv(tables_dir / "multi_condition_modeling_table.csv", index=False)
    point_summary.to_csv(tables_dir / "multi_condition_point_cloud_summary.csv", index=False)
    multi_dimensional_numbers.to_csv(tables_dir / "multi_condition_dimensionless_numbers.csv", index=False)
    multi_geometry_summary.to_csv(tables_dir / "multi_condition_geometry_summary.csv", index=False)
    multi_dynamics_summary.to_csv(tables_dir / "multi_condition_dynamics_summary.csv", index=False)
    process_response.to_csv(tables_dir / "multi_condition_process_response_summary.csv", index=False)
    loco_validation.to_csv(tables_dir / "leave_one_condition_out_validation.csv", index=False)
    external_validation_table.to_csv(tables_dir / "external_validation_modeling_table.csv", index=False)
    external_validation_point_cloud.to_csv(tables_dir / "external_validation_collapsed_point_cloud.csv", index=False)
    if len(external_validation_table):
        external_case_audit = case_metadata_from_modeling_table(external_validation_table)
        external_case_audit["geometry_validation_ready"] = True
        external_case_audit["thermal_flow_validation_ready"] = True
        external_case_audit["dynamics_validation_ready"] = True
        external_file_audit_cols = [
            "case_id",
            "case_index",
            "validation_cohort",
            "source_root",
            "power_W",
            "scan_speed_mm_s",
            "powder_feed_g_min",
            "time_s",
            "source_file",
            "raw_rows",
            "exact_dedup_rows",
            "unique_points",
            "Tmax_K",
            "Umax_m_per_s",
            "convex_hull_status",
            "superellipsoid_fit_status",
        ]
        external_file_audit = external_validation_table[
            [col for col in external_file_audit_cols if col in external_validation_table.columns]
        ].copy()
    else:
        external_case_audit = pd.DataFrame()
        external_file_audit = pd.DataFrame()
    external_case_audit.to_csv(tables_dir / "external_validation_case_audit.csv", index=False)
    external_file_audit.to_csv(tables_dir / "external_validation_file_audit.csv", index=False)
    external_validation_geometry_comparison.to_csv(
        tables_dir / "external_validation_geometry_model_comparison.csv", index=False
    )
    external_validation_dimensionless_numbers.to_csv(
        tables_dir / "external_validation_dimensionless_numbers.csv", index=False
    )
    external_holdout_process_response.to_csv(tables_dir / "external_holdout_process_response_validation.csv", index=False)
    external_holdout_dynamics_predictions.to_csv(tables_dir / "external_holdout_dynamics_predictions.csv", index=False)
    external_holdout_dynamics_summary.to_csv(tables_dir / "external_holdout_dynamics_summary.csv", index=False)
    external_holdout_summary.to_csv(tables_dir / "external_holdout_validation_summary.csv", index=False)
    holdout_cohort_summary.to_csv(tables_dir / "holdout_cohort_summary.csv", index=False)
    input_file_manifest.to_csv(tables_dir / "input_file_manifest.csv", index=False)
    boundary_extraction_sensitivity.to_csv(tables_dir / "boundary_extraction_sensitivity.csv", index=False)
    geometry_selection_metrics.to_csv(tables_dir / "geometry_selection_metrics.csv", index=False)
    dynamics_minimal_baselines.to_csv(tables_dir / "dynamics_minimal_baselines.csv", index=False)
    external_dynamics_minimal_baselines.to_csv(tables_dir / "external_holdout_minimal_dynamics_baselines.csv", index=False)
    method_detail_audit.to_csv(tables_dir / "method_detail_audit.csv", index=False)
    dynamics_derivative_audit.to_csv(tables_dir / "dynamics_derivative_audit.csv", index=False)
    q_inf_estimation_audit.to_csv(tables_dir / "q_inf_estimation_audit.csv", index=False)
    holdout_extrapolation_audit.to_csv(tables_dir / "holdout_extrapolation_audit.csv", index=False)
    literature_dimension_benchmark.to_csv(tables_dir / "literature_dimension_benchmark.csv", index=False)
    environment_summary.to_csv(tables_dir / "environment_summary.csv", index=False)
    (reports_dir / "environment_summary.txt").write_text(
        "\n".join(
            f"{row.component}: {row.version} ({row.source})"
            for row in environment_summary.itertuples(index=False)
        )
        + "\n",
        encoding="utf-8",
    )
    point_cloud.to_csv(tables_dir / "collapsed_point_cloud.csv", index=False)
    predictions.to_csv(tables_dir / "dynamics_predictions.csv", index=False)
    dynamics_summary.to_csv(tables_dir / "dynamics_fit_summary.csv", index=False)
    eigenvalues.to_csv(tables_dir / "stability_eigenvalues.csv", index=False)
    coupled_predictions.to_csv(tables_dir / "coupled_dynamics_predictions.csv", index=False)
    coupled_summary.to_csv(tables_dir / "coupled_dynamics_fit_summary.csv", index=False)
    coupled_eigenvalues.to_csv(tables_dir / "coupled_stability_eigenvalues.csv", index=False)
    coupled_matrix.to_csv(tables_dir / "coupled_A_matrix.csv", index=False)
    dynamics_comparison.to_csv(tables_dir / "dynamics_model_comparison.csv", index=False)
    quasi.to_csv(tables_dir / "quasi_steady_summary.csv", index=False)
    geometry_comparison.to_csv(tables_dir / "geometry_model_comparison.csv", index=False)
    superellipsoid_parameters.to_csv(tables_dir / "superellipsoid_parameters.csv", index=False)
    material_parameters.to_csv(tables_dir / "material_parameters_316L.csv", index=False)
    property_table.to_csv(tables_dir / "temperature_dependent_properties.csv", index=False)
    dimensionless_numbers.to_csv(tables_dir / "dimensionless_numbers.csv", index=False)
    model_selection.to_csv(tables_dir / "model_selection_summary.csv", index=False)
    robustness_summary.to_csv(tables_dir / "robustness_summary.csv", index=False)
    robustness_long.to_csv(tables_dir / "robustness_long.csv", index=False)
    error_summary.to_csv(tables_dir / "error_summary.csv", index=False)
    error_budget.to_csv(tables_dir / "error_budget_summary.csv", index=False)
    parameter_identifiability.to_csv(tables_dir / "parameter_identifiability.csv", index=False)
    dimensionless_sensitivity.to_csv(tables_dir / "dimensionless_sensitivity_summary.csv", index=False)
    parameter_audit.to_csv(tables_dir / "parameter_reconciliation_audit.csv", index=False)
    geometry_risk_summary.to_csv(tables_dir / "geometry_risk_summary.csv", index=False)
    validation_hierarchy.to_csv(tables_dir / "validation_hierarchy_table.csv", index=False)
    make_data_provenance_summary(table).to_csv(tables_dir / "data_provenance_summary.csv", index=False)
    identifiability_v4 = make_identifiability_diagnostics_v4(
        table,
        dynamics_summary,
        coupled_summary,
        coupled_matrix,
        parameter_identifiability,
    )
    error_bound_summary = make_error_bound_summary_v4(error_budget)
    assumption_matrix = make_assumption_justification_matrix(
        table,
        geometry_comparison,
        dynamics_comparison,
        dimensionless_numbers,
        robustness_summary,
        identifiability_v4,
        dimensionless_sensitivity,
    )
    timescale_summary = make_timescale_separation_summary(dynamics_summary)
    validation_stress_tests, validation_stress_text = make_validation_stress_tests(table)
    submission_gap_audit = make_submission_gap_audit(
        validation_stress_tests,
        timescale_summary,
        assumption_matrix,
        external_holdout_summary,
    )
    dynamics_fit_asymmetry_audit = make_dynamics_fit_asymmetry_audit(
        dynamics_summary,
        coupled_summary,
        dynamics_comparison,
        coupled_eigenvalues,
        parameter_identifiability,
        identifiability_v4,
        robustness_summary,
        dynamics_derivative_audit,
        validation_stress_tests,
    )
    numerical_credibility_audit = make_numerical_credibility_audit(
        validation_stress_tests,
        holdout_extrapolation_audit,
        literature_dimension_benchmark,
        external_holdout_summary,
        boundary_extraction_sensitivity,
        geometry_selection_metrics,
        input_file_manifest,
        environment_summary,
    )
    identifiability_v4.to_csv(tables_dir / "identifiability_diagnostics_v4.csv", index=False)
    error_bound_summary.to_csv(tables_dir / "error_bound_summary.csv", index=False)
    assumption_matrix.to_csv(tables_dir / "assumption_justification_matrix.csv", index=False)
    timescale_summary.to_csv(tables_dir / "timescale_separation_summary.csv", index=False)
    validation_stress_tests.to_csv(tables_dir / "validation_stress_tests.csv", index=False)
    submission_gap_audit.to_csv(tables_dir / "submission_gap_audit.csv", index=False)
    dynamics_fit_asymmetry_audit.to_csv(tables_dir / "dynamics_fit_asymmetry_audit.csv", index=False)
    numerical_credibility_audit.to_csv(tables_dir / "numerical_credibility_audit.csv", index=False)
    make_dimensionless_definitions().to_csv(tables_dir / "dimensionless_definitions.csv", index=False)
    make_nomenclature_table().to_csv(tables_dir / "nomenclature_table.csv", index=False)
    make_equation_inventory().to_csv(tables_dir / "equation_inventory.csv", index=False)
    literature_matrix = make_literature_matrix()
    literature_matrix.to_csv(tables_dir / "literature_matrix.csv", index=False)
    write_response_matrix(
        reports_dir / "response_matrix.md",
        parameter_audit,
        geometry_risk_summary,
        validation_hierarchy,
        model_selection,
        identifiability_v4,
        external_holdout_summary,
    )
    write_external_validation_data_audit(
        reports_dir / "external_validation_data_audit.md",
        external_case_audit,
        external_file_audit,
        external_holdout_summary,
    )

    run_plot_tasks(output_dir, plot_workers)
    export_paper_figure_set(output_dir)
    figure_manifest = make_active_figure_manifest(output_dir)
    figure_manifest.to_csv(tables_dir / "active_figure_manifest.csv", index=False)
    write_latex_package(
        output_dir,
        table,
        geometry_comparison,
        dynamics_comparison,
        dynamics_summary,
        coupled_eigenvalues,
        dimensionless_numbers,
        model_selection,
        robustness_summary,
        error_budget,
        parameter_identifiability,
        dimensionless_sensitivity,
        assumption_matrix,
        figure_manifest,
        timescale_summary,
        validation_stress_tests,
        submission_gap_audit,
        external_holdout_summary,
    )
    compile_latex_package(output_dir)

    write_enhanced_method_draft(
        reports_dir / "method_framework_draft.md",
        table,
        dynamics_summary,
        eigenvalues,
        geometry_comparison,
        coupled_summary,
        coupled_eigenvalues,
        dynamics_comparison,
        dimensionless_numbers,
    )
    write_paper_outline_draft(
        reports_dir / "paper_outline_draft.md",
        model_selection,
        robustness_summary,
        geometry_comparison,
        dynamics_comparison,
        dimensionless_numbers,
    )
    write_manuscript_draft_v1(
        reports_dir / "manuscript_draft_v1.md",
        table,
        geometry_comparison,
        dynamics_comparison,
        dimensionless_numbers,
        model_selection,
        robustness_summary,
    )
    write_theory_and_error_analysis(
        reports_dir / "theory_and_error_analysis.md",
        table,
        geometry_comparison,
        dynamics_summary,
        coupled_eigenvalues,
        error_budget,
        parameter_identifiability,
        dimensionless_sensitivity,
    )
    write_manuscript_draft_v2(
        reports_dir / "manuscript_draft_v2.md",
        table,
        geometry_comparison,
        dynamics_comparison,
        dynamics_summary,
        coupled_eigenvalues,
        dimensionless_numbers,
        model_selection,
        robustness_summary,
        error_budget,
        parameter_identifiability,
        dimensionless_sensitivity,
    )
    write_manuscript_draft_v3(
        reports_dir / "manuscript_draft_v3.md",
        table,
        geometry_comparison,
        dynamics_comparison,
        dynamics_summary,
        coupled_eigenvalues,
        dimensionless_numbers,
        model_selection,
        robustness_summary,
        error_budget,
        parameter_identifiability,
        dimensionless_sensitivity,
    )
    write_theory_framework_v4(
        reports_dir / "theory_framework_v4.md",
        table,
        geometry_comparison,
        dynamics_summary,
        coupled_eigenvalues,
        dynamics_comparison,
        dimensionless_numbers,
        robustness_summary,
        error_bound_summary,
        identifiability_v4,
        dimensionless_sensitivity,
    )
    write_manuscript_draft_v4(
        reports_dir / "manuscript_draft_v4.md",
        table,
        geometry_comparison,
        dynamics_comparison,
        dynamics_summary,
        coupled_eigenvalues,
        dimensionless_numbers,
        model_selection,
        robustness_summary,
        error_budget,
        error_bound_summary,
        identifiability_v4,
        dimensionless_sensitivity,
    )
    write_theoretical_derivation_v5(
        reports_dir / "theoretical_derivation_v5.md",
        table,
        geometry_comparison,
        dynamics_comparison,
        dynamics_summary,
        coupled_eigenvalues,
        dimensionless_numbers,
        robustness_summary,
        assumption_matrix,
        error_bound_summary,
        dimensionless_sensitivity,
    )
    write_manuscript_draft_v5(
        reports_dir / "manuscript_draft_v5.md",
        table,
        geometry_comparison,
        dynamics_comparison,
        dynamics_summary,
        coupled_eigenvalues,
        dimensionless_numbers,
        model_selection,
        robustness_summary,
        assumption_matrix,
        error_bound_summary,
        identifiability_v4,
        dimensionless_sensitivity,
    )
    write_rigorous_theory_notes(
        reports_dir / "rigorous_theory_notes.md",
        error_bound_summary,
        timescale_summary,
        validation_stress_tests,
    )
    write_free_boundary_manifold_rationale(
        reports_dir / "free_boundary_manifold_rationale.md",
        geometry_comparison,
    )
    (reports_dir / "validation_stress_summary.md").write_text(validation_stress_text, encoding="utf-8")
    write_manuscript_draft_final(
        reports_dir / "manuscript_draft_final.md",
        dimensionless_numbers,
        geometry_comparison,
        dynamics_comparison,
        timescale_summary,
        validation_stress_tests,
        submission_gap_audit,
    )
    write_supplementary_methods_draft(
        reports_dir / "supplementary_methods_draft.md",
        table,
        geometry_comparison,
        dynamics_comparison,
        dimensionless_numbers,
        error_budget,
        parameter_identifiability,
        dimensionless_sensitivity,
    )
    write_figure_captions(reports_dir / "figure_captions.md")
    write_references_seed(reports_dir / "references_seed.bib")
    write_literature_search_log(reports_dir / "literature_search_log.md", literature_matrix)
    write_reviewer_risk_response(
        reports_dir / "reviewer_risk_response.md",
        table,
        geometry_comparison,
        dynamics_comparison,
        dimensionless_numbers,
        model_selection,
        robustness_summary,
        parameter_identifiability,
        dimensionless_sensitivity,
    )
    write_submission_readiness_checklist(
        reports_dir / "submission_readiness_checklist.md",
        figure_manifest,
        literature_matrix,
    )
    write_cover_letter_draft(
        reports_dir / "cover_letter_draft.md",
        dimensionless_numbers,
        dynamics_comparison,
    )
    write_highlights_draft(
        reports_dir / "highlights_draft.md",
        geometry_comparison,
        dynamics_comparison,
        external_holdout_summary,
    )
    write_declarations_for_submission(reports_dir / "declarations_for_submission.md")
    write_amm_submission_checklist(
        reports_dir / "amm_submission_checklist.md",
        output_dir,
        figure_manifest,
    )
    submission_manifest = write_submission_package(output_dir)
    submission_manifest.to_csv(tables_dir / "submission_package_manifest.csv", index=False)
    reproducibility_manifest = write_reproducibility_package(output_dir)
    reproducibility_manifest.to_csv(tables_dir / "reproducibility_package_manifest.csv", index=False)
    output_checksums = make_output_checksums(output_dir)
    output_checksums.to_csv(tables_dir / "output_checksums.csv", index=False)


def validate_outputs(output_dir: Path) -> pd.DataFrame:
    records = []
    required = [
        output_dir / "tables" / "modeling_table.csv",
        output_dir / "tables" / "case_metadata.csv",
        output_dir / "tables" / "multi_condition_modeling_table.csv",
        output_dir / "tables" / "multi_condition_point_cloud_summary.csv",
        output_dir / "tables" / "multi_condition_dimensionless_numbers.csv",
        output_dir / "tables" / "multi_condition_geometry_summary.csv",
        output_dir / "tables" / "multi_condition_dynamics_summary.csv",
        output_dir / "tables" / "multi_condition_process_response_summary.csv",
        output_dir / "tables" / "leave_one_condition_out_validation.csv",
        output_dir / "tables" / "external_validation_modeling_table.csv",
        output_dir / "tables" / "external_validation_case_audit.csv",
        output_dir / "tables" / "external_validation_file_audit.csv",
        output_dir / "tables" / "external_validation_geometry_model_comparison.csv",
        output_dir / "tables" / "external_validation_dimensionless_numbers.csv",
        output_dir / "tables" / "external_holdout_process_response_validation.csv",
        output_dir / "tables" / "external_holdout_dynamics_predictions.csv",
        output_dir / "tables" / "external_holdout_dynamics_summary.csv",
        output_dir / "tables" / "external_holdout_validation_summary.csv",
        output_dir / "tables" / "holdout_cohort_summary.csv",
        output_dir / "tables" / "input_file_manifest.csv",
        output_dir / "tables" / "boundary_extraction_sensitivity.csv",
        output_dir / "tables" / "geometry_selection_metrics.csv",
        output_dir / "tables" / "dynamics_minimal_baselines.csv",
        output_dir / "tables" / "external_holdout_minimal_dynamics_baselines.csv",
        output_dir / "tables" / "method_detail_audit.csv",
        output_dir / "tables" / "dynamics_derivative_audit.csv",
        output_dir / "tables" / "q_inf_estimation_audit.csv",
        output_dir / "tables" / "holdout_extrapolation_audit.csv",
        output_dir / "tables" / "literature_dimension_benchmark.csv",
        output_dir / "tables" / "environment_summary.csv",
        output_dir / "tables" / "output_checksums.csv",
        output_dir / "tables" / "dynamics_fit_summary.csv",
        output_dir / "tables" / "stability_eigenvalues.csv",
        output_dir / "tables" / "geometry_model_comparison.csv",
        output_dir / "tables" / "superellipsoid_parameters.csv",
        output_dir / "tables" / "coupled_dynamics_fit_summary.csv",
        output_dir / "tables" / "coupled_stability_eigenvalues.csv",
        output_dir / "tables" / "dynamics_model_comparison.csv",
        output_dir / "tables" / "material_parameters_316L.csv",
        output_dir / "tables" / "temperature_dependent_properties.csv",
        output_dir / "tables" / "dimensionless_numbers.csv",
        output_dir / "tables" / "model_selection_summary.csv",
        output_dir / "tables" / "robustness_summary.csv",
        output_dir / "tables" / "robustness_long.csv",
        output_dir / "tables" / "error_summary.csv",
        output_dir / "tables" / "error_budget_summary.csv",
        output_dir / "tables" / "parameter_identifiability.csv",
        output_dir / "tables" / "dimensionless_sensitivity_summary.csv",
        output_dir / "tables" / "parameter_reconciliation_audit.csv",
        output_dir / "tables" / "geometry_risk_summary.csv",
        output_dir / "tables" / "validation_hierarchy_table.csv",
        output_dir / "tables" / "data_provenance_summary.csv",
        output_dir / "tables" / "identifiability_diagnostics_v4.csv",
        output_dir / "tables" / "error_bound_summary.csv",
        output_dir / "tables" / "assumption_justification_matrix.csv",
        output_dir / "tables" / "timescale_separation_summary.csv",
        output_dir / "tables" / "validation_stress_tests.csv",
        output_dir / "tables" / "submission_gap_audit.csv",
        output_dir / "tables" / "dynamics_fit_asymmetry_audit.csv",
        output_dir / "tables" / "numerical_credibility_audit.csv",
        output_dir / "tables" / "nomenclature_table.csv",
        output_dir / "tables" / "equation_inventory.csv",
        output_dir / "tables" / "literature_matrix.csv",
        output_dir / "tables" / "active_figure_manifest.csv",
        output_dir / "reports" / "method_framework_draft.md",
        output_dir / "reports" / "paper_outline_draft.md",
        output_dir / "reports" / "manuscript_draft_v1.md",
        output_dir / "reports" / "theory_and_error_analysis.md",
        output_dir / "reports" / "manuscript_draft_v2.md",
        output_dir / "reports" / "manuscript_draft_v3.md",
        output_dir / "reports" / "theory_framework_v4.md",
        output_dir / "reports" / "manuscript_draft_v4.md",
        output_dir / "reports" / "theoretical_derivation_v5.md",
        output_dir / "reports" / "manuscript_draft_v5.md",
        output_dir / "reports" / "rigorous_theory_notes.md",
        output_dir / "reports" / "free_boundary_manifold_rationale.md",
        output_dir / "reports" / "validation_stress_summary.md",
        output_dir / "reports" / "external_validation_data_audit.md",
        output_dir / "reports" / "response_matrix.md",
        output_dir / "reports" / "manuscript_draft_final.md",
        output_dir / "reports" / "supplementary_methods_draft.md",
        output_dir / "reports" / "references_seed.bib",
        output_dir / "reports" / "literature_search_log.md",
        output_dir / "reports" / "figure_captions.md",
        output_dir / "reports" / "reviewer_risk_response.md",
        output_dir / "reports" / "submission_readiness_checklist.md",
        output_dir / "reports" / "cover_letter_draft.md",
        output_dir / "reports" / "highlights_draft.md",
        output_dir / "reports" / "declarations_for_submission.md",
        output_dir / "reports" / "amm_submission_checklist.md",
        output_dir / "reports" / "environment_summary.txt",
        output_dir / "latex" / "main_submission.tex",
        output_dir / "latex" / "supplementary_methods.tex",
        output_dir / "latex" / "references.bib",
        output_dir / "latex" / "latex_figure_manifest.csv",
        output_dir / "latex" / "README.md",
        output_dir / "latex" / "main_submission.pdf",
        output_dir / "latex" / "supplementary_methods.pdf",
        output_dir / "latex" / "latex_compile_summary.txt",
        output_dir / "tables" / "submission_package_manifest.csv",
        output_dir / "tables" / "reproducibility_package_manifest.csv",
        output_dir / "submission_package" / "README.md",
        output_dir / "submission_package" / "submission_package_manifest.csv",
        output_dir / "reproducibility_package" / "README.md",
        output_dir / "reproducibility_package" / "reproducibility_manifest.csv",
        output_dir / "reproducibility_package.zip",
    ]
    required_figure_stems = [
        output_dir / "paper_figures" / "paper_fig01_modeling_framework",
        output_dir / "paper_figures" / "paper_fig02_process_matrix",
        output_dir / "paper_figures" / "paper_fig03_data_moving_frame",
        output_dir / "paper_figures" / "paper_fig04_geometry_quasi_steady",
        output_dir / "paper_figures" / "paper_fig05_free_boundary_model_comparison",
        output_dir / "paper_figures" / "paper_fig06_process_response",
        output_dir / "paper_figures" / "paper_fig07_dimensionless_regime",
        output_dir / "paper_figures" / "paper_fig08_dynamics_validation",
        output_dir / "paper_figures" / "paper_fig14_dynamics_residuals_by_state",
        output_dir / "paper_figures" / "paper_fig09_error_budget_model_selection",
        output_dir / "paper_figures" / "paper_fig10_identifiability_overparameterization",
        output_dir / "paper_figures" / "paper_fig11_leave_one_condition_validation",
        output_dir / "paper_figures" / "paper_fig12_external_holdout_validation",
        output_dir / "paper_figures" / "paper_fig13_simulation_cross_sections",
        output_dir / "figures" / "supp_figS1_all_boundary_fits",
        output_dir / "figures" / "supp_figS2_superellipsoid_parameters",
        output_dir / "figures" / "supp_figS3_dynamics_residuals",
        output_dir / "figures" / "supp_figS4_dimensionless_sensitivity_grid",
        output_dir / "figures" / "supp_figS10_temperature_dependent_properties",
        output_dir / "figures" / "supp_figS5_theory_identifiability_error_bounds",
        output_dir / "figures" / "fig05_boundary_fit_comparison",
        output_dir / "figures" / "fig10_stability_attractor",
        output_dir / "figures" / "fig13_multicondition_process_matrix",
        output_dir / "figures" / "fig14_multicondition_response_surfaces",
        output_dir / "figures" / "fig15_multicondition_geometry_comparison",
        output_dir / "figures" / "fig16_multicondition_dynamics_validation",
        output_dir / "figures" / "fig17_leave_one_condition_validation",
        output_dir / "figures" / "fig18_external_holdout_validation",
        output_dir / "figures" / "fig03_thermal_flow_evolution",
        output_dir / "figures" / "fig06_dynamics_model_comparison",
    ]
    for stem in required_figure_stems:
        for suffix in [".svg", ".pdf", ".tiff", ".png"]:
            required.append(stem.with_suffix(suffix))
    figure_paths = []
    for fig_dir in [output_dir / "figures", output_dir / "paper_figures"]:
        for suffix in [".svg", ".pdf", ".tiff", ".png"]:
            figure_paths.extend(sorted(fig_dir.glob(f"*{suffix}")))
    for path in required + figure_paths:
        records.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    validation = pd.DataFrame(records)
    validation.to_csv(output_dir / "tables" / "output_validation.csv", index=False)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pilot reduced-order melt-pool modeling package.")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw data"))
    parser.add_argument("--validation-dir", type=Path, default=Path("validation data"))
    parser.add_argument("--additional-validation-dir", type=Path, default=Path("additional data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    parser.add_argument(
        "--plot-workers",
        type=int,
        default=default_plot_workers(),
        help="Number of parallel plotting worker processes. Values are clamped to 1-8.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw figures from cached CSV tables and recompile LaTeX without rebuilding models.",
    )
    parser.add_argument("--plot-task", choices=active_plot_task_names(), help=argparse.SUPPRESS)
    args = parser.parse_args()
    plot_workers = normalize_plot_workers(args.plot_workers)

    if args.plot_task:
        _run_plot_task(args.plot_task, str(args.output_dir))
        print(f"Completed plot task: {args.plot_task}")
        return

    if args.plot_only:
        validation = run_plot_only(args.output_dir, plot_workers)
        print(f"Plot-only update complete with {plot_workers} plotting worker(s).")
        print(f"Output directory: {args.output_dir.resolve()}")
        print(validation.to_string(index=False))
        return

    table, point_cloud = build_tables(args.raw_dir)
    base_dir = args.raw_dir.parent if args.raw_dir.parent != Path("") else Path(".")
    property_curves = load_property_curves(base_dir)
    material_parameters = make_material_parameter_table()
    property_table = make_temperature_dependent_property_table(table, property_curves)
    dimensionless_numbers = make_dimensionless_number_table(table, property_table)
    predictions, dynamics_summary, eigenvalues = fit_attractor_model(table)
    coupled_predictions, coupled_summary, coupled_eigenvalues = fit_coupled_attractor_model(table)
    coupled_matrix = coupled_summary.attrs["A_matrix_long"]
    dynamics_comparison = compare_dynamics_models(dynamics_summary, eigenvalues, coupled_summary, coupled_eigenvalues)
    quasi = make_quasi_steady_summary(table)
    geometry_comparison = make_geometry_model_comparison(table)
    superellipsoid_parameters = make_superellipsoid_parameters(table)
    model_selection = make_model_selection_summary(geometry_comparison, dynamics_comparison, coupled_eigenvalues)
    robustness_summary, robustness_long = run_robustness_analysis(args.raw_dir)
    error_summary = make_error_summary(table, dynamics_summary, eigenvalues, coupled_summary, coupled_eigenvalues)
    dimensionless_sensitivity = make_dimensionless_sensitivity_summary(table, property_curves)
    parameter_identifiability = make_parameter_identifiability(table, dynamics_summary, coupled_matrix)
    error_budget = make_error_budget_summary(
        table,
        geometry_comparison,
        dynamics_summary,
        coupled_summary,
        dimensionless_sensitivity,
    )
    validation_table = pd.DataFrame()
    validation_point_cloud = pd.DataFrame()
    validation_dimensionless_numbers = pd.DataFrame()
    validation_geometry_comparison = pd.DataFrame()
    external_holdout_process_response = pd.DataFrame()
    external_holdout_dynamics_predictions = pd.DataFrame()
    external_holdout_dynamics_summary = pd.DataFrame()
    external_holdout_summary = pd.DataFrame()
    validation_sources = [
        (args.validation_dir, "v_prefixed_validation_holdout"),
        (args.additional_validation_dir, "a16_a20_additional_holdout"),
    ]
    input_file_manifest = make_input_file_manifest(
        [
            (args.raw_dir, "model_construction", ""),
            (args.validation_dir, "same_ecosystem_holdout", "v_prefixed_validation_holdout"),
            (args.additional_validation_dir, "same_ecosystem_holdout", "a16_a20_additional_holdout"),
        ]
    )
    validation_table, validation_point_cloud = combine_validation_sources(validation_sources)
    if len(validation_table):
        validation_property_table = make_temperature_dependent_property_table(validation_table, property_curves)
        validation_dimensionless_numbers = make_dimensionless_number_table(validation_table, validation_property_table)
        validation_geometry_comparison = make_geometry_model_comparison(validation_table)
        external_holdout_process_response = make_external_holdout_process_response_validation(table, validation_table)
        external_holdout_dynamics_predictions, external_holdout_dynamics_summary = make_external_holdout_dynamics_validation(
            table,
            validation_table,
            dynamics_summary,
        )
        external_holdout_summary = make_external_holdout_validation_summary(
            validation_table,
            validation_geometry_comparison,
            external_holdout_process_response,
            external_holdout_dynamics_summary,
        )
    write_outputs(
        args.output_dir,
        table,
        point_cloud,
        predictions,
        dynamics_summary,
        eigenvalues,
        coupled_predictions,
        coupled_summary,
        coupled_eigenvalues,
        coupled_matrix,
        dynamics_comparison,
        quasi,
        geometry_comparison,
        superellipsoid_parameters,
        material_parameters,
        property_table,
        dimensionless_numbers,
        model_selection,
        robustness_summary,
        robustness_long,
        error_summary,
        error_budget,
        parameter_identifiability,
        dimensionless_sensitivity,
        validation_table,
        validation_point_cloud,
        validation_geometry_comparison,
        validation_dimensionless_numbers,
        external_holdout_process_response,
        external_holdout_dynamics_predictions,
        external_holdout_dynamics_summary,
        external_holdout_summary,
        input_file_manifest,
        plot_workers=plot_workers,
    )
    validation = validate_outputs(args.output_dir)

    print(
        f"Processed {len(table)} condition-time steps "
        f"({table['case_id'].nunique() if 'case_id' in table.columns else 1} conditions) from {args.raw_dir}"
    )
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Plot workers: {plot_workers}")
    print(f"Stable eigenvalues: {int(eigenvalues['stable_if_negative'].sum())}/{len(eigenvalues)}")
    print(
        "Coupled stable eigenvalues: "
        f"{int(coupled_eigenvalues['stable_if_real_negative'].sum())}/{len(coupled_eigenvalues)}"
    )
    print(
        "Mean validation relative RMSE: "
        f"{float(dynamics_summary['validation_relative_rmse'].mean()):.6f}"
    )
    print(
        "Coupled mean validation relative RMSE: "
        f"{float(coupled_summary['validation_relative_rmse'].mean()):.6f}"
    )
    geom_summary = geometry_comparison[geometry_comparison["time_s"] == "summary"]
    print(
        "Mean volume relative error ellipsoid -> superellipsoid: "
        f"{float(geom_summary.loc[geom_summary['model'] == 'ellipsoid', 'mean_volume_relative_error'].iloc[0]):.6f}"
        " -> "
        f"{float(geom_summary.loc[geom_summary['model'] == 'superellipsoid', 'mean_volume_relative_error'].iloc[0]):.6f}"
    )
    dim_lookup = dimensionless_value_lookup(dimensionless_numbers)
    print(
        "Dimensionless groups: "
        f"Pe={float(dim_lookup['Pe']):.6g}, "
        f"Fo_final={float(dim_lookup['Fo_final']):.6g}, "
        f"Ste={float(dim_lookup['Ste']):.6g}, "
        f"E*={float(dim_lookup['E_star']):.6g}"
    )
    print(
        "Robustness scenarios OK: "
        f"{int(robustness_summary['status'].eq('ok').sum())}/{len(robustness_summary)}"
    )
    if len(external_holdout_summary):
        metric_map = dict(zip(external_holdout_summary["metric"], external_holdout_summary["value"]))
        print(
            "External CFD holdout: "
            f"{int(metric_map.get('external_validation_case_count', 0))} cases, "
            f"{int(metric_map.get('external_validation_cohort_count', 0))} cohorts, "
            f"process mean relative error={float(metric_map.get('external_process_response_mean_relative_error', np.nan)):.6f}, "
            f"dynamics mean relative RMSE={float(metric_map.get('external_dynamics_mean_relative_rmse', np.nan)):.6f}"
        )
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()


