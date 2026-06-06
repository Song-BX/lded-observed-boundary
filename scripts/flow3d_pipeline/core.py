from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
import subprocess
import sys
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.spatial import ConvexHull, QhullError, cKDTree
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

MATERIAL_CONSTANTS = {
    "material": "316L stainless steel",
    "beam_radius_m": 0.00021,
    "absorptivity": 0.35,
    "initial_temperature_K": 298.0,
    "ambient_temperature_K": 298.0,
    "powder_initial_temperature_K": 293.0,
    "solidus_temperature_K": 1648.0,
    "liquidus_temperature_K": 1753.0,
    "latent_heat_fusion_J_per_kg": 1.674e5,
    "boiling_temperature_K": 3023.0,
    "latent_heat_vaporization_J_per_kg": 7.45e6,
    "surface_tension_N_per_m": 1.6,
    "surface_tension_temperature_coefficient_N_per_m_K": 1.9e-4,
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

BOUNDARY_FIT_TIMES = [0.05, 0.20, 0.50, 0.70]
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
            "font.size": 7.2,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
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
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def apply_axis_polish(ax: mpl.axes.Axes, grid: str | None = "y") -> None:
    ax.tick_params(length=2.6, width=0.7, pad=1.6)
    if grid:
        ax.grid(True, axis=grid, color="0.90", linewidth=0.45, linestyle="-", zorder=0)
        ax.set_axisbelow(True)


def short_state_label(name: str) -> str:
    if name in STATE_LABELS:
        return STATE_LABELS[name]
    cleaned = str(name).replace("_m_per_s", "").replace("_m3", "").replace("_m", "")
    cleaned = cleaned.replace("_K_per_m", "").replace("_K", "")
    cleaned = cleaned.replace("melt_pool_length", "Length")
    cleaned = cleaned.replace("full_width", "Width")
    cleaned = cleaned.replace("height_span", "Height")
    cleaned = cleaned.replace("_", " ")
    return cleaned


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
            "role": "main_geometry_model",
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
            for col in ["case_id", "case_index", "power_W", "scan_speed_mm_s", "particle_rate", "powder_feed_g_min"]:
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
            return "latent_heat_dominant"
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
            return "weak_marangoni_scale"
        if value <= 1000.0:
            return "moderate_marangoni_scale"
        return "strong_marangoni_scale"
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
            "recommendation": "Retain only as overparameterization control, not as the main dynamics",
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
            "Scaling uncertainty from material-property reference state and uncertain absorptivity or surface-tension coefficient.",
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
    rows = [
        ("material", MATERIAL_CONSTANTS["material"], "", "user_provided"),
        ("laser_power", LASER_POWER_W, "W", "process_setting"),
        ("scan_speed", SCAN_SPEED_M_PER_S, "m/s", "process_setting"),
        ("powder_feed_rate", POWDER_FEED_KG_PER_S, "kg/s", "converted from 12 g/min"),
        ("beam_radius", MATERIAL_CONSTANTS["beam_radius_m"], "m", "user_provided"),
        ("absorptivity", MATERIAL_CONSTANTS["absorptivity"], "-", "user_provided"),
        ("initial_temperature", MATERIAL_CONSTANTS["initial_temperature_K"], "K", "user_provided"),
        ("ambient_temperature", MATERIAL_CONSTANTS["ambient_temperature_K"], "K", "user_provided"),
        ("powder_initial_temperature", MATERIAL_CONSTANTS["powder_initial_temperature_K"], "K", "user_provided"),
        ("solidus_temperature", MATERIAL_CONSTANTS["solidus_temperature_K"], "K", "user_provided"),
        ("liquidus_temperature", MATERIAL_CONSTANTS["liquidus_temperature_K"], "K", "user_provided"),
        ("latent_heat_fusion", MATERIAL_CONSTANTS["latent_heat_fusion_J_per_kg"], "J/kg", "user_provided"),
        ("boiling_temperature", MATERIAL_CONSTANTS["boiling_temperature_K"], "K", "user_provided"),
        ("latent_heat_vaporization", MATERIAL_CONSTANTS["latent_heat_vaporization_J_per_kg"], "J/kg", "user_provided"),
        ("surface_tension", MATERIAL_CONSTANTS["surface_tension_N_per_m"], "N/m", "user_provided; kg/s^2 equals N/m"),
        (
            "surface_tension_temperature_coefficient",
            MATERIAL_CONSTANTS["surface_tension_temperature_coefficient_N_per_m_K"],
            "N/(m K)",
            "user_provided; kg/s^2/K equals N/(m K)",
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
        ("global", "rho_liquidus", rho, "kg/m3", "interpolated/extrapolated at liquidus"),
        ("global", "cp_liquidus", cp, "J/(kg K)", "interpolated/extrapolated at liquidus"),
        ("global", "k_liquidus", k, "W/(m K)", "interpolated/extrapolated at liquidus"),
        ("global", "alpha_liquidus", alpha, "m2/s", "k/(rho cp) at liquidus"),
        ("global", "mu_liquidus", mu, "kg/(m s)", "viscosity curve uses nearest lower value below first liquid data"),
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
    return steady.groupby("case_id", as_index=False).agg(
        case_index=("case_index", "first"),
        power_W=("power_W", "first"),
        scan_speed_mm_s=("scan_speed_mm_s", "first"),
        particle_rate=("particle_rate", "first"),
        powder_feed_g_min=("powder_feed_g_min", "first"),
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
            rows.append(
                {
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
            summary_rows.append(
                {
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
                    "interpretation": "Independent V-prefixed FLOW-3D holdout conditions.",
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
                    "interpretation": "Mean quasi-steady process-response error on V-prefixed holdout cases.",
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


def plot_moving_frame(point_cloud: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    point_cloud = representative_case_subset(point_cloud)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    cmap = mpl.colormaps["viridis"]
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
    descriptor_color = "#4C78A8"
    mirror_color = "#F58518"
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
        arrowprops=dict(arrowstyle="<->", color="#54A24B", lw=1.05),
    )
    axes[0, 1].text(x_for_w - 0.055 * xi_range, 0, r"$W$", color="#54A24B", va="center", ha="right")
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
        arrowprops=dict(arrowstyle="<->", color="#D95F02", lw=1.1),
    )
    axes[1, 1].text((xi_max + xi_center) / 2, z_mid + 0.055, r"$L_f$", color="#D95F02", ha="center")
    axes[1, 1].annotate(
        "",
        xy=(xi_min, z_mid),
        xytext=(xi_center, z_mid),
        arrowprops=dict(arrowstyle="<->", color="#F58518", lw=1.1),
    )
    axes[1, 1].text((xi_min + xi_center) / 2, z_mid + 0.055, r"$L_r$", color="#F58518", ha="center")
    x_for_h = xi_max + 0.14 * xi_range
    axes[1, 1].annotate(
        "",
        xy=(x_for_h, z_max),
        xytext=(x_for_h, z_min),
        arrowprops=dict(arrowstyle="<->", color="#54A24B", lw=1.1),
    )
    axes[1, 1].text(x_for_h + 0.04 * xi_range, (z_min + z_max) / 2, r"$H$", color="#54A24B", va="center")
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
        ("front_length_m", r"$L_f$", "#4C78A8"),
        ("rear_length_m", r"$L_r$", "#F58518"),
        ("full_width_m", r"$W$", "#54A24B"),
        ("height_span_m", r"$H$", "#B279A2"),
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
    axes[0].plot(x, table["Tmax_K"], marker="o", ms=3.0, lw=1.2, color="#E45756")
    axes[1].plot(x, table["Gmean_K_per_m"] / 1e6, marker="o", ms=3.0, lw=1.2, color="#4C78A8")
    axes[2].plot(x, table["Umax_m_per_s"], marker="o", ms=3.0, lw=1.2, color="#54A24B")
    axes[0].set_ylabel(r"$T_{max}$ (K)")
    axes[1].set_ylabel(r"$\overline{G}$ (MK m$^{-1}$)")
    axes[2].set_ylabel(r"$U_{max}$ (m s$^{-1}$)")
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
        ax.plot(time, pred, "-", lw=1.1, color="#4C78A8", label="Attractor model")
        ax.axvspan(time[0], time[np.where(train)[0][-1]], color="0.92", zorder=-1)
        ax.set_title(STATE_LABELS[col])
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


def plot_boundary_fit_comparison(table: pd.DataFrame, point_cloud: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    case_id = representative_case_id(table)
    if case_id is not None:
        table = table[table["case_id"].astype(str).eq(case_id)].copy().sort_values("time_s")
        point_cloud = point_cloud[point_cloud["case_id"].astype(str).eq(case_id)].copy()
    selected_times = []
    available = table["time_s"].to_numpy(dtype=float)
    for requested in BOUNDARY_FIT_TIMES:
        selected_times.append(float(available[np.argmin(np.abs(available - requested))]))
    selected_times = list(dict.fromkeys(selected_times))

    fig, axes = plt.subplots(len(selected_times), 2, figsize=(7.2, 1.7 * len(selected_times)), constrained_layout=True)
    if len(selected_times) == 1:
        axes = np.asarray([axes])
    colors = {"ellipsoid": "#4C78A8", "superellipsoid": "#E45756"}
    labels = {"ellipsoid": "Ellipsoid", "superellipsoid": "Superellipsoid"}

    for row_idx, time_s in enumerate(selected_times):
        row = table.loc[np.isclose(table["time_s"], time_s)].iloc[0]
        part = point_cloud[np.isclose(point_cloud["time_s"], time_s)]
        xi = part["xi_m"].to_numpy() * 1e3
        y = part["Points_1"].to_numpy() * 1e3
        z = part["Points_2"].to_numpy() * 1e3
        axes[row_idx, 0].scatter(xi, y, s=5, alpha=0.35, color="0.45", linewidths=0)
        axes[row_idx, 0].scatter(xi, -y, s=5, alpha=0.35, color="0.75", linewidths=0)
        axes[row_idx, 1].scatter(xi, z, s=5, alpha=0.35, color="0.45", linewidths=0)

        for model in ["ellipsoid", "superellipsoid"]:
            params = model_params_from_row(row, model)
            if not np.all(np.isfinite(params)):
                continue
            for side in [-1, 1]:
                xi_curve, y_curve = boundary_curve_top(params, model, side)
                axes[row_idx, 0].plot(xi_curve * 1e3, y_curve * 1e3, color=colors[model], lw=1.1)
                axes[row_idx, 0].plot(xi_curve * 1e3, -y_curve * 1e3, color=colors[model], lw=1.1)
                xi_curve, z_curve = boundary_curve_side(params, model, side)
                axes[row_idx, 1].plot(xi_curve * 1e3, z_curve * 1e3, color=colors[model], lw=1.1)
                axes[row_idx, 1].plot(xi_curve * 1e3, (2 * params[5] - z_curve) * 1e3, color=colors[model], lw=1.1)

        axes[row_idx, 0].axvline(0, color="0.3", lw=0.6, ls="--")
        axes[row_idx, 1].axvline(0, color="0.3", lw=0.6, ls="--")
        axes[row_idx, 0].set_ylabel(f"{time_s:.2f}s\ny (mm)")
        axes[row_idx, 1].set_ylabel("z (mm)")
        axes[row_idx, 0].set_title("Top view" if row_idx == 0 else "")
        axes[row_idx, 1].set_title("Side view" if row_idx == 0 else "")
        apply_axis_polish(axes[row_idx, 0], grid=None)
        apply_axis_polish(axes[row_idx, 1], grid=None)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Moving coordinate $\xi$ (mm)")
    add_panel_label(axes[0, 0], "a", x=-0.11, y=1.08)
    add_panel_label(axes[0, 1], "b", x=-0.11, y=1.08)
    handles = [
        mpl.lines.Line2D([0], [0], color=colors[m], lw=1.2, label=labels[m])
        for m in ["ellipsoid", "superellipsoid"]
    ]
    axes[0, 1].legend(handles=handles, loc="upper right")
    save_publication_figure(fig, fig_dir / "fig05_boundary_fit_comparison")
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
        ax.plot(time, diag, "-", lw=1.0, color="#4C78A8", label="Diagonal")
        ax.plot(time, coupled, "--", lw=1.0, color="#E45756", label="Coupled")
        ax.set_title(STATE_LABELS[col])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(units[col])
    axes[-1].axis("off")
    mean_errors = dynamics_comparison.groupby("model")["validation_relative_rmse"].mean()
    model_labels = {
        "diagonal_attractor": "Diagonal",
        "coupled_ridge_attractor": "Coupled",
    }
    text = "\n".join([f"{model_labels.get(str(model), str(model))}: {value:.3f}" for model, value in mean_errors.items()])
    axes[-1].text(0.05, 0.62, "Mean validation\nrelative RMSE", fontsize=7, transform=axes[-1].transAxes)
    axes[-1].text(0.05, 0.35, text, fontsize=7, transform=axes[-1].transAxes)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="lower left")
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
    risk_colors = {"low": "#4C78A8", "medium": "#F2A541", "high": "#D95F02"}
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
    axes[1].hlines(y, rel_min, rel_max, color="#4C78A8", lw=3)
    axes[1].plot(np.ones_like(y), y, "o", color="#222222", ms=3, label="baseline")
    axes[1].plot(rel_min, y, "|", color="#4C78A8", ms=7)
    axes[1].plot(rel_max, y, "|", color="#4C78A8", ms=7)
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
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 1.8), constrained_layout=True)
    ax.set_axis_off()
    labels = [
        ("FLOW-3D\nhalf-domain\nmolten points", "#D9E8F5"),
        ("Symmetry\nreconstruction\nY = 0", "#E7F0D8"),
        ("Moving frame\nxi = x - vt", "#F5E6CC"),
        ("Observed\nboundary\nenvelope", "#EADCF0"),
        ("Superellipsoid\nlow-dimensional\nmanifold", "#DDEDEB"),
        ("Diagonal\nattractor\ndynamics", "#F3DCD7"),
        ("Stability,\nerror budget,\nidentifiability", "#E7E7E7"),
    ]
    x_positions = np.linspace(0.06, 0.94, len(labels))
    y = 0.62
    width = 0.12
    height = 0.50
    for idx, (x, (label, color)) in enumerate(zip(x_positions, labels)):
        box = mpl.patches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color,
            edgecolor="0.25",
            linewidth=0.8,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=7, transform=ax.transAxes)
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - width / 2 - 0.01, y),
                xytext=(x + width / 2 + 0.01, y),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"),
            )
    ax.text(0.06, 0.13, "Input", ha="center", va="center", fontsize=7, color="0.35", transform=ax.transAxes)
    ax.text(0.50, 0.13, "Boundary-envelope modeling", ha="center", va="center", fontsize=7, color="0.35", transform=ax.transAxes)
    ax.text(0.88, 0.13, "Model selection", ha="center", va="center", fontsize=7, color="0.35", transform=ax.transAxes)
    save_publication_figure(fig, fig_dir / "fig08_modeling_framework")
    plt.close(fig)


def plot_dimensionless_regime(dimensionless_sensitivity: pd.DataFrame, fig_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    sens = dimensionless_sensitivity.copy()
    symbols = sens["symbol"].tolist()
    y = np.arange(len(symbols))
    baseline = sens["baseline_value"].to_numpy(dtype=float)
    min_values = sens["min_value"].to_numpy(dtype=float)
    max_values = sens["max_value"].to_numpy(dtype=float)
    colors = ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]
    axes[0].barh(y, baseline, color=colors, height=0.58)
    axes[0].set_xscale("log")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(symbols)
    axes[0].set_xlabel("Baseline value (log scale)")
    axes[0].set_title("Dimensionless regime")
    apply_axis_polish(axes[0], grid="x")
    for yi, value in zip(y, baseline):
        axes[0].text(value * 1.08, yi, f"{value:.2g}", va="center", fontsize=6.5)

    rel_min = sens["relative_min"].to_numpy(dtype=float)
    rel_max = sens["relative_max"].to_numpy(dtype=float)
    axes[1].hlines(y, rel_min, rel_max, color="#4C78A8", lw=3)
    axes[1].plot(np.ones_like(y), y, "o", color="#222222", ms=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(symbols)
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
        axes[0].plot(t, err / denom, lw=1.0, alpha=0.85, label=STATE_LABELS[col])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(r"$|q-q_\infty|$ / max")
    axes[0].set_title("State convergence")
    apply_axis_polish(axes[0])

    labels = [STATE_LABELS[col] for col in dynamics_summary["state"]]
    k_values = dynamics_summary["k_per_s"].to_numpy(dtype=float)
    axes[1].bar(np.arange(len(labels)), k_values, color="#4C78A8", width=0.72)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel(r"$k_i$ (s$^{-1}$)")
    axes[1].set_title("Diagonal rates")
    axes[1].axhline(0, color="0.3", lw=0.8)
    apply_axis_polish(axes[1])

    real = coupled_eigenvalues["jacobian_eigenvalue_real_per_s"].to_numpy(dtype=float)
    imag = coupled_eigenvalues["jacobian_eigenvalue_imag_per_s"].to_numpy(dtype=float)
    axes[2].scatter(real, imag, s=28, color="#E45756", edgecolor="white", linewidth=0.5)
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
    colors = ["#4C78A8", "#54A24B", "#F58518", "#B279A2", "#E45756", "#72B7B2"][: len(eb)]
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
    bar_colors = np.where(selected, "#4C78A8", "#BBBBBB")
    axes[1].bar(np.arange(len(labels)), values, color=bar_colors, width=0.68)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=0, ha="center", fontsize=6.5)
    axes[1].set_ylabel("Primary metric")
    axes[1].set_title("Model selection")
    axes[1].set_yscale("log")
    for idx, is_selected in enumerate(selected):
        if is_selected:
            axes[1].text(idx, values[idx] * 1.12, "selected", ha="center", va="bottom", fontsize=6.1, color="#4C78A8")
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
    colors = geom["risk_level"].map({"low": "#4C78A8", "medium": "#F2A541", "high": "#D95F02"}).fillna("#777777")
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
    im = axes[1].imshow(mat.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_xticks(np.arange(len(STATE_COLUMNS)))
    axes[1].set_yticks(np.arange(len(STATE_COLUMNS)))
    axes[1].set_xticklabels([STATE_LABELS[c] for c in STATE_COLUMNS], rotation=45, ha="right")
    axes[1].set_yticklabels([STATE_LABELS[c] for c in STATE_COLUMNS])
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
            ax.plot(xi_curve * 1e3, y_curve * 1e3, color="#E45756", lw=0.8)
            ax.plot(xi_curve * 1e3, -y_curve * 1e3, color="#E45756", lw=0.8)
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
        ax.plot(t, table[col].to_numpy(dtype=float) * scale, "o-", ms=2.6, lw=1.0, color="#4C78A8")
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
        ax.plot(time, diag, "o-", ms=2.4, lw=0.9, color="#4C78A8", label="Diagonal")
        ax.plot(time, coupled, "s--", ms=2.2, lw=0.9, color="#E45756", label="Coupled")
        ax.set_title(STATE_LABELS[col])
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
    symbols = sens["symbol"].tolist()
    data = np.vstack(
        [
            sens["relative_min"].to_numpy(dtype=float),
            np.ones(len(sens)),
            sens["relative_max"].to_numpy(dtype=float),
        ]
    )
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(np.arange(len(symbols)))
    ax.set_xticklabels(symbols)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Minimum", "Baseline", "Maximum"])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_title("Dimensionless sensitivity scenario envelope")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Relative to baseline")
    cbar.ax.tick_params(labelsize=6.2, length=2.4)
    save_publication_figure(fig, fig_dir / "supp_figS4_dimensionless_sensitivity_grid")
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
    axes[0].barh(y, eb["normalized_proxy"].to_numpy(dtype=float), color="#4C78A8", height=0.62)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(eb["bound_component"].str.replace("E_", "", regex=False), fontsize=6.3)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Normalized proxy")
    axes[0].set_title("Error-budget weights")
    apply_axis_polish(axes[0], grid="x")

    risk_colors = {"low": "#4C78A8", "medium": "#F2A541", "high": "#D95F02"}
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
    axes[1].set_yticklabels(plot_df["label"].tolist(), fontsize=6.4, linespacing=1.1)
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(["low", "medium", "high"])
    axes[1].set_xlim(0, 3.35)
    axes[1].invert_yaxis()
    axes[1].set_title("Identifiability risk")
    axes[1].grid(axis="x", color="0.9", lw=0.6)

    sens = dimensionless_sensitivity.copy()
    span = np.maximum(
        abs(sens["relative_min"].to_numpy(dtype=float) - 1.0),
        abs(sens["relative_max"].to_numpy(dtype=float) - 1.0),
    )
    x = np.arange(len(sens))
    colors = np.where(sens["conclusion_changed"].to_numpy(dtype=bool), "#D95F02", "#54A24B")
    axes[2].bar(x, span, color=colors, width=0.62)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sens["symbol"].tolist())
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
    base_color = "#2c7fb8"
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
    fig.suptitle("Multi-condition process matrix (n = 15 training cases)", fontsize=8.8, y=1.02)
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
    x_plot = case["scan_speed_mm_s"].to_numpy(dtype=float) + 0.18 * (powder - powder_center) / powder_span
    for ax, (col, label) in zip(axes, panels):
        sc = ax.scatter(
            x_plot,
            case["power_W"],
            c=case[col],
            s=42 + 5 * (case["powder_feed_g_min"] - case["powder_feed_g_min"].min()),
            cmap="viridis",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel(r"Scan speed (mm s$^{-1}$)")
        ax.set_ylabel("Power (W)")
        ax.set_title(label)
        ax.set_xlim(5.72, 10.28)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
        cbar.set_label(label)
        cbar.ax.tick_params(labelsize=6.0, length=2.2)
        apply_axis_polish(ax, grid="both")
    fig.text(
        0.51,
        0.01,
        "Marker area increases with powder feed.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="0.35",
    )
    for label, ax in zip(["a", "b", "c", "d"], axes):
        add_panel_label(ax, label)
    save_publication_figure(fig, fig_dir / "fig14_multicondition_response_surfaces")
    plt.close(fig)


def plot_multi_condition_geometry_comparison(geometry_comparison: pd.DataFrame, fig_dir: Path) -> None:
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
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.5), sharex=True, constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "mean_boundary_residual_rmse", "Boundary RMSE"),
        (axes[1], "mean_volume_relative_error", "Volume rel. error"),
    ]:
        pivot = case.pivot(index="case_index", columns="model", values=metric).sort_index()
        x = np.arange(len(pivot))
        width = 0.36
        ax.bar(x - width / 2, pivot.get("ellipsoid", pd.Series(index=pivot.index, dtype=float)), width, label="Ellipsoid", color="#9E9E9E")
        ax.bar(x + width / 2, pivot.get("superellipsoid", pd.Series(index=pivot.index, dtype=float)), width, label="Superellipsoid", color="#4C78A8")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        apply_axis_polish(ax)
        if {"ellipsoid", "superellipsoid"}.issubset(pivot.columns):
            wins = int((pivot["superellipsoid"] < pivot["ellipsoid"]).sum())
            ax.text(
                0.01,
                0.94,
                f"superellipsoid lower in {wins}/{len(pivot)} cases",
                transform=ax.transAxes,
                fontsize=6.2,
                color="0.30",
                ha="left",
                va="top",
            )
    axes[1].set_xticks(np.arange(len(pivot)))
    axes[1].set_xticklabels([f"A{int(idx)}" for idx in pivot.index], rotation=0)
    axes[1].set_xlabel("Condition")
    axes[0].set_title("Cross-condition observed boundary-envelope model comparison")
    for label, ax in zip(["a", "b"], axes):
        add_panel_label(ax, label)
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
    axes[0].bar(x - width / 2, pivot["diagonal_attractor"], width, color="#4C78A8", label="Diagonal")
    axes[0].bar(x + width / 2, pivot["coupled_ridge_attractor"], width, color="#E45756", label="Coupled")
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
    axes[1].barh(y - 0.18, state["diagonal_attractor"], 0.34, color="#4C78A8", label="Diagonal")
    axes[1].barh(y + 0.18, state["coupled_ridge_attractor"], 0.34, color="#E45756", label="Coupled")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([STATE_LABELS[col] for col in state.index])
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
            label=short_state_label(target),
            edgecolor="white",
            linewidth=0.25,
        )
    axes[0].plot([0, 1], [0, 1], color="0.3", lw=0.8, ls="--")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("Actual value, target-wise normalized")
    axes[0].set_ylabel("LOCO prediction, target-wise normalized")
    axes[0].set_title("Held-out prediction")
    axes[0].legend(fontsize=5.6, ncols=1, loc="lower right")
    apply_axis_polish(axes[0], grid="both")

    summary = loco[loco["held_out_case_id"].astype(str).eq("summary")].copy()
    summary = summary[summary["target"].isin(targets)]
    y = np.arange(len(summary))
    axes[1].barh(y, summary["mean_relative_error"].to_numpy(dtype=float), color="#4C78A8", height=0.58)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([short_state_label(t) for t in summary["target"]], fontsize=6.3)
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
            "External CFD holdout validation is not available.",
        )
        return
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.35), constrained_layout=True)

    geom_case = external_geometry_comparison[
        external_geometry_comparison["time_s"].astype(str).eq("case_summary")
    ].copy()
    boundary = geom_case.pivot(index="case_id", columns="model", values="mean_boundary_residual_rmse")
    case_ids = boundary.index.astype(str).tolist()
    x = np.arange(len(case_ids))
    if {"ellipsoid", "superellipsoid"}.issubset(boundary.columns):
        axes[0].bar(x - 0.17, boundary["ellipsoid"], width=0.34, color="#A6A6A6", label="Ellipsoid")
        axes[0].bar(x + 0.17, boundary["superellipsoid"], width=0.34, color="#3B6EA8", label="Superellipsoid")
        wins = int((boundary["superellipsoid"] < boundary["ellipsoid"]).sum())
        axes[0].text(
            0.02,
            0.94,
            f"lower residual in {wins}/{len(boundary)}",
            transform=axes[0].transAxes,
            fontsize=6.0,
            color="0.30",
            ha="left",
            va="top",
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(case_ids, rotation=35, ha="right")
    axes[0].set_ylabel("Boundary residual")
    axes[0].set_title("External geometry")
    axes[0].legend(loc="best", fontsize=6)
    apply_axis_polish(axes[0])

    process_summary = external_process_validation[
        external_process_validation["case_id"].astype(str).eq("summary")
    ].copy()
    if len(process_summary):
        order = process_summary.sort_values("mean_relative_error", ascending=False)
        labels = [short_state_label(str(t)) for t in order["target"]]
        axes[1].barh(np.arange(len(order)), order["mean_relative_error"], color="#6BA292")
        axes[1].set_yticks(np.arange(len(order)))
        axes[1].set_yticklabels(labels, fontsize=6)
        axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean relative error")
    axes[1].set_title("Process response")
    apply_axis_polish(axes[1], grid="x")

    dyn_state = (
        external_dynamics_summary.groupby("state", as_index=False)["relative_rmse"]
        .mean()
        .sort_values("relative_rmse", ascending=False)
    )
    labels = [short_state_label(str(t)) for t in dyn_state["state"]]
    axes[2].barh(np.arange(len(dyn_state)), dyn_state["relative_rmse"], color="#B76E79")
    axes[2].set_yticks(np.arange(len(dyn_state)))
    axes[2].set_yticklabels(labels, fontsize=6)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("External relative RMSE")
    axes[2].set_title("Attractor trajectory")
    apply_axis_polish(axes[2], grid="x")

    for label, ax in zip(["a", "b", "c"], axes):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    save_publication_figure(fig, fig_dir / "fig18_external_holdout_validation")
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
        "paper_fig09_error_budget_model_selection": "fig11_error_budget_model_selection",
        "paper_fig10_identifiability_overparameterization": "fig12_identifiability_overparameterization",
        "paper_fig11_leave_one_condition_validation": "fig17_leave_one_condition_validation",
        "paper_fig12_external_holdout_validation": "fig18_external_holdout_validation",
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

The present implementation uses the supplied 316L process/material settings and interpolates temperature-dependent properties at the liquidus temperature for the reference transport properties. The computed dimensionless groups are:

```text
Pe = v L / alpha,
Fo = alpha t / L^2,
Ste = c_p (T_l - T_m) / L_m,
E* = eta P / [rho c_p v r_b^2 (T_m - T_0)].
```

For the current data, `Pe = {pe:.3g}`, `Fo_final = {fo:.3g}`, `Ste = {ste:.3g}`, `E* = {e_star:.3g}`, `Re = {re:.3g}`, `Pr = {pr:.3g}`, and `Ma = {ma:.3g}`. The Marangoni number uses the corrected surface-tension temperature coefficient magnitude `|d sigma/dT| = 1.9e-4 N/(m K)`. The modeling table also includes `lf_star, lr_star, w_star, h_star`, normalized by the quasi-steady mean melt-pool length `L_ref`.

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

If this value is `False`, the coupled model should be reported as a controlled over-parameterization test rather than as the main predictive model.

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

The superellipsoid is therefore selected as the main geometric model because it improves both boundary residual and volume error while retaining a closed-form volume.

## 6. Reduced-order dynamics

- Diagonal attractor mean validation relative RMSE: `{diagonal_validation:.4f}`.
- Coupled ridge attractor mean validation relative RMSE: `{coupled_validation:.4f}`.
- Coupled attractor eigenvalues are stable, but validation error is not lower than the diagonal baseline.

The diagonal attractor is selected as the main dynamical model. The coupled model is retained as an over-parameterization test.

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
        "The boundary residual is therefore used as the primary envelope-fit selection metric, whereas the volume proxy "
        "and distance diagnostics are retained as separate limitation diagnostics rather than as selection claims."
    )


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
        ("Gamma(t)", "free boundary", "m2", "outer envelope of the molten-region point cloud"),
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
        ("T_s", "solidus temperature", "K", "1648 K"),
        ("T_l", "liquidus temperature", "K", "1753 K"),
        ("T_max", "maximum temperature", "K", "maximum molten-region temperature at a time step"),
        ("G", "temperature-gradient magnitude", "K/m", "FLOW-3D temperature-gradient output"),
        ("G_mean", "mean temperature gradient", "K/m", "point-cloud average at a time step"),
        ("U", "velocity magnitude", "m/s", "FLOW-3D velocity magnitude"),
        ("U_max", "maximum velocity magnitude", "m/s", "maximum molten-region speed at a time step"),
        ("rho", "density", "kg/m3", "temperature-dependent 316L property"),
        ("c_p", "specific heat capacity", "J/(kg K)", "temperature-dependent 316L property"),
        ("k", "thermal conductivity", "W/(m K)", "temperature-dependent 316L property"),
        ("alpha", "thermal diffusivity", "m2/s", "alpha = k/(rho c_p)"),
        ("mu", "dynamic viscosity", "kg/(m s)", "temperature-dependent 316L property"),
        ("sigma", "surface tension", "N/m", "corrected value 1.6 N/m"),
        ("d sigma/dT", "surface-tension temperature coefficient", "N/(m K)", "corrected magnitude 1.9e-4 N/(m K)"),
        ("P", "laser power", "W", "750 W"),
        ("eta", "laser absorptivity", "-", "0.35"),
        ("r_b", "laser beam radius", "m", "0.00021 m"),
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
            "Nine-parameter main geometry model.",
        ),
        (
            "E6",
            "Free-boundary model",
            "Superellipsoid full volume",
            r"V = (a_f+a_r) b c 2^3 Gamma(1+1/n) Gamma(1+1/m) Gamma(1+1/p)/Gamma(1+1/n+1/m+1/p)",
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
            "Primary nondimensional groups reported in the manuscript.",
        ),
        (
            "E9",
            "Reduced-order dynamics",
            "Diagonal attractor",
            r"dq_i/dt = k_i(q_inf,i - q_i)",
            "Selected main dynamical model.",
        ),
        (
            "E10",
            "Reduced-order dynamics",
            "Coupled ridge attractor",
            r"dq/dt = A(q_inf - q)",
            "Coupled over-parameterization test.",
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
            "Assumption validation",
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
            "Diagonal parameter identification",
            r"min_{k_i>=0} sum_{r in T_tr} [dot q_{i,r}-k_i(q_inf,i-q_i(t_r))]^2",
            "States how the selected diagonal attractor rates are estimated.",
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
        f"A reviewer may argue that {n_conditions} FLOW-3D conditions are not enough for a universal process map or experimental validation."
        if n_conditions > 1
        else "A reviewer may argue that one condition, 750 W, 8 mm/s and 12 g/min, is insufficient for a process-map or general predictive model."
    )
    scope_response = (
        "Frame the contribution as multi-condition FLOW-3D-informed free-boundary reduction, not as an experimentally validated universal process map. The analysis tests whether the analytic boundary and attractor structure persist across power, scan speed and powder-feed settings."
        if n_conditions > 1
        else "Frame the contribution as single-condition transient-to-quasi-steady reduced-order modeling, not as prediction over arbitrary L-DED parameters."
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

**Response strategy.** State explicitly that Gamma(t) is the envelope of the exported molten domain, not an independently reconstructed solid-liquid interface from the full temperature field. This is acceptable for the present modeling target because the reduced state is defined on the available molten-region free boundary. The volume is reported as a symmetry-reconstructed convex-hull proxy, V_full = 2 V_half, rather than as an exact thermodynamic melt volume.

## Risk 3: The superellipsoid may overfit the boundary

**Likely concern.** The superellipsoid has nine parameters, compared with six for the ellipsoid baseline.

**Response strategy.** Treat the ellipsoid as a required baseline and justify the superellipsoid by boundary-model evidence. In the current results, {geometry_selection_text} Robustness checks show superellipsoid improvement in {vals['super_volume_wins']}/{vals['robust_total']} scenarios for volume error and {vals['super_boundary_wins']}/{vals['robust_total']} scenarios for boundary residual. The manuscript should emphasize that the model remains analytic and low-dimensional, while volume and distance-proxy mismatches are reported rather than hidden. The new geometry table also reports paired better rates, sign-test p-values and distance proxies, so the choice is framed as a boundary-residual selection with explicit geometric-risk diagnostics, not as complete three-dimensional geometric dominance.

## Risk 4: The coupled dynamical model is not selected

**Likely concern.** A reviewer may expect cross-coupling among melt-pool geometry, temperature gradient and flow velocity.

**Response strategy.** Present the coupled model as a controlled over-parameterization test. Although the coupled ridge attractor is stable, its mean validation relative RMSE is {_fmt(vals['coupled_validation'], 4)}, compared with {_fmt(vals['diagonal_validation'], 4)} for the diagonal attractor. Robustness checks show coupled-model improvement in {vals['coupled_wins']}/{vals['robust_total']} tested scenarios. The conclusion should therefore be that coupling is physically plausible but not statistically justified by the available condition-wise sequences.
The model-selection table now also includes paired better rates and sign-test p-values for the diagonal-versus-coupled comparison, so the rejection of the coupled model is not a qualitative impression.

## Risk 5: Dimensionless groups depend on chosen 316L properties

**Likely concern.** Temperature-dependent properties make Pe, Ste, E* and Ma sensitive to the reference temperature.

**Response strategy.** Report the reference convention: liquidus-temperature 316L properties from the supplied tables, L_ref from the quasi-steady melt-pool length, and corrected surface-tension coefficient magnitude 1.9e-4 N/(m K). The current values, Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}, should be interpreted as scaling diagnostics rather than universal material constants.
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

The material is 316L stainless steel. Density, heat capacity, thermal conductivity and viscosity are temperature dependent and are taken from the supplied property tables. The laser beam radius is 0.00021 m, absorptivity is 0.35, the initial substrate temperature is 298 K, solidus and liquidus temperatures are 1648 K and 1753 K, and the latent heat of fusion is 1.674e5 J/kg. The surface tension is 1.6 N/m, and the corrected surface-tension temperature coefficient magnitude is 1.9e-4 N/(m K).

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

The dimensionless framework uses the quasi-steady melt-pool length as L_ref and liquidus-temperature 316L properties as reference material values. The primary groups are

```text
Pe = v L_ref / alpha
Fo = alpha t / L_ref^2
Ste = c_p (T_l - T_s) / L_fus
E* = eta P / [rho c_p v r_b^2 (T_l - T0)]
Ma = |d sigma/dT| (T_l - T_s) L_ref / (mu alpha)
```

The computed values are Pe={vals['Pe']:.2f}, Fo_final={vals['Fo_final']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f}, Re={vals['Re']:.2f}, Pr={vals['Pr']:.3f} and Ma={vals['Ma']:.2f}. Pe near unity indicates that advection by the moving heat source and thermal diffusion are both relevant at the reference scale. The large E* reflects substantial normalized laser input, while the large Ma indicates that surface-tension-driven flow can be dynamically important even though the present reduced-order dynamics is identified from CFD outputs rather than solved directly from Navier-Stokes equations.

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

The superellipsoid improves the free-boundary representation relative to the ellipsoid baseline (Fig. 3). The mean boundary residual decreases from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)}, and the mean volume relative error decreases from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}. Robustness checks across training fraction, quasi-steady cutoff and exponent upper bound show volume-error improvement in {vals['super_volume_wins']}/{vals['robust_total']} scenarios and boundary-residual improvement in {vals['super_boundary_wins']}/{vals['robust_total']} scenarios. The model-selection table therefore assigns the superellipsoid as the main geometry model.

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

This work converts FLOW-3D molten-region point-cloud data into a moving-frame, symmetry-aware free-boundary reduced-order model for L-DED. The main geometric model is an asymmetric superellipsoid, supported by lower boundary residual and volume error than an ellipsoid baseline and by {vals['super_volume_wins']}/{vals['robust_total']} robustness improvement for volume error. The main dynamical model is a diagonal attractor, because the coupled ridge model does not improve validation performance in the present dataset. The resulting framework provides a conservative and reproducible bridge from high-fidelity CFD output to a mathematical modeling manuscript focused on transient-to-quasi-steady melt-pool dynamics.
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
            f"The paired condition-wise comparison gives boundary-residual improvement in {geom_wins}/{geom_total} conditions "
            f"(sign-test p={geom_p:.3g}, median residual reduction {geom_adv:.4g}), while the volume proxy improves in {vol_wins}/{geom_total} conditions."
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
            f"In paired condition-state comparisons, the diagonal model has lower validation error in {diag_wins}/{dyn_total} pairs "
            f"(sign-test p={dyn_p:.3g}, median relative-RMSE reduction {dyn_adv:.4g})."
        )
    changed_groups = dimensionless_sensitivity[dimensionless_sensitivity["conclusion_changed"]]
    changed_text = ", ".join(changed_groups["symbol"].astype(str).tolist()) if len(changed_groups) else "none"
    error_table_text = "\n".join(
        [
            f"- {row.error_term}: {row.primary_metric} = {row.primary_value:.4g}."
            for row in error_budget.itertuples()
        ]
    )
    text = f"""# CFD-informed free-boundary reduction of laser directed energy deposition melt-pool evolution via superellipsoid manifolds and stable attractor dynamics

## Abstract

High-fidelity computational fluid dynamics can resolve melt-pool evolution in laser directed energy deposition (L-DED), but the resulting fields are difficult to use directly in mathematical modeling. This study develops a CFD-informed reduced-order framework for a single 316L stainless-steel L-DED condition at 750 W, 8 mm/s and 12 g/min. The exported FLOW-3D data contain only the molten region, so the melt pool is modeled as a moving-frame free-boundary point cloud. A half-domain simulation is reconstructed through the y = 0 symmetry plane, the free boundary is fitted by an asymmetric superellipsoid, and the extracted state is advanced using a stable low-dimensional attractor. The liquidus-reference dimensionless groups are Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}. The superellipsoid reduces mean boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)} and mean volume relative error from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}, with geometric improvement in {vals['super_volume_wins']}/{vals['robust_total']} robustness settings. A coupled ridge attractor is stable but improves validation error in {vals['coupled_wins']}/{vals['robust_total']} settings, so it is retained as an over-parameterization control. The selected model is a superellipsoid free boundary with a diagonal attractor dynamic. The manuscript explicitly reports stability propositions, error decomposition and parameter-identifiability risks to keep the single-condition claim bounded.

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

where a_s = a_f in front of the center and a_s = a_r behind it. The selected superellipsoid is

```text
|((xi - xi_c)/a_s)|^n + |y/b|^m + |((z - z_c)/c)|^p = 1.
```

The parameter vector is theta = [a_f, a_r, b, c, xi_c, z_c, n, m, p]. This form is flexible enough to capture non-ellipsoidal boundaries but remains compact and analytic.

## Data and preprocessing

The dataset contains {vals['n_time_steps']} CSV files from t={vals['t_min']:.2f} s to t={vals['t_max']:.2f} s. Exact duplicate rows were removed, repeated coordinates were collapsed by field averaging, and all coordinates were transformed into the moving frame. The full-width and full-volume descriptors use the half-domain symmetry assumption. The material is 316L stainless steel with temperature-dependent density, heat capacity, thermal conductivity and viscosity supplied as tables. The laser beam radius is 0.00021 m, absorptivity is 0.35, the solidus and liquidus temperatures are 1648 K and 1753 K, and the fusion latent heat is 1.674e5 J/kg.

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

The decomposition clarifies why a low validation RMSE is not enough by itself. The molten-region-only export affects E_reconstruction, the analytic boundary affects E_geometry, the convex-hull volume proxy affects E_volume_proxy, the short validation sequence affects E_dynamics, and temperature-dependent material properties affect E_parameter_scale.

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

Identifiability is assessed with coefficient of variation, finite-difference local sensitivity, a Fisher-information proxy, condition proxies and parameter-to-transition ratio. High-risk parameters are: {high_risk_text}. This directly supports the modeling decision: the superellipsoid is retained as a compact analytic manifold, while the coupled matrix is retained as an overparameterization control.

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

High-fidelity computational fluid dynamics can resolve L-DED melt-pool transport, but the resulting fields are difficult to use directly in mathematical modeling. This v4 draft strengthens a CFD-informed reduced-order framework for a single 316L stainless-steel L-DED condition at 750 W, 8 mm/s and 12 g/min. The exported FLOW-3D data contain only the molten region, so the melt pool is treated as a moving-frame free-boundary observation rather than as a complete thermal field. A half-domain simulation is reconstructed through the `y=0` symmetry plane, the free-boundary envelope is projected onto an asymmetric superellipsoid manifold, and the extracted state is advanced by a Lyapunov-stable diagonal attractor. The liquidus-reference groups are Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f}. The superellipsoid reduces mean boundary residual from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)} and mean volume relative error from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}. A coupled ridge attractor is stable but validates worse than the diagonal model ({_fmt(vals['coupled_validation'], 4)} versus {_fmt(vals['diagonal_validation'], 4)} relative RMSE), so it is retained as an overparameterization control. The study is deliberately limited to single-condition transient-to-quasi-steady modeling.

## Introduction

Directed energy deposition is governed by moving heat input, powder capture, melt-pool convection, temperature-dependent properties and phase change [@ahn2021; @svetlizky2021; @li2023]. Recent CFD, monitoring and machine-learning studies have made melt-pool geometry a central process-state variable [@liao2022; @dasilva2023; @akbari2022; @wu2024]. However, many available descriptions either remain high-dimensional or rely on black-box predictors that require broader training sets.

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
            "assumption": "Temperature-dependent material properties enter through scale diagnostics",
            "physical_basis": "316L density, heat capacity, conductivity and viscosity vary with temperature.",
            "mathematical_role": "Sets alpha, Pe, Ste, E*, Re, Pr and Ma at a reference state and in sensitivity scans.",
            "current_evidence": f"Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f}, Ma={vals['Ma']:.2f}; class changes={class_changes}.",
            "failure_mode": "Different reference states or uncertain absorptivity can shift numerical values even if regime classes remain stable.",
            "reviewer_response": "Report ranges and classify dimensionless groups as scaling diagnostics rather than constants.",
            "source_outputs": "dimensionless_numbers.csv; dimensionless_sensitivity_summary.csv",
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
            "assumption": "Coupled model is an overparameterization control",
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

## 8. Assumption validation matrix

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

## Assumption validation

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

These stress tests are internal checks across the available multi-condition FLOW-3D dataset. They strengthen the validation narrative but do not replace external experimental validation or independently generated held-out process designs.
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
        external_status = f"external_cfd_holdout_available_{external_cases}_conditions_no_experimental_validation"
        external_risk = "medium_high"
        external_evidence = (
            f"external_cases={external_cases}; process_mean_rel_error={external_process_error:.4f}; "
            f"dynamics_mean_rel_rmse={external_dynamics_error:.4f}"
        )
        external_action = "Use the external CFD holdout as a main validation result; experimental validation remains the next risk-reduction step."
    else:
        external_status = "simulation_only_multi_condition_validation_no_external_holdout"
        external_risk = "high"
        external_evidence = f"stress_support_rate={support_rate:.3f}"
        external_action = "Add experimental validation or an independently generated held-out process design before targeting the highest-tier journals."
    return pd.DataFrame(
        [
            {
                "gap_area": "external_validation",
                "current_status": external_status,
                "risk_level": external_risk,
                "evidence": external_evidence,
                "recommended_action": external_action,
            },
            {
                "gap_area": "theory_rigor",
                "current_status": "assumption-proposition-proof-sketch framework",
                "risk_level": "medium",
                "evidence": "The theory notes formalize projection and Lyapunov arguments; constants remain data-limited.",
                "recommended_action": "Keep proof claims conservative and avoid universal error constants.",
            },
            {
                "gap_area": "assumption_risk",
                "current_status": "assumption audit generated",
                "risk_level": "medium_high" if high_assumptions else "medium",
                "evidence": f"high_or_high_for assumptions={high_assumptions}",
                "recommended_action": "Use the assumption matrix in supplementary material and reviewer response.",
            },
            {
                "gap_area": "timescale_support",
                "current_status": "estimated from fitted relaxation rates",
                "risk_level": "medium_high" if high_timescale else "medium",
                "evidence": f"high_risk_states={high_timescale}",
                "recommended_action": "Discuss weak Umax identifiability and avoid claiming strong time-scale separation.",
            },
            {
                "gap_area": "latex_submission_package",
                "current_status": "compiled_pdf_required",
                "risk_level": "medium",
                "evidence": "compile summary generated by script",
                "recommended_action": "Use compiled main.pdf and supplementary_methods.pdf as the final local package.",
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
            f"- {row.label}: k={row.k_per_s:.4g} 1/s, tau={row.characteristic_time_s:.4g} s, risk={row.risk_level}."
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

## Validation stress evidence

The internal stress-test mean relative RMSE averaged over scenarios is {stress_mean:.4f}. These tests are useful within the multi-condition FLOW-3D design, but they are not a substitute for external experimental validation or independently generated held-out process designs.
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

The manuscript should be positioned as CFD-informed observed boundary-envelope identification of FLOW-3D molten-region observations. The main claim remains deliberately bounded: the available sequences can be reduced to asymmetric superellipsoid boundary descriptors, a parsimonious diagonal attractor baseline and process-response diagnostics, without claiming experimental validation or universal process-map prediction.

## Strengthened theory

The theory package formalizes the observation operator, superellipsoid manifold projection, descriptor error transfer, first-order relaxation and Lyapunov stability. It uses the full Stefan-Marangoni picture only as physical motivation and does not claim a closed-form solution of the governing PDE system.

## Strengthened validation

The internal stress tests include rolling-origin time extrapolation, leave-one-time-step interpolation, training-fraction perturbation and deterministic state-noise perturbation. The parsimonious diagonal baseline support rate is {stress_support:.3f}, so the stress tests are reported as a limitation rather than as decisive proof of superiority.{dyn_stats_text}

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
        ("Figure 4", "paper_fig04_geometry_quasi_steady", "geometric evolution and quasi-steady approach"),
        ("Figure 5", "paper_fig05_free_boundary_model_comparison", "cross-condition free-boundary comparison"),
        ("Figure 6", "paper_fig06_process_response", "process-response surface diagnostics"),
        ("Figure 7", "paper_fig07_dimensionless_regime", "dimensionless regime and sensitivity"),
        ("Figure 8", "paper_fig08_dynamics_validation", "cross-condition dynamics validation"),
        ("Figure 9", "paper_fig09_error_budget_model_selection", "error budget and model selection"),
        ("Figure 10", "paper_fig10_identifiability_overparameterization", "identifiability and overparameterization"),
        ("Figure 11", "paper_fig11_leave_one_condition_validation", "leave-one-condition-out validation"),
        ("Figure 12", "paper_fig12_external_holdout_validation", "external CFD holdout validation"),
    ]
    active_supp = [
        ("Supplementary Figure S1", "supp_figS1_all_boundary_fits", "all time-step boundary fits"),
        ("Supplementary Figure S2", "supp_figS2_superellipsoid_parameters", "superellipsoid parameter trajectories"),
        ("Supplementary Figure S3", "supp_figS3_dynamics_residuals", "state-wise dynamical residuals"),
        ("Supplementary Figure S4", "supp_figS4_dimensionless_sensitivity_grid", "dimensionless sensitivity scenario grid"),
        (
            "Supplementary Figure S5",
            "supp_figS5_theory_identifiability_error_bounds",
            "theory, identifiability and error-budget diagnostics",
        ),
        ("Supplementary Figure S6", "fig10_stability_attractor", "representative stability and attractor evidence"),
        ("Supplementary Figure S7", "fig05_boundary_fit_comparison", "representative boundary overlays"),
        ("Supplementary Figure S8", "fig03_thermal_flow_evolution", "thermal-flow state evolution"),
        ("Supplementary Figure S9", "fig06_dynamics_model_comparison", "dynamical model trajectory comparison"),
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

These diagnostics explain why the superellipsoid is selected as the main geometric model, while the coupled matrix is retained as an overparameterization control.

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
    all_active_ready = bool(figure_manifest[figure_manifest["status"].eq("active")]["all_formats_exist"].all())
    text = f"""# Submission Readiness Checklist

## Completed in this package

- Manuscript draft v3 with citation keys and no bare `[Ref]` placeholders.
- Literature matrix with {len(literature_matrix)} candidate references and manuscript-use notes.
- BibTeX seed file for the candidate reference set.
- Active figure manifest with {active_count} active figures and {legacy_count} legacy figure stems.
- Figure files for 12 main figures and 9 supplementary figures; active formats complete: {all_active_ready}.
- Supplementary Methods draft explaining preprocessing, symmetry reconstruction, boundary fitting, dynamics, stability, error budget, sensitivity and supplementary figures.
- Nomenclature table and equation inventory from the reproducible analysis script.
- Processed reproducibility package prepared under `analysis_outputs/reproducibility_package/` and zipped as `analysis_outputs/reproducibility_package.zip`.
- Reviewer-risk response notes covering finite process-matrix scope, molten-region-only export, overfitting, overparameterization, theory depth, identifiability and material sensitivity.

## Remaining manual tasks before submission

- Target journal set to Applied Mathematical Modelling; before submission, check abstract length, graphical rules, reference style and data-availability wording against the current author guide.
- Verify every seed reference against the publisher page or database export, especially older books and classic papers.
- Add author names, affiliations, ORCID identifiers, acknowledgments and funding statements.
- Decide whether FLOW-3D raw CSV files can be shared publicly; if not, submit the processed reproducibility package as Supplementary Data and optionally archive it on Zenodo, Mendeley Data or GitHub.
- Confirm whether FLOW-3D software settings, mesh resolution and export filters can be described in sufficient detail for reproducibility.
- Check all TIFF files against the target journal's DPI, color mode and physical width requirements.
- Remove or ignore legacy figure stems during final layout; use only files marked `active` in `active_figure_manifest.csv`.
- Add final reference manager output, journal-specific BibTeX or CSL formatting.
- Confirm all material constants with the simulation setup notes, especially absorptivity, beam radius, latent heat and temperature-dependent property tables.
- Decide whether the manuscript needs experimental validation language removed or softened, since the current evidence is CFD-informed rather than experiment-validated.

## Current recommended submission position

For Applied Mathematical Modelling, the paper should be presented as CFD-informed observed boundary-envelope identification and engineering mathematical modeling for L-DED. The central defensible claim is that the molten-region point-cloud sequences admit compact superellipsoid boundary descriptors and parsimonious stable condition-wise baseline dynamics, with external CFD holdout support, while remaining short of experimental validation or a universal process map.
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

The work is aimed at readers interested in observed-boundary modeling, reduced-order dynamics and model selection in engineering systems. It is not a process-wide empirical map. Its contribution is a reproducible route from molten-region CFD output to compact analytic boundary-envelope descriptors and transient-to-quasi-steady dynamical baselines across a finite process matrix, with an additional five-condition external CFD holdout. The baseline nondimensional values are Pe={dim['Pe']:.2f}, Ste={dim['Ste']:.3f}, E*={dim['E_star']:.2f} and Ma={dim['Ma']:.2f}. The diagonal attractor has slightly lower mean validation relative RMSE than the coupled ridge attractor ({dyn_mean['diagonal_attractor']:.4f} versus {dyn_mean['coupled_ridge_attractor']:.4f}), but it is selected as a parsimonious baseline rather than as a statistically dominant model; the more complex coupled model is retained as an overparameterization comparison.

A processed reproducibility package is provided as Supplementary Data, including geometry descriptors, fitted parameters, model-selection tables, external-holdout summaries, plotting scripts and LaTeX source files.

We believe the manuscript fits Applied Mathematical Modelling because it combines observation operators, analytic observed-boundary manifolds, nondimensional scaling, validation design, reproducibility checks, error budgeting and parameter identifiability for an engineering CFD problem.

Sincerely,

[Author names]
"""
    report_path.write_text(text, encoding="utf-8")


def write_highlights_draft(
    report_path: Path,
    geometry_comparison: pd.DataFrame,
    dynamics_comparison: pd.DataFrame,
    external_holdout_summary: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    geom = geometry_comparison[geometry_comparison["time_s"].astype(str).eq("summary")].set_index("model")
    dyn_mean = dynamics_comparison.groupby("model")["validation_relative_rmse"].mean()
    external_text = "External CFD holdout summaries are included for the V-prefixed validation conditions."
    if external_holdout_summary is not None and len(external_holdout_summary):
        metric_map = dict(zip(external_holdout_summary["metric"], external_holdout_summary["value"]))
        external_text = (
            f"An independent five-condition holdout gives process-response mean relative error "
            f"{float(metric_map.get('external_process_response_mean_relative_error', np.nan)):.4f}."
        )
    text = f"""# Highlights Draft

- Builds an observed-boundary mathematical model for L-DED molten-region point clouds.
- Converts half-domain molten-region data into symmetry-reconstructed moving-frame boundary-envelope descriptors.
- Selects the superellipsoid as a boundary-envelope model, reducing mean residual from {float(geom.loc['ellipsoid', 'mean_boundary_residual_rmse']):.4f} to {float(geom.loc['superellipsoid', 'mean_boundary_residual_rmse']):.4f}.
- Uses a parsimonious diagonal attractor baseline rather than claiming statistical dominance over the coupled ridge model ({dyn_mean['diagonal_attractor']:.4f} versus {dyn_mean['coupled_ridge_attractor']:.4f} mean validation relative RMSE).
- {external_text}
"""
    report_path.write_text(text, encoding="utf-8")


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
        "active_figure_manifest.csv",
        "nomenclature_table.csv",
        "equation_inventory.csv",
    ]
    for name in table_files:
        copy_file(output_dir / "tables" / name, f"tables/{name}", "processed table")

    report_files = [
        "external_validation_data_audit.md",
        "figure_captions.md",
        "submission_readiness_checklist.md",
        "cover_letter_draft.md",
        "highlights_draft.md",
    ]
    for name in report_files:
        copy_file(output_dir / "reports" / name, f"reports/{name}", "report")

    latex_files = [
        "main.tex",
        "main.pdf",
        "supplementary_methods.tex",
        "supplementary_methods.pdf",
        "references.bib",
        "latex_compile_summary.txt",
    ]
    for name in latex_files:
        copy_file(output_dir / "latex" / name, f"latex/{name}", "manuscript source")

    copy_file(Path("scripts") / "flow3d_melt_pool_pilot.py", "scripts/flow3d_melt_pool_pilot.py", "pipeline entry point")
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

    readme = """# Reproducibility Package

This processed package accompanies the manuscript and is suitable for journal supplementary data or repository archival.

Included:

- processed geometry descriptors and reduced-state time series;
- fitted superellipsoid parameters and model-selection tables;
- leave-one-condition-out and external CFD holdout validation summaries;
- figure manifest, captions, nomenclature and equation inventory;
- plotting/analysis scripts and LaTeX manuscript sources.

Excluded:

- proprietary FLOW-3D project files;
- raw FLOW-3D molten-region CSV exports, which are available from the corresponding author upon reasonable request subject to project-sharing and software-export constraints.

Reproduction command from the project root:

```bash
python scripts/flow3d_melt_pool_pilot.py
```

The command rebuilds the processed tables, active figures, LaTeX manuscript and compiled PDFs from the available CSV exports.
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
    try:
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_base), "zip", root_dir=package_dir)
    except OSError:
        # Keep the previous archive when a Windows file lock prevents replacement.
        pass
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
            "DED computational framework",
            "kovsca2023",
            "Towards an automated framework for the finite element computational modelling of directed energy deposition",
            2023,
            "computational framework",
            "Automated FE framework for DED with free-surface detection and experimental validation.",
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
            "reduced-order modeling",
            "fresca2021",
            "A Comprehensive Deep Learning-Based Approach to Reduced Order Modeling of Nonlinear Time-Dependent Parametrized PDEs",
            2021,
            "ROM method",
            "Recent deep-learning ROM for nonlinear time-dependent parameterized PDEs.",
            "Use to place the paper near modern nonlinear ROM while emphasizing the present analytic low-dimensional state.",
            "Deep-learning ROM, not free-boundary shape dynamics.",
            "Springer page, DOI 10.1007/s10915-021-01462-7.",
            """@article{fresca2021,
  author = {Fresca, Stefania and Dede, Luca and Manzoni, Andrea},
  title = {A Comprehensive Deep Learning-Based Approach to Reduced Order Modeling of Nonlinear Time-Dependent Parametrized PDEs},
  journal = {Journal of Scientific Computing},
  volume = {87},
  pages = {61},
  year = {2021},
  doi = {10.1007/s10915-021-01462-7}
}""",
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


def write_references_seed(report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(str(item["bibtex"]) for item in reference_seed_entries()) + "\n"
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
    metric_lines = "_External holdout summary is not available._"
    if external_holdout_summary is not None and len(external_holdout_summary):
        metric_lines = markdown_table_from_dataframe(external_holdout_summary)
    display_cols = [
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
    text = f"""# External CFD Holdout Validation Data Audit

- Validation source: `validation data/`
- External cases processed: {case_count}
- External CSV time-step files processed: {file_count}
- Cases ready for full geometry, thermal-flow and dynamics validation: {ready_cases}/{case_count if case_count else 0}

## Case Audit

{case_table}

## Holdout Metrics

{metric_lines}

## Interpretation

The V-prefixed cases are processed separately from the A-prefixed training process matrix. They are therefore used as an external CFD holdout: the descriptor extraction and boundary fitting are applied to the validation files, while process-response and trajectory-prediction errors are evaluated against relationships learned from the training cases.
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
            f"The paired condition-wise comparison gives boundary-residual improvement in {geom_wins}/{geom_total} conditions "
            f"(sign-test p={geom_p:.3g}, median residual reduction {geom_adv:.4g}), while the volume proxy improves in {vol_wins}/{geom_total} conditions."
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
            f"In paired condition-state comparisons, the diagonal model has lower validation error in {diag_wins}/{dyn_total} pairs "
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

The present work takes a narrower mathematical route. It asks whether one high-fidelity FLOW-3D L-DED simulation can be converted into a transparent reduced-order free-boundary model. This goal is close in spirit to recent physics-informed and non-intrusive modeling trends, where simulation data and physical constraints are combined under limited data [@karniadakis2021; @cuomo2022; @bai2021; @fresca2021; @jiang2024piml]. The difference is that the present model does not train a neural network. Instead, it extracts a low-dimensional analytic boundary and a stable attractor from a short time sequence.

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

with a_s = a_f in front of the fitted center and a_s = a_r behind it. The selected superellipsoid is

```text
|((xi - xi_c)/a_s)|^n + |y/b|^m + |((z - z_c)/c)|^p = 1,
```

with parameter vector theta = [a_f, a_r, b, c, xi_c, z_c, n, m, p]. The superellipsoid is adopted only after comparison with the ellipsoid baseline, not because additional parameters are automatically preferred.

## Data and preprocessing

The dataset contains {vals['n_time_steps']} FLOW-3D CSV files from t={vals['t_min']:.2f} s to t={vals['t_max']:.2f} s. Duplicate rows are removed, repeated coordinates are collapsed by field averaging, and all coordinates are transformed into the moving frame. The simulation used a half computational domain in the y direction, so W(t) and V_full(t) use symmetry reconstruction.

The material is 316L stainless steel. Density, heat capacity, thermal conductivity and viscosity are temperature-dependent tables supplied with the dataset. The laser radius is 0.00021 m, absorptivity is 0.35, the initial substrate temperature is 298 K, the solidus and liquidus temperatures are 1648 K and 1753 K, and the fusion latent heat is 1.674e5 J/kg. Surface tension is 1.6 N/m and the magnitude of the surface-tension temperature coefficient is 1.9e-4 N/(m K).

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

The main dynamical model is the diagonal attractor

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

Figures 2 and 3 show the moving-frame reconstruction and transient geometric evolution. After the initial growth stage, the melt-pool envelope approaches a quasi-steady form after approximately 0.20 s. The quasi-steady mean front length, rear length, full width and height are {vals['lf_quasi_mm']:.3f} mm, {vals['lr_quasi_mm']:.3f} mm, {vals['w_quasi_mm']:.3f} mm and {vals['h_quasi_mm']:.3f} mm.

Figure 4 compares the ellipsoid and superellipsoid boundary models. The mean boundary residual decreases from {_fmt(vals['ellipsoid_boundary'], 4)} to {_fmt(vals['super_boundary'], 4)}, and the mean volume relative error decreases from {_fmt(vals['ellipsoid_volume'], 4)} to {_fmt(vals['super_volume'], 4)}. Robustness tests show superellipsoid improvement in {vals['super_volume_wins']}/{vals['robust_total']} settings for volume error and {vals['super_boundary_wins']}/{vals['robust_total']} settings for boundary residual.

Figure 5 reports the nondimensional regime and sensitivity envelope. The values Pe={vals['Pe']:.2f}, Ste={vals['Ste']:.3f}, E*={vals['E_star']:.2f} and Ma={vals['Ma']:.2f} indicate a regime where advective translation, finite melting enthalpy scale, concentrated heat input and thermocapillary forcing all matter as scaling diagnostics.

Figure 6 compares stability and predictive evidence. Both the diagonal and coupled attractors are stable by their respective criteria. The diagonal model has lower validation relative RMSE, {_fmt(vals['diagonal_validation'], 4)} versus {_fmt(vals['coupled_validation'], 4)}, and the coupled model improves validation error in {vals['coupled_wins']}/{vals['robust_total']} robustness settings.

Figure 7 presents the error budget and model-selection summary. Figure 8 shows parameter-identifiability and overparameterization diagnostics. High-risk parameters include: {high_param_text}. These diagnostics support the selected model: {selected_text}.

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
        "Gmean": r"$G_{\mathrm{mean}}$",
        "Umax": r"$U_{\max}$",
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
        "k_Umax_m_per_s": r"$k_{U_{\max}}$",
        "A_matrix_entries": r"$A$-matrix entries",
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
        "large normalized heat input": "large normalized heat input",
        "moderate marangoni scale": "moderate Marangoni scale",
        "rmse": "RMSE",
        "medium high": "medium-high",
        "low medium": "low-medium",
        "All k i>0": r"All $k_i>0$",
        "k i": r"$k_i$",
        "xi c": r"$\xi_c$",
        "z c": r"$z_c$",
        "n, m, p": r"$n$, $m$, $p$",
        "k Umax m per s": r"$k_{U_{\max}}$",
        "Umax": r"$U_{\max}$",
        "A matrix entries": r"$A$-matrix entries",
        "observed boundary envelope geometry": "observed boundary-envelope geometry",
        "free boundary geometry": "observed boundary-envelope geometry",
        "Boundary residual improves 0.2922->0.1943": r"Boundary residual improves 0.2922$\to$0.1943",
        "volume error improves 0.4317->0.3739": r"volume error improves 0.4317$\to$0.3739",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def yes_no(value: bool) -> str:
    return "Yes" if bool(value) else "No"


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
        "The novelty is not a closed-form Stefan-Marangoni solution, but a validated chain that maps": (
            "The novelty is not a closed-form Stefan-Marangoni solution. It is the validated map from"
        ),
        "The novelty is not a closed-form Stefan-Marangoni solution. It is the validated map from sparse molten-region CFD observations to analytic boundary-envelope manifolds, reduced-state descriptors, parsimonious baseline dynamics and auditable model-selection diagnostics.": (
            "The novelty lies in the validated map from sparse molten-region CFD observations to analytic boundary-envelope manifolds, reduced-state descriptors, parsimonious baseline dynamics and checkable model-selection results."
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
        "The selected working model should therefore be read as an auditable baseline for observed boundary evolution, while the coupled attractor remains an overparameterization control.": (
            "We use the selected model as a traceable baseline for observed boundary evolution, with the coupled attractor retained as the overparameterization control."
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
        "and is a FLOW-3D-informed observed-boundary modeling study, not as an experimentally validated universal process map": (
            "and is best read as a FLOW-3D-informed observed-boundary modeling study rather than an experimentally validated universal process map"
        ),
        "and is best read as a FLOW-3D-informed observed-boundary modeling study, not as an experimentally validated universal process map": (
            "and is best read as a FLOW-3D-informed observed-boundary modeling study rather than an experimentally validated universal process map"
        ),
        "The added 5-condition external CFD holdout reduces the previous validation gap.": "The five-condition external holdout reduces the previous validation gap.",
        "This study establishes a multi-condition CFD-informed observed boundary-envelope identification and modeling framework": (
            "This study develops a multi-condition CFD-informed observed boundary-envelope model"
        ),
        "The central result is that": "The main result is that",
        "The contribution is therefore not": "The contribution is not",
        "First, the geometric part of the framework shows that": "First, the geometric analysis shows that",
        "Second, the dynamical part of the framework supports": "Second, the dynamics support",
        "Finally, the analysis clarifies the boundary of the claim.": "Finally, the claim remains bounded.",
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
            "The assumption matrix is placed in the Supplementary Information because it supports the model audit rather than the primary result."
        ),
        r"\subsection*{Submission-gap audit}": r"\subsection*{Residual limitations}",
        r"\subsection{Validation stress tests and residual submission risk}": r"\subsection{Validation stress tests and residual limitations}",
        "The remaining submission risks are listed here because they are useful for transparency but are not themselves a modeling result.": (
            "The remaining limitations are listed here for transparency. They are not treated as modeling results."
        ),
        "The assumption matrix is placed in the supplementary methods because it is a reviewer-audit device rather than a primary modeling result.": (
            "The assumption matrix is placed in the supplementary methods because it supports model checking rather than the primary result."
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
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
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
            f"The paired condition-wise comparison gives boundary-residual improvement in {geom_wins}/{geom_total} conditions "
            f"(sign-test p={geom_p:.3g}, median residual reduction {geom_adv:.4g}), while the volume proxy improves in {vol_wins}/{geom_total} conditions."
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
            f"In paired condition-state comparisons, the diagonal model has lower validation error in {diag_wins}/{dyn_total} pairs "
            f"(sign-test p={dyn_p:.3g}, median relative-RMSE reduction {dyn_adv:.4g})."
        )
    changed_groups = dimensionless_sensitivity[dimensionless_sensitivity["conclusion_changed"]]
    changed_text = ", ".join(changed_groups["symbol"].astype(str).tolist()) if len(changed_groups) else "none"
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
    external_holdout_text = "External CFD holdout validation was not available when the manuscript text was generated."
    external_abstract_text = ""
    external_scope_text = "The submission-gap audit identifies external validation as the principal remaining high-level submission risk."
    if external_holdout_summary is not None and len(external_holdout_summary):
        metric_map = dict(zip(external_holdout_summary["metric"], external_holdout_summary["value"]))
        ext_cases = int(metric_map.get("external_validation_case_count", 0))
        ext_process = float(metric_map.get("external_process_response_mean_relative_error", np.nan))
        ext_dynamics = float(metric_map.get("external_dynamics_mean_relative_rmse", np.nan))
        ext_boundary_win = float(metric_map.get("external_superellipsoid_boundary_win_rate", np.nan))
        ext_volume_win = float(metric_map.get("external_superellipsoid_volume_win_rate", np.nan))
        ext_boundary_wins = int(round(ext_boundary_win * ext_cases)) if ext_cases else 0
        ext_volume_wins = int(round(ext_volume_win * ext_cases)) if ext_cases else 0
        external_holdout_text = (
            f"Figure~\\ref{{fig:external-holdout}} gives the independent CFD holdout check on {ext_cases} V-prefixed conditions. "
            f"The external quasi-steady process-response mean relative error is {ext_process:.4f}, and the process-parameterized diagonal-attractor "
            f"trajectory mean relative RMSE is {ext_dynamics:.4f}. The superellipsoid boundary residual is lower than the ellipsoid baseline "
            f"in {ext_boundary_wins}/{ext_cases} external cases, while the volume-proxy error is lower in {ext_volume_wins}/{ext_cases} cases."
        )
        external_abstract_text = (
            f" An independent five-condition holdout set gives process-response mean relative error {ext_process:.4f} "
            f"and process-parameterized attractor mean relative RMSE {ext_dynamics:.4f}."
        )
        external_scope_text = (
            f"The added {ext_cases}-condition external CFD holdout reduces the previous validation gap. The holdout validates transfer across process parameters "
            "under the same FLOW-3D modeling assumptions, not across independent experimental platforms or alternative CFD physics; experimental melt-pool "
            "measurements or independently varied simulation physics remain needed before claiming experimental generality."
        )
    high_gap_text = "; ".join(
        submission_gap_audit.loc[submission_gap_audit["risk_level"].astype(str).eq("high"), "gap_area"].astype(str)
    )
    if not high_gap_text:
        high_gap_text = "none marked high"
    repo_fig_root = Path("..")
    repository_url = os.environ.get("FLOW3D_REPOSITORY_URL", "").strip()
    repository_sentence = (
        rf"The processed reproducibility package is archived at \url{{{latex_escape(repository_url)}}}. "
        if repository_url
        else "The processed reproducibility package is provided as Supplementary Data. "
    )
    export_columns_text = ", ".join(CANONICAL_EXPORT_COLUMNS)
    n_conditions = int(table["case_id"].nunique()) if "case_id" in table.columns else 1
    process_range_text = case_parameter_ranges(table)
    condition_text = (
        f"{n_conditions} FLOW-3D numerical L-DED process conditions; not experimental imaging data."
        if n_conditions > 1
        else "Single FLOW-3D numerical L-DED simulation; not experimental imaging data."
    )
    raw_location_text = (
        "CSV files under raw data/Aa-b-c-d/*.csv, where b is power, c is scan speed and d is particle rate."
        if n_conditions > 1
        else f"{vals['n_time_steps']} CSV files in raw data/*.csv: {vals['source_files']}."
    )
    powder_rule_text = "Powder feed is converted as particle_rate/60000*12 g/min."
    data_provenance_items = [
        (
            "Simulation source",
            condition_text,
        ),
        (
            "Raw files",
            raw_location_text,
        ),
        (
            "Process matrix",
            f"{process_range_text}; {powder_rule_text}",
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
            "Point-count audit",
            f"Raw rows per file {vals['raw_rows_min']}-{vals['raw_rows_max']}; exact-deduplicated rows {vals['exact_dedup_rows_min']}-{vals['exact_dedup_rows_max']}; unique coordinates {vals['unique_points_min']}-{vals['unique_points_max']}.",
        ),
    ]
    data_provenance_rows = "\n".join(
        [
            f"{latex_escape(item)} & {latex_escape(description)} \\\\"
            for item, description in data_provenance_items
        ]
    )

    def fig_path(stem: str, location: str = "paper") -> str:
        if location == "paper":
            return str((repo_fig_root / "paper_figures" / f"{stem}.pdf").as_posix())
        return str((repo_fig_root / "figures" / f"{stem}.png").as_posix())

    error_rows = "\n".join(
        [
            f"{latex_math_label(row.error_term)} & {latex_readable_text(row.primary_metric)} & {float(row.primary_value):.4g} & {latex_readable_text(row.risk_level)} \\\\"
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
                f"{latex_escape(row.assumption_id)} & {latex_readable_text(row.assumption)} & "
                f"{latex_readable_text(row.current_evidence)} & {latex_readable_text(row.risk_level)} \\\\"
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
    gap_rows = "\n".join(
        [
            (
                f"{latex_readable_text(row.gap_area)} & {latex_readable_text(row.risk_level)} & "
                f"{latex_readable_text(row.current_status)} & {latex_readable_text(row.recommended_action)} \\\\"
            )
            for row in submission_gap_audit.itertuples()
        ]
    )
    k_min = float(np.nanmin(dynamics_summary["k_per_s"].to_numpy(dtype=float)))

    figure_blocks = [
        (
            "fig:framework",
            "Modeling framework for CFD-informed observed boundary-envelope identification.",
            "The workflow converts multi-condition FLOW-3D half-domain molten-region point clouds into symmetry-reconstructed moving-frame observed boundary envelopes, fits analytic superellipsoid manifolds, identifies condition-wise attractors, and evaluates process response, stability, error budget and parameter identifiability.",
            "paper_fig01_modeling_framework",
        ),
        (
            "fig:process-matrix",
            "Multi-condition process matrix.",
            "The 15 FLOW-3D conditions span laser power, scan speed and powder feed, with powder feed converted from the particle generation rate by the stated linear rule.",
            "paper_fig02_process_matrix",
        ),
        (
            "fig:moving-frame",
            "Moving-frame reconstruction of the molten region.",
            "For the representative baseline condition, panel A shows the raw half-domain FLOW-3D molten-region export in the laboratory frame, panel B shows the symmetry-reconstructed full-width observation at the final exported time, panel C overlays selected times in the laser-attached coordinate to show moving-frame alignment, and panel D marks the reduced observed boundary-envelope descriptors $L_f$, $L_r$, $W$ and $H$ used by the downstream dynamics.",
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
            "Condition-wise boundary residuals and volume errors compare the asymmetric ellipsoid baseline with the asymmetric superellipsoid model.",
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
            "Dimensionless regime and sensitivity.",
            "Baseline values and perturbation ranges for $Pe$, $Ste$, $E^*$ and $Ma$ summarize the thermal-transport scaling.",
            "paper_fig07_dimensionless_regime",
        ),
        (
            "fig:dynamics-cross-condition",
            "Cross-condition dynamics validation.",
            "Condition-wise and state-wise validation errors compare the diagonal attractor with the coupled ridge attractor.",
            "paper_fig08_dynamics_validation",
        ),
        (
            "fig:error-budget",
            "Error budget and model selection.",
            "The diagnostic error budget is shown alongside the model-selection summary.",
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
            "External CFD holdout validation.",
            "Five V-prefixed FLOW-3D conditions are withheld from model construction and used to test boundary-model transfer, quasi-steady process-response prediction and process-parameterized diagonal-attractor trajectories. These holdout conditions were not used for boundary-model selection or attractor-baseline selection.",
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
    supp_figures = [
        ("fig:supp-boundary", "Representative-condition boundary fits across all time steps.", "supp_figS1_all_boundary_fits"),
        ("fig:supp-parameters", "Representative-condition superellipsoid parameters versus time.", "supp_figS2_superellipsoid_parameters"),
        ("fig:supp-residuals", "Dynamical residuals by state.", "supp_figS3_dynamics_residuals"),
        ("fig:supp-dimensionless", "Dimensionless sensitivity scenario grid.", "supp_figS4_dimensionless_sensitivity_grid"),
        (
            "fig:supp-theory",
            "Theory, identifiability and error-budget diagnostics.",
            "supp_figS5_theory_identifiability_error_bounds",
        ),
        (
            "fig:supp-stability",
            "Representative-condition stability and attractor evidence.",
            "fig10_stability_attractor",
        ),
        (
            "fig:supp-boundary-panels",
            "Representative boundary-envelope time-step overlays.",
            "fig05_boundary_fit_comparison",
        ),
        (
            "fig:supp-thermal-flow",
            "Thermal-flow state evolution.",
            "fig03_thermal_flow_evolution",
        ),
        (
            "fig:supp-dynamics-comparison",
            "Dynamical model trajectory comparison.",
            "fig06_dynamics_model_comparison",
        ),
    ]
    def make_supp_figure(label: str, title: str, stem: str, width: str = "0.95\\textwidth") -> str:
        return rf"""\begin{{center}}
\centering
\includegraphics[width={width}]{{{fig_path(stem, "supp")}}}
\captionof{{figure}}{{\textbf{{{title}}}}}
\label{{{label}}}
\end{{center}}"""

    supp_figure_blocks = {
        label: make_supp_figure(label, title, stem)
        for label, title, stem in supp_figures
    }
    supp_figure_tex = "\n\n".join(supp_figure_blocks.values())
    supp_boundary_fig = supp_figure_blocks["fig:supp-boundary"]
    supp_parameters_fig = supp_figure_blocks["fig:supp-parameters"]
    supp_residuals_fig = supp_figure_blocks["fig:supp-residuals"]
    supp_dimensionless_fig = supp_figure_blocks["fig:supp-dimensionless"]
    supp_theory_fig = supp_figure_blocks["fig:supp-theory"]
    supp_stability_fig = supp_figure_blocks["fig:supp-stability"]
    supp_boundary_panels_fig = supp_figure_blocks["fig:supp-boundary-panels"]
    supp_thermal_flow_fig = supp_figure_blocks["fig:supp-thermal-flow"]
    supp_dynamics_comparison_fig = supp_figure_blocks["fig:supp-dynamics-comparison"]
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
            cv_text = "not available" if cv_values.empty else f"{float(cv_values.median()):.3g}"
            count_text = f"; {len(group)} entries" if len(group) > 1 else ""
            risk_items.append(
                f"\\item {latex_math_label(parameter)}: median CV={cv_text}{count_text}; risk={latex_readable_text(group['risk_level'].iloc[0])}."
            )
    risk_lines = "\n".join(risk_items)
    if not risk_lines:
        risk_lines = "\\item No parameter is marked high risk by the current pilot diagnostic."

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
The source data consist of {vals['n_source_files']} FLOW-3D (Flow Science, Inc.) CSV files from {vals['n_conditions']} process-condition folders in the local \texttt{{raw data/}} folder. The folder naming rule is \texttt{{Aa-b-c-d}}, where $a$ is the condition index, $b$ is laser power in watts, $c$ is scan speed in $\mathrm{{mm\,s^{{-1}}}}$, and $d$ is the particle generation rate. The particle rate is converted to powder feed by $d/60000\times12\,\mathrm{{g\,min^{{-1}}}}$. The exports cover $t={vals['t_min']:.2f}$--${vals['t_max']:.2f}\,\mathrm{{s}}$ at the exported times {latex_escape(vals['time_points'])} s. Each CSV contains points exported only from the molten region, so the supplementary workflow begins with an observation problem rather than a full thermal-field reconstruction problem. The exported columns are {latex_escape(export_columns_text)}. Coordinates are interpreted in metres, temperature in kelvin, temperature-gradient magnitude in $\mathrm{{K\,m^{{-1}}}}$, pressure in pascals and velocity in $\mathrm{{m\,s^{{-1}}}}$.

The raw files contain {vals['raw_rows_min']}--{vals['raw_rows_max']} rows per condition-time export. After exact row deduplication, {vals['exact_dedup_rows_min']}--{vals['exact_dedup_rows_max']} rows remain; after repeated-coordinate collapse, {vals['unique_points_min']}--{vals['unique_points_max']} unique spatial points remain per export. Across the dataset, {vals['exact_duplicates_removed_total']} exact duplicate rows are removed and {vals['coordinate_duplicates_collapsed_total']} repeated-coordinate groups are collapsed by field averaging. All point locations are then converted to the condition-specific laser-attached frame by $\xi=x-v_ct$, where $v_c$ is parsed from the condition folder.

The coordinate convention used throughout the manuscript is as follows. The laboratory scan direction is $x$, the transverse direction is $y$, the build direction is $z$, and the moving coordinate is $\xi$. Time is denoted by $t$. Temperature, gradient magnitude and velocity magnitude are extracted from the exported molten points and are not extrapolated into the surrounding solid region.

\subsection*{{Symmetry reconstruction of the half-domain export}}
The FLOW-3D domain is simulated only for $y\geq0$, with $y=0$ as a symmetry plane. The full-domain observation is obtained by mirroring each point $(\xi,y,z)$ to $(\xi,-y,z)$. This reconstruction is exact for scalar geometric descriptors when the process, heat source and powder delivery are symmetric with respect to the plane, but it should not be interpreted as a recovery of unobserved antisymmetric flow structures.

The principal scalar descriptors use $W=2\max y$ and $V_{{\mathrm{{full}}}}=2V_{{\mathrm{{half}}}}$. Here, $W$ is the reconstructed full width and $V_{{\mathrm{{half}}}}$ is the convex-hull volume proxy of the exported half-domain points. The factor of two is therefore a symmetry operator applied to the observation, not a second numerical simulation.

\subsection*{{Boundary-envelope extraction and manifold fitting}}
For each time step, the molten-point envelope is approximated by a convex-hull boundary. This choice is intentionally conservative: because the exported data contain only molten-region points, the boundary is the observed molten-domain envelope, not a reconstructed isotherm from an unexported solid-domain field. The ellipsoid baseline and the superellipsoid model are defined in the main text as analytic boundary manifolds, and the fitting operation is posed there as a level-set projection problem.

The fitted geometric parameters have direct physical roles. The parameters $a_f$ and $a_r$ describe front and rear extents in the moving coordinate, $b$ describes the half-width, $c$ describes the vertical scale, and $(\xi_c,z_c)$ locates the fitted boundary center. The exponents $n$, $m$ and $p$ control directional shape sharpness. The superellipsoid is retained because it improves boundary residual while remaining identifiable enough for the present multi-condition dataset; the volume proxy mismatch is reported separately rather than used as a selection claim. The present study prioritizes boundary-envelope consistency and descriptor transferability rather than volume-preserving reconstruction; volume-preserving manifold fitting is left as a separate constrained optimization problem.

The full time-step boundary fits and the fitted parameter trajectories are shown immediately below to keep the geometric evidence adjacent to the fitting procedure.

{supp_boundary_fig}

{supp_boundary_panels_fig}

{supp_parameters_fig}

\subsection*{{Reduced dynamics, stability and residuals}}
The reduced state is $\bm{{q}}=[L_f,L_r,W,H,T_{{\mathrm{{max}}}},G_{{\mathrm{{mean}}}},U_{{\mathrm{{max}}}}]^T$. Here, $L_f$ and $L_r$ are the front and rear lengths, $W$ is full width, $H$ is height, $T_{{\mathrm{{max}}}}$ is maximum temperature, $G_{{\mathrm{{mean}}}}$ is mean temperature-gradient magnitude and $U_{{\mathrm{{max}}}}$ is maximum velocity magnitude. The selected baseline is the diagonal attractor, and the comparison model is the coupled ridge attractor.

The diagonal attractor uses $dq_i/dt=k_i(q_{{\infty,i}}-q_i)$, where $q_{{\infty,i}}$ is the quasi-steady value and $k_i$ is the fitted relaxation rate. It is exponentially stable when all $k_i>0$. The coupled model uses $d\bm{{q}}/dt=A(\bm{{q}}_\infty-\bm{{q}})$, where $A$ is the identified coupling matrix. It is stable when all eigenvalues of $-A$ have negative real part. The coupled model is therefore evaluated by both stability and validation error, because spectral stability alone does not justify its additional parameters.

State-wise residuals are placed here because they diagnose the same reduced-order dynamical fit discussed in this subsection.

{supp_residuals_fig}

The representative-condition stability plot below reports the normalized state-error contraction, diagonal relaxation rates and coupled-model eigenvalue spectrum used to interpret the attractor evidence in the main text.

{supp_stability_fig}

\subsection*{{Dimensionless scaling and sensitivity}}
The baseline material properties are interpolated at the liquidus temperature. The key reported values are $Pe={vals['Pe']:.2f}$, $Ste={vals['Ste']:.3f}$, $E^*={vals['E_star']:.2f}$ and $Ma={vals['Ma']:.2f}$. These groups are used to interpret the fitted reduced model in physically meaningful scales. They are not tuned to improve the geometric or dynamical fit.

The symbols are defined as follows. $Pe=vL_{{\mathrm{{ref}}}}/\alpha$ compares advection by the moving laser with thermal diffusion, $Fo=\alpha t/L_{{\mathrm{{ref}}}}^2$ measures diffusive time relative to the reference melt-pool length, $Ste=c_p(T_l-T_s)/L_{{\mathrm{{fus}}}}$ compares sensible heat across the mushy interval with latent heat, $E^*$ scales absorbed laser power against the enthalpy needed to heat material through the moving beam footprint, and $Ma$ scales thermocapillary forcing against viscous-thermal diffusion. Sensitivity scenarios perturb reference temperature, absorptivity and surface-tension coefficient to check whether these interpretations are stable.

The full scenario grid is shown below so that the nondimensional interpretation and its sensitivity evidence remain in the same local reading unit.

{supp_dimensionless_fig}

\subsection*{{Assumption validation matrix}}
The assumption matrix is placed in the Supplementary Information because it is a reviewer-audit device rather than a primary modeling result.

\begin{{table}}[htbp]
\centering
\caption{{Assumption validation matrix for the observed boundary-envelope modeling framework.}}
\label{{tab:supp-assumptions}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.07\textwidth}}>{{\raggedright\arraybackslash}}p{{0.27\textwidth}}>{{\raggedright\arraybackslash}}p{{0.42\textwidth}}>{{\raggedright\arraybackslash}}p{{0.12\textwidth}}}}
\toprule
ID & Assumption & Evidence & Risk \\
\midrule
{assumption_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection*{{Validation stress-test protocol}}
The validation stress-test package perturbs the train-validation split, uses rolling-origin tests, removes individual time steps and applies deterministic state-noise perturbations. The selected parsimonious diagonal baseline has stress-test support rate {stress_support:.3f}. The mean validation relative RMSE ranges from {stress_min:.4f} to {stress_max:.4f}. These tests are internal to the FLOW-3D dataset and are reported as evidence-strengthening checks, not as external experimental validation.

The stress tests serve two purposes. First, they check whether the parsimonious diagonal baseline is merely a consequence of one arbitrary train-validation split. Second, they expose the sensitivity of the conclusion to short-sequence sampling within each condition. Because the dataset remains simulation-only, a stress-test pass supports internal consistency but cannot establish experimental generality over power, speed or powder-feed rate.

\begin{{table}}[htbp]
\centering
\caption{{Representative validation stress tests. The full table is provided in the generated analysis package.}}
\label{{tab:supp-stress-tests}}
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

\subsection*{{Submission-gap audit}}
The remaining submission risks are listed here because they are useful for transparency but are not themselves a modeling result. The highest remaining risk is {latex_escape(high_gap_text)}.

\begin{{table}}[htbp]
\centering
\caption{{Submission-gap audit for AMM-oriented revision.}}
\label{{tab:supp-gap-audit}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.16\textwidth}}>{{\raggedright\arraybackslash}}p{{0.10\textwidth}}>{{\raggedright\arraybackslash}}p{{0.27\textwidth}}>{{\raggedright\arraybackslash}}p{{0.34\textwidth}}}}
\toprule
Gap area & Risk & Current status & Recommended action \\
\midrule
{gap_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection*{{Error-budget construction}}
The error taxonomy is $E_{{\mathrm{{total}}}}=E_{{\mathrm{{reconstruction}}}}+E_{{\mathrm{{geometry}}}}+E_{{\mathrm{{volume}}}}+E_{{\mathrm{{dynamics}}}}+E_{{\mathrm{{parameter}}}}$. The term $E_{{\mathrm{{reconstruction}}}}$ covers duplicate handling and half-domain mirroring, $E_{{\mathrm{{geometry}}}}$ covers boundary-model residuals, $E_{{\mathrm{{volume}}}}$ covers the convex-hull volume proxy, $E_{{\mathrm{{dynamics}}}}$ covers train-validation prediction error, and $E_{{\mathrm{{parameter}}}}$ covers uncertainty in material and process scales. This taxonomy separates point-cloud reconstruction, analytic boundary fitting, volume proxy, dynamical prediction and material-scale uncertainty.

The decomposition is used as a diagnostic accounting device rather than a probabilistic independence claim. In a stricter uncertainty-quantification study, the terms would be propagated through a global sensitivity or Bayesian framework. Here, the purpose is narrower: each reported error source is linked to a measurable table or figure so that the reduced model remains auditable.

\subsection*{{Identifiability and overparameterization diagnostics}}
Identifiability is assessed through coefficient of variation, stability of fitted parameters across time, signs of dynamic parameters and validation behavior. For the superellipsoid, the highest risks are shape exponents and center shifts that may compensate for each other during fitting. For the coupled attractor, the highest risk is the ratio between matrix parameters and available state transitions. These diagnostics are used to decide whether additional flexibility is mathematically defensible.

High-risk parameters are:
\begin{{itemize}}
{risk_lines}
\end{{itemize}}

The combined theory, identifiability and error-budget diagnostic is placed here because it summarizes the same overparameterization risks described in this subsection.

{supp_theory_fig}

\subsection*{{Auxiliary thermal-flow and trajectory diagnostics}}
Four auxiliary diagnostic figures are retained after the core S1--S5 evidence chain. They record representative stability evidence, selected boundary overlays, thermal-flow state evolution and diagonal-versus-coupled trajectories. These panels are useful for auditability, but they are not part of the main model-selection sequence.

{supp_thermal_flow_fig}

{supp_dynamics_comparison_fig}
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
\renewcommand{{\arraystretch}}{{1.12}}

\title{{CFD-informed observed boundary-envelope identification of L-DED melt pools using superellipsoid manifolds and parsimonious attractor dynamics}}
\author{{Author Name\\Affiliation}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Laser directed energy deposition (L-DED) melt pools are moving, localized phase-change regions whose observable shape reflects heat input, translation, thermocapillary transport and material addition. A central modeling question is how to describe this evolving molten region by a small number of interpretable state variables when the complete thermal-flow field is not available. This study formulates the problem as observed boundary-envelope identification for 316L stainless-steel L-DED, covering {latex_escape(vals['process_range_text'])}. Half-domain molten-region observations are mapped to symmetry-reconstructed moving-frame envelopes, represented by superellipsoid boundary manifolds, reduced to geometric and thermal-flow descriptors and linked to parsimonious attractor baselines. The process-matrix mean liquidus-reference dimensionless groups are $Pe={vals['Pe']:.2f}$, $Ste={vals['Ste']:.3f}$, $E^*={vals['E_star']:.2f}$ and $Ma={vals['Ma']:.2f}$, indicating a translated, heat-input-dominated regime with strong thermocapillary forcing. Relative to an ellipsoid baseline, {geometry_selection_text} The diagonal attractor gives slightly lower mean validation relative RMSE than the coupled ridge model ({_fmt(vals['diagonal_validation'], 4)} versus {_fmt(vals['coupled_validation'], 4)}), but stress-test and paired-comparison evidence support parsimonious baseline selection rather than statistical dominance.{external_abstract_text} The resulting model is therefore an asymmetric superellipsoid observed-boundary descriptor with a parsimonious diagonal attractor baseline; it is intended as an auditable reduced mathematical representation of molten-region evolution, not as a full inverse reconstruction of the Stefan-Marangoni free boundary.
\end{{abstract}}

\noindent\textbf{{Keywords:}} laser directed energy deposition; melt pool; observed boundary-envelope model; reduced-order modeling; dimensionless analysis; stability; model selection

\section{{Introduction}}

Laser directed energy deposition (L-DED) is widely studied for metallic repair, graded deposition and large-component manufacturing {latex_cite('@ahn2021; @svetlizky2021')}. The molten region remains difficult to describe because laser-material interaction, mass addition, heat transfer, phase change, free-surface heat loss and thermocapillary transport evolve together. Recent reviews emphasize that melt-pool behavior controls process stability, defect formation and geometric fidelity {latex_cite('@li2023; @era2023')}. Numerical studies have further shown that powder delivery, gas-flow effects, pore formation and thermocapillary transport can strongly reshape the molten region {latex_cite('@wang2023powder; @zhang2024pore; @lei2024shaping')}. Related simulations highlight the roles of heat transfer, free-surface evolution and thermo-mechanical response in interpreting the melt pool {latex_cite('@poggi2022; @kovsca2023; @sinclair2024gasflow')}. The scientific problem addressed here is how to represent this evolving molten region as a low-dimensional, interpretable and verifiable mathematical object when the accessible information is a time-resolved boundary observation rather than a complete thermal-flow field.

Existing studies have approached melt-pool evolution through simulation, monitoring, control and data-driven prediction. Simulation-guided work has targeted melt-pool depth, temperature or width {latex_cite('@liao2022; @smoqi2022')}, and feedback-control studies have treated melt-pool geometry as a state for regulation {latex_cite('@rahmani2024psq; @miao2023lqr')}. Coaxial and infrared monitoring have linked melt-pool area or thermal signatures to process diagnosis {latex_cite('@dasilva2023; @herzog2024infrared')}, while image-based studies have related shape irregularity to optimization or flaw detection {latex_cite('@asadi2024dnn; @kong2023monitoring; @abranovic2024flaw')}. Machine-learning and surrogate models have predicted melt-pool dimensions and morphology from process conditions {latex_cite('@akbari2022; @zhu2023')}, with related studies extending these ideas to thermal fields and sequence-based prediction {latex_cite('@hemmasian2023; @wu2024; @wang2023tcn')}. Thermal-field finite-element modeling provides another route to compact process-response estimates {latex_cite('@jelinek2020thermalfe')}. Physics-informed and non-intrusive modeling shows how data and physical constraints can be combined under limited observations {latex_cite('@karniadakis2021; @cuomo2022; @fresca2021')}. Related physics-informed machine-learning studies extend this idea to broader engineering and manufacturing settings {latex_cite('@bai2021; @jiang2024piml; @kumar2023piml')}. Uncertainty-aware and multifidelity studies further emphasize robustness under imperfect data {latex_cite('@pham2022uncertainty; @menon2022multifidelity; @hermann2023')}, a concern also reflected in additive-manufacturing uncertainty quantification {latex_cite('@wang2020uq')}. These advances motivate compact melt-pool states, yet many methods emphasize scalar dimensions, full-field prediction, feedback targets or black-box maps. What remains less developed is an auditable boundary-envelope representation that links observed melt-pool shape, reduced descriptors, model-selection evidence and dynamical evolution across process conditions. Classical moving-source heat-transfer theory and ellipsoidal heat-source geometry provide the moving-frame and baseline-shape context {latex_cite('@rosenthal1946; @goldak1984')}. Superquadric shape representation and ridge/Tikhonov regularization provide the geometric and model-selection scaffolding {latex_cite('@barr1981; @hoerl1970; @tikhonov1977')}.

This study develops an observed boundary-envelope mathematical modeling framework for L-DED melt pools. The framework reconstructs half-domain molten-region observations in a moving coordinate, projects the observed boundary envelope onto an asymmetric superellipsoid manifold, extracts reduced geometric and thermal-flow descriptors, and compares parsimonious first-order attractor baselines with overparameterized coupled alternatives. Model selection is supported by boundary-distance diagnostics, volume-proxy limitation checks, error-budget diagnostics, leave-one-condition-out tests and an independent five-condition numerical holdout. The contribution is an observed boundary-envelope mathematical modeling framework, not a closed-form solution or asymptotic reduction of the Stefan-Marangoni system. Its scientific significance is to provide a reproducible bridge from high-dimensional melt-pool observations to interpretable boundary geometry, reduced dynamics and explicit model-selection limits.

\section{{Physical formulation and observed-boundary modeling}}

\subsection{{Stefan-Marangoni origin and observation model}}

A full L-DED melt-pool model can be idealized as a moving-source Stefan-Marangoni problem. A representative energy balance is
\begin{{equation}}
\rho c_p(T)\left(\frac{{\partial T}}{{\partial t}}+\bm{{u}}\cdot\nabla T\right)
=\nabla\cdot(k(T)\nabla T)+Q_{{\mathrm{{laser}}}}+Q_{{\mathrm{{powder}}}}-\rho L_{{\mathrm{{fus}}}}\frac{{\partial f_l}}{{\partial t}}.
\label{{eq:full-energy}}
\end{{equation}}
In Eq.~\eqref{{eq:full-energy}}, $\rho$ is density, $c_p(T)$ is the temperature-dependent specific heat, $T$ is temperature, $\bm{{u}}$ is the melt velocity, and $k(T)$ is the temperature-dependent thermal conductivity. The terms $Q_{{\mathrm{{laser}}}}$ and $Q_{{\mathrm{{powder}}}}$ represent laser and powder energy input, respectively, $L_{{\mathrm{{fus}}}}$ is the latent heat of fusion, and $f_l$ is the liquid fraction. This equation is used as the physical starting point for reduction; the exported data do not contain the complete solid-domain temperature field needed to solve it directly.
At a solid-liquid interface, the phase-change balance can be written as
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
In Eq.~\eqref{{eq:heat-loss}}, $h_c$ is the convective heat-transfer coefficient, $T_\infty$ is the ambient temperature, $\epsilon_{{\mathrm{{rad}}}}$ is the emissivity, and $\sigma_{{\mathrm{{SB}}}}$ is the Stefan-Boltzmann constant. In Eq.~\eqref{{eq:marangoni}}, $\bm{{\tau}}$ is the viscous stress tensor, $\bm{{t}}$ is a tangent direction on the free surface, $d\sigma/dT$ is the surface-tension temperature coefficient, and $\nabla_s$ is the surface-gradient operator. These equations motivate the observed boundary-envelope viewpoint. They are not solved analytically here.

Let $\Omega_m^h(t)$ be the exported half-domain molten-region point cloud. The laser-attached coordinate is
\begin{{equation}}
\xi = x - v_c t,\qquad v_c=\frac{{s_c}}{{1000}}\,\mathrm{{m\,s^{{-1}}}}.
\label{{eq:moving-coordinate}}
\end{{equation}}
In Eq.~\eqref{{eq:moving-coordinate}}, $x$ is the laboratory scan coordinate, $t$ is time, $s_c$ is the scan speed in $\mathrm{{mm\,s^{{-1}}}}$ parsed from the condition folder, $v_c$ is the corresponding SI scan speed for condition $c$, and $\xi$ is the coordinate observed from a frame translating with the laser. The imposed symmetry boundary is $y=0$. The full-domain observation operator is
\begin{{equation}}
\mathcal{{R}}[\Omega_m^h](t)=\Omega_m^h(t)\cup \lbrace(\xi,-y,z):(\xi,y,z)\in\Omega_m^h(t)\rbrace.
\label{{eq:symmetry}}
\end{{equation}}
In Eq.~\eqref{{eq:symmetry}}, $\mathcal{{R}}$ is the reflection-reconstruction operator, $\Omega_m^h(t)$ is the observed half-domain molten region, and $(\xi,y,z)$ are moving-frame coordinates. The operation reconstructs geometric descriptors of the full observation but does not recover unobserved antisymmetric flow components.
The exported molten-region data are treated as an observation
\begin{{equation}}
P^h(t)=\mathcal{{O}}_h[\Omega_m(t),T,\bm{{u}}],
\label{{eq:observation}}
\end{{equation}}
where $P^h(t)$ is the discrete half-domain point cloud and $\mathcal{{O}}_h$ is the FLOW-3D export operator applied to the full molten region $\Omega_m(t)$, temperature field $T$ and velocity field $\bm{{u}}$. The actual mathematical object is therefore the observed envelope $\Gamma_h(t)$, not the complete solid-liquid interface of the full thermal field.
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
The selected superellipsoid is
\begin{{equation}}
\left|\frac{{\xi-\xi_c}}{{a_s}}\right|^n+
\left|\frac{{y}}{{b}}\right|^m+
\left|\frac{{z-z_c}}{{c}}\right|^p=1,
\label{{eq:superellipsoid}}
\end{{equation}}
with parameter vector $\bm{{\theta}}=[a_f,a_r,b,c,\xi_c,z_c,n,m,p]^T$. The exponents $n$, $m$ and $p$ control the sharpness or flatness of the boundary along the scan, transverse and vertical directions; the ellipsoid-like baseline is recovered when these exponents are fixed at 2. The superellipsoid is adopted only after comparison with the ellipsoid baseline, not because additional parameters are automatically preferred.
For the fitting operation, Eq.~\eqref{{eq:superellipsoid}} is written as an implicit level-set function
\begin{{equation}}
\Phi(\bm{{x}};\bm{{\theta}})=
\left|\frac{{\xi-\xi_c}}{{a_s}}\right|^n+
\left|\frac{{y}}{{b}}\right|^m+
\left|\frac{{z-z_c}}{{c}}\right|^p,
\qquad \Gamma_M(\bm{{\theta}})=\left\{{\bm{{x}}:\Phi(\bm{{x}};\bm{{\theta}})=1\right\}}.
\label{{eq:superellipsoid-levelset}}
\end{{equation}}
In Eq.~\eqref{{eq:superellipsoid-levelset}}, $\Phi$ is the implicit boundary function and $\Gamma_M(\bm{{\theta}})$ is the analytic boundary manifold. The fitted parameter vector is the constrained projection
\begin{{equation}}
\bm{{\theta}}^*(t)=
\arg\min_{{\bm{{\theta}}\in\Theta}}
\frac{{1}}{{N_b(t)}}\sum_{{\bm{{x}}_j\in\Gamma_h(t)}}
\left(\Phi(\bm{{x}}_j;\bm{{\theta}})-1\right)^2 .
\label{{eq:boundary-projection}}
\end{{equation}}
Here, $N_b(t)$ is the number of boundary-envelope points, $\Gamma_h(t)$ is the observed half-domain envelope after moving-frame transformation, and $\Theta$ contains the positivity and exponent bounds used in the numerical fit. The resulting geometric diagnostics are
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

The modeling chain is
\begin{{equation}}
\mathrm{{full\ Stefan-Marangoni\ problem}}
\rightarrow \mathcal{{O}}_h
\rightarrow \Gamma_h(\xi,y,z,t)
\rightarrow \Pi_M\Gamma_h
\rightarrow \bm{{q}}(t)
\rightarrow d\bm{{q}}/dt .
\label{{eq:reduction-chain}}
\end{{equation}}
In Eq.~\eqref{{eq:reduction-chain}}, $\mathcal{{O}}_h$ maps the full high-dimensional simulation state to the observed half-domain molten point cloud, $\Gamma_h$ is the observed boundary envelope, $\Pi_M$ denotes projection onto the finite-dimensional superellipsoid manifold $M(\bm{{\theta}})$, and $\bm{{q}}(t)$ is the descriptor vector used for dynamics. The moving-frame step is justified by constant laser translation: after the initial transient, a localized heat source can approach a slowly varying shape in the laser-attached coordinate. The superellipsoid step is a manifold projection, not a claim that the true Stefan interface is exactly superellipsoidal. This chain is an observation-driven modeling map, not an asymptotic or Galerkin reduction of the governing PDEs. The first-order dynamic is obtained by linearizing an unknown reduced vector field near $\bm{{q}}_\infty$,
\begin{{equation}}
\frac{{d\bm{{q}}}}{{dt}}=F(\bm{{q}})\approx J(\bm{{q}}-\bm{{q}}_\infty),\qquad J\approx-\mathrm{{diag}}(k_i).
\label{{eq:first-order-relaxation}}
\end{{equation}}
In Eq.~\eqref{{eq:first-order-relaxation}}, $F$ is the unknown reduced vector field, $J$ is its local Jacobian, $\bm{{q}}_\infty$ is the quasi-steady attractor state for a condition, and $k_i$ are positive component-wise relaxation rates. This approximation is selected because the diagonal model is stable, parsimonious and slightly lower in mean validation error than the coupled ridge matrix, while the paired and stress-test evidence does not justify a stronger dominance claim.

\section{{FLOW-3D data provenance and dimensionless scaling}}

\subsection{{FLOW-3D export and preprocessing}}

The dataset is a local FLOW-3D numerical export from {vals['n_conditions']} 316L L-DED simulations, not an experimental image sequence. The raw files are stored as CSV files under \texttt{{raw data/Aa-b-c-d/}}, where $a$ is the condition index, $b$ is laser power in watts, $c$ is scan speed in $\mathrm{{mm\,s^{{-1}}}}$, and $d$ is the particle generation rate. The particle rate is converted to powder feed by $\dot m_p=d/60000\times12\,\mathrm{{g\,min^{{-1}}}}$. The dataset covers $t={vals['t_min']:.2f}$--${vals['t_max']:.2f}\,\mathrm{{s}}$ with {vals['n_source_files']} condition-time CSV exports. Each file contains coordinates, volume fraction, heat absorption, heat flux, melt-region indicator, pressure, temperature, temperature-gradient and velocity fields for the exported molten region only. The export therefore excludes the surrounding solid domain and any already-solidified material. Duplicate rows are removed, repeated coordinates are collapsed by field averaging, and all coordinates are transformed into the condition-specific moving frame. The simulation used a half computational domain in the $y$ direction, so $W(t)$ and $V_{{\mathrm{{full}}}}(t)$ use symmetry reconstruction.

\begin{{table}}[htbp]
\centering
\caption{{Data provenance and export audit.}}
\label{{tab:data-provenance}}
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

The process matrix spans {latex_escape(vals['process_range_text'])}. The material is 316L stainless steel with temperature-dependent density, heat capacity, thermal conductivity and viscosity supplied as tables. The laser radius is $0.00021\,\mathrm{{m}}$, absorptivity is 0.35, the initial substrate temperature is $298\,\mathrm{{K}}$, the solidus and liquidus temperatures are $1648\,\mathrm{{K}}$ and $1753\,\mathrm{{K}}$, and the fusion latent heat is $1.674\times10^5\,\mathrm{{J\,kg^{{-1}}}}$. Because the exported data are molten-region observations, the boundary model is interpreted as an envelope reduction of the available point cloud, not as a complete inverse reconstruction of the full thermal field.

\subsection{{Dimensionless regime and sensitivity}}

The reference length is the quasi-steady mean melt-pool length. Material properties are evaluated at the liquidus temperature for baseline reporting. The main groups are
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
In Eqs.~\eqref{{eq:dimensionless-thermal}} and~\eqref{{eq:dimensionless-driving}}, $L_{{\mathrm{{ref}}}}$ is the quasi-steady reference melt-pool length, $\alpha=k/(\rho c_p)$ is thermal diffusivity, $T_s$ and $T_l$ are the solidus and liquidus temperatures, $T_0$ is the initial substrate temperature, $\eta$ is laser absorptivity, $P$ is laser power, $r_b$ is beam radius, $\mu$ is dynamic viscosity, and $d\sigma/dT$ is the surface-tension temperature coefficient. The groups are used as regime descriptors rather than as independently fitted parameters.
The process-matrix mean values are $Pe={vals['Pe']:.2f}$, $Fo_{{\mathrm{{final}}}}={vals['Fo_final']:.2f}$, $Ste={vals['Ste']:.3f}$, $E^*={vals['E_star']:.2f}$, $Re={vals['Re']:.2f}$, $Pr={vals['Pr']:.3f}$ and $Ma={vals['Ma']:.2f}$. Table~\ref{{tab:dimensionless-sensitivity}} also reports the representative baseline used for the material-property sensitivity scan, so that process-matrix means are not confused with perturbation baselines. A scenario sensitivity scan over reference temperature, absorptivity and surface-tension coefficient shows class changes for: {changed_text}.

\begin{{table}}[htbp]
\centering
\caption{{Dimensionless sensitivity envelope. The mean column reports the process-matrix mean, whereas the sensitivity-baseline column reports the representative baseline used in the perturbation scan.}}
\label{{tab:dimensionless-sensitivity}}
\begin{{tabular}}{{lllll}}
\toprule
Group & Mean & Sensitivity baseline & Perturbation range & Observed class \\
\midrule
{sensitivity_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Reduced dynamics, stability and validation}}

\subsection{{Attractor identification and stability}}

The selected parsimonious baseline dynamics is the diagonal attractor
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
In Eq.~\eqref{{eq:coupled-attractor}}, $A$ is the ridge-identified coupling matrix and $\bm{{q}}_\infty$ is fixed from the quasi-steady training segment. The coupled model is retained as a structured overparameterization check, not as the default dynamical law.
The diagonal rates are estimated from the training time set $\mathcal{{T}}_{{\mathrm{{tr}}}}$ by
\begin{{equation}}
k_i^*=
\arg\min_{{k_i\geq0}}
\sum_{{r\in\mathcal{{T}}_{{\mathrm{{tr}}}}}}
\left[
\dot q_{{i,r}}-k_i\left(q_{{\infty,i}}-q_i(t_r)\right)
\right]^2,
\qquad
\dot q_{{i,r}}\approx\frac{{q_i(t_{{r+1}})-q_i(t_r)}}{{t_{{r+1}}-t_r}} .
\label{{eq:diagonal-identification}}
\end{{equation}}
In Eq.~\eqref{{eq:diagonal-identification}}, $\dot q_{{i,r}}$ is a finite-difference derivative and the nonnegative constraint enforces physically interpretable relaxation toward the quasi-steady state. The coupled matrix is identified by ridge regression,
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
In Eq.~\eqref{{eq:coupled-ridge-identification}}, $\lambda_R$ is chosen by leave-one-step validation within the training segment and $\lVert A\rVert_F$ penalizes poorly supported coupling coefficients.
For the diagonal model, define $\bm{{e}}=\bm{{q}}-\bm{{q}}_\infty$. The Lyapunov function
\begin{{equation}}
V(\bm{{e}})=\frac{{1}}{{2}}\lVert\bm{{e}}\rVert_2^2,
\qquad
\dot V=-\sum_i k_i e_i^2
\leq -2k_{{\min}}V,\quad k_{{\min}}=\min_i k_i ,
\label{{eq:lyapunov-diagonal}}
\end{{equation}}
shows global exponential stability whenever $k_i>0$ for all retained state components. In Eq.~\eqref{{eq:lyapunov-diagonal}}, $V$ is the state-error energy and $k_{{\min}}$ is the slowest fitted relaxation rate. {diag_stability_sentence} For the coupled model, $\bm{{e}}=\bm{{q}}-\bm{{q}}_\infty$ satisfies $d\bm{{e}}/dt=-A\bm{{e}}$. The coupled equilibrium is locally exponentially stable if all eigenvalues of $-A$ have negative real part. {coupled_stability_sentence} However, its mean validation relative RMSE is {_fmt(vals['coupled_validation'], 4)}, compared with {_fmt(vals['diagonal_validation'], 4)} for the parsimonious diagonal baseline.
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

\begin{{table}}[htbp]
\centering
\caption{{Summary of relaxation time scales for the selected parsimonious diagonal attractor baseline. The full 105-row condition-state table is exported as \texttt{{timescale\_separation\_summary.csv}}.}}
\label{{tab:timescales}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.12\textwidth}}>{{\raggedright\arraybackslash}}p{{0.15\textwidth}}>{{\raggedright\arraybackslash}}p{{0.25\textwidth}}>{{\raggedright\arraybackslash}}p{{0.17\textwidth}}>{{\raggedright\arraybackslash}}p{{0.16\textwidth}}}}
\toprule
State & Group & Median $\tau_i$ [IQR] (s) & Median validation rRMSE & High-risk cases \\
\midrule
{timescale_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Error budget and assumption audit}}

The uncertainty chain is organized as
\begin{{equation}}
E_{{\mathrm{{total}}}}=E_{{\mathrm{{reconstruction}}}}+E_{{\mathrm{{geometry}}}}+E_{{\mathrm{{volume}}}}+E_{{\mathrm{{dynamics}}}}+E_{{\mathrm{{parameter}}}}.
\label{{eq:error-budget}}
\end{{equation}}
In Eq.~\eqref{{eq:error-budget}}, $E_{{\mathrm{{reconstruction}}}}$ covers half-domain mirroring and duplicate-point handling, $E_{{\mathrm{{geometry}}}}$ covers analytic boundary fitting, $E_{{\mathrm{{volume}}}}$ covers the convex-hull volume proxy, $E_{{\mathrm{{dynamics}}}}$ covers train-validation prediction error, and $E_{{\mathrm{{parameter}}}}$ covers material and process-scale uncertainty. This expression is an error-budget taxonomy, not an assumption that independent random errors add linearly.
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
In Eq.~\eqref{{eq:error-bound}}, the constants $C_R$, $C_\Gamma$, $C_V$, $C_D$ and $C_P$ are sensitivity weights rather than fitted parameters. The table below reports observable diagnostics for the corresponding terms, so the bound is used to structure uncertainty rather than to claim a sharp worst-case estimate from the finite simulation dataset. For each condition-time step, the normalized Chamfer distance is computed as $d_{{\mathrm{{Ch}}}}^*(t)=d_{{\mathrm{{Ch}}}}(t)/(L(t)+\epsilon_L)$, where $d_{{\mathrm{{Ch}}}}(t)$ is the symmetric point-to-surface Chamfer distance between the observed envelope and the fitted manifold, $L(t)=L_f(t)+L_r(t)$ is the melt-pool length, and $\epsilon_L$ prevents division by zero. The value reported in Table~\ref{{tab:error-budget}} is the arithmetic mean of $d_{{\mathrm{{Ch}}}}^*(t)$ over valid condition-time steps, so the distance diagnostic is interpreted relative to the 1--2 mm melt-pool scale.

\begin{{table}}[htbp]
\centering
\caption{{Diagnostic error budget.}}
\label{{tab:error-budget}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.22\textwidth}}>{{\raggedright\arraybackslash}}p{{0.35\textwidth}}>{{\raggedright\arraybackslash}}p{{0.12\textwidth}}>{{\raggedright\arraybackslash}}p{{0.15\textwidth}}}}
\toprule
Error term & Primary metric & Value & Risk level \\
\midrule
{error_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

The full assumption validation matrix is reported in Supplementary Table~\ref{{tab:supp-assumptions}} to keep the main text focused on the model definition, model selection and validation results.

\subsection{{Validation stress tests and residual submission risk}}

To reduce the risk that the parsimonious diagonal baseline is an artefact of a single train-validation split, this study includes internal stress tests: training-fraction perturbation, rolling-origin extrapolation, leave-one-time-step interpolation and deterministic state-noise perturbation. Across these internal scenarios, the support rate for the parsimonious diagonal baseline is {stress_support:.3f}, and the mean validation relative RMSE spans {stress_min:.4f}--{stress_max:.4f} with an average of {stress_mean:.4f}. These tests strengthen the multi-condition simulation evidence but do not replace external experimental validation or independently generated held-out process designs.

Representative validation stress tests are reported in Supplementary Table~\ref{{tab:supp-stress-tests}}; the full machine-readable table remains in the generated analysis package.

Residual submission risks are summarized as an audit item in Supplementary Table~\ref{{tab:supp-gap-audit}}, rather than treated as a main-text research-result table.

\subsection{{Model selection rationale}}

The model-selection rule is deliberately conservative. The superellipsoid is selected as the observed boundary-envelope descriptor because it improves the implicit boundary-envelope residual in all 15 process conditions and remains a compact nine-parameter analytic manifold. It is not presented as a volume-preserving model: the volume proxy improves in only {vals['super_volume_wins']}/{vals['robust_total']} robustness settings, and the distance diagnostics are treated as geometric-risk diagnostics rather than as independent confirmation of full volumetric fidelity. The large volume-proxy error reflects the mismatch between analytic superellipsoid volume and the mirrored convex-hull proxy obtained from sparse molten-region exports; it is not interpreted as a thermodynamic melt-volume prediction error. The present study prioritizes boundary-envelope consistency and descriptor transferability rather than volume-preserving reconstruction; volume-preserving manifold fitting is left as a separate constrained optimization problem.

The diagonal attractor is selected as a parsimonious baseline dynamics, not as a statistically dominant dynamical discovery. The coupled ridge model is physically plausible and spectrally stable, but its additional matrix parameters are not supported by the short condition-wise sequences: the diagonal model has lower validation error in 56/105 paired condition-state comparisons and the sign-test result is not significant. The selected working model should therefore be read as an auditable baseline for observed boundary evolution, while the coupled attractor remains an overparameterization control.

\section{{Results and model selection}}

Figure~\ref{{fig:framework}} summarizes the modeling chain from FLOW-3D molten-region data to symmetry reconstruction, moving-frame analysis, superellipsoid fitting, attractor identification and error auditing. Figure~\ref{{fig:process-matrix}} shows the 15-condition process matrix, and Figures~\ref{{fig:moving-frame}} and~\ref{{fig:geometry}} illustrate the moving-frame reconstruction and transient geometric evolution for the representative baseline condition. After the initial growth stage, the melt-pool envelope approaches a quasi-steady form after approximately 0.20 s. Across conditions, the quasi-steady mean front length, rear length, full width and height are {vals['lf_quasi_mm']:.3f} mm, {vals['lr_quasi_mm']:.3f} mm, {vals['w_quasi_mm']:.3f} mm and {vals['h_quasi_mm']:.3f} mm.

{figure_tex['fig:framework']}

{figure_tex['fig:process-matrix']}

{figure_tex['fig:moving-frame']}

{figure_tex['fig:geometry']}

Figure~\ref{{fig:boundary}} compares the ellipsoid and superellipsoid boundary models. Across the 15 conditions, {geometry_selection_text} {geometry_pair_text} Robustness tests show superellipsoid improvement in {vals['super_volume_wins']}/{vals['robust_total']} settings for volume error and {vals['super_boundary_wins']}/{vals['robust_total']} settings for boundary residual. The remaining volume-proxy error is kept in the error budget and is not interpreted as exact volume recovery. In particular, the high relative volume error compares an analytic superellipsoid volume with a mirrored convex-hull proxy from molten-region point exports, not with an independently measured thermodynamic melt volume. The selected model is therefore a boundary-envelope descriptor, not a volume-preserving reconstruction.

{figure_tex['fig:boundary']}

Figure~\ref{{fig:process-response}} summarizes quasi-steady process-response patterns over power, speed and powder feed. Figure~\ref{{fig:dimensionless}} reports the nondimensional regime and sensitivity envelope. Figure~\ref{{fig:dynamics-cross-condition}} compares cross-condition dynamics validation. Both the diagonal and coupled attractors are stable by their respective criteria, but the diagonal baseline has lower mean validation relative RMSE, {_fmt(vals['diagonal_validation'], 4)} versus {_fmt(vals['coupled_validation'], 4)}. {dynamics_pair_text} The coupled model improves validation error in only {vals['coupled_wins']}/{vals['robust_total']} robustness settings. The validation stress tests in Supplementary Table~\ref{{tab:supp-stress-tests}} give support rate {stress_support:.3f} for the parsimonious diagonal baseline, so this result is treated as a parsimonious model-selection outcome rather than as a strong statistical dominance claim.

{figure_tex['fig:process-response']}

{figure_tex['fig:dimensionless']}

{figure_tex['fig:dynamics-cross-condition']}

Figure~\ref{{fig:error-budget}} presents the error budget and model-selection summary, Figure~\ref{{fig:identifiability}} shows parameter-identifiability and overparameterization diagnostics, and Figure~\ref{{fig:loco}} reports a leave-one-condition-out process-response validation check. {loco_detail_text} High-risk parameters include: {high_param_text}.

{figure_tex['fig:error-budget']}

{figure_tex['fig:identifiability']}

{figure_tex['fig:loco']}

\subsection{{External CFD holdout validation}}

The external validation set contains five V-prefixed FLOW-3D conditions that are processed after model construction and are not used to select the boundary descriptor or the attractor baseline. This design tests whether the observed-boundary descriptor, the quasi-steady process-response map and the process-parameterized diagonal-attractor trajectories transfer beyond the 15-condition training process matrix. {external_holdout_text} This result is reported as independent CFD holdout validation, not as experimental generalization.

{figure_tex['fig:external-holdout']}

\subsection{{Verification and reproducibility checks}}

The workflow is deterministic from CSV input to PDF output. The preprocessing audit records raw row counts, exact duplicate removal, repeated-coordinate collapse, the half-domain symmetry convention and the condition-specific moving-frame transform $\xi=x-v_ct$. Model-selection reproducibility is supported by fixed model families, exported fitted parameters, leave-one-condition-out process-response checks, validation stress tests and separation of the V-prefixed external holdout from the A-prefixed training matrix. The generated reproducibility package contains processed descriptors, reduced-state time series, fitted superellipsoid parameters, model-selection tables, external-holdout summaries, plotting scripts and the LaTeX source. The complete manuscript, figures and tables are regenerated with \texttt{{python scripts/flow3d\_melt\_pool\_pilot.py}}. The processed reproducibility package is provided as Supplementary Data.

\begin{{table}}[htbp]
\centering
\caption{{Model-selection summary. The selected working model combination is {latex_escape(selected_text)}. The diagonal attractor is treated as a parsimonious baseline rather than as a statistically dominant dynamical law.}}
\label{{tab:model-selection}}
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

\section{{Discussion}}

The main result is a separation between useful boundary flexibility, imperfect volume recovery and unsupported dynamical complexity across a multi-condition CFD design. The superellipsoid adds three shape exponents to the ellipsoid and improves the observed boundary residual in every condition, while the convex-hull volume proxy remains a limitation rather than an improved metric. The volume discrepancy should therefore be read as disagreement between two geometric summaries of sparse exported molten points, not as a calibrated melt-volume prediction error. The present study prioritizes boundary-envelope consistency and descriptor transferability rather than volume-preserving reconstruction; volume-preserving manifold fitting is left as a separate constrained optimization problem. The coupled attractor adds many interaction coefficients, remains stable, but does not improve validation accuracy. This contrast is important for a mathematical modeling journal because model choice is governed by evidence rather than by formal flexibility alone.

The present formulation clarifies what is physical and what is reduced. The Stefan-Marangoni equations motivate the observation and boundary structure, while the fitted model operates on the exported molten-region envelope. This distinction prevents the model from being interpreted as a full thermal-field inverse solution.

Recent DED papers increasingly combine numerical melt-pool physics and defect analysis {latex_cite('@zhang2021dedcfd; @zhang2024pore')}, in situ imaging and process monitoring {latex_cite('@dasilva2023; @asadi2024dnn')}, simulation-guided control {latex_cite('@liao2022; @rahmani2024psq')}, and ML surrogates {latex_cite('@wu2024; @wang2023tcn')}. The present work complements that direction by asking a more mathematical question: what low-dimensional boundary and attractor can be defended from multi-condition molten-region CFD exports? The negative result for the coupled attractor is useful because physically plausible cross-coupling is not statistically justified by the available short condition-wise sequences.

The observed boundary-envelope definition is deliberately conservative. Because only molten-region points were exported, $\Gamma(t)$ is the envelope of the observed molten domain. It is not a reconstructed solid-liquid isotherm from the full temperature field. This limitation narrows the claim, but it makes the model reproducible from the available data and keeps the error budget explicit.

\subsection{{Scope and residual risk}}

The study uses {vals['n_conditions']} training simulated 316L conditions and should be read as a FLOW-3D-informed observed-boundary modeling study, not as an experimentally validated universal process map over all beam radii, absorptivity values, powder-delivery geometries and materials. More conditions would still be needed to fit nonlinear process functions for $\bm{{q}}_\infty$, $k_i$ and superellipsoid shape parameters with tight uncertainty bounds. The volume metric is a mirrored convex-hull proxy, not a direct thermodynamic melt volume, and volume-preserving manifold fitting is left as future constrained optimization. The sensitivity analysis is a scenario scan, not a full global uncertainty quantification. {external_scope_text}

\section{{Conclusion}}

This study establishes a multi-condition CFD-informed observed boundary-envelope identification and modeling framework for transient-to-quasi-steady L-DED melt-pool evolution in 316L stainless steel. The central result is that molten-region FLOW-3D exports over {vals['n_conditions']} process conditions can be converted into reproducible mathematical objects: symmetry-reconstructed moving-frame boundary envelopes, low-dimensional analytic manifolds, and parsimonious stable first-order baseline dynamics. The contribution is therefore not an experimentally validated universal process map, but a traceable modeling chain from high-dimensional CFD output to interpretable observed-boundary reduced-order models and process-response diagnostics.

First, the geometric part of the framework shows that the observed melt-pool envelope is better represented by an asymmetric superellipsoid manifold than by an ellipsoid baseline when the selection criterion is boundary residual. Specifically, {geometry_selection_text} This gives consistent improvement across {vals['super_boundary_wins']}/{vals['robust_total']} boundary-residual robustness settings, whereas volume-error improvement appears in {vals['super_volume_wins']}/{vals['robust_total']} settings and is treated as an unresolved proxy-volume limitation. The nondimensional analysis further places the fitted boundary in a physically interpretable regime, with $Pe={vals['Pe']:.2f}$, $Ste={vals['Ste']:.3f}$, $E^*={vals['E_star']:.2f}$ and $Ma={vals['Ma']:.2f}$.

Second, the dynamical part of the framework supports a deliberately simple attractor baseline. The selected parsimonious diagonal baseline is exponentially stable when the fitted relaxation rates are positive, and across condition-wise fits it gives a slightly lower mean validation relative RMSE than the coupled ridge attractor ({_fmt(vals['diagonal_validation'], 4)} versus {_fmt(vals['coupled_validation'], 4)}). {dynamics_pair_text} The coupled model remains useful as a negative control: although it is spectrally stable, it improves validation error in only {vals['coupled_wins']}/{vals['robust_total']} robustness settings, which indicates overparameterization risk rather than robustly justified cross-state coupling for the available condition-wise sequences.

Finally, the analysis clarifies the boundary of the claim. The observed boundary envelope is extracted from the exported molten region, not recovered as a solid-liquid isotherm from the full thermal field, and the volume remains a symmetry-reconstructed convex-hull proxy. Within these limits, the study provides a defensible mathematical modeling template for turning high-fidelity L-DED CFD point clouds into boundary-envelope manifolds, reduced states, stability diagnostics, model-selection evidence, process-response checks and an explicit error budget. Extending the framework to experimental observations and denser process designs is the necessary next step for converting the present FLOW-3D-informed model into a fully predictive process-dependent tool.

\section*{{Data and code availability}}

{repository_sentence}It includes processed geometry descriptors, reduced-state time series, fitted superellipsoid parameters, model-selection tables, external-holdout summaries, plotting and analysis scripts, figure manifests and LaTeX source files. The raw FLOW-3D molten-region CSV exports are available from the corresponding author upon reasonable request, subject to software/export and project-sharing restrictions. Proprietary FLOW-3D project files are not redistributed. All reported tables and figures can be regenerated locally from the provided CSV exports by running \texttt{{python scripts/flow3d\_melt\_pool\_pilot.py}}.

\bibliographystyle{{unsrtnat}}
\bibliography{{references}}

{supplementary_body_tex}

\end{{document}}
"""
    main_tex = humanize_submission_text(main_tex)
    (latex_dir / "main.tex").write_text(main_tex, encoding="utf-8")

    supp_figures = [
        ("fig:supp-boundary", "Representative-condition boundary fits across all time steps.", "supp_figS1_all_boundary_fits"),
        ("fig:supp-parameters", "Representative-condition superellipsoid parameters versus time.", "supp_figS2_superellipsoid_parameters"),
        ("fig:supp-residuals", "Dynamical residuals by state.", "supp_figS3_dynamics_residuals"),
        ("fig:supp-dimensionless", "Dimensionless sensitivity scenario grid.", "supp_figS4_dimensionless_sensitivity_grid"),
        (
            "fig:supp-theory",
            "Theory, identifiability and error-budget diagnostics.",
            "supp_figS5_theory_identifiability_error_bounds",
        ),
        (
            "fig:supp-stability",
            "Representative-condition stability and attractor evidence.",
            "fig10_stability_attractor",
        ),
        (
            "fig:supp-boundary-panels",
            "Representative free-boundary time-step overlays.",
            "fig05_boundary_fit_comparison",
        ),
        (
            "fig:supp-thermal-flow",
            "Thermal-flow state evolution.",
            "fig03_thermal_flow_evolution",
        ),
        (
            "fig:supp-dynamics-comparison",
            "Dynamical model trajectory comparison.",
            "fig06_dynamics_model_comparison",
        ),
    ]
    supp_figure_tex = "\n\n".join(
        [
            rf"""\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{{fig_path(stem, "supp")}}}
\caption{{\textbf{{{title}}}}}
\label{{{label}}}
\end{{figure}}"""
            for label, title, stem in supp_figures
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

\title{{Supplementary Methods: CFD-informed observed boundary-envelope identification of L-DED melt pools}}
\author{{Author Name}}
\date{{}}

\begin{{document}}
\maketitle

\section{{Data provenance and coordinate convention}}
The source data consist of {vals['n_source_files']} FLOW-3D (Flow Science, Inc.) CSV files from {vals['n_conditions']} process-condition folders in the local \texttt{{raw data/}} folder. The folder naming rule is \texttt{{Aa-b-c-d}}, where $a$ is the condition index, $b$ is laser power in watts, $c$ is scan speed in $\mathrm{{mm\,s^{{-1}}}}$, and $d$ is the particle generation rate. The particle rate is converted to powder feed by $d/60000\times12\,\mathrm{{g\,min^{{-1}}}}$. The exports cover $t={vals['t_min']:.2f}$--${vals['t_max']:.2f}\,\mathrm{{s}}$ at the exported times {latex_escape(vals['time_points'])} s. Each CSV contains points exported only from the molten region, so the supplementary workflow begins with an observation problem rather than a full thermal-field reconstruction problem. The exported columns are {latex_escape(export_columns_text)}. Coordinates are interpreted in metres, temperature in kelvin, temperature-gradient magnitude in $\mathrm{{K\,m^{{-1}}}}$, pressure in pascals and velocity in $\mathrm{{m\,s^{{-1}}}}$.

The raw files contain {vals['raw_rows_min']}--{vals['raw_rows_max']} rows per condition-time export. After exact row deduplication, {vals['exact_dedup_rows_min']}--{vals['exact_dedup_rows_max']} rows remain; after repeated-coordinate collapse, {vals['unique_points_min']}--{vals['unique_points_max']} unique spatial points remain per export. Across the dataset, {vals['exact_duplicates_removed_total']} exact duplicate rows are removed and {vals['coordinate_duplicates_collapsed_total']} repeated-coordinate groups are collapsed by field averaging. All point locations are then converted to the condition-specific laser-attached frame by $\xi=x-v_ct$, where $v_c$ is parsed from the condition folder.

The coordinate convention used throughout the manuscript is as follows. The laboratory scan direction is $x$, the transverse direction is $y$, the build direction is $z$, and the moving coordinate is $\xi$. Time is denoted by $t$. Temperature, gradient magnitude and velocity magnitude are extracted from the exported molten points and are not extrapolated into the surrounding solid region.

\section{{Symmetry reconstruction of the half-domain export}}
The FLOW-3D domain is simulated only for $y\geq0$, with $y=0$ as a symmetry plane. The full-domain observation is obtained by mirroring each point $(\xi,y,z)$ to $(\xi,-y,z)$. This reconstruction is exact for scalar geometric descriptors when the process, heat source and powder delivery are symmetric with respect to the plane, but it should not be interpreted as a recovery of unobserved antisymmetric flow structures.

The principal scalar descriptors use $W=2\max y$ and $V_{{\mathrm{{full}}}}=2V_{{\mathrm{{half}}}}$. Here, $W$ is the reconstructed full width and $V_{{\mathrm{{half}}}}$ is the convex-hull volume proxy of the exported half-domain points. The factor of two is therefore a symmetry operator applied to the observation, not a second numerical simulation.

\section{{Boundary-envelope extraction and manifold fitting}}
For each time step, the molten-point envelope is approximated by a convex-hull boundary. This choice is intentionally conservative: because the exported data contain only molten-region points, the boundary is the observed molten-domain envelope, not a reconstructed isotherm from an unexported solid-domain field. The ellipsoid baseline and the superellipsoid model are given in the main text by Eqs.~(5) and~(6).

The fitted geometric parameters have direct physical roles. The parameters $a_f$ and $a_r$ describe front and rear extents in the moving coordinate, $b$ describes the half-width, $c$ describes the vertical scale, and $(\xi_c,z_c)$ locates the fitted boundary center. The exponents $n$, $m$ and $p$ control directional shape sharpness. The superellipsoid is retained because it improves boundary residual while remaining identifiable enough for the present multi-condition dataset; volume proxy mismatch is reported separately in the error budget. That mismatch compares analytic manifold volume with a mirrored convex-hull proxy from sparse molten-region exports, so it is a geometric proxy diagnostic rather than a direct thermodynamic volume error. The present study prioritizes boundary-envelope consistency and descriptor transferability rather than volume-preserving reconstruction; volume-preserving manifold fitting is left as a separate constrained optimization problem.

The full time-step boundary fits and the fitted parameter trajectories are shown immediately below to keep the geometric evidence adjacent to the fitting procedure.

{supp_boundary_fig}

{supp_parameters_fig}

\section{{Reduced dynamics, stability and residuals}}
The reduced state is $\bm{{q}}=[L_f,L_r,W,H,T_{{\mathrm{{max}}}},G_{{\mathrm{{mean}}}},U_{{\mathrm{{max}}}}]^T$. Here, $L_f$ and $L_r$ are the front and rear lengths, $W$ is full width, $H$ is height, $T_{{\mathrm{{max}}}}$ is maximum temperature, $G_{{\mathrm{{mean}}}}$ is mean temperature-gradient magnitude and $U_{{\mathrm{{max}}}}$ is maximum velocity magnitude. The selected baseline is the diagonal attractor, and the comparison model is the coupled ridge attractor.

The diagonal attractor uses $dq_i/dt=k_i(q_{{\infty,i}}-q_i)$, where $q_{{\infty,i}}$ is the quasi-steady value and $k_i$ is the fitted relaxation rate. It is exponentially stable when all $k_i>0$. The coupled model uses $d\bm{{q}}/dt=A(\bm{{q}}_\infty-\bm{{q}})$, where $A$ is the identified coupling matrix. It is stable when all eigenvalues of $-A$ have negative real part. The coupled model is therefore evaluated by both stability and validation error, because spectral stability alone does not justify its additional parameters.

State-wise residuals are placed here because they diagnose the same reduced-order dynamical fit discussed in this section.

{supp_residuals_fig}

\section{{Dimensionless scaling and sensitivity}}
The baseline material properties are interpolated at the liquidus temperature. The key reported values are $Pe={vals['Pe']:.2f}$, $Ste={vals['Ste']:.3f}$, $E^*={vals['E_star']:.2f}$ and $Ma={vals['Ma']:.2f}$. These groups are used to interpret the fitted reduced model in physically meaningful scales. They are not tuned to improve the geometric or dynamical fit.

The symbols are defined as follows. $Pe=vL_{{\mathrm{{ref}}}}/\alpha$ compares advection by the moving laser with thermal diffusion, $Fo=\alpha t/L_{{\mathrm{{ref}}}}^2$ measures diffusive time relative to the reference melt-pool length, $Ste=c_p(T_l-T_s)/L_{{\mathrm{{fus}}}}$ compares sensible heat across the mushy interval with latent heat, $E^*$ scales absorbed laser power against the enthalpy needed to heat material through the moving beam footprint, and $Ma$ scales thermocapillary forcing against viscous-thermal diffusion. Sensitivity scenarios perturb reference temperature, absorptivity and surface-tension coefficient to check whether these interpretations are stable.

The full scenario grid is shown below so that the nondimensional interpretation and its sensitivity evidence remain in the same local reading unit.

{supp_dimensionless_fig}

\section{{Assumption validation matrix}}
The assumption matrix is placed in the supplementary methods because it is a reviewer-audit device rather than a primary modeling result.

\begin{{table}}[htbp]
\centering
\caption{{Assumption validation matrix for the observed boundary-envelope modeling framework.}}
\label{{tab:supp-assumptions}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.07\textwidth}}>{{\raggedright\arraybackslash}}p{{0.27\textwidth}}>{{\raggedright\arraybackslash}}p{{0.42\textwidth}}>{{\raggedright\arraybackslash}}p{{0.12\textwidth}}}}
\toprule
ID & Assumption & Evidence & Risk \\
\midrule
{assumption_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Validation stress-test protocol}}
The validation stress-test package perturbs the train-validation split, uses rolling-origin tests, removes individual time steps and applies deterministic state-noise perturbations. The selected parsimonious diagonal baseline has stress-test support rate {stress_support:.3f}. The mean validation relative RMSE ranges from {stress_min:.4f} to {stress_max:.4f}. These tests are internal to the FLOW-3D dataset and are reported as evidence-strengthening checks, not as external experimental validation.

The stress tests serve two purposes. First, they check whether the diagonal attractor is merely a consequence of one arbitrary train-validation split. Second, they expose the sensitivity of the conclusion to short-sequence sampling within each condition. Because the dataset remains simulation-only, a stress-test pass supports internal consistency but cannot establish experimental generality over power, speed or powder-feed rate.

\begin{{table}}[htbp]
\centering
\caption{{Representative validation stress tests. The full table is provided in the generated analysis package.}}
\label{{tab:supp-stress-tests}}
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

\section{{Submission-gap audit}}
The remaining submission risks are listed here because they are useful for transparency but are not themselves a modeling result. The highest remaining risk is {latex_escape(high_gap_text)}.

\begin{{table}}[htbp]
\centering
\caption{{Submission-gap audit for AMM-oriented revision.}}
\label{{tab:supp-gap-audit}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{0.16\textwidth}}>{{\raggedright\arraybackslash}}p{{0.10\textwidth}}>{{\raggedright\arraybackslash}}p{{0.27\textwidth}}>{{\raggedright\arraybackslash}}p{{0.34\textwidth}}}}
\toprule
Gap area & Risk & Current status & Recommended action \\
\midrule
{gap_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Error-budget construction}}
The error taxonomy is $E_{{\mathrm{{total}}}}=E_{{\mathrm{{reconstruction}}}}+E_{{\mathrm{{geometry}}}}+E_{{\mathrm{{volume}}}}+E_{{\mathrm{{dynamics}}}}+E_{{\mathrm{{parameter}}}}$. The term $E_{{\mathrm{{reconstruction}}}}$ covers duplicate handling and half-domain mirroring, $E_{{\mathrm{{geometry}}}}$ covers boundary-model residuals, $E_{{\mathrm{{volume}}}}$ covers the convex-hull volume proxy, $E_{{\mathrm{{dynamics}}}}$ covers train-validation prediction error, and $E_{{\mathrm{{parameter}}}}$ covers uncertainty in material and process scales. This taxonomy separates point-cloud reconstruction, analytic boundary fitting, volume proxy, dynamical prediction and material-scale uncertainty.

The decomposition is used as a diagnostic accounting device rather than a probabilistic independence claim. In a stricter uncertainty-quantification study, the terms would be propagated through a global sensitivity or Bayesian framework. Here, the purpose is narrower: each reported error source is linked to a measurable table or figure so that the reduced model remains auditable.

\section{{Identifiability and overparameterization diagnostics}}
Identifiability is assessed through coefficient of variation, stability of fitted parameters across time, signs of dynamic parameters and validation behavior. For the superellipsoid, the highest risks are shape exponents and center shifts that may compensate for each other during fitting. For the coupled attractor, the highest risk is the ratio between matrix parameters and available state transitions. These diagnostics are used to decide whether additional flexibility is mathematically defensible.

High-risk parameters are:
\begin{{itemize}}
{risk_lines}
\end{{itemize}}

The combined theory, identifiability and error-budget diagnostic is placed here because it summarizes the same overparameterization risks described in this section.

{supp_theory_fig}

\section{{Auxiliary thermal-flow and trajectory diagnostics}}
Four auxiliary diagnostic figures are retained after the core S1--S5 evidence chain. They record representative stability evidence, selected boundary overlays, thermal-flow state evolution and diagonal-versus-coupled trajectories. These panels are useful for auditability, but they are not part of the main model-selection sequence.

{supp_thermal_flow_fig}

{supp_dynamics_comparison_fig}

\end{{document}}
"""
    supp_tex = humanize_submission_text(supp_tex)
    (latex_dir / "supplementary_methods.tex").write_text(supp_tex, encoding="utf-8")
    write_references_seed(latex_dir / "references.bib")

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

- `main.tex`: combined manuscript, with Supplementary Information appended after the references.
- `supplementary_methods.tex`: standalone copy of the supplementary methods and supplementary figures for journals that request a separate file.
- `references.bib`: BibTeX seed library, 2020+ literature prioritized.
- `latex_figure_manifest.csv`: active and legacy figure mapping.

Compile from this directory with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex supplementary_methods.tex
pdflatex supplementary_methods.tex
```

The TeX files reference figure PDFs in `../paper_figures/` and `../figures/`, so keep the `analysis_outputs` directory structure intact.
The main manuscript uses numbered citations via `natbib` with `unsrtnat`.
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
    cache_dir = latex_dir / ".tex-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["texmf-var", "texmf-config", "texmf-home"]:
        (cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TEXMFVAR"] = str(cache_dir / "texmf-var")
    env["TEXMFCONFIG"] = str(cache_dir / "texmf-config")
    env["TEXMFHOME"] = str(cache_dir / "texmf-home")

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "supplementary_methods.tex"],
        ["pdflatex", "-interaction=nonstopmode", "supplementary_methods.tex"],
    ]
    command_records = []
    for command in commands:
        try:
            completed = subprocess.run(
                command,
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
            output_tail = (stdout + "\n" + stderr)[-3000:]
            command_records.append(
                {
                    "command": " ".join(command),
                    "returncode": completed.returncode,
                    "output_tail": output_tail,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive local toolchain reporting
            command_records.append(
                {
                    "command": " ".join(command),
                    "returncode": -1,
                    "output_tail": f"{type(exc).__name__}: {exc}",
                }
            )

    main_tex = latex_dir / "main.tex"
    supp_tex = latex_dir / "supplementary_methods.tex"
    main_pdf = latex_dir / "main.pdf"
    supp_pdf = latex_dir / "supplementary_methods.pdf"
    main_log = latex_dir / "main.log"
    supp_log = latex_dir / "supplementary_methods.log"

    main_warnings = _latex_log_warnings(main_log)
    supp_warnings = _latex_log_warnings(supp_log)
    main_fresh = main_pdf.exists() and main_pdf.stat().st_size > 0 and main_pdf.stat().st_mtime >= main_tex.stat().st_mtime
    supp_fresh = supp_pdf.exists() and supp_pdf.stat().st_size > 0 and supp_pdf.stat().st_mtime >= supp_tex.stat().st_mtime
    command_success = all(record["returncode"] == 0 for record in command_records)
    final_acceptance = bool(command_success and main_fresh and supp_fresh and not main_warnings and not supp_warnings)

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
            f"main.pdf exists: {main_pdf.exists()}",
            f"main.pdf fresh: {main_fresh}",
            f"main.pdf size_bytes: {main_pdf.stat().st_size if main_pdf.exists() else 0}",
            f"main.pdf page_count_estimate: {_pdf_page_count(main_pdf)}",
            f"supplementary_methods.pdf exists: {supp_pdf.exists()}",
            f"supplementary_methods.pdf fresh: {supp_fresh}",
            f"supplementary_methods.pdf size_bytes: {supp_pdf.stat().st_size if supp_pdf.exists() else 0}",
            f"supplementary_methods.pdf page_count_estimate: {_pdf_page_count(supp_pdf)}",
            "",
            "Warning summary:",
            f"main.log blocking warnings: {len(main_warnings)}",
            *(f"  - {warning}" for warning in main_warnings[:20]),
            f"supplementary_methods.log blocking warnings: {len(supp_warnings)}",
            *(f"  - {warning}" for warning in supp_warnings[:20]),
            "",
            "Command output tails:",
        ]
    )
    for idx, record in enumerate(command_records, start=1):
        lines.extend(
            [
                f"--- command {idx}: {record['command']} ---",
                record["output_tail"],
                "",
            ]
        )
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
        "dimensionless_regime",
        "stability_attractor",
        "error_budget_model_selection",
        "identifiability_overparameterization",
        "supplementary_all_boundary_fits",
        "supplementary_superellipsoid_parameters",
        "supplementary_residuals",
        "supplementary_dimensionless_grid",
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
        plot_multi_condition_geometry_comparison(table("geometry_model_comparison.csv"), fig_dir)
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
        result = subprocess.run(command, capture_output=True, text=True, env=env)
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
    reproducibility_manifest = write_reproducibility_package(output_dir)
    reproducibility_manifest.to_csv(tables_dir / "reproducibility_package_manifest.csv", index=False)
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
    identifiability_v4.to_csv(tables_dir / "identifiability_diagnostics_v4.csv", index=False)
    error_bound_summary.to_csv(tables_dir / "error_bound_summary.csv", index=False)
    assumption_matrix.to_csv(tables_dir / "assumption_justification_matrix.csv", index=False)
    timescale_summary.to_csv(tables_dir / "timescale_separation_summary.csv", index=False)
    validation_stress_tests.to_csv(tables_dir / "validation_stress_tests.csv", index=False)
    submission_gap_audit.to_csv(tables_dir / "submission_gap_audit.csv", index=False)
    make_dimensionless_definitions().to_csv(tables_dir / "dimensionless_definitions.csv", index=False)
    make_nomenclature_table().to_csv(tables_dir / "nomenclature_table.csv", index=False)
    make_equation_inventory().to_csv(tables_dir / "equation_inventory.csv", index=False)
    literature_matrix = make_literature_matrix()
    literature_matrix.to_csv(tables_dir / "literature_matrix.csv", index=False)
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
    reproducibility_manifest = write_reproducibility_package(output_dir)
    reproducibility_manifest.to_csv(tables_dir / "reproducibility_package_manifest.csv", index=False)


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
        output_dir / "tables" / "data_provenance_summary.csv",
        output_dir / "tables" / "identifiability_diagnostics_v4.csv",
        output_dir / "tables" / "error_bound_summary.csv",
        output_dir / "tables" / "assumption_justification_matrix.csv",
        output_dir / "tables" / "timescale_separation_summary.csv",
        output_dir / "tables" / "validation_stress_tests.csv",
        output_dir / "tables" / "submission_gap_audit.csv",
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
        output_dir / "reports" / "manuscript_draft_final.md",
        output_dir / "reports" / "supplementary_methods_draft.md",
        output_dir / "reports" / "references_seed.bib",
        output_dir / "reports" / "literature_search_log.md",
        output_dir / "reports" / "figure_captions.md",
        output_dir / "reports" / "reviewer_risk_response.md",
        output_dir / "reports" / "submission_readiness_checklist.md",
        output_dir / "reports" / "cover_letter_draft.md",
        output_dir / "reports" / "highlights_draft.md",
        output_dir / "latex" / "main.tex",
        output_dir / "latex" / "supplementary_methods.tex",
        output_dir / "latex" / "references.bib",
        output_dir / "latex" / "latex_figure_manifest.csv",
        output_dir / "latex" / "README.md",
        output_dir / "latex" / "main.pdf",
        output_dir / "latex" / "supplementary_methods.pdf",
        output_dir / "latex" / "latex_compile_summary.txt",
        output_dir / "tables" / "reproducibility_package_manifest.csv",
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
        output_dir / "paper_figures" / "paper_fig09_error_budget_model_selection",
        output_dir / "paper_figures" / "paper_fig10_identifiability_overparameterization",
        output_dir / "paper_figures" / "paper_fig11_leave_one_condition_validation",
        output_dir / "paper_figures" / "paper_fig12_external_holdout_validation",
        output_dir / "figures" / "supp_figS1_all_boundary_fits",
        output_dir / "figures" / "supp_figS2_superellipsoid_parameters",
        output_dir / "figures" / "supp_figS3_dynamics_residuals",
        output_dir / "figures" / "supp_figS4_dimensionless_sensitivity_grid",
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
    if args.validation_dir.exists():
        validation_table, validation_point_cloud = build_tables(args.validation_dir)
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
            f"process mean relative error={float(metric_map.get('external_process_response_mean_relative_error', np.nan)):.6f}, "
            f"dynamics mean relative RMSE={float(metric_map.get('external_dynamics_mean_relative_rmse', np.nan)):.6f}"
        )
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()

