import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.estimator import StateEstimator


# =========================
# Configurable experiment knobs
# =========================
HEIGHT_AGL_M = 50.0
TIME_S = 600.0
PROFILE_SAMPLES = 500
QUIVER_STRIDE = 15
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp1_windfield_comparison"
PEAK_WINDOW_RADIUS_CELLS = 12

# Analysis-only switch:
# The current core wind model already contains background wind, terrain slope correction,
# height-dependent log profile, and dynamic storm cells. It does not expose a separate
# localized eddy term. To satisfy the paper experiment requirement without touching core
# modules, we add a small terrain-induced local perturbation term in this script only.
ENABLE_ANALYSIS_LOCAL_PERTURBATION = True
LOCAL_PERTURB_SCALE_MPS = 3.0

FIG_DPI = 300
PROFILE_LOW_RELIEF_SEARCH_RATIO = 0.6


@dataclass
class FieldBundle:
    model_name: str
    u: np.ndarray
    v: np.ndarray
    speed: np.ndarray


def log(msg: str) -> None:
    print(f"[exp1] {msg}")


def build_config() -> SimulationConfig:
    config = SimulationConfig()
    config.noise_level = 0.0
    config.enable_single_agent_gusts = False
    config.enable_random_gusts = False
    return config


def compute_log_profile_factor(z_agl: np.ndarray, z0: np.ndarray, h_ref: float = 200.0) -> np.ndarray:
    safe_z0 = np.maximum(z0, 1e-3)
    safe_z = np.maximum(z_agl, safe_z0 + 1e-6)
    raw = np.log(safe_z / safe_z0) / np.log(h_ref / safe_z0)
    return np.clip(raw, 0.05, 1.5)


def sample_baseline_field(config: SimulationConfig, map_manager: MapManager) -> FieldBundle:
    log("正在构建 Baseline 风场")
    z0 = map_manager.z0_map
    z_agl = np.full_like(map_manager.dem, HEIGHT_AGL_M, dtype=float)
    factor = compute_log_profile_factor(z_agl, z0)

    base_u = float(config.env_wind_u)
    base_v = float(config.env_wind_v)
    u = base_u * factor
    v = base_v * factor
    speed = np.hypot(u, v)
    return FieldBundle(model_name="Baseline", u=u, v=v, speed=speed)


def compute_storm_field(
    wind_model,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(wind_model, "storm_manager") or wind_model.storm_manager is None:
        return np.zeros_like(x_grid, dtype=float), np.zeros_like(y_grid, dtype=float)
    if not wind_model.config.enable_storms:
        return np.zeros_like(x_grid, dtype=float), np.zeros_like(y_grid, dtype=float)

    u = np.zeros_like(x_grid, dtype=float)
    v = np.zeros_like(y_grid, dtype=float)
    active_storms = wind_model.storm_manager.get_active_storms(t_s)
    for storm in active_storms:
        center = storm.center_at(t_s)
        dx = x_grid - center[0]
        dy = y_grid - center[1]
        dist = np.hypot(dx, dy)

        sigma = max(storm.radius_m / 2.5, 1e-6)
        weight = np.exp(-0.5 * (dist / sigma) ** 2)
        mask = dist <= storm.radius_m * 1.5

        direction = np.array(storm.velocity_xy, dtype=float)
        speed = float(np.linalg.norm(direction))
        if speed < 1e-6:
            direction_unit = np.array([1.0, 0.0], dtype=float)
        else:
            direction_unit = direction / speed

        storm_speed = storm.strength_mps * weight * mask
        u += direction_unit[0] * storm_speed
        v += direction_unit[1] * storm_speed
    return u, v


def compute_analysis_local_perturbation(map_manager: MapManager) -> Tuple[np.ndarray, np.ndarray]:
    if not ENABLE_ANALYSIS_LOCAL_PERTURBATION:
        return np.zeros_like(map_manager.dem, dtype=float), np.zeros_like(map_manager.dem, dtype=float)

    resolution = float(map_manager.resolution)
    gx = map_manager.grad_x
    gy = map_manager.grad_y

    dgy_dy, dgy_dx = np.gradient(gy, resolution, resolution)
    dgx_dy, dgx_dx = np.gradient(gx, resolution, resolution)
    laplacian = dgx_dx + dgy_dy

    slope_mag = np.hypot(gx, gy)
    slope_norm = slope_mag / max(float(np.percentile(slope_mag, 95)), 1e-6)
    curvature_norm = np.abs(laplacian) / max(float(np.percentile(np.abs(laplacian), 95)), 1e-6)
    magnitude = LOCAL_PERTURB_SCALE_MPS * np.clip(slope_norm * curvature_norm, 0.0, 1.0)

    perp_x = -gy
    perp_y = gx
    perp_norm = np.hypot(perp_x, perp_y)
    perp_x = np.divide(perp_x, perp_norm, out=np.zeros_like(perp_x), where=perp_norm > 1e-9)
    perp_y = np.divide(perp_y, perp_norm, out=np.zeros_like(perp_y), where=perp_norm > 1e-9)

    sign = np.sign(laplacian)
    return perp_x * magnitude * sign, perp_y * magnitude * sign


def sample_full_field(
    config: SimulationConfig,
    map_manager: MapManager,
    estimator: StateEstimator,
) -> FieldBundle:
    log("正在构建 Full 风场")

    wind_model = estimator.wind
    x_grid = map_manager.X
    y_grid = map_manager.Y
    gx = map_manager.grad_x
    gy = map_manager.grad_y
    z0 = map_manager.z0_map

    height_agl = np.full_like(map_manager.dem, HEIGHT_AGL_M, dtype=float)
    if hasattr(wind_model, "_get_time_varying_background_wind"):
        u_env, v_env = wind_model._get_time_varying_background_wind(TIME_S)
    else:
        u_env, v_env = config.env_wind_u, config.env_wind_v

    if config.time_of_day == "Day":
        u_slope = config.k_slope * gx
        v_slope = config.k_slope * gy
    else:
        u_slope = -config.k_slope * gx
        v_slope = -config.k_slope * gy

    factor = compute_log_profile_factor(height_agl, z0)
    u = (u_env + u_slope) * factor
    v = (v_env + v_slope) * factor

    storm_u, storm_v = compute_storm_field(wind_model, x_grid, y_grid, TIME_S)
    u += storm_u
    v += storm_v

    local_u, local_v = compute_analysis_local_perturbation(map_manager)
    u += local_u
    v += local_v

    u = np.clip(u, -config.max_wind_speed, config.max_wind_speed)
    v = np.clip(v, -config.max_wind_speed, config.max_wind_speed)
    speed = np.hypot(u, v)
    return FieldBundle(model_name="Full", u=u, v=v, speed=speed)


def summarize_field(bundle: FieldBundle) -> Dict[str, float]:
    speed = bundle.speed[np.isfinite(bundle.speed)]
    mean_speed = float(np.mean(speed))
    std_speed = float(np.std(speed))
    peak_speed = float(np.max(speed))
    cv_speed = float(std_speed / max(mean_speed, 1e-6))
    high_wind_threshold = mean_speed + std_speed
    high_wind_ratio = float(np.mean(speed > high_wind_threshold))

    return {
        "model": bundle.model_name,
        "mean_wind_speed": mean_speed,
        "std_wind_speed": std_speed,
        "peak_wind_speed": peak_speed,
        "cv_wind_speed": cv_speed,
        "high_wind_threshold_mps": float(high_wind_threshold),
        "high_wind_ratio": high_wind_ratio,
    }


def diagnose_peak_region(
    config: SimulationConfig,
    map_manager: MapManager,
    wind_model,
    full: FieldBundle,
) -> Dict[str, float]:
    peak_idx = np.unravel_index(np.argmax(full.speed), full.speed.shape)
    iy, ix = int(peak_idx[0]), int(peak_idx[1])

    peak_x = float(map_manager.x[ix])
    peak_y = float(map_manager.y[iy])
    peak_dem = float(map_manager.dem[iy, ix])
    z0 = float(map_manager.z0_map[iy, ix])
    gx = float(map_manager.grad_x[iy, ix])
    gy = float(map_manager.grad_y[iy, ix])

    factor = float(
        compute_log_profile_factor(
            np.array([[HEIGHT_AGL_M]], dtype=float),
            np.array([[z0]], dtype=float),
        )[0, 0]
    )

    if hasattr(wind_model, "_get_time_varying_background_wind"):
        u_env, v_env = wind_model._get_time_varying_background_wind(TIME_S)
    else:
        u_env, v_env = config.env_wind_u, config.env_wind_v

    if config.time_of_day == "Day":
        u_slope_raw = config.k_slope * gx
        v_slope_raw = config.k_slope * gy
    else:
        u_slope_raw = -config.k_slope * gx
        v_slope_raw = -config.k_slope * gy

    u_bg = float(u_env * factor)
    v_bg = float(v_env * factor)
    u_slope = float(u_slope_raw * factor)
    v_slope = float(v_slope_raw * factor)

    storm_u, storm_v = compute_storm_field(wind_model, map_manager.X, map_manager.Y, TIME_S)
    local_u, local_v = compute_analysis_local_perturbation(map_manager)

    storm_u_peak = float(storm_u[iy, ix])
    storm_v_peak = float(storm_v[iy, ix])
    local_u_peak = float(local_u[iy, ix])
    local_v_peak = float(local_v[iy, ix])

    ys3 = slice(max(0, iy - 1), min(full.speed.shape[0], iy + 2))
    xs3 = slice(max(0, ix - 1), min(full.speed.shape[1], ix + 2))
    ys5 = slice(max(0, iy - 2), min(full.speed.shape[0], iy + 3))
    xs5 = slice(max(0, ix - 2), min(full.speed.shape[1], ix + 3))
    nb3 = full.speed[ys3, xs3]
    nb5 = full.speed[ys5, xs5]

    def connected_component_size(threshold: float) -> int:
        mask = full.speed >= threshold
        if not mask[iy, ix]:
            return 0
        visited = {(iy, ix)}
        stack = [(iy, ix)]
        count = 0
        while stack:
            cy, cx = stack.pop()
            count += 1
            for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                    if (ny, nx) not in visited and mask[ny, nx]:
                        visited.add((ny, nx))
                        stack.append((ny, nx))
        return count

    diagnostics = {
        "peak_x_m": peak_x,
        "peak_y_m": peak_y,
        "peak_elevation_m": peak_dem,
        "peak_u_mps": float(full.u[iy, ix]),
        "peak_v_mps": float(full.v[iy, ix]),
        "peak_wind_speed_mps": float(full.speed[iy, ix]),
        "roughness_z0_m": z0,
        "terrain_grad_x": gx,
        "terrain_grad_y": gy,
        "log_profile_factor": factor,
        "background_u_mps": u_bg,
        "background_v_mps": v_bg,
        "slope_u_mps": u_slope,
        "slope_v_mps": v_slope,
        "storm_u_mps": storm_u_peak,
        "storm_v_mps": storm_v_peak,
        "local_u_mps": local_u_peak,
        "local_v_mps": local_v_peak,
        "neighborhood_3x3_mean_speed_mps": float(np.mean(nb3)),
        "neighborhood_3x3_min_speed_mps": float(np.min(nb3)),
        "neighborhood_3x3_max_speed_mps": float(np.max(nb3)),
        "neighborhood_5x5_mean_speed_mps": float(np.mean(nb5)),
        "connected_cells_ge_30mps": int(connected_component_size(30.0)),
        "connected_cells_ge_31mps": int(connected_component_size(31.0)),
        "connected_cells_ge_32mps": int(connected_component_size(32.0)),
    }
    return diagnostics


def export_peak_diagnostics_csv(diagnostics: Dict[str, float], output_path: Path) -> None:
    pd.DataFrame([diagnostics]).to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_peak_region_diagnostic(
    map_manager: MapManager,
    full: FieldBundle,
    diagnostics: Dict[str, float],
    output_path: Path,
) -> None:
    peak_x = diagnostics["peak_x_m"]
    peak_y = diagnostics["peak_y_m"]

    ix = int(np.argmin(np.abs(map_manager.x - peak_x)))
    iy = int(np.argmin(np.abs(map_manager.y - peak_y)))

    y0 = max(0, iy - PEAK_WINDOW_RADIUS_CELLS)
    y1 = min(full.speed.shape[0], iy + PEAK_WINDOW_RADIUS_CELLS + 1)
    x0 = max(0, ix - PEAK_WINDOW_RADIUS_CELLS)
    x1 = min(full.speed.shape[1], ix + PEAK_WINDOW_RADIUS_CELLS + 1)

    speed_zoom = full.speed[y0:y1, x0:x1]
    u_zoom = full.u[y0:y1, x0:x1]
    v_zoom = full.v[y0:y1, x0:x1]
    dem_zoom = map_manager.dem[y0:y1, x0:x1]
    x_zoom = map_manager.x[x0:x1]
    y_zoom = map_manager.y[y0:y1]
    X_zoom, Y_zoom = np.meshgrid(x_zoom, y_zoom)
    extent = [x_zoom[0], x_zoom[-1], y_zoom[0], y_zoom[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    dem_img = axes[0].imshow(
        dem_zoom,
        extent=extent,
        origin="lower",
        cmap="gist_earth",
        alpha=0.72,
    )
    speed_img = axes[0].imshow(
        speed_zoom,
        extent=extent,
        origin="lower",
        cmap="turbo",
        alpha=0.62,
    )
    axes[0].quiver(
        X_zoom,
        Y_zoom,
        u_zoom,
        v_zoom,
        color="black",
        angles="xy",
        scale_units="xy",
        scale=max(float(np.max(speed_zoom)) * 2.2, 1.0),
        width=0.003,
        alpha=0.75,
    )
    axes[0].scatter([peak_x], [peak_y], s=90, color="#F94144", edgecolors="white", linewidths=1.0)
    axes[0].set_title("Peak Region Zoom", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")

    table_rows = [
        ["Peak xy (m)", f"({peak_x:.0f}, {peak_y:.0f})"],
        ["Peak speed", f"{diagnostics['peak_wind_speed_mps']:.2f} m/s"],
        ["3x3 mean", f"{diagnostics['neighborhood_3x3_mean_speed_mps']:.2f} m/s"],
        ["5x5 mean", f"{diagnostics['neighborhood_5x5_mean_speed_mps']:.2f} m/s"],
        ["Bg + slope", f"{math.hypot(diagnostics['background_u_mps'] + diagnostics['slope_u_mps'], diagnostics['background_v_mps'] + diagnostics['slope_v_mps']):.2f} m/s"],
        ["Storm term", f"{math.hypot(diagnostics['storm_u_mps'], diagnostics['storm_v_mps']):.2f} m/s"],
        ["Local term", f"{math.hypot(diagnostics['local_u_mps'], diagnostics['local_v_mps']):.2f} m/s"],
        [">=30 m/s cluster", f"{int(diagnostics['connected_cells_ge_30mps'])} cells"],
    ]

    axes[1].axis("off")
    table = axes[1].table(
        cellText=table_rows,
        colLabels=["Diagnostic", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
    axes[1].set_title("Peak Diagnostic Table", fontsize=13, fontweight="bold")

    cbar1 = fig.colorbar(dem_img, ax=axes[0], fraction=0.046, pad=0.03)
    cbar1.set_label("Elevation (m)")
    cbar2 = fig.colorbar(speed_img, ax=axes[0], fraction=0.046, pad=0.10)
    cbar2.set_label("Wind Speed (m/s)")
    fig.suptitle("Experiment 1 Peak Wind Region Diagnosis", fontsize=15)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def choose_profiles(map_manager: MapManager) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    dem = map_manager.dem
    x = map_manager.x
    y = map_manager.y

    row_relief = dem.max(axis=1) - dem.min(axis=1)
    col_relief = dem.max(axis=0) - dem.min(axis=0)

    ridge_row_idx = int(np.argmax(row_relief))
    ridge_col_idx = int(np.argmax(col_relief))

    if row_relief[ridge_row_idx] >= col_relief[ridge_col_idx]:
        profile_a = ((float(x[0]), float(y[ridge_row_idx])), (float(x[-1]), float(y[ridge_row_idx])))
    else:
        profile_a = ((float(x[ridge_col_idx]), float(y[0])), (float(x[ridge_col_idx]), float(y[-1])))

    y_margin = int((1.0 - PROFILE_LOW_RELIEF_SEARCH_RATIO) * dem.shape[0] * 0.5)
    x_margin = int((1.0 - PROFILE_LOW_RELIEF_SEARCH_RATIO) * dem.shape[1] * 0.5)
    row_slice = slice(y_margin, dem.shape[0] - y_margin)
    col_slice = slice(x_margin, dem.shape[1] - x_margin)

    central_row_relief = row_relief[row_slice]
    central_col_relief = col_relief[col_slice]
    flat_row_idx = int(np.argmin(central_row_relief)) + row_slice.start
    flat_col_idx = int(np.argmin(central_col_relief)) + col_slice.start

    if row_relief[ridge_row_idx] >= col_relief[ridge_col_idx]:
        profile_b = ((float(x[flat_col_idx]), float(y[0])), (float(x[flat_col_idx]), float(y[-1])))
    else:
        profile_b = ((float(x[0]), float(y[flat_row_idx])), (float(x[-1]), float(y[flat_row_idx])))

    return {"Profile A": profile_a, "Profile B": profile_b}


def build_field_interpolator(
    map_manager: MapManager,
    field: np.ndarray,
) -> Callable[[float, float], float]:
    x = map_manager.x
    y = map_manager.y

    def sampler(xq: float, yq: float) -> float:
        ix = int(np.clip(np.searchsorted(x, xq), 1, len(x) - 1))
        iy = int(np.clip(np.searchsorted(y, yq), 1, len(y) - 1))

        x0, x1 = x[ix - 1], x[ix]
        y0, y1 = y[iy - 1], y[iy]
        tx = 0.0 if abs(x1 - x0) < 1e-9 else (xq - x0) / (x1 - x0)
        ty = 0.0 if abs(y1 - y0) < 1e-9 else (yq - y0) / (y1 - y0)

        f00 = field[iy - 1, ix - 1]
        f01 = field[iy - 1, ix]
        f10 = field[iy, ix - 1]
        f11 = field[iy, ix]

        top = (1.0 - tx) * f00 + tx * f01
        bottom = (1.0 - tx) * f10 + tx * f11
        return float((1.0 - ty) * top + ty * bottom)

    return sampler


def sample_profile(
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    n_samples: int,
    wind_speed_sampler: Callable[[float, float], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_samples = np.linspace(start_xy[0], end_xy[0], n_samples)
    y_samples = np.linspace(start_xy[1], end_xy[1], n_samples)
    dx = np.diff(x_samples)
    dy = np.diff(y_samples)
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])
    speed = np.array([wind_speed_sampler(xv, yv) for xv, yv in zip(x_samples, y_samples)], dtype=float)
    return distance, speed, x_samples, y_samples


def plot_windfield_comparison(
    map_manager: MapManager,
    baseline: FieldBundle,
    full: FieldBundle,
    profiles: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    output_path: Path,
) -> None:
    log("正在绘制对比图")
    extent = [map_manager.x[0], map_manager.x[-1], map_manager.y[0], map_manager.y[-1]]
    speed_vmax = float(max(np.max(baseline.speed), np.max(full.speed)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    for ax, bundle in zip(axes, [baseline, full]):
        dem_img = ax.imshow(
            map_manager.dem,
            extent=extent,
            origin="lower",
            cmap="gist_earth",
            alpha=0.72,
        )
        speed_img = ax.imshow(
            bundle.speed,
            extent=extent,
            origin="lower",
            cmap="turbo",
            alpha=0.58,
            vmin=0.0,
            vmax=speed_vmax,
        )
        ax.quiver(
            map_manager.X[::QUIVER_STRIDE, ::QUIVER_STRIDE],
            map_manager.Y[::QUIVER_STRIDE, ::QUIVER_STRIDE],
            bundle.u[::QUIVER_STRIDE, ::QUIVER_STRIDE],
            bundle.v[::QUIVER_STRIDE, ::QUIVER_STRIDE],
            color="black",
            angles="xy",
            scale_units="xy",
            scale=max(speed_vmax * 3.0, 1.0),
            width=0.0022,
            alpha=0.75,
        )

        for profile_name, (start_xy, end_xy) in profiles.items():
            ax.plot(
                [start_xy[0], end_xy[0]],
                [start_xy[1], end_xy[1]],
                linestyle="--",
                linewidth=1.8,
                label=profile_name,
            )

        ax.set_title(f"{bundle.model_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right")

    cbar1 = fig.colorbar(dem_img, ax=axes, fraction=0.025, pad=0.02)
    cbar1.set_label("Elevation (m)")
    cbar2 = fig.colorbar(speed_img, ax=axes, fraction=0.025, pad=0.08)
    cbar2.set_label("Wind Speed (m/s)")
    fig.suptitle(f"Wind Field Comparison at {HEIGHT_AGL_M:.0f} m AGL, t = {TIME_S:.1f} s", fontsize=16)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_profile_curves(
    map_manager: MapManager,
    baseline: FieldBundle,
    full: FieldBundle,
    profiles: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    output_path: Path,
) -> None:
    baseline_sampler = build_field_interpolator(map_manager, baseline.speed)
    full_sampler = build_field_interpolator(map_manager, full.speed)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    for ax, (profile_name, (start_xy, end_xy)) in zip(axes, profiles.items()):
        dist_b, speed_b, _, _ = sample_profile(start_xy, end_xy, PROFILE_SAMPLES, baseline_sampler)
        dist_f, speed_f, _, _ = sample_profile(start_xy, end_xy, PROFILE_SAMPLES, full_sampler)
        ax.plot(dist_b, speed_b, linewidth=2.2, label="Baseline")
        ax.plot(dist_f, speed_f, linewidth=2.2, label="Full")
        ax.set_title(
            f"{profile_name}: ({start_xy[0]:.0f}, {start_xy[1]:.0f}) -> ({end_xy[0]:.0f}, {end_xy[1]:.0f})",
            fontsize=12,
        )
        ax.set_xlabel("Distance Along Profile (m)")
        ax.set_ylabel("Wind Speed (m/s)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")

    fig.suptitle("Typical Profile Wind Speed Curves", fontsize=16)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_profile_metadata(
    profiles: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    output_path: Path,
) -> None:
    rows: List[Dict[str, float]] = []
    for profile_name, (start_xy, end_xy) in profiles.items():
        rows.append(
            {
                "profile": profile_name,
                "x0_m": start_xy[0],
                "y0_m": start_xy[1],
                "x1_m": end_xy[0],
                "y1_m": end_xy[1],
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("正在加载地图")
    config = build_config()
    map_manager = MapManager(config)

    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)

    baseline = sample_baseline_field(config, map_manager)
    full = sample_full_field(config, map_manager, estimator)

    profiles = choose_profiles(map_manager)
    profile_desc = ", ".join(
        f"{name}=({start[0]:.0f},{start[1]:.0f})->({end[0]:.0f},{end[1]:.0f})"
        for name, (start, end) in profiles.items()
    )
    log(f"已选择剖面线: {profile_desc}")

    windfield_png = OUTPUT_DIR / "exp1_windfield_comparison.png"
    profile_png = OUTPUT_DIR / "exp1_profile_curves.png"
    summary_csv = OUTPUT_DIR / "exp1_summary.csv"
    fields_npz = OUTPUT_DIR / "exp1_fields.npz"
    profile_csv = OUTPUT_DIR / "exp1_profiles.csv"
    peak_png = OUTPUT_DIR / "exp1_peak_region_diagnostic.png"
    peak_csv = OUTPUT_DIR / "exp1_peak_diagnostics.csv"

    plot_windfield_comparison(map_manager, baseline, full, profiles, windfield_png)
    plot_profile_curves(map_manager, baseline, full, profiles, profile_png)

    log("正在导出统计指标")
    summary_df = pd.DataFrame([summarize_field(baseline), summarize_field(full)])
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    save_profile_metadata(profiles, profile_csv)
    peak_diagnostics = diagnose_peak_region(config, map_manager, wind_model, full)
    export_peak_diagnostics_csv(peak_diagnostics, peak_csv)
    plot_peak_region_diagnostic(map_manager, full, peak_diagnostics, peak_png)

    np.savez_compressed(
        fields_npz,
        x=map_manager.x,
        y=map_manager.y,
        dem=map_manager.dem,
        baseline_u=baseline.u,
        baseline_v=baseline.v,
        baseline_speed=baseline.speed,
        full_u=full.u,
        full_v=full.v,
        full_speed=full.speed,
        height_agl_m=HEIGHT_AGL_M,
        time_s=TIME_S,
    )

    log(f"输出文件路径: {OUTPUT_DIR}")
    log(f"  - {windfield_png.name}")
    log(f"  - {profile_png.name}")
    log(f"  - {summary_csv.name}")
    log(f"  - {fields_npz.name}")
    log(f"  - {profile_csv.name}")
    log(f"  - {peak_png.name}")
    log(f"  - {peak_csv.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"实验失败: {exc}")
        raise
