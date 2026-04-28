import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import SimulationConfig
from core.physics import PhysicsEngine
from core.estimator import StateEstimator
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory


# =========================
# Configurable experiment knobs
# =========================
HEIGHT_AGL_M = 50.0
TIME_S = 600.0
ROUTE_SAMPLES = 260
ROUTE_EDGE_MARGIN_M = 700.0
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp4_route_load_analysis"

# Analysis-only local perturbation used in exp1.
LOCAL_PERTURB_SCALE_MPS = 3.0

FIG_DPI = 300


@dataclass
class RouteSpec:
    name: str
    control_points_xy: List[Tuple[float, float]]
    description: str


def log(msg: str) -> None:
    print(f"[exp4] {msg}")


def build_environment() -> Tuple[SimulationConfig, MapManager, PhysicsEngine]:
    log("正在加载环境")
    config = SimulationConfig()
    config.noise_level = 0.0
    config.enable_random_gusts = False
    config.enable_single_agent_gusts = False
    map_manager = MapManager(config)
    physics = PhysicsEngine(config)
    return config, map_manager, physics


def compute_log_profile_factor(z_agl: float, z0: float, h_ref: float = 200.0) -> float:
    safe_z0 = max(z0, 1e-3)
    safe_z = max(z_agl, safe_z0 + 1e-6)
    raw = math.log(safe_z / safe_z0) / math.log(h_ref / safe_z0)
    return float(np.clip(raw, 0.05, 1.5))


def compute_analysis_local_perturbation(map_manager: MapManager) -> Tuple[np.ndarray, np.ndarray]:
    if LOCAL_PERTURB_SCALE_MPS <= 0.0:
        zeros = np.zeros_like(map_manager.dem, dtype=float)
        return zeros, zeros

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


def build_model_samplers(
    config: SimulationConfig,
    map_manager: MapManager,
) -> Dict[str, callable]:
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)

    local_u, local_v = compute_analysis_local_perturbation(map_manager)
    interp_local_u = RegularGridInterpolator(
        (map_manager.y, map_manager.x), local_u, bounds_error=False, fill_value=0.0
    )
    interp_local_v = RegularGridInterpolator(
        (map_manager.y, map_manager.x), local_v, bounds_error=False, fill_value=0.0
    )

    def baseline_sampler(x: float, y: float, z_abs: float, t_s: float) -> np.ndarray:
        ground_alt = map_manager.get_altitude(x, y)
        z_agl = max(z_abs - ground_alt, 0.1)
        z0 = map_manager.get_roughness(x, y)
        factor = compute_log_profile_factor(z_agl, z0)
        return np.array([config.env_wind_u * factor, config.env_wind_v * factor], dtype=float)

    def full_sampler(x: float, y: float, z_abs: float, t_s: float) -> np.ndarray:
        wind_xy = estimator.get_wind(x, y, z=z_abs, t_s=t_s)
        wind_xy = wind_xy + np.array(
            [float(interp_local_u((y, x))), float(interp_local_v((y, x)))],
            dtype=float,
        )
        return wind_xy

    return {"Baseline": baseline_sampler, "Full": full_sampler}


def build_fixed_routes(map_manager: MapManager) -> Dict[str, RouteSpec]:
    log("正在构建固定航迹")
    dem = map_manager.dem
    x = map_manager.x
    y = map_manager.y
    min_x, max_x, min_y, max_y = map_manager.get_bounds()

    row_relief = dem.max(axis=1) - dem.min(axis=1)
    col_relief = dem.max(axis=0) - dem.min(axis=0)
    ridge_row_idx = int(np.argmax(row_relief))
    ridge_col_idx = int(np.argmax(col_relief))

    y_margin = int(0.2 * dem.shape[0])
    x_margin = int(0.2 * dem.shape[1])
    row_slice = slice(y_margin, dem.shape[0] - y_margin)
    col_slice = slice(x_margin, dem.shape[1] - x_margin)
    flat_row_idx = int(np.argmin(row_relief[row_slice])) + row_slice.start
    flat_col_idx = int(np.argmin(col_relief[col_slice])) + col_slice.start

    if row_relief[ridge_row_idx] >= col_relief[ridge_col_idx]:
        y_ridge = float(y[ridge_row_idx])
        y_valley = float(y[flat_row_idx])
        y_mid = float(0.5 * (y_ridge + y_valley))

        routes = {
            "Route A": RouteSpec(
                name="Route A",
                control_points_xy=[
                    (min_x + ROUTE_EDGE_MARGIN_M, y_ridge),
                    (max_x - ROUTE_EDGE_MARGIN_M, y_ridge),
                ],
                description="Cross-ridge route across the highest-relief horizontal band.",
            ),
            "Route B": RouteSpec(
                name="Route B",
                control_points_xy=[
                    (min_x + ROUTE_EDGE_MARGIN_M, y_valley),
                    (max_x - ROUTE_EDGE_MARGIN_M, y_valley),
                ],
                description="Valley / gentle route along the lowest-relief horizontal band.",
            ),
            "Route C": RouteSpec(
                name="Route C",
                control_points_xy=[
                    (min_x + ROUTE_EDGE_MARGIN_M, y_mid),
                    (max_x - ROUTE_EDGE_MARGIN_M, y_mid),
                ],
                description="Intermediate route between ridge-crossing and valley-following bands.",
            ),
        }
    else:
        x_ridge = float(x[ridge_col_idx])
        x_valley = float(x[flat_col_idx])
        x_mid = float(0.5 * (x_ridge + x_valley))

        routes = {
            "Route A": RouteSpec(
                name="Route A",
                control_points_xy=[
                    (x_ridge, min_y + ROUTE_EDGE_MARGIN_M),
                    (x_ridge, max_y - ROUTE_EDGE_MARGIN_M),
                ],
                description="Cross-ridge route across the highest-relief vertical band.",
            ),
            "Route B": RouteSpec(
                name="Route B",
                control_points_xy=[
                    (x_valley, min_y + ROUTE_EDGE_MARGIN_M),
                    (x_valley, max_y - ROUTE_EDGE_MARGIN_M),
                ],
                description="Valley / gentle route along the lowest-relief vertical band.",
            ),
            "Route C": RouteSpec(
                name="Route C",
                control_points_xy=[
                    (x_mid, min_y + ROUTE_EDGE_MARGIN_M),
                    (x_mid, max_y - ROUTE_EDGE_MARGIN_M),
                ],
                description="Intermediate route between ridge-crossing and valley-following bands.",
            ),
        }

    return routes


def sample_route_points(route: RouteSpec, n_samples: int) -> np.ndarray:
    points = np.array(route.control_points_xy, dtype=float)
    seg_vecs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)
    total_length = float(np.sum(seg_lengths))

    if total_length <= 1e-9:
        raise ValueError(f"Route {route.name} has zero length.")

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    target_dist = np.linspace(0.0, total_length, n_samples)
    sampled = np.zeros((n_samples, 2), dtype=float)

    seg_idx = 0
    for i, dist in enumerate(target_dist):
        while seg_idx < len(seg_lengths) - 1 and dist > cumulative[seg_idx + 1]:
            seg_idx += 1
        seg_start = points[seg_idx]
        seg_end = points[seg_idx + 1]
        seg_length = seg_lengths[seg_idx]
        local_dist = dist - cumulative[seg_idx]
        ratio = 0.0 if seg_length <= 1e-9 else local_dist / seg_length
        sampled[i] = seg_start + ratio * (seg_end - seg_start)

    return sampled


def compute_ground_velocity_along_route(route_xyz: np.ndarray, cruise_speed_mps: float) -> Tuple[np.ndarray, np.ndarray]:
    p0 = route_xyz[:-1]
    p1 = route_xyz[1:]
    seg_vec = p1 - p0
    seg_len = np.linalg.norm(seg_vec, axis=1)
    direction = np.divide(seg_vec, seg_len[:, None], out=np.zeros_like(seg_vec), where=seg_len[:, None] > 1e-9)
    ground_velocity = direction * cruise_speed_mps
    return ground_velocity, seg_len


def evaluate_route_under_model(
    route: RouteSpec,
    route_xy: np.ndarray,
    map_manager: MapManager,
    physics: PhysicsEngine,
    wind_sampler,
    model_name: str,
    config: SimulationConfig,
) -> Dict[str, float]:
    route_z = np.array([map_manager.get_altitude(x, y) + HEIGHT_AGL_M for x, y in route_xy], dtype=float)
    route_xyz = np.column_stack([route_xy, route_z])

    ground_velocity, seg_len = compute_ground_velocity_along_route(route_xyz, config.cruise_speed_mps)
    mid_xyz = 0.5 * (route_xyz[:-1] + route_xyz[1:])

    rel_airspeed = []
    power_values = []
    energy_values = []
    wind_speed_values = []

    for idx in range(len(seg_len)):
        wind_xy = wind_sampler(float(mid_xyz[idx, 0]), float(mid_xyz[idx, 1]), float(mid_xyz[idx, 2]), TIME_S)
        wind_xyz = np.array([wind_xy[0], wind_xy[1], 0.0], dtype=float)
        seg_power = physics.estimate_power_from_vectors(ground_velocity[idx], wind_xyz)
        seg_time = seg_len[idx] / max(config.cruise_speed_mps, 1e-9)
        seg_energy = seg_power * seg_time
        v_rel = ground_velocity[idx] - wind_xyz

        rel_airspeed.append(float(np.linalg.norm(v_rel)))
        power_values.append(float(seg_power))
        energy_values.append(float(seg_energy))
        wind_speed_values.append(float(np.linalg.norm(wind_xy)))

    seg_weights = np.maximum(seg_len, 1e-9)
    total_energy = float(np.sum(energy_values))
    route_length = float(np.sum(seg_len))
    energy_per_km = (total_energy / 1000.0) / max(route_length / 1000.0, 1e-9)

    return {
        "route_name": route.name,
        "model": model_name,
        "route_length_m": route_length,
        "avg_relative_airspeed": float(np.average(rel_airspeed, weights=seg_weights)),
        "peak_relative_airspeed": float(np.max(rel_airspeed)),
        "mean_power": float(np.average(power_values, weights=seg_weights)),
        "peak_power": float(np.max(power_values)),
        "total_energy_j": total_energy,
        "energy_per_km": energy_per_km,
        "mean_wind_speed_along_route": float(np.average(wind_speed_values, weights=seg_weights)),
    }


def export_route_summary_csv(summary_df: pd.DataFrame, output_path: Path) -> None:
    log("正在导出 CSV")
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def export_route_definitions_csv(routes: Dict[str, RouteSpec], output_path: Path) -> None:
    rows = []
    for route in routes.values():
        for idx, (xv, yv) in enumerate(route.control_points_xy):
            rows.append(
                {
                    "route_name": route.name,
                    "waypoint_index": idx,
                    "x_m": float(xv),
                    "y_m": float(yv),
                    "description": route.description,
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_route_load_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    log("正在绘制图")
    route_order = ["Route A", "Route B", "Route C"]
    model_order = ["Baseline", "Full"]
    x = np.arange(len(route_order))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    metrics = [
        ("energy_per_km", "Energy per km (kJ/km)", axes[0]),
        ("peak_power", "Peak Power (W)", axes[1]),
    ]
    colors = {"Baseline": "#4ECDC4", "Full": "#F57575"}

    for metric_name, ylabel, ax in metrics:
        for offset_idx, model_name in enumerate(model_order):
            vals = []
            for route_name in route_order:
                row = summary_df[(summary_df["route_name"] == route_name) & (summary_df["model"] == model_name)]
                vals.append(float(row.iloc[0][metric_name]) if not row.empty else np.nan)
            ax.bar(
                x + (offset_idx - 0.5) * width,
                vals,
                width=width,
                label=model_name,
                color=colors[model_name],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(route_order)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="best")

    axes[0].set_title("Route Load Comparison: Energy per km", fontsize=12, fontweight="bold")
    axes[1].set_title("Route Load Comparison: Peak Power", fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config, map_manager, physics = build_environment()
    samplers = build_model_samplers(config, map_manager)
    routes = build_fixed_routes(map_manager)

    rows: List[Dict[str, float]] = []
    for route_name in ["Route A", "Route B", "Route C"]:
        route = routes[route_name]
        route_xy = sample_route_points(route, ROUTE_SAMPLES)
        log(f"正在评估 {route.name}")
        for model_name in ["Baseline", "Full"]:
            log(f"当前模型为 {model_name}")
            row = evaluate_route_under_model(
                route=route,
                route_xy=route_xy,
                map_manager=map_manager,
                physics=physics,
                wind_sampler=samplers[model_name],
                model_name=model_name,
                config=config,
            )
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_csv = OUTPUT_DIR / "exp4_route_load_summary.csv"
    compare_png = OUTPUT_DIR / "exp4_route_load_comparison.png"
    route_csv = OUTPUT_DIR / "exp4_routes.csv"

    export_route_summary_csv(summary_df, summary_csv)
    export_route_definitions_csv(routes, route_csv)
    plot_route_load_comparison(summary_df, compare_png)

    log(f"输出文件路径: {OUTPUT_DIR}")
    log(f"  - {summary_csv.name}")
    log(f"  - {compare_png.name}")
    log(f"  - {route_csv.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"实验失败: {exc}")
        raise
