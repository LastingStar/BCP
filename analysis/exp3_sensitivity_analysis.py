import copy
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
from core.estimator import StateEstimator
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory


# =========================
# Configurable experiment knobs
# =========================
HEIGHT_AGL_M = 50.0
TIME_S = 600.0
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp3_sensitivity_analysis"

# The current project's exponential risk surrogate is numerically very small
# under default settings, so we reuse the informative threshold used in exp2.
HIGH_RISK_THRESHOLD = 1e-6

ALPHA_CONFIG_FIELD = "k_slope"
BETA_EXPERIMENT_FIELD = "analysis_local_perturb_scale_mps"

ALPHA_DEFAULT = SimulationConfig().k_slope
BETA_DEFAULT = 3.0

ALPHA_VALUES = [
    0.0,
    0.5 * ALPHA_DEFAULT,
    1.0 * ALPHA_DEFAULT,
    1.5 * ALPHA_DEFAULT,
    2.0 * ALPHA_DEFAULT,
]

BETA_VALUES = [
    0.0,
    0.5 * BETA_DEFAULT,
    1.0 * BETA_DEFAULT,
    1.5 * BETA_DEFAULT,
    2.0 * BETA_DEFAULT,
]

FIG_DPI = 300
CURVE_METRICS = ["std_wind_speed", "peak_tke", "high_risk_ratio"]
CURVE_LABELS = {
    "std_wind_speed": "Wind Speed Std (m/s)",
    "peak_tke": "Peak TKE (m$^2$/s$^2$)",
    "high_risk_ratio": "High Risk Ratio",
}
CURVE_COLORS = {
    "std_wind_speed": "#277DA1",
    "peak_tke": "#F3722C",
    "high_risk_ratio": "#F94144",
}


def log(msg: str) -> None:
    print(f"[exp3] {msg}")


def compute_log_profile_factor(z_agl: np.ndarray, z0: np.ndarray, h_ref: float = 200.0) -> np.ndarray:
    safe_z0 = np.maximum(z0, 1e-3)
    safe_z = np.maximum(z_agl, safe_z0 + 1e-6)
    raw = np.log(safe_z / safe_z0) / np.log(h_ref / safe_z0)
    return np.clip(raw, 0.05, 1.5)


def compute_storm_field(wind_model, x_grid: np.ndarray, y_grid: np.ndarray, t_s: float) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(wind_model, "storm_manager") or wind_model.storm_manager is None:
        return np.zeros_like(x_grid, dtype=float), np.zeros_like(y_grid, dtype=float)
    if not wind_model.config.enable_storms:
        return np.zeros_like(x_grid, dtype=float), np.zeros_like(y_grid, dtype=float)

    u = np.zeros_like(x_grid, dtype=float)
    v = np.zeros_like(y_grid, dtype=float)
    for storm in wind_model.storm_manager.get_active_storms(t_s):
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


def compute_analysis_local_perturbation(
    map_manager: MapManager,
    beta_scale_mps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if beta_scale_mps <= 0.0:
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
    magnitude = beta_scale_mps * np.clip(slope_norm * curvature_norm, 0.0, 1.0)

    perp_x = -gy
    perp_y = gx
    perp_norm = np.hypot(perp_x, perp_y)
    perp_x = np.divide(perp_x, perp_norm, out=np.zeros_like(perp_x), where=perp_norm > 1e-9)
    perp_y = np.divide(perp_y, perp_norm, out=np.zeros_like(perp_y), where=perp_norm > 1e-9)
    sign = np.sign(laplacian)
    return perp_x * magnitude * sign, perp_y * magnitude * sign


class SensitivityEstimator(StateEstimator):
    def __init__(
        self,
        map_manager: MapManager,
        wind_model,
        config: SimulationConfig,
        local_u: np.ndarray,
        local_v: np.ndarray,
    ):
        super().__init__(map_manager, wind_model, config)
        self.interp_local_u = RegularGridInterpolator(
            (map_manager.y, map_manager.x),
            local_u,
            bounds_error=False,
            fill_value=0.0,
        )
        self.interp_local_v = RegularGridInterpolator(
            (map_manager.y, map_manager.x),
            local_v,
            bounds_error=False,
            fill_value=0.0,
        )

    def get_wind(
        self,
        x: float,
        y: float,
        z: float = -1.0,
        t_s: float = 0.0,
    ) -> np.ndarray:
        wind_vec = super().get_wind(x, y, z=z, t_s=t_s)
        local_u = float(self.interp_local_u((y, x)))
        local_v = float(self.interp_local_v((y, x)))
        return wind_vec + np.array([local_u, local_v], dtype=float)


@dataclass
class CaseResult:
    parameter_name: str
    parameter_value: float
    metrics: Dict[str, float]


def build_environment() -> Tuple[SimulationConfig, MapManager]:
    log("正在加载环境")
    config = SimulationConfig()
    config.noise_level = 0.0
    config.enable_random_gusts = False
    config.enable_single_agent_gusts = False
    map_manager = MapManager(config)
    return config, map_manager


def clone_config_with_parameter(
    base_config: SimulationConfig,
    parameter_name: str,
    parameter_value: float,
) -> Tuple[SimulationConfig, float]:
    config = copy.deepcopy(base_config)
    beta_scale = float(BETA_DEFAULT)

    if parameter_name == "alpha":
        setattr(config, ALPHA_CONFIG_FIELD, float(parameter_value))
    elif parameter_name == "beta":
        beta_scale = float(parameter_value)
    else:
        raise ValueError(f"Unsupported parameter_name: {parameter_name}")

    return config, beta_scale


def compute_wind_speed_field(
    config: SimulationConfig,
    map_manager: MapManager,
    wind_model,
    local_u: np.ndarray,
    local_v: np.ndarray,
) -> np.ndarray:
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

    storm_u, storm_v = compute_storm_field(wind_model, map_manager.X, map_manager.Y, TIME_S)
    u += storm_u + local_u
    v += storm_v + local_v

    u = np.clip(u, -config.max_wind_speed, config.max_wind_speed)
    v = np.clip(v, -config.max_wind_speed, config.max_wind_speed)
    return np.hypot(u, v)


def compute_tke_and_risk_fields(
    config: SimulationConfig,
    map_manager: MapManager,
    estimator: SensitivityEstimator,
) -> Tuple[np.ndarray, np.ndarray]:
    tke_field = np.zeros_like(map_manager.dem, dtype=float)
    z_abs = map_manager.dem + HEIGHT_AGL_M

    flat_x = map_manager.X.ravel()
    flat_y = map_manager.Y.ravel()
    flat_z = z_abs.ravel()
    tke_flat = np.zeros_like(flat_x, dtype=float)

    for idx, (xv, yv, zv) in enumerate(zip(flat_x, flat_y, flat_z)):
        tke_flat[idx] = estimator.get_tke(float(xv), float(yv), float(zv), t_s=TIME_S)

    tke_field[:, :] = tke_flat.reshape(map_manager.dem.shape)

    v_eff = max(float(config.cruise_speed_mps), 1.0)
    exposure_time = float(map_manager.resolution) / v_eff
    n0 = float(config.drone_response_freq_N0)
    k_robust = float(config.drone_robustness_K)
    safe_tke = np.maximum(tke_field, 1e-6)
    exponent = -n0 * exposure_time * np.exp(-k_robust / safe_tke)
    risk_field = 1.0 - np.exp(exponent)
    return tke_field, np.clip(risk_field, 0.0, 1.0)


def compute_summary_metrics(
    wind_speed_field: np.ndarray,
    tke_field: np.ndarray,
    risk_field: np.ndarray,
) -> Dict[str, float]:
    valid_mask = (
        np.isfinite(wind_speed_field)
        & np.isfinite(tke_field)
        & np.isfinite(risk_field)
    )

    return {
        "mean_wind_speed": float(np.mean(wind_speed_field[valid_mask])),
        "std_wind_speed": float(np.std(wind_speed_field[valid_mask])),
        "peak_wind_speed": float(np.max(wind_speed_field[valid_mask])),
        "mean_tke": float(np.mean(tke_field[valid_mask])),
        "peak_tke": float(np.max(tke_field[valid_mask])),
        "mean_risk": float(np.mean(risk_field[valid_mask])),
        "peak_risk": float(np.max(risk_field[valid_mask])),
        "high_risk_ratio": float(np.mean(risk_field[valid_mask] > HIGH_RISK_THRESHOLD)),
    }


def run_single_sensitivity_case(
    base_config: SimulationConfig,
    map_manager: MapManager,
    parameter_name: str,
    parameter_value: float,
) -> CaseResult:
    config, beta_scale = clone_config_with_parameter(base_config, parameter_name, parameter_value)

    if parameter_name == "alpha":
        beta_scale = float(BETA_DEFAULT)
    elif parameter_name == "beta":
        beta_scale = float(parameter_value)

    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    local_u, local_v = compute_analysis_local_perturbation(map_manager, beta_scale)
    estimator = SensitivityEstimator(map_manager, wind_model, config, local_u, local_v)

    wind_speed_field = compute_wind_speed_field(config, map_manager, wind_model, local_u, local_v)
    tke_field, risk_field = compute_tke_and_risk_fields(config, map_manager, estimator)
    metrics = compute_summary_metrics(wind_speed_field, tke_field, risk_field)
    return CaseResult(parameter_name=parameter_name, parameter_value=float(parameter_value), metrics=metrics)


def run_parameter_sweep(
    base_config: SimulationConfig,
    map_manager: MapManager,
    parameter_name: str,
    values: List[float],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    log(f"正在执行 {parameter_name} 灵敏度分析")
    for value in values:
        log(f"当前 {parameter_name} = {value:.6g}")
        case = run_single_sensitivity_case(base_config, map_manager, parameter_name, float(value))
        row = {"parameter_name": case.parameter_name, "parameter_value": case.parameter_value}
        row.update(case.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sensitivity_curves(
    alpha_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    output_path: Path,
) -> None:
    log("正在导出图和 CSV")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    panels = [
        (axes[0], alpha_df, "Sensitivity to alpha", "alpha"),
        (axes[1], beta_df, "Sensitivity to beta", "beta"),
    ]

    for ax, df, title, xlabel in panels:
        x = df["parameter_value"].to_numpy(dtype=float)
        for metric in CURVE_METRICS:
            ax.plot(
                x,
                df[metric].to_numpy(dtype=float),
                marker="o",
                linewidth=2.2,
                markersize=6,
                label=CURVE_LABELS[metric],
                color=CURVE_COLORS[metric],
            )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Metric Value")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")

    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def export_sensitivity_csv(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_config, map_manager = build_environment()
    alpha_df = run_parameter_sweep(base_config, map_manager, "alpha", ALPHA_VALUES)
    beta_df = run_parameter_sweep(base_config, map_manager, "beta", BETA_VALUES)

    summary_df = pd.concat([alpha_df, beta_df], ignore_index=True)
    fig_path = OUTPUT_DIR / "exp3_sensitivity_analysis.png"
    csv_path = OUTPUT_DIR / "exp3_sensitivity_summary.csv"

    plot_sensitivity_curves(alpha_df, beta_df, fig_path)
    export_sensitivity_csv(summary_df, csv_path)

    log(f"输出路径为 {OUTPUT_DIR}")
    log(f"  - {fig_path.name}")
    log(f"  - {csv_path.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"实验失败: {exc}")
        raise
