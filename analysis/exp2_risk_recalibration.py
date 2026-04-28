import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.exp2_riskfield_analysis import build_environment_from_config, compute_tke_field


# =========================
# Configurable knobs
# =========================
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp2_risk_recalibration"
N0_FIXED = 3.0
K_VALUES = [20.0, 40.0, 80.0, 200.0]
HEIGHT_NOTE = 50.0
TIME_NOTE = 600.0
HIGH_RISK_THRESHOLD = 0.7
TKE_CURVE_MAX = 30.0
TKE_CURVE_POINTS = 600
FIG_DPI = 300


def log(msg: str) -> None:
    print(f"[exp2b] {msg}")


def compute_risk_indicator_from_tke(
    tke_field: np.ndarray,
    exposure_time_s: float,
    n0: float,
    k_robust: float,
) -> np.ndarray:
    safe_tke = np.maximum(tke_field, 1e-6)
    exponent = -n0 * exposure_time_s * np.exp(-k_robust / safe_tke)
    risk_indicator = 1.0 - np.exp(exponent)
    return np.clip(risk_indicator, 0.0, 1.0)


def summarize_risk_field(
    risk_field: np.ndarray,
    k_value: float,
    n0_value: float,
) -> Dict[str, float]:
    vals = risk_field[np.isfinite(risk_field)]
    return {
        "K_value": float(k_value),
        "N0_value": float(n0_value),
        "risk_min": float(np.min(vals)),
        "risk_mean": float(np.mean(vals)),
        "risk_median": float(np.median(vals)),
        "risk_p95": float(np.percentile(vals, 95)),
        "risk_p99": float(np.percentile(vals, 99)),
        "risk_max": float(np.max(vals)),
        "high_risk_threshold": float(HIGH_RISK_THRESHOLD),
        "high_risk_ratio": float(np.mean(vals > HIGH_RISK_THRESHOLD)),
    }


def plot_risk_field_comparison(
    map_manager,
    risk_fields: Dict[float, np.ndarray],
    output_path: Path,
) -> None:
    log("正在绘制 K 重标定风险场对比图")
    extent = [map_manager.x[0], map_manager.x[-1], map_manager.y[0], map_manager.y[-1]]
    vmax = max(float(np.max(field)) for field in risk_fields.values())
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes_flat = axes.flatten()
    dem_levels = np.linspace(np.min(map_manager.dem), np.max(map_manager.dem), 12)

    last_im = None
    for ax, k_value in zip(axes_flat, K_VALUES):
        field = risk_fields[k_value]
        last_im = ax.imshow(
            field,
            extent=extent,
            origin="lower",
            cmap="turbo",
            vmin=0.0,
            vmax=vmax,
            alpha=0.92,
        )
        ax.contour(
            map_manager.X,
            map_manager.Y,
            map_manager.dem,
            levels=dem_levels,
            colors="white",
            linewidths=0.5,
            alpha=0.55,
        )
        ax.set_title(f"Risk Indicator Field (K = {k_value:.0f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Risk Indicator")
    fig.suptitle(
        f"Risk Surrogate Recalibration Comparison (N0 = {N0_FIXED:.1f}, height = {HEIGHT_NOTE:.0f} m AGL, t = {TIME_NOTE:.0f} s)",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mapping_curves(
    tke_field: np.ndarray,
    exposure_time_s: float,
    output_path: Path,
) -> None:
    log("正在绘制 TKE 到风险映射曲线")
    tke_axis = np.linspace(0.01, TKE_CURVE_MAX, TKE_CURVE_POINTS)
    tke_vals = tke_field[np.isfinite(tke_field)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for k_value in K_VALUES:
        risk_axis = compute_risk_indicator_from_tke(tke_axis, exposure_time_s, N0_FIXED, k_value)
        axes[0].plot(tke_axis, risk_axis, linewidth=2.2, label=f"K = {k_value:.0f}")

    axes[0].axvline(float(np.percentile(tke_vals, 95)), color="#277DA1", linestyle="--", linewidth=1.4, label="TKE p95")
    axes[0].axvline(float(np.max(tke_vals)), color="#577590", linestyle="--", linewidth=1.4, label="TKE max")
    axes[0].set_title("TKE to Risk Mapping", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("TKE")
    axes[0].set_ylabel("Risk Indicator")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best")

    axes[1].hist(tke_vals, bins=60, color="#4D908E", alpha=0.78)
    axes[1].set_title("TKE Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("TKE")
    axes[1].set_ylabel("Cell Count")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        f"Risk Mapping Diagnostic for K Recalibration (N0 = {N0_FIXED:.1f}, exposure = {exposure_time_s:.2f} s)",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("正在加载环境并计算基准 TKE 场")
    config, map_manager, wind_model, estimator = build_environment_from_config()
    tke_field = compute_tke_field(config, map_manager, estimator)
    exposure_time_s = float(map_manager.resolution) / max(float(config.cruise_speed_mps), 1.0)

    risk_fields: Dict[float, np.ndarray] = {}
    rows: List[Dict[str, float]] = []

    log("正在执行 K 重标定对比")
    for k_value in K_VALUES:
        log(f"当前 K = {k_value:.0f}, 固定 N0 = {N0_FIXED:.1f}")
        risk_field = compute_risk_indicator_from_tke(tke_field, exposure_time_s, N0_FIXED, k_value)
        risk_fields[k_value] = risk_field
        rows.append(summarize_risk_field(risk_field, k_value, N0_FIXED))

    summary_df = pd.DataFrame(rows)
    summary_csv = OUTPUT_DIR / "exp2_risk_recalibration_summary.csv"
    fields_png = OUTPUT_DIR / "exp2_risk_recalibration_fields.png"
    curves_png = OUTPUT_DIR / "exp2_risk_recalibration_curves.png"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    plot_risk_field_comparison(map_manager, risk_fields, fields_png)
    plot_mapping_curves(tke_field, exposure_time_s, curves_png)

    log(f"输出文件路径: {OUTPUT_DIR}")
    log(f"  - {summary_csv.name}")
    log(f"  - {fields_png.name}")
    log(f"  - {curves_png.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"实验失败: {exc}")
        raise
