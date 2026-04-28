import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHOD_ORDER = ["astar", "teacher", "rl_residual"]
BAR_COLORS = ["#F57575", "#60C0BA", "#5AAFC2"]


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes"])


def load_metrics_from_raw(raw_df: pd.DataFrame, methods: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    raw_df = raw_df.copy()
    raw_df["success"] = _to_bool(raw_df["success"])

    for method in methods:
        subset = raw_df[raw_df["method"] == method]
        if subset.empty:
            continue
        success_df = subset[subset["success"] == True]
        metrics[method] = {
            "success_rate_pct": float(subset["success"].mean() * 100.0),
            "success_avg_path_m": float(success_df["path_length_m"].mean()) if not success_df.empty else np.nan,
            "success_avg_energy_kj": float((success_df["energy_j"] / 1000.0).mean()) if not success_df.empty else np.nan,
            "success_avg_energy_kj_per_km": float(success_df["energy_kj_per_km"].mean()) if not success_df.empty else np.nan,
        }
    return metrics


def load_metrics_from_summary(summary_df: pd.DataFrame, methods: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}

    if "method" in summary_df.columns:
        for method in methods:
            row = summary_df[summary_df["method"] == method]
            if row.empty:
                continue
            rec = row.iloc[0]
            metrics[method] = {
                "success_rate_pct": float(rec.get("success_rate_pct", np.nan)),
                "success_avg_path_m": float(rec.get("success_avg_path_m", np.nan)),
                "success_avg_energy_kj": float(rec.get("success_avg_energy_kj", np.nan)),
                "success_avg_energy_kj_per_km": float(rec.get("success_avg_energy_kj_per_km", np.nan)),
            }
        return metrics

    if "Method" in summary_df.columns:
        for method in methods:
            row = summary_df[summary_df["Method"] == method]
            if row.empty:
                continue
            rec = row.iloc[0]
            metrics[method] = {
                "success_rate_pct": float(rec.get("Success Rate (%)", np.nan)),
                "success_avg_path_m": float(rec.get("Success Avg Path (m)", np.nan)),
                "success_avg_energy_kj": float(rec.get("Success Avg Energy (kJ)", np.nan)),
                "success_avg_energy_kj_per_km": float(rec.get("Success Energy/km (kJ/km)", np.nan)),
            }
        return metrics

    raise ValueError("Unsupported summary csv format.")


def load_metrics(csv_path: Path, methods: List[str]) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(csv_path)
    cols = set(df.columns)
    raw_cols = {"seed", "method", "success", "path_length_m", "energy_j", "energy_kj_per_km"}
    if raw_cols.issubset(cols):
        return load_metrics_from_raw(df, methods)
    return load_metrics_from_summary(df, methods)


def plot_2x2(metrics: Dict[str, Dict[str, float]], methods: List[str], output_path: Path) -> None:
    ordered_methods = [m for m in methods if m in metrics]
    if not ordered_methods:
        raise ValueError("No method data found to plot.")

    def collect(key: str) -> List[float]:
        return [metrics[m][key] for m in ordered_methods]

    panels = [
        ("Success Rate (%)", "success_rate_pct"),
        ("Success Avg Path (m)", "success_avg_path_m"),
        ("Success Avg Energy (kJ)", "success_avg_energy_kj"),
        ("Success Energy/km (kJ/km)", "success_avg_energy_kj_per_km"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#EAEAEA")
    axes_flat = axes.flatten()

    for idx, (title, key) in enumerate(panels):
        ax = axes_flat[idx]
        values = collect(key)
        bars = ax.bar(
            ordered_methods,
            values,
            color=[BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(ordered_methods))],
            edgecolor="#1F1F1F",
            linewidth=1.4,
        )
        ax.set_title(title, fontsize=20, weight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_facecolor("#E0E0E0")

        y_max = np.nanmax(values) if np.isfinite(values).any() else 1.0
        if np.isfinite(y_max) and y_max > 0:
            ax.set_ylim(0, y_max * 1.15)

        for b, v in zip(bars, values):
            if np.isnan(v):
                label = "nan"
            elif key == "success_rate_pct":
                label = f"{v:.1f}"
            else:
                label = f"{v:.1f}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + max(0.01 * (y_max if np.isfinite(y_max) else 1.0), 0.1),
                label,
                ha="center",
                va="bottom",
                fontsize=17,
                weight="bold",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark 2x2 chart directly from csv.")
    parser.add_argument("--csv", required=True, help="Input csv path: raw_runs.csv or summary.csv")
    parser.add_argument("--output", default=None, help="Output png path")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHOD_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"csv not found: {csv_path}")

    output_path = Path(args.output) if args.output else csv_path.with_name("comparison_2x2.png")
    metrics = load_metrics(csv_path, methods=args.methods)
    plot_2x2(metrics, methods=args.methods, output_path=output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
