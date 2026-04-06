import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from adapters.rl_adapter import run_env_episode_to_mission_result
from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from rl_env.drone_env import GuidedDroneEnv
from simulation.mission_executor import MissionExecutor

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

METHODS = ["astar", "teacher", "rl_residual"]


class TeacherPolicy:
    """A* reference-following policy used as the teacher baseline."""

    def predict(self, env: GuidedDroneEnv) -> np.ndarray:
        teacher = env._teacher_reference()
        current_ground_alt = env.estimator.get_altitude(env.current_pos[0], env.current_pos[1])
        current_agl = max(env.min_clearance_agl, env.current_pos[2] - current_ground_alt)

        raw_delta_heading = (teacher["heading_deg"] - env.current_heading + 180.0) % 360.0 - 180.0
        raw_target_speed = teacher["speed_mps"]
        raw_delta_agl = teacher["agl_m"] - current_agl

        cfg = env.config
        norm_heading = np.clip(raw_delta_heading / cfg.rl_heading_delta_max_deg, -1.0, 1.0)
        speed_mid = 0.5 * (cfg.rl_speed_min + cfg.rl_speed_max)
        speed_half = 0.5 * (cfg.rl_speed_max - cfg.rl_speed_min)
        norm_speed = np.clip((raw_target_speed - speed_mid) / max(speed_half, 1e-6), -1.0, 1.0)
        norm_agl = np.clip(raw_delta_agl / cfg.rl_agl_delta_max_m, -1.0, 1.0)
        return np.array([norm_heading, norm_speed, norm_agl], dtype=np.float32)


def normalize_failure_reason(reason: str) -> str:
    if reason in ["overload"]:
        return "overload"
    if reason in ["battery_depleted", "battery below reserve threshold", "replanned path is not battery feasible"]:
        return "battery"
    if reason in ["storm_risk_too_high"]:
        return "storm_risk"
    if reason in ["terrain_or_nfz"]:
        return "terrain_or_nfz"
    if "planner failed" in reason or reason in ["no_path"]:
        return "planner_no_path"
    if reason in ["goal_reached", "None", "", None]:
        return "goal_reached"
    return "other"


def path_length_m(path_xyz: List[Tuple[float, float, float]]) -> float:
    if not path_xyz or len(path_xyz) < 2:
        return 0.0
    path_arr = np.array(path_xyz, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1)))


def build_fixed_task(seed: int, curriculum_stage: int, custom_task: Optional[dict] = None):
    config = SimulationConfig()
    config.curriculum_stage = curriculum_stage
    config.enable_single_agent_gusts = True
    env = GuidedDroneEnv(config)

    if custom_task:
        options = {"start_xy": tuple(custom_task["start_xy"]), "goal_xy": tuple(custom_task["goal_xy"])}
        env.reset(seed=seed, options=options)
    else:
        env.reset(seed=seed)

    fixed_config = env.config
    fixed_config.wind_seed = seed
    fixed_config.enable_single_agent_gusts = True
    fixed_config.collect_ablation_telemetry = True
    return tuple(env.current_pos[:2]), tuple(env.goal_pos[:2]), fixed_config


def run_pure_astar(seed: int, start_xy: Tuple[float, float], goal_xy: Tuple[float, float], config: SimulationConfig) -> Dict:
    metrics = {"seed": seed, "method": "astar", "success": False}
    try:
        map_manager = MapManager(config)
        wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
        estimator = StateEstimator(map_manager, wind_model, config)
        physics = PhysicsEngine(config)
        battery = BatteryManager(config)
        planner = AStarPlanner(config, estimator, physics)
        executor = MissionExecutor(config, estimator, physics, battery, planner)
        res = executor.execute_mission(start_xy, goal_xy)
        metrics.update(
            {
                "success": res.success,
                "time_s": res.total_mission_time_s,
                "energy_j": res.total_energy_used_j,
                "path_length_m": path_length_m(res.actual_flown_path_xyz),
                "failure_reason_raw": res.failure_reason or "goal_reached",
            }
        )
    except Exception as exc:
        metrics["failure_reason_raw"] = f"Error: {exc}"
        metrics["time_s"] = 0.0
        metrics["energy_j"] = 0.0
        metrics["path_length_m"] = 0.0
    metrics["failure_bucket"] = normalize_failure_reason(metrics["failure_reason_raw"])
    metrics["energy_kj_per_km"] = (
        (metrics["energy_j"] / 1000.0) / (metrics["path_length_m"] / 1000.0)
        if metrics["success"] and metrics["path_length_m"] > 0
        else np.nan
    )
    return metrics


def run_env_method(
    seed: int,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    config: SimulationConfig,
    method_kind: str,
    policy,
) -> Dict:
    metrics = {"seed": seed, "method": method_kind, "success": False}
    try:
        env = GuidedDroneEnv(copy.deepcopy(config))
        res, ext = run_env_episode_to_mission_result(
            env,
            policy,
            "teacher" if method_kind == "teacher" else "rl",
            seed,
            options={"start_xy": start_xy, "goal_xy": goal_xy},
        )
        metrics.update(
            {
                "success": res.success,
                "time_s": res.total_mission_time_s,
                "energy_j": res.total_energy_used_j,
                "path_length_m": path_length_m(res.actual_flown_path_xyz),
                "failure_reason_raw": ext.get("terminated_reason", res.failure_reason or "goal_reached"),
            }
        )
    except Exception as exc:
        metrics["failure_reason_raw"] = f"Error: {exc}"
        metrics["time_s"] = 0.0
        metrics["energy_j"] = 0.0
        metrics["path_length_m"] = 0.0
    metrics["failure_bucket"] = normalize_failure_reason(metrics["failure_reason_raw"])
    metrics["energy_kj_per_km"] = (
        (metrics["energy_j"] / 1000.0) / (metrics["path_length_m"] / 1000.0)
        if metrics["success"] and metrics["path_length_m"] > 0
        else np.nan
    )
    return metrics


def summarize_raw_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for method in METHODS:
        subset = raw_df[raw_df["method"] == method]
        if subset.empty:
            continue
        success_df = subset[subset["success"] == True]
        failure_counts = subset[subset["success"] == False]["failure_bucket"].value_counts().to_dict()
        summary_rows.append(
            {
                "method": method,
                "success_rate_pct": float(subset["success"].mean() * 100.0),
                "success_avg_energy_kj_per_km": float(success_df["energy_kj_per_km"].mean()) if not success_df.empty else np.nan,
                "success_avg_time_s": float(success_df["time_s"].mean()) if not success_df.empty else np.nan,
                "success_avg_energy_kj": float((success_df["energy_j"] / 1000.0).mean()) if not success_df.empty else np.nan,
                "failure_breakdown": " | ".join(f"{k}:{v}" for k, v in failure_counts.items()) if failure_counts else "None",
            }
        )
    return pd.DataFrame(summary_rows)


def plot_success_energy(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    methods = summary_df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.38

    success_vals = summary_df["success_rate_pct"].fillna(0.0).to_numpy()
    energy_vals = summary_df["success_avg_energy_kj_per_km"].fillna(0.0).to_numpy()

    ax1.bar(x - width / 2, success_vals, width=width, color="#4ECDC4", label="Success Rate (%)")
    ax2.bar(x + width / 2, energy_vals, width=width, color="#FF6B6B", label="Success Energy/km (kJ/km)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("Success Rate (%)")
    ax2.set_ylabel("Success Energy/km (kJ/km)")
    ax1.set_title("Control Ablation: Success Rate vs Energy Efficiency")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_failure_breakdown(raw_df: pd.DataFrame, output_path: Path) -> None:
    failures = raw_df[raw_df["success"] == False]
    if failures.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Control Ablation: Failure Breakdown")
        ax.text(0.5, 0.5, "No failures", ha="center", va="center", fontsize=16)
        ax.axis("off")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    pivot = failures.pivot_table(
        index="method",
        columns="failure_bucket",
        values="seed",
        aggfunc="count",
        fill_value=0,
    ).reindex(METHODS, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(pivot.index))
    for bucket in pivot.columns:
        values = pivot[bucket].to_numpy(dtype=float)
        ax.bar(pivot.index, values, bottom=bottom, label=bucket)
        bottom += values

    ax.set_title("Control Ablation: Failure Breakdown")
    ax.set_ylabel("Failure Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_benchmark(
    rl_model_path: str,
    seeds: List[int],
    output_dir: Path,
    curriculum_stage: int = 3,
    custom_task: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting fixed benchmark with %d seeds", len(seeds))

    rl_model = PPO.load(rl_model_path, device="cpu") if PPO and Path(rl_model_path).exists() else None
    if rl_model is None:
        logger.warning("RL model unavailable: %s", rl_model_path)

    raw_rows = []
    for seed in seeds:
        start_xy, goal_xy, fixed_config = build_fixed_task(seed, curriculum_stage, custom_task=custom_task)
        raw_rows.append(run_pure_astar(seed, start_xy, goal_xy, copy.deepcopy(fixed_config)))
        raw_rows.append(run_env_method(seed, start_xy, goal_xy, fixed_config, "teacher", TeacherPolicy()))
        if rl_model is not None:
            rl_row = run_env_method(seed, start_xy, goal_xy, fixed_config, "rl_residual", rl_model)
            rl_row["method"] = "rl_residual"
            raw_rows.append(rl_row)

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize_raw_results(raw_df)
    raw_df.to_csv(output_dir / "raw_runs.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    plot_success_energy(summary_df, output_dir / "control_success_energy_dual_axis.png")
    plot_failure_breakdown(raw_df, output_dir / "control_failure_breakdown_stacked_bar.png")
    return raw_df, summary_df


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed benchmark for astar / teacher / rl_residual")
    parser.add_argument("--rl-model", default="models/ppo_drone_stage3_obs31_run1_best/best_model.zip")
    parser.add_argument("--output-dir", default="results/benchmark_stage3_obs31")
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--seed-count", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        rl_model_path=args.rl_model,
        seeds=list(range(args.seed_count)),
        output_dir=Path(args.output_dir),
        curriculum_stage=args.curriculum_stage,
    )
