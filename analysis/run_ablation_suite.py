import argparse
import copy
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.benchmark_fixed_suite import run_benchmark
from analysis.showcase_support_flank_screen import inject_flank_storm
from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from rl_env.drone_env import GuidedDroneEnv
from rl_training.train_ppo import evaluate_model, seed_everything, train_ppo
from simulation.mission_executor import MissionExecutor
from simulation.swarm_mission_executor import SwarmMissionExecutor
from utils.animation_builder import MissionAnimator
from utils.visualizer_core import Visualizer

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


def create_suite_root(output_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_root = Path(output_root) / f"ablation_suite_{timestamp}"
    suite_root.mkdir(parents=True, exist_ok=True)
    return suite_root


def load_rl_model(rl_model_path: str):
    if not PPO or not Path(rl_model_path).exists():
        return None
    return PPO.load(rl_model_path, device="cpu")


def build_single_agent_config(seed: int) -> SimulationConfig:
    config = SimulationConfig()
    config.curriculum_stage = 3
    config.wind_seed = seed
    config.enable_single_agent_gusts = True
    config.collect_ablation_telemetry = True
    return config


def build_swarm_components(config: SimulationConfig):
    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery = BatteryManager(config)
    return map_manager, wind_model, estimator, physics, battery


def generate_long_distance_tasks(task_count: int, min_distance_m: float = 12000.0) -> List[dict]:
    config = SimulationConfig()
    config.curriculum_stage = 3
    env = GuidedDroneEnv(config)
    min_x, max_x, min_y, max_y = env.estimator.get_bounds()
    margin_m = 1000.0
    rng = np.random.default_rng(2025)
    tasks = []

    while len(tasks) < task_count:
        sx = float(rng.uniform(min_x + margin_m, max_x - margin_m))
        sy = float(rng.uniform(min_y + margin_m, max_y - margin_m))
        if not env._is_safe_location(sx, sy, 0.0):
            continue
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        distance = float(rng.uniform(min_distance_m, min_distance_m + 2500.0))
        gx = float(np.clip(sx + distance * math.cos(angle), min_x + margin_m, max_x - margin_m))
        gy = float(np.clip(sy + distance * math.sin(angle), min_y + margin_m, max_y - margin_m))
        actual_dist = math.hypot(gx - sx, gy - sy)
        if actual_dist < min_distance_m:
            continue
        if not env._is_safe_location(gx, gy, 0.0):
            continue
        tasks.append(
            {
                "task_id": len(tasks),
                "start_xy": (sx, sy),
                "goal_xy": (gx, gy),
                "wind_seed": int(rng.integers(1000, 100000)),
                "distance_m": actual_dist,
            }
        )
    return tasks


def run_exp1_control(args, suite_root: Path) -> Dict:
    output_dir = suite_root / "exp1_control"
    raw_df, summary_df = run_benchmark(
        rl_model_path=args.rl_model,
        seeds=list(range(args.control_seeds)),
        output_dir=output_dir,
        curriculum_stage=3,
        custom_task=None,
    )
    return {
        "raw_rows": len(raw_df),
        "summary_rows": len(summary_df),
        "output_dir": str(output_dir),
    }


def run_exp2_observation(args, suite_root: Path) -> Dict:
    output_dir = suite_root / "exp2_observation"
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = ["full", "no_future", "no_radar"]
    run_rows = []
    learning_curves: Dict[str, List[pd.DataFrame]] = {variant: [] for variant in variants}

    for variant in variants:
        for seed in args.obs_train_seeds:
            seed_everything(seed)
            run_name = f"{variant}_seed{seed}"
            model_root = output_dir / "models" / run_name
            log_root = output_dir / "logs" / run_name

            config = build_single_agent_config(seed)
            config.obs_ablation_mode = variant

            _, best_model_path = train_ppo(
                config=config,
                total_timesteps=args.obs_total_timesteps,
                n_envs=1,
                model_save_root=str(model_root),
                log_root=str(log_root),
                run_name=run_name,
                seed=seed,
                load_model_path=None,
                from_scratch=True,
            )

            eval_raw_path = output_dir / "evals" / f"{run_name}_final_eval_raw.csv"
            summary, eval_df = evaluate_model(
                best_model_path,
                config,
                n_episodes=config.ablation_eval_episodes,
                eval_seed=seed,
                raw_csv_path=str(eval_raw_path),
            )
            with open(output_dir / "evals" / f"{run_name}_final_eval_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            eval_metrics_path = Path(str(model_root) + "_eval") / "eval_metrics.csv"
            if eval_metrics_path.exists():
                curve_df = pd.read_csv(eval_metrics_path)
                curve_df["variant"] = variant
                curve_df["seed"] = seed
                learning_curves[variant].append(curve_df)

            run_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "success_rate": summary["success_rate"],
                    "storm_risk_fail_rate": summary["storm_risk_fail_rate"],
                    "max_p_crash_mean": summary["max_p_crash_mean"],
                    "max_p_crash_max": summary["max_p_crash_max"],
                    "mean_reward": summary["mean_reward"],
                    "mean_length": summary["mean_length"],
                }
            )

    raw_df = pd.DataFrame(run_rows)
    raw_df.to_csv(output_dir / "raw_runs.csv", index=False, encoding="utf-8-sig")
    summary_df = raw_df.groupby("variant", as_index=False).agg(
        success_rate_mean=("success_rate", "mean"),
        success_rate_std=("success_rate", "std"),
        storm_risk_fail_rate_mean=("storm_risk_fail_rate", "mean"),
        storm_risk_fail_rate_std=("storm_risk_fail_rate", "std"),
        max_p_crash_mean=("max_p_crash_mean", "mean"),
        max_p_crash_max=("max_p_crash_max", "max"),
    )
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, curves in learning_curves.items():
        if not curves:
            continue
        merged = pd.concat(curves, ignore_index=True)
        grouped = merged.groupby("timesteps")["success_rate"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["timesteps"], grouped["mean"], label=variant)
        ax.fill_between(
            grouped["timesteps"],
            grouped["mean"] - grouped["std"].fillna(0.0),
            grouped["mean"] + grouped["std"].fillna(0.0),
            alpha=0.2,
        )
    ax.set_title("Observation Ablation: Success Rate Learning Curves")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Success Rate")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "learning_curve_success_rate.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(summary_df["variant"], summary_df["success_rate_mean"], yerr=summary_df["success_rate_std"].fillna(0.0), color="#4ECDC4")
    ax.set_title("Observation Ablation: Final Success Rate")
    ax.set_ylabel("Success Rate")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "observation_final_success_bar.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        summary_df["variant"],
        summary_df["storm_risk_fail_rate_mean"],
        yerr=summary_df["storm_risk_fail_rate_std"].fillna(0.0),
        color="#FF6B6B",
    )
    ax.set_title("Observation Ablation: Storm Risk Failure Rate")
    ax.set_ylabel("Fail Rate")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "observation_storm_fail_bar.png", dpi=300)
    plt.close(fig)

    return {"runs": len(raw_df), "output_dir": str(output_dir)}


def run_exp3_topology(args, suite_root: Path) -> Dict:
    output_dir = suite_root / "exp3_topology"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = generate_long_distance_tasks(args.topology_seeds, min_distance_m=12000.0)
    with open(output_dir / "tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    raw_rows = []
    for task in tasks:
        for topology_mode in ["full", "no_relay"]:
            config = SimulationConfig()
            config.curriculum_stage = 3
            config.wind_seed = task["wind_seed"]
            config.swarm_topology_mode = topology_mode
            config.collect_ablation_telemetry = True
            _, _, estimator, physics, battery = build_swarm_components(config)
            executor = SwarmMissionExecutor(
                config=config,
                true_estimator=estimator,
                physics=physics,
                battery_manager=battery,
                master_mode="astar",
                rl_model=None,
            )
            result = executor.execute_mission(tuple(task["start_xy"]), tuple(task["goal_xy"]))
            raw_rows.append(
                {
                    "task_id": task["task_id"],
                    "distance_m": task["distance_m"],
                    "topology_mode": topology_mode,
                    "success": result.success,
                    "first_warning_distance_m": result.first_warning_distance_m,
                    "total_replans": result.total_replans,
                    "warning_count": result.warning_count,
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(output_dir / "raw_runs.csv", index=False, encoding="utf-8-sig")
    summary_df = raw_df.groupby("topology_mode", as_index=False).agg(
        success_rate=("success", "mean"),
        first_warning_distance_mean=("first_warning_distance_m", "mean"),
        total_replans_mean=("total_replans", "mean"),
    )
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [
        raw_df.loc[raw_df["topology_mode"] == mode, "first_warning_distance_m"].dropna().tolist()
        for mode in ["full", "no_relay"]
    ]
    ax.boxplot(box_data, tick_labels=["full", "no_relay"])
    ax.set_title("Topology Ablation: First Warning Distance")
    ax.set_ylabel("Distance (m)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "warning_distance_boxplot.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(summary_df["topology_mode"], summary_df["total_replans_mean"], color="#45B7D1")
    ax.set_title("Topology Ablation: Total Replans")
    ax.set_ylabel("Mean replans")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "total_replans_bar.png", dpi=300)
    plt.close(fig)

    return {"runs": len(raw_df), "output_dir": str(output_dir)}


def run_exp4_shield(args, suite_root: Path) -> Dict:
    output_dir = suite_root / "exp4_shield"
    gifs_dir = output_dir / "gifs"
    keyframes_dir = output_dir / "keyframes"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    rl_model = load_rl_model(args.rl_model)

    raw_rows = []
    rep_results = {}
    for seed in range(args.shield_seeds):
        for shield_enabled in [True, False]:
            config = SimulationConfig()
            config.curriculum_stage = 3
            config.wind_seed = seed
            config.enable_support_shield_mode = shield_enabled
            config.collect_ablation_telemetry = True
            start_xy = (-6500.0, -5000.0)
            goal_xy = (6200.0, 5200.0)

            map_manager = MapManager(config)
            wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
            inject_flank_storm(config, wind_model, start_xy, goal_xy)
            estimator = StateEstimator(map_manager, wind_model, config)
            physics = PhysicsEngine(config)
            battery = BatteryManager(config)
            master_mode = "rl" if rl_model is not None else "astar"
            executor = SwarmMissionExecutor(
                config=config,
                true_estimator=estimator,
                physics=physics,
                battery_manager=battery,
                master_mode=master_mode,
                rl_model=rl_model if rl_model is not None else None,
            )
            result = executor.execute_mission(start_xy, goal_xy)
            raw_rows.append(
                {
                    "seed": seed,
                    "shield_enabled": shield_enabled,
                    "master_mode": master_mode,
                    "success": result.success,
                    "peak_master_power_w": max(result.master_power_history_w) if result.master_power_history_w else 0.0,
                    "peak_master_risk": max(result.master_risk_history) if result.master_risk_history else 0.0,
                    "warning_count": result.warning_count,
                }
            )

            if seed == 91:
                label = "ours" if shield_enabled else "no_shield"
                rep_dir = output_dir / f"representative_{label}"
                rep_dir.mkdir(parents=True, exist_ok=True)
                vis = Visualizer(config, estimator)
                vis.plot_swarm_execution(result, start_xy, goal_xy, save_dir=str(rep_dir))
                vis.plot_swarm_elevation_profile(result, save_dir=str(rep_dir))
                animator = MissionAnimator(config, estimator)
                gif_path = rep_dir / f"{label}.gif"
                animator.generate_swarm_gif(result, start_xy, goal_xy, filename=str(gif_path))
                rep_results[label] = {"dir": rep_dir, "result": result, "gif": gif_path}

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(output_dir / "raw_runs.csv", index=False, encoding="utf-8-sig")
    summary_df = raw_df.groupby("shield_enabled", as_index=False).agg(
        success_rate=("success", "mean"),
        peak_master_power_w_mean=("peak_master_power_w", "mean"),
        peak_master_risk_mean=("peak_master_risk", "mean"),
    )
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    if "ours" in rep_results and "no_shield" in rep_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ours_power = rep_results["ours"]["result"].master_power_history_w
        no_shield_power = rep_results["no_shield"]["result"].master_power_history_w
        ax.plot(ours_power, label="Shield ON")
        ax.plot(no_shield_power, label="Shield OFF")
        ax.set_title("Support Shield Ablation: Master Power")
        ax.set_xlabel("Swarm step")
        ax.set_ylabel("Power (W)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "shield_master_power_timeseries.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ours_risk = rep_results["ours"]["result"].master_risk_history
        no_shield_risk = rep_results["no_shield"]["result"].master_risk_history
        ax.plot(ours_risk, label="Shield ON")
        ax.plot(no_shield_risk, label="Shield OFF")
        ax.set_title("Support Shield Ablation: Master Risk")
        ax.set_xlabel("Swarm step")
        ax.set_ylabel("p_crash")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "shield_master_risk_timeseries.png", dpi=300)
        plt.close(fig)

        shutil.copyfile(rep_results["ours"]["dir"] / "swarm_static_trajectories.png", keyframes_dir / "ours.png")
        shutil.copyfile(rep_results["no_shield"]["dir"] / "swarm_static_trajectories.png", keyframes_dir / "no_shield.png")
        shutil.copyfile(rep_results["ours"]["gif"], gifs_dir / "ours.gif")
        shutil.copyfile(rep_results["no_shield"]["gif"], gifs_dir / "no_shield.gif")

    return {"runs": len(raw_df), "output_dir": str(output_dir)}


def run_exp5_planner_time(args, suite_root: Path) -> Dict:
    output_dir = suite_root / "exp5_planner_time"
    output_dir.mkdir(parents=True, exist_ok=True)
    start_xy = (-6500.0, -5000.0)
    goal_xy = (6200.0, 5200.0)
    rows = []
    results = {}

    for planner_time_mode in ["4d", "frozen_3d"]:
        config = SimulationConfig()
        config.curriculum_stage = 3
        config.wind_seed = 91
        config.disable_periodic_replan = True
        config.planner_time_mode = planner_time_mode
        config.frozen_reference_time_s = 0.0
        config.collect_ablation_telemetry = True
        config.enable_single_agent_gusts = False

        map_manager = MapManager(config)
        wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
        inject_flank_storm(config, wind_model, start_xy, goal_xy)
        estimator = StateEstimator(map_manager, wind_model, config)
        physics = PhysicsEngine(config)
        battery = BatteryManager(config)
        planner = AStarPlanner(config, estimator, physics)
        executor = MissionExecutor(config, estimator, physics, battery, planner)
        result = executor.execute_mission(start_xy, goal_xy)
        results[planner_time_mode] = (result, estimator)

        nearest_dist = min(result.risk_history) if result.risk_history else np.nan
        rows.append(
            {
                "planner_time_mode": planner_time_mode,
                "success": result.success,
                "failure_reason": result.failure_reason or "goal_reached",
                "mission_time_s": result.total_mission_time_s,
                "energy_j": result.total_energy_used_j,
                "path_length_m": float(len(result.actual_flown_path_xyz)),
                "nearest_risk_value": nearest_dist,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "planner_time_case_metrics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6))
    for mode, (result, _) in results.items():
        path = np.array(result.actual_flown_path_xyz, dtype=float)
        if path.size == 0:
            continue
        ax.plot(path[:, 0] / 1000.0, path[:, 1] / 1000.0, label=mode)
    ax.scatter([start_xy[0] / 1000.0, goal_xy[0] / 1000.0], [start_xy[1] / 1000.0, goal_xy[1] / 1000.0], c=["gold", "red"])
    ax.set_title("Planner Time Ablation Overlay")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "planner_time_overlay.png", dpi=300)
    plt.close(fig)

    for mode, (result, estimator) in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contourf(estimator.map.X / 1000.0, estimator.map.Y / 1000.0, estimator.map.dem, 30, cmap="gist_earth", alpha=0.7)
        path = np.array(result.actual_flown_path_xyz, dtype=float)
        if path.size > 0:
            ax.plot(path[:, 0] / 1000.0, path[:, 1] / 1000.0, color="lime", linewidth=2.5)
        ax.set_title(f"Planner Time Mode: {mode}")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.tight_layout()
        fig.savefig(output_dir / f"planner_time_{mode}.png", dpi=300)
        plt.close(fig)

    return {"runs": len(metrics_df), "output_dir": str(output_dir)}


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end ablation suite controller")
    parser.add_argument("--exp", choices=["all", "control", "observation", "topology", "shield", "planner_time"], default="all")
    parser.add_argument("--rl-model", default="models/ppo_drone_stage3_obs31_run1_best/best_model.zip")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--control-seeds", type=int, default=50)
    parser.add_argument("--obs-train-seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--obs-total-timesteps", type=int, default=200000)
    parser.add_argument("--topology-seeds", type=int, default=50)
    parser.add_argument("--shield-seeds", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    suite_root = create_suite_root(args.output_root)
    manifest = vars(args).copy()
    manifest["suite_root"] = str(suite_root)
    with open(suite_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    suite_summary = {}
    if args.exp in {"all", "control"}:
        suite_summary["exp1_control"] = run_exp1_control(args, suite_root)
    if args.exp in {"all", "observation"}:
        suite_summary["exp2_observation"] = run_exp2_observation(args, suite_root)
    if args.exp in {"all", "topology"}:
        suite_summary["exp3_topology"] = run_exp3_topology(args, suite_root)
    if args.exp in {"all", "shield"}:
        suite_summary["exp4_shield"] = run_exp4_shield(args, suite_root)
    if args.exp in {"all", "planner_time"}:
        suite_summary["exp5_planner_time"] = run_exp5_planner_time(args, suite_root)

    with open(suite_root / "suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2, ensure_ascii=False)

    print(f"Ablation suite finished. Output: {suite_root}")


if __name__ == "__main__":
    main()
