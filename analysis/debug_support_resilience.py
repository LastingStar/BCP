import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from analysis.showcase_support_flank_screen import inject_flank_storm
from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from simulation.swarm_mission_executor import SwarmMissionExecutor
from utils.animation_builder import MissionAnimator
from utils.visualizer_core import Visualizer

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


START_XY = (-6500.0, -5000.0)
GOAL_XY = (6200.0, 5200.0)


@dataclass(frozen=True)
class SupportResilienceCase:
    name: str
    failure_mode: str
    allow_support_takeover: bool
    allow_support_bridge: bool
    allow_support_scan: bool
    description: str


def build_cases() -> List[SupportResilienceCase]:
    return [
        SupportResilienceCase(
            name="normal",
            failure_mode="none",
            allow_support_takeover=True,
            allow_support_bridge=True,
            allow_support_scan=True,
            description="Baseline four-drone swarm with full support roles enabled.",
        ),
        SupportResilienceCase(
            name="relay_fail",
            failure_mode="relay_fail",
            allow_support_takeover=False,
            allow_support_bridge=False,
            allow_support_scan=False,
            description="Relay fails and support is not allowed to compensate.",
        ),
        SupportResilienceCase(
            name="relay_fail_with_support",
            failure_mode="relay_fail",
            allow_support_takeover=True,
            allow_support_bridge=True,
            allow_support_scan=True,
            description="Relay fails and support is allowed to bridge/take over.",
        ),
        SupportResilienceCase(
            name="scout_fail",
            failure_mode="scout_fail",
            allow_support_takeover=False,
            allow_support_bridge=False,
            allow_support_scan=False,
            description="Scout sensing fails and support is not allowed to compensate.",
        ),
        SupportResilienceCase(
            name="scout_fail_with_support",
            failure_mode="scout_fail",
            allow_support_takeover=True,
            allow_support_bridge=False,
            allow_support_scan=True,
            description="Scout sensing fails and support is allowed to move forward and provide backup scanning.",
        ),
        SupportResilienceCase(
            name="scout_fail_with_full_support",
            failure_mode="scout_fail",
            allow_support_takeover=True,
            allow_support_bridge=True,
            allow_support_scan=True,
            description="Scout sensing fails and support is allowed to scan, bridge, and move forward as a redundant node.",
        ),
    ]


def build_config(
    seed: int,
    case: SupportResilienceCase,
    failure_trigger_time_s: float,
    enable_random_gusts: bool = False,
) -> SimulationConfig:
    config = SimulationConfig()
    config.curriculum_stage = 3
    config.wind_seed = seed
    config.enable_storms = True
    config.storm_count = 1
    config.enable_random_gusts = enable_random_gusts
    config.enable_support_shield_mode = False
    config.collect_ablation_telemetry = True
    config.failure_mode = case.failure_mode
    config.failure_trigger_time_s = float(failure_trigger_time_s)
    config.allow_support_takeover = case.allow_support_takeover
    config.allow_support_bridge = case.allow_support_bridge
    config.allow_support_scan = case.allow_support_scan
    config.local_debug_seed = seed
    return config


def load_rl_model(master_mode: str, rl_model_path: str):
    if master_mode == "astar":
        return None, "astar"
    if not PPO:
        if master_mode == "rl":
            raise RuntimeError("stable_baselines3 is not installed, cannot force RL mode.")
        print("⚠️ stable_baselines3 unavailable, falling back to ASTAR.")
        return None, "astar"

    model_path = Path(rl_model_path)
    if not model_path.exists():
        if master_mode == "rl":
            raise FileNotFoundError(f"Cannot find RL model at {model_path}")
        print(f"⚠️ RL model not found at {model_path}, falling back to ASTAR.")
        return None, "astar"
    return PPO.load(str(model_path), device="cpu"), "rl"


def build_swarm_stack(config: SimulationConfig):
    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    inject_flank_storm(config, wind_model, START_XY, GOAL_XY)
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery = BatteryManager(config)
    return map_manager, wind_model, estimator, physics, battery


def _support_formation_target(executor, master_xy: np.ndarray, support_xy: np.ndarray, current_path) -> np.ndarray:
    forward_vec = executor._get_path_forward_vector(master_xy, current_path)
    right_flank = np.array([forward_vec[1], -forward_vec[0]], dtype=float)
    target = (
        master_xy
        - forward_vec * executor.support_trail_offset_m
        + right_flank * executor.support_flank_offset_m
    )
    executor.support_mode = "FORMATION"
    return executor._project_vehicle_to_safe_xy(target, fallback_xy=support_xy)


def inject_failure(executor: SwarmMissionExecutor, case: SupportResilienceCase, failure_trigger_time_s: float) -> None:
    executor._debug_case_name = case.name
    executor._debug_failure_mode = case.failure_mode
    executor._debug_failure_trigger_time_s = float(failure_trigger_time_s)
    executor._debug_allow_support_takeover = case.allow_support_takeover
    executor._debug_allow_support_bridge = case.allow_support_bridge
    executor._debug_allow_support_scan = case.allow_support_scan
    executor._debug_current_time_s = 0.0
    executor._debug_latest_scout_xy = None
    executor._debug_latest_support_xy = None
    executor._debug_support_detected_storm_ids = set()
    executor._debug_support_detected_threat_count = 0
    executor._debug_support_first_post_failure_detect_time_s = None
    executor._debug_plan_call_count = 0
    executor._debug_post_failure_replan_count = 0

    original_move_scout = executor._move_scout
    original_scout_scan = executor._scout_radar_scan
    original_support_scan = executor._support_radar_scan
    original_relay_target = executor._compute_relay_target
    original_support_target = executor._compute_support_target
    original_link_status = executor._evaluate_link_status
    original_plan_in_belief_world = executor._plan_in_belief_world

    def move_scout_wrapper(self, scout_pos, master_xy, path, dt):
        current_t = getattr(self, "_debug_current_time_s", 0.0)
        if self._debug_failure_mode == "scout_fail" and current_t >= self._debug_failure_trigger_time_s:
            next_pos = np.array(scout_pos, dtype=float)
        else:
            next_pos = np.array(original_move_scout(scout_pos, master_xy, path, dt), dtype=float)
        self._debug_latest_scout_xy = next_pos.copy()
        return next_pos

    def scout_scan_wrapper(self, scout_pos, current_t):
        self._debug_current_time_s = current_t
        if self._debug_failure_mode == "scout_fail" and current_t >= self._debug_failure_trigger_time_s:
            return []
        return original_scout_scan(scout_pos, current_t)

    def support_scan_wrapper(self, support_pos, current_t):
        self._debug_current_time_s = current_t
        if not self._debug_allow_support_scan:
            return []
        finds = original_support_scan(support_pos, current_t)
        if current_t >= self._debug_failure_trigger_time_s:
            new_support_finds = [
                storm_id for storm_id in finds if storm_id not in self._debug_support_detected_storm_ids
            ]
            if new_support_finds:
                self._debug_support_detected_storm_ids.update(new_support_finds)
                self._debug_support_detected_threat_count += len(new_support_finds)
                if self._debug_support_first_post_failure_detect_time_s is None:
                    self._debug_support_first_post_failure_detect_time_s = current_t
        return finds

    def relay_target_wrapper(self, master_xy, scout_xy, current_t):
        self._debug_current_time_s = current_t
        scout_failed = self._debug_failure_mode == "scout_fail" and current_t >= self._debug_failure_trigger_time_s
        if scout_failed and (self._debug_allow_support_takeover or self._debug_allow_support_bridge):
            if self._debug_latest_support_xy is not None:
                return original_relay_target(master_xy, self._debug_latest_support_xy, current_t)
        return original_relay_target(master_xy, scout_xy, current_t)

    def support_target_wrapper(self, master_xy, relay_xy, support_xy, current_path, current_t):
        self._debug_current_time_s = current_t
        self._debug_latest_support_xy = np.array(support_xy, dtype=float)
        support_z = self.true_estimator.get_altitude(support_xy[0], support_xy[1]) + self.config.takeoff_altitude_agl
        p_crash_sup, _ = self.true_estimator.get_risk(
            support_xy[0], support_xy[1], support_z, 10.0, current_t
        )
        if p_crash_sup > self.safe_risk_threshold:
            self.support_mode = "ESCAPE"
            escape_vec = support_xy - master_xy
            if np.linalg.norm(escape_vec) < 1e-6:
                escape_vec = np.array([1.0, 0.0], dtype=float)
            escape_vec = escape_vec / np.linalg.norm(escape_vec)
            return self._project_vehicle_to_safe_xy(
                support_xy + escape_vec * 0.5 * self.comm_range_m,
                fallback_xy=support_xy,
            )

        scout_failed = self._debug_failure_mode == "scout_fail" and current_t >= self._debug_failure_trigger_time_s
        relay_failed = self._debug_failure_mode == "relay_fail" and current_t >= self._debug_failure_trigger_time_s
        if scout_failed and self._debug_allow_support_scan and (
            self._debug_allow_support_takeover or self._debug_allow_support_bridge
        ):
            self.support_mode = "BRIDGE"
            forward_vec = self._get_path_forward_vector(master_xy, current_path)
            forward_target = master_xy + forward_vec * min(0.75 * self.comm_range_m, 0.85 * self.support_radar_radius_m)
            return self._project_vehicle_to_safe_xy(forward_target, fallback_xy=support_xy)

        if relay_failed:
            if self._debug_allow_support_takeover and self._debug_latest_scout_xy is not None:
                self.support_mode = "BRIDGE"
                takeover_target = master_xy + (self._debug_latest_scout_xy - master_xy) * 0.5
                return self._project_vehicle_to_safe_xy(takeover_target, fallback_xy=support_xy)
            return _support_formation_target(self, master_xy, support_xy, current_path)

        if not self._debug_allow_support_bridge and not self._debug_allow_support_takeover:
            return _support_formation_target(self, master_xy, support_xy, current_path)

        return original_support_target(master_xy, relay_xy, support_xy, current_path, current_t)

    def link_status_wrapper(self, master_xy, scout_xy, relay_xy=None, support_xy=None):
        status = original_link_status(master_xy, scout_xy, relay_xy, support_xy)
        current_t = getattr(self, "_debug_current_time_s", 0.0)
        scout_failed = self._debug_failure_mode == "scout_fail" and current_t >= self._debug_failure_trigger_time_s

        if not self._debug_allow_support_bridge:
            status["path_support"] = False

        if scout_failed:
            status["m_s"] = False
            status["r_s"] = False
            status["path_direct"] = False
            status["path_relay"] = False
            status["path_support"] = False
            if support_xy is not None and self._debug_allow_support_scan:
                status["m_sup"] = np.linalg.norm(master_xy - support_xy) <= self.comm_range_m
                if relay_xy is not None:
                    status["sup_r"] = np.linalg.norm(support_xy - relay_xy) <= self.comm_range_m
                if self._debug_allow_support_takeover:
                    status["path_direct"] = status["m_sup"]
                if self._debug_allow_support_bridge and relay_xy is not None:
                    status["path_support"] = status["m_r"] and status["sup_r"]

        relay_failed = self._debug_failure_mode == "relay_fail" and current_t >= self._debug_failure_trigger_time_s
        if relay_failed:
            status["m_r"] = False
            status["r_s"] = False
            status["path_relay"] = False
            if self._debug_allow_support_bridge and support_xy is not None:
                m_sup = np.linalg.norm(master_xy - support_xy) <= self.comm_range_m
                sup_s = np.linalg.norm(support_xy - scout_xy) <= self.comm_range_m
                status["m_sup"] = m_sup
                status["path_support"] = m_sup and sup_s
            else:
                status["path_support"] = False

        status["network_active"] = bool(
            status.get("path_direct", False)
            or status.get("path_relay", False)
            or status.get("path_support", False)
        )
        return status

    def plan_in_belief_world_wrapper(self, start_xy, goal_xy, t_s):
        if self._debug_plan_call_count > 0 and t_s >= self._debug_failure_trigger_time_s:
            self._debug_post_failure_replan_count += 1
        self._debug_plan_call_count += 1
        return original_plan_in_belief_world(start_xy, goal_xy, t_s)

    executor._move_scout = MethodType(move_scout_wrapper, executor)
    executor._scout_radar_scan = MethodType(scout_scan_wrapper, executor)
    executor._support_radar_scan = MethodType(support_scan_wrapper, executor)
    executor._compute_relay_target = MethodType(relay_target_wrapper, executor)
    executor._compute_support_target = MethodType(support_target_wrapper, executor)
    executor._evaluate_link_status = MethodType(link_status_wrapper, executor)
    executor._plan_in_belief_world = MethodType(plan_in_belief_world_wrapper, executor)


def collect_metrics(case: SupportResilienceCase, seed: int, failure_trigger_time_s: float, result) -> Dict:
    link_history = result.link_status_history or []
    time_history = result.swarm_time_history or []
    network_active = [bool(snapshot.get("network_active", False)) for snapshot in link_history]
    path_direct = [bool(snapshot.get("path_direct", False)) for snapshot in link_history]
    path_relay = [bool(snapshot.get("path_relay", False)) for snapshot in link_history]
    path_support = [bool(snapshot.get("path_support", False)) for snapshot in link_history]

    paired_steps = list(zip(time_history, link_history))
    post_failure_steps = [
        snapshot for time_s, snapshot in paired_steps if time_s >= failure_trigger_time_s
    ]
    post_failure_network_active = [bool(snapshot.get("network_active", False)) for snapshot in post_failure_steps]
    post_failure_path_support = [bool(snapshot.get("path_support", False)) for snapshot in post_failure_steps]

    disconnect_count = sum(
        1 for prev, curr in zip(network_active, network_active[1:]) if prev and not curr
    )
    reconnect_count = sum(
        1 for prev, curr in zip(network_active, network_active[1:]) if (not prev) and curr
    )

    power_history = result.master_power_history_w or []
    risk_history = result.master_risk_history or []
    return {
        "case_name": case.name,
        "seed": seed,
        "failure_mode": case.failure_mode,
        "allow_support_takeover": case.allow_support_takeover,
        "allow_support_bridge": case.allow_support_bridge,
        "allow_support_scan": case.allow_support_scan,
        "success": result.success,
        "failure_reason": result.failure_reason or "goal_reached",
        "mission_time_s": result.total_mission_time_s,
        "energy_j": result.total_energy_used_j,
        "total_replans": result.total_replans,
        "first_warning_time_s": result.first_warning_time_s,
        "first_warning_distance_m": result.first_warning_distance_m,
        "first_warning_route": result.first_warning_route,
        "warning_count": result.warning_count,
        "post_failure_replan_count": getattr(result, "post_failure_replan_count", np.nan),
        "network_disconnect_count": disconnect_count,
        "network_reconnect_count": reconnect_count,
        "network_active_ratio": float(np.mean(network_active)) if network_active else np.nan,
        "post_failure_network_active_ratio": float(np.mean(post_failure_network_active)) if post_failure_network_active else np.nan,
        "path_direct_ratio": float(np.mean(path_direct)) if path_direct else np.nan,
        "path_relay_ratio": float(np.mean(path_relay)) if path_relay else np.nan,
        "path_support_ratio": float(np.mean(path_support)) if path_support else np.nan,
        "post_failure_path_support_ratio": float(np.mean(post_failure_path_support)) if post_failure_path_support else np.nan,
        "support_detected_threat_count": getattr(result, "support_detected_threat_count", np.nan),
        "support_first_post_failure_detect_time_s": getattr(result, "support_first_post_failure_detect_time_s", np.nan),
        "mean_master_risk": float(np.mean(risk_history)) if risk_history else np.nan,
        "max_master_risk": float(np.max(risk_history)) if risk_history else np.nan,
        "mean_master_power_w": float(np.mean(power_history)) if power_history else np.nan,
        "max_master_power_w": float(np.max(power_history)) if power_history else np.nan,
    }


def run_single_case(
    case: SupportResilienceCase,
    seed: int,
    failure_trigger_time_s: float,
    master_mode: str,
    rl_model,
    enable_random_gusts: bool = False,
):
    config = build_config(seed, case, failure_trigger_time_s, enable_random_gusts=enable_random_gusts)
    _, _, estimator, physics, battery = build_swarm_stack(config)
    executor = SwarmMissionExecutor(
        config=config,
        true_estimator=estimator,
        physics=physics,
        battery_manager=battery,
        master_mode=master_mode,
        rl_model=rl_model,
    )
    inject_failure(executor, case, failure_trigger_time_s)
    result = executor.execute_mission(START_XY, GOAL_XY)
    result.support_detected_threat_count = executor._debug_support_detected_threat_count
    result.support_first_post_failure_detect_time_s = executor._debug_support_first_post_failure_detect_time_s
    result.post_failure_replan_count = executor._debug_post_failure_replan_count
    metrics = collect_metrics(case, seed, failure_trigger_time_s, result)
    return metrics, result, config, estimator


def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    summary_df = raw_df.groupby("case_name", as_index=False).agg(
        runs=("seed", "count"),
        success_rate=("success", "mean"),
        mission_time_s_mean=("mission_time_s", "mean"),
        energy_j_mean=("energy_j", "mean"),
        total_replans_mean=("total_replans", "mean"),
        first_warning_distance_m_mean=("first_warning_distance_m", "mean"),
        warning_count_mean=("warning_count", "mean"),
        post_failure_replan_count_mean=("post_failure_replan_count", "mean"),
        network_disconnect_count_mean=("network_disconnect_count", "mean"),
        network_reconnect_count_mean=("network_reconnect_count", "mean"),
        network_active_ratio_mean=("network_active_ratio", "mean"),
        post_failure_network_active_ratio_mean=("post_failure_network_active_ratio", "mean"),
        path_direct_ratio_mean=("path_direct_ratio", "mean"),
        path_relay_ratio_mean=("path_relay_ratio", "mean"),
        path_support_ratio_mean=("path_support_ratio", "mean"),
        post_failure_path_support_ratio_mean=("post_failure_path_support_ratio", "mean"),
        support_detected_threat_count_mean=("support_detected_threat_count", "mean"),
        support_first_post_failure_detect_time_s_mean=("support_first_post_failure_detect_time_s", "mean"),
        mean_master_risk_mean=("mean_master_risk", "mean"),
        max_master_risk_mean=("max_master_risk", "mean"),
        mean_master_power_w_mean=("mean_master_power_w", "mean"),
        max_master_power_w_mean=("max_master_power_w", "mean"),
    )
    failure_mode = (
        raw_df.groupby("case_name")["failure_reason"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
        .rename("dominant_failure_reason")
        .reset_index()
    )
    warning_route = (
        raw_df.groupby("case_name")["first_warning_route"]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "")
        .rename("dominant_warning_route")
        .reset_index()
    )
    summary_df = summary_df.merge(failure_mode, on="case_name", how="left")
    summary_df = summary_df.merge(warning_route, on="case_name", how="left")
    return summary_df


def plot_bar(summary_df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary_df[x_col], summary_df[y_col], color="#4ECDC4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_comparison(summary_df: pd.DataFrame, output_root: Path) -> None:
    if summary_df.empty:
        return
    plot_bar(
        summary_df,
        "case_name",
        "success_rate",
        "Support Resilience Debug: Success Rate",
        "Success Rate",
        output_root / "success_rate_bar.png",
    )
    plot_bar(
        summary_df,
        "case_name",
        "first_warning_distance_m_mean",
        "Support Resilience Debug: First Warning Distance",
        "Distance (m)",
        output_root / "first_warning_distance_bar.png",
    )
    plot_bar(
        summary_df,
        "case_name",
        "network_active_ratio_mean",
        "Support Resilience Debug: Network Active Ratio",
        "Active Ratio",
        output_root / "network_active_ratio_bar.png",
    )
    plot_bar(
        summary_df,
        "case_name",
        "post_failure_network_active_ratio_mean",
        "Support Resilience Debug: Post-failure Network Active Ratio",
        "Active Ratio",
        output_root / "post_failure_network_active_ratio_bar.png",
    )
    plot_bar(
        summary_df,
        "case_name",
        "support_detected_threat_count_mean",
        "Support Resilience Debug: Support-detected Threat Count",
        "Detected Threats",
        output_root / "support_detected_threat_count_bar.png",
    )


def save_representative_outputs(
    representatives: Dict[str, Dict],
    output_root: Path,
    generate_gif: bool,
) -> None:
    for case_name, payload in representatives.items():
        rep_dir = output_root / f"representative_{case_name}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        result = payload["result"]
        config = payload["config"]
        estimator = payload["estimator"]

        vis = Visualizer(config, estimator)
        vis.plot_swarm_execution(result, START_XY, GOAL_XY, save_dir=str(rep_dir))
        vis.plot_swarm_elevation_profile(result, save_dir=str(rep_dir))

        times = result.swarm_time_history[1 : 1 + len(result.master_risk_history)] or list(range(len(result.master_risk_history)))
        if result.master_risk_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times, result.master_risk_history, color="#FF6B6B", linewidth=2.0)
            ax.set_title(f"{case_name}: Master Risk Time Series")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("p_crash")
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            fig.savefig(rep_dir / f"master_risk_timeseries_{case_name}.png", dpi=300)
            plt.close(fig)

        if result.master_power_history_w:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times[: len(result.master_power_history_w)], result.master_power_history_w, color="#45B7D1", linewidth=2.0)
            ax.set_title(f"{case_name}: Master Power Time Series")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Power (W)")
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            fig.savefig(rep_dir / f"master_power_timeseries_{case_name}.png", dpi=300)
            plt.close(fig)

        if generate_gif:
            animator = MissionAnimator(config, estimator)
            animator.generate_swarm_gif(
                mission_result=result,
                start_xy=START_XY,
                goal_xy=GOAL_XY,
                filename=str(rep_dir / f"{case_name}.gif"),
            )


def save_outputs(
    output_root: Path,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    manifest: Dict,
    representatives: Dict[str, Dict],
    generate_gif: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_root / "raw_runs.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_root / "summary.csv", index=False, encoding="utf-8-sig")
    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    plot_comparison(summary_df, output_root)
    save_representative_outputs(representatives, output_root, generate_gif)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug script for support redundancy and degraded swarm resilience.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--failure-trigger-time", type=float, default=120.0)
    parser.add_argument("--output-root", default="results/support_resilience_debug")
    parser.add_argument("--representative-seed", type=int, default=0)
    parser.add_argument("--generate-gif", action="store_true")
    parser.add_argument("--enable-random-gusts", action="store_true")
    parser.add_argument("--master-mode", choices=["astar", "auto", "rl"], default="astar")
    parser.add_argument("--rl-model", default="models/ppo_drone_stage3_obs31_run1_best/best_model.zip")
    return parser.parse_args()


def main():
    args = parse_args()
    cases = build_cases()
    rep_seed = args.representative_seed if args.representative_seed in args.seeds else args.seeds[0]
    rl_model, resolved_master_mode = load_rl_model(args.master_mode, args.rl_model)

    raw_rows: List[Dict] = []
    representatives: Dict[str, Dict] = {}
    for seed in args.seeds:
        for case in cases:
            metrics, result, config, estimator = run_single_case(
                case=case,
                seed=seed,
                failure_trigger_time_s=args.failure_trigger_time,
                master_mode=resolved_master_mode,
                rl_model=rl_model,
                enable_random_gusts=args.enable_random_gusts,
            )
            raw_rows.append(metrics)
            if seed == rep_seed and case.name not in representatives:
                representatives[case.name] = {
                    "result": result,
                    "config": config,
                    "estimator": estimator,
                }

    raw_df = pd.DataFrame(raw_rows)
    summary_df = build_summary(raw_df)
    output_root = Path(args.output_root)
    manifest = {
        "seeds": args.seeds,
        "representative_seed": rep_seed,
        "failure_trigger_time_s": args.failure_trigger_time,
        "generate_gif": args.generate_gif,
        "enable_random_gusts": args.enable_random_gusts,
        "requested_master_mode": args.master_mode,
        "resolved_master_mode": resolved_master_mode,
        "rl_model": args.rl_model,
        "cases": [asdict(case) for case in cases],
    }
    save_outputs(
        output_root=output_root,
        raw_df=raw_df,
        summary_df=summary_df,
        manifest=manifest,
        representatives=representatives,
        generate_gif=args.generate_gif,
    )
    print("=" * 72)
    print("Support resilience debug finished")
    print(f"Master mode: {resolved_master_mode.upper()}")
    print(f"Output: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
"""
现在默认是 astar，你改成 RL 用这个命令：

python analysis/debug_support_resilience.py --master-mode rl
如果还想同时出 GIF：

python analysis/debug_support_resilience.py --master-mode rl --seeds 0 1 2 --failure-trigger-time 120 --generate-gif
如果你想“有模型就用 RL，没有就自动退回 A*”，用：

python analysis/debug_support_resilience.py --master-mode auto
"""
