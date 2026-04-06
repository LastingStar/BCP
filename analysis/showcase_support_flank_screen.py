import os
import sys
from pathlib import Path

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from environment.map_manager import MapManager
from environment.wind_models import StatelessStormCell, WindModelFactory
from simulation.swarm_mission_executor import SwarmMissionExecutor
from utils.animation_builder import MissionAnimator
from utils.visualizer_core import Visualizer

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


def inject_flank_storm(config: SimulationConfig, wind_model, start_xy, goal_xy) -> None:
    if not hasattr(wind_model, "storm_manager") or wind_model.storm_manager is None:
        return

    route_vec = np.array(goal_xy, dtype=float) - np.array(start_xy, dtype=float)
    route_dist = np.linalg.norm(route_vec)
    if route_dist < 1.0:
        return

    forward = route_vec / route_dist
    right_flank = np.array([forward[1], -forward[0]], dtype=float)

    intercept_progress = 0.45
    intercept_xy = np.array(start_xy, dtype=float) + route_vec * intercept_progress

    cross_offset_m = 900.0
    storm_speed_mps = 15.0
    encounter_time_s = (route_dist * intercept_progress) / max(config.cruise_speed_mps, 1.0)
    birth_time_s = max(0.0, encounter_time_s - cross_offset_m / storm_speed_mps)

    custom_storm = StatelessStormCell(
        start_center_xy=intercept_xy + right_flank * cross_offset_m,
        velocity_xy=-right_flank * storm_speed_mps,
        radius_m=700.0,
        strength_mps=18.0,
        birth_time_s=birth_time_s,
        actual_lifetime_s=260.0,
    )
    wind_model.storm_manager.storm_slots = [[custom_storm]]


def load_rl_model():
    if not PPO:
        print("⚠️ 未安装 stable_baselines3，剧本 A 回退为 ASTAR 版。")
        return None
    model_path = Path("models/ppo_drone_stage3_obs31_run1_best/best_model.zip")
    if not model_path.exists():
        print(f"⚠️ 找不到 RL 模型: {model_path}，剧本 A 回退为 ASTAR 版。")
        return None
    print("🧠 加载 RL 模型权重用于剧本 A 展示...")
    return PPO.load(str(model_path), device="cpu")


def main():
    config = SimulationConfig()
    config.wind_seed = 91
    config.curriculum_stage = 3
    config.enable_storms = True
    config.storm_count = 1
    config.enable_random_gusts = False
    config.enable_support_shield_mode = True
    config.support_shield_master_radius_m = 1400.0
    config.support_shield_offset_m = 450.0

    start_xy = (-6500.0, -5000.0)
    goal_xy = (6200.0, 5200.0)

    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    inject_flank_storm(config, wind_model, start_xy, goal_xy)

    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery = BatteryManager(config)
    rl_model = load_rl_model()
    master_mode = "rl" if rl_model is not None else "astar"

    executor = SwarmMissionExecutor(
        config=config,
        true_estimator=estimator,
        physics=physics,
        battery_manager=battery,
        master_mode=master_mode,
        rl_model=rl_model,
    )

    mission_result = executor.execute_mission(start_xy, goal_xy)

    output_dir = Path("results/showcase_support_flank_screen")
    output_dir.mkdir(parents=True, exist_ok=True)

    vis = Visualizer(config, estimator)
    vis.plot_swarm_execution(mission_result, start_xy, goal_xy, save_dir=str(output_dir))
    vis.plot_swarm_elevation_profile(mission_result, save_dir=str(output_dir))

    animator = MissionAnimator(config, estimator)
    animator.generate_swarm_gif(
        mission_result=mission_result,
        start_xy=start_xy,
        goal_xy=goal_xy,
        filename=str(output_dir / "support_flank_screen.gif"),
    )

    print("\n" + "=" * 72)
    print("Support flank-screen showcase finished")
    print(f"Master mode: {master_mode.upper()}")
    print(f"Success: {mission_result.success}")
    print(f"Total time: {mission_result.total_mission_time_s:.1f}s")
    print(f"Replans: {mission_result.total_replans}")
    print(f"Output: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
