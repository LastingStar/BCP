# --- START OF FILE test_swarm_standalone.py ---
import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.battery_manager import BatteryManager
from simulation.swarm_mission_executor import SwarmMissionExecutor
from utils.visualizer_core import Visualizer
from utils.animation_builder import MissionAnimator

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

def load_rl_model():
    if not PPO:
        print("⚠️ 未安装 stable_baselines3，跳过 RL 实验。")
        return None
    model_path = "models/ppo_drone_stage3_obs31_run1_best/best_model.zip"
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到 RL 模型: {model_path}，跳过 RL 实验。")
        return None
    print("🧠 加载 RL 模型权重...")
    return PPO.load(model_path, device="cpu")


def run_experiment(mode_name: str, disturbance_enabled: bool = False, rl_model=None):
    variant_name = f"{mode_name}_{'gust' if disturbance_enabled else 'clean'}"
    print(f"\n" + "="*70)
    print(
        f"🎬 正在执行实验组: [ FANET Swarm + Master {mode_name.upper()} "
        f"+ {'Random Gusts' if disturbance_enabled else 'Clean Baseline'} ]"
    )
    print("="*70)

    # 1. 统一环境底座 (Seed 锁定 37 保证风暴位置绝对公平)
    config = SimulationConfig()
    config.wind_seed = 37               
    config.curriculum_stage = 3         
    config.enable_storms = True         
    config.storm_count = 3
    config.enable_support_shield_mode = True

    if disturbance_enabled:
        config.enable_random_gusts = True
        config.gust_trigger_prob = 0.02
        config.gust_duration_s = 8.0
        config.gust_min_speed_mps = 4.0
        config.gust_max_speed_mps = 8.0
        config.gust_obs_noise_std = 0.01 if mode_name == "rl" else 0.0

    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery = BatteryManager(config)

    # 2. 注入指定策略
    swarm_executor = SwarmMissionExecutor(
        config, estimator, physics, battery, 
        master_mode=mode_name, 
        rl_model=rl_model
    )

    start_xy = (-8000.0, -8000.0)
    goal_xy = (6000.0, 7500.0)

    # 3. 轰鸣起飞
    mission_result = swarm_executor.execute_mission(start_xy, goal_xy)

    # 4. 结果汇报
    print("\n" + "-"*40)
    print(f"[{variant_name.upper()}] 任务结果: {'✅ 成功' if mission_result.success else '❌ 失败'}")
    print(f"   终止原因: {mission_result.failure_reason or 'None'}")
    print(f"   总耗时: {mission_result.total_mission_time_s:.1f} s")
    print(f"   总能耗: {mission_result.total_energy_used_j/1000:.1f} kJ")
    print(f"   重规划: {mission_result.total_replans} 次")
    print("-"*40)

    # 5. 专属出图保存
    output_dir = Path(f"results/swarm_test_{variant_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis = Visualizer(config, estimator)
    vis.plot_swarm_execution(
        mission_result=mission_result,
        start_xy=start_xy, goal_xy=goal_xy,
        save_dir=str(output_dir)
    )
    vis.plot_swarm_elevation_profile(
        mission_result=mission_result,
        save_dir=str(output_dir)
    )

    animator = MissionAnimator(config, estimator)
    animator.generate_swarm_gif(
        mission_result=mission_result, 
        start_xy=start_xy, goal_xy=goal_xy, 
        filename=str(output_dir / f"swarm_dynamic_{variant_name}.gif")
    )
    print(f"📁 结果已归档: {output_dir}\n")

if __name__ == "__main__":
    # --- 实验 1: 四版对比矩阵 ---
    run_experiment("astar", disturbance_enabled=False, rl_model=None)
    run_experiment("astar", disturbance_enabled=True, rl_model=None)

    rl_model = load_rl_model()
    if rl_model:
        run_experiment("rl", disturbance_enabled=False, rl_model=rl_model)
        run_experiment("rl", disturbance_enabled=True, rl_model=rl_model)
# --- END OF FILE test_swarm_standalone.py ---
