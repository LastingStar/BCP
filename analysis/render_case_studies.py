# --- START OF FILE analysis/render_case_studies.py ---
import os, sys, copy
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import SimulationConfig
from rl_env.drone_env import GuidedDroneEnv
from adapters.rl_adapter import run_env_episode_to_mission_result
from utils.visualizer_core import Visualizer
from utils.animation_builder import MissionAnimator

# 🌟 新增：为了独立运行 A* 所需的核心物理与规划组件
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.physics import PhysicsEngine
from core.estimator import StateEstimator
from core.planner import AStarPlanner
from core.battery_manager import BatteryManager
from simulation.mission_executor import MissionExecutor

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

CASES_DIR = PROJECT_ROOT / "results" / "cases"

class TeacherPolicy:
    """内部定义的 Teacher 基线 (A* Reference Following)"""
    def predict(self, env: GuidedDroneEnv) -> np.ndarray:
        teacher = env._teacher_reference()
        current_ground_alt = env.estimator.get_altitude(env.current_pos[0], env.current_pos[1])
        current_agl = max(env.min_clearance_agl, env.current_pos[2] - current_ground_alt)

        # 1. 计算原始物理偏差
        raw_delta_heading = (teacher["heading_deg"] - env.current_heading + 180.0) % 360.0 - 180.0
        raw_target_speed = teacher["speed_mps"]
        raw_delta_agl = teacher["agl_m"] - current_agl

        # 2. 动态读取环境配置，进行归一化映射 [-1, 1]
        cfg = env.config
        norm_heading = np.clip(raw_delta_heading / cfg.rl_heading_delta_max_deg, -1.0, 1.0)
        speed_mid = 0.5 * (cfg.rl_speed_min + cfg.rl_speed_max)
        speed_half = 0.5 * (cfg.rl_speed_max - cfg.rl_speed_min)
        norm_speed = np.clip((raw_target_speed - speed_mid) / speed_half, -1.0, 1.0)
        norm_agl = np.clip(raw_delta_agl / cfg.rl_agl_delta_max_m, -1.0, 1.0)

        return np.array([norm_heading, norm_speed, norm_agl], dtype=np.float32)

def render_custom_case(
    method_name: str, 
    method_kind: str, 
    model_path: str, 
    stage: int,
    wind_seed: int,
    start_xy: tuple, 
    goal_xy: tuple, 
    tag: str
):
    print(f"\n🎬 正在渲染定制案例: [{method_name}] -> {tag}")
    
    # 1. 初始化独立配置
    config = SimulationConfig()
    config.curriculum_stage = stage
    config.wind_seed = wind_seed
    
    # ==============================================================
    # 🌟 核心分流：根据 method_kind 决定走底层执行器还是强化学习环境
    # ==============================================================
    if method_kind == "astar":
        # 【分支 A】纯 A*：直接调用底层物理引擎与重规划执行器
        map_manager = MapManager(config)
        wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
        estimator = StateEstimator(map_manager, wind_model, config)
        physics_engine = PhysicsEngine(config)
        battery = BatteryManager(config)
        planner = AStarPlanner(config, estimator, physics_engine)
        
        executor = MissionExecutor(config, estimator, physics_engine, battery, planner)
        mission_result = executor.execute_mission(start_xy, goal_xy)
        
    elif method_kind in ["teacher", "rl"]:
        # 【分支 B】环境模型：初始化 GuidedDroneEnv
        env = GuidedDroneEnv(copy.deepcopy(config))
        
        if method_kind == "teacher":
            policy = TeacherPolicy()
        else:
            model_file = PROJECT_ROOT / model_path
            if not model_file.exists():
                print(f"❌ 找不到 RL 模型: {model_file}")
                return
            policy = PPO.load(str(model_file))
            
        env_options = {
            "start_xy": start_xy,
            "goal_xy": goal_xy
        }
        
        mission_result, _ = run_env_episode_to_mission_result(
            env, policy, method_kind, seed=wind_seed, options=env_options
        )
        estimator = env.estimator
        physics_engine = env.physics
    else:
        print(f"❌ 未知的 method_kind: {method_kind}")
        return

    # ==============================================================
    # 4. 共享渲染逻辑：无论是 A* 还是 RL，只要出了 mission_result 就直接画图
    # ==============================================================
    case_dir = CASES_DIR / tag
    case_dir.mkdir(parents=True, exist_ok=True)
    
    vis = Visualizer(config, estimator)
    
    # 🌟 修改调用，使用纯净版画图函数
    print(" -> 正在生成 2D 地形单路径图...")
    vis.plot_single_mission_execution(
        mission_result=mission_result,
        start_xy=start_xy, goal_xy=goal_xy,
        method_name=method_name,
        save_dir=str(case_dir)
    )

    print(" -> 正在生成高度剖面图...")
    vis.plot_elevation_profile(
        mission_result=mission_result,
        save_dir=str(case_dir),
        method_name=method_name
    )
    
    print(" -> 正在生成功率与能耗物理分析图...")
    vis.plot_power_energy_comparison(
        mission_result=mission_result,
        physics_engine=physics_engine,
        save_dir=str(case_dir),
        method_name=method_name,     # 传入当前画的是什么算法
        show_baseline=False          # 🌟 关键：关闭红线对比，实现纯净单线展示
    )
    
    # 🌟 修改：去掉 if mission_result.success 限制！无论成功还是坠机，都强行生成 GIF！
    print(" -> 正在生成 4D 动态飞行追风 GIF 动画...")
    animator = MissionAnimator(config, estimator)
    animator.generate_gif(
        mission_result=mission_result,
        start_xy=start_xy, goal_xy=goal_xy,
        filename=str(case_dir / "flight_with_storms.gif"),
        physics_engine=physics_engine
    )
    print(f"✅ 案例 {tag} 渲染完成！结果已存入 {case_dir}")


# ========== 统一设置区：所有会经常改的参数都放这里 ==========
SETTINGS = {
    "task": {
        "start_xy": (-300.0, -600.0),
        "goal_xy": (-5000.0, 1250.0),
        "wind_seed": 37,
        "stage": 3,
    },
    "model_path": "models/ppo_drone_stage3_obs31_run1_best/best_model.zip",
    "astar_tag": "stage3_obs31_run1_case_valley_crossing_astar",      # 🌟 新增 A* 标签
    "teacher_tag": "stage3_obs31_run1_case_valley_crossing_teacher",
    "rl_tag": "stage3_obs31_run1_case_valley_crossing_rl_ours",
}

if __name__ == "__main__":
    task = SETTINGS["task"]

    print("🚀 开始出图！启动定制化场景渲染...")

    # 🌟 画第一组：纯全局规划 A*
    render_custom_case(
        method_name="Pure A* (Global Replanning)",
        method_kind="astar",
        model_path="",
        stage=task["stage"],
        wind_seed=task["wind_seed"],
        start_xy=task["start_xy"],
        goal_xy=task["goal_xy"],
        tag=SETTINGS["astar_tag"],
    )

    # 画第二组：Teacher 跟随
    render_custom_case(
        method_name="Teacher (A* Reference Following)",
        method_kind="teacher",
        model_path="",
        stage=task["stage"],
        wind_seed=task["wind_seed"],
        start_xy=task["start_xy"],
        goal_xy=task["goal_xy"],
        tag=SETTINGS["teacher_tag"],
    )

    # 画第三组：我们的 RL 模型
    render_custom_case(
        method_name="RL (Our 4D Spatio-Temporal Method)",
        method_kind="rl",
        model_path=SETTINGS["model_path"],
        stage=task["stage"],
        wind_seed=task["wind_seed"],
        start_xy=task["start_xy"],
        goal_xy=task["goal_xy"],
        tag=SETTINGS["rl_tag"],
    )
# --- END OF FILE analysis/render_case_studies.py ---