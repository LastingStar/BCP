import sys, os
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from utils.animation_builder import MissionAnimator
from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.physics import PhysicsEngine
from core.estimator import StateEstimator
from core.planner import AStarPlanner
from core.battery_manager import BatteryManager
from simulation.mission_executor import MissionExecutor
from utils.visualizer_core import Visualizer
from analysis.mission_metrics import summarize_mission_result, format_summary_text

def build_system():
    """构建系统核心组件"""
    config = SimulationConfig()

    # M4/M5 快速演示参数
    config.max_replans = 4
    config.max_mission_time_s = 240.0
    config.mission_update_interval_s = 20.0
    config.cruise_speed_mps = 18.0
    config.max_steps = 12000

    if hasattr(config, "planner_verbose"):
        config.planner_verbose = False

    if hasattr(config, "k_wind"):
        config.k_wind = min(config.k_wind, 0.8)
    if hasattr(config, "risk_factor"):
        config.risk_factor = min(config.risk_factor, 0.3)

    config.battery_capacity_j = max(config.battery_capacity_j, 800000.0)

    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery_manager = BatteryManager(config)
    planner = AStarPlanner(config, estimator, physics)
    visualizer = Visualizer(config, estimator)

    return config, map_manager, estimator, physics, battery_manager, planner, visualizer


def select_start_goal(estimator: StateEstimator):
    """选择起终点（局部短距离任务）"""
    min_x, max_x, min_y, max_y = estimator.get_bounds()

    start_xy = (min_x + 300.0, min_y + 300.0)
    goal_xy = (min_x + 1200.0, min_y + 900.0)

    goal_x = min(goal_xy[0], max_x - 300.0)
    goal_y = min(goal_xy[1], max_y - 300.0)
    goal_xy = (goal_x, goal_y)

    return start_xy, goal_xy


def run_dynamic_mission(config, estimator, physics, battery_manager, planner, start_xy, goal_xy):
    """执行动态任务"""
    executor = MissionExecutor(
        config=config,
        estimator=estimator,
        physics=physics,
        battery_manager=battery_manager,
        planner=planner,
    )
    return executor.execute_mission(start_xy, goal_xy)


def print_mission_summary(mission_result):
    """输出任务摘要"""
    summary = summarize_mission_result(mission_result)
    print("\n" + format_summary_text(summary) + "\n")


def main():
    # 1. 初始化系统
    config, map_manager, estimator, physics, battery_manager, planner, visualizer = build_system()

    # 2. 设置起终点
    start_xy, goal_xy = select_start_goal(estimator)

    # 3. 执行动态任务（4D 预测规划）
    print("\n--- 正在执行动态任务（4D Dynamic Replanning Mission）---")
    mission_result = run_dynamic_mission(
        config=config,
        estimator=estimator,
        physics=physics,
        battery_manager=battery_manager,
        planner=planner,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )

    # 4. 输出任务摘要
    print_mission_summary(mission_result)

    # 5. 可视化：三路径对比静态图
    print("--- 正在绘制三路径对比图（Mission Comparison）---")
    visualizer.plot_mission_comparison(
        mission_result=mission_result,
        start_xy=start_xy,
        goal_xy=goal_xy,
        wind_time_s=0.0,
        show_replanned_paths=True,
        save_path="mission_comparison.png",
    )

    # 6. 🌟 大创杀手锏：渲染动态飞行与躲避风暴 GIF
    if mission_result.success:
        print("\n--- 正在渲染动态飞行过程，准备生成 GIF... ---")
        animator = MissionAnimator(config, estimator)
        animator.generate_gif(
            mission_result=mission_result, 
            start_xy=start_xy, 
            goal_xy=goal_xy, 
            filename="flight_with_storms.gif"
        )
    else:
        print("\n[提示] 任务未能成功抵达终点，跳过 GIF 动画生成。")

    # 7. 统一显示所有弹出的 matplotlib 静态图表
    print("正在显示所有静态图表 (关闭图表窗口即可结束程序)...")
    plt.show()


if __name__ == "__main__":
    main()