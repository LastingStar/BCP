import numpy as np
import matplotlib.pyplot as plt

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
    """
    构建系统核心组件（快速演示版配置）。
    """
    config = SimulationConfig()

    # =========================
    # M4/M5 快速演示参数
    # 目的：先保证动态闭环跑通、可视化可生成
    # =========================

    # 动态任务执行参数
    config.max_replans = 4
    config.max_mission_time_s = 240.0
    config.mission_update_interval_s = 20.0
    config.cruise_speed_mps = 18.0

    # 规划器参数（先偏保守，避免搜索空间过大）
    config.max_steps = 12000

    # 可选：如果你在 config.py 里已经加了这个开关，就关闭详细输出
    if hasattr(config, "planner_verbose"):
        config.planner_verbose = False

    # 可选：为了让演示更容易成功，可以稍微弱化风/风险惩罚
    # 如果你觉得路径太“拧”，可以把这些值继续调低
    if hasattr(config, "k_wind"):
        config.k_wind = min(config.k_wind, 0.8)
    if hasattr(config, "risk_factor"):
        config.risk_factor = min(config.risk_factor, 0.3)

    # 电池参数（演示阶段先保证能跑通）
    config.battery_capacity_j = max(config.battery_capacity_j, 800000.0)

    map_manager = MapManager(config)
    wind_model = WindModelFactory.create("slope", config)
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery_manager = BatteryManager(config)
    planner = AStarPlanner(config, estimator, physics)
    visualizer = Visualizer(config, estimator)

    return config, map_manager, estimator, physics, battery_manager, planner, visualizer


def select_start_goal(estimator: StateEstimator):
    """
    选择起终点（快速演示版：局部短距离任务）。

    说明：
    - 不再用全图大范围起终点
    - 先用局部任务保证动态闭环能跑通
    """
    min_x, max_x, min_y, max_y = estimator.get_bounds()

    # 先选一个局部范围任务，避免 3D A* 搜索空间过大
    start_xy = (min_x + 300.0, min_y + 300.0)
    goal_xy = (min_x + 1200.0, min_y + 900.0)

    # 保险：防止超出地图边界，留一点 margin
    goal_x = min(goal_xy[0], max_x - 300.0)
    goal_y = min(goal_xy[1], max_y - 300.0)
    goal_xy = (goal_x, goal_y)

    return start_xy, goal_xy


def run_dynamic_mission(
    config: SimulationConfig,
    estimator: StateEstimator,
    physics: PhysicsEngine,
    battery_manager: BatteryManager,
    planner: AStarPlanner,
    start_xy,
    goal_xy,
):
    """
    执行动态任务（M4）。
    """
    executor = MissionExecutor(
        config=config,
        estimator=estimator,
        physics=physics,
        battery_manager=battery_manager,
        planner=planner,
    )

    mission_result = executor.execute_mission(start_xy, goal_xy)
    return mission_result


def print_mission_summary(mission_result):
    """
    输出任务摘要（基于 M6 结构化指标）。
    """
    summary = summarize_mission_result(mission_result)
    print("\n" + format_summary_text(summary) + "\n")


def main():
    # 1. 初始化系统
    config, map_manager, estimator, physics, battery_manager, planner, visualizer = build_system()

    # 2. 设置起终点
    start_xy, goal_xy = select_start_goal(estimator)

    # 3. 执行动态任务（M4）
    print("\n--- 正在执行动态任务（Dynamic Replanning Mission）---")
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

    # 5. 三路径对比可视化（M5）
    print("--- 正在绘制三路径对比图（Mission Comparison）---")
    visualizer.plot_mission_comparison(
        mission_result=mission_result,
        start_xy=start_xy,
        goal_xy=goal_xy,
        wind_time_s=0.0,                 # 背景风场显示时刻，可后续改成别的时刻
        show_replanned_paths=True,       # 是否显示每次中间重规划路径
        save_path="mission_comparison.png",
    )

    # 6. 统一显示
    print("正在显示所有图表...")
    plt.show()


if __name__ == "__main__":
    main()