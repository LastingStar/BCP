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
    
    # 🌟 如果觉得当前的雷暴位置把路全堵死了，可以随时换个种子！
    # config.wind_seed = 100 # 换个种子，风暴位置就会变

    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery_manager = BatteryManager(config)
    planner = AStarPlanner(config, estimator, physics)
    visualizer = Visualizer(config, estimator)

    return config, map_manager, estimator, physics, battery_manager, planner, visualizer

def select_start_goal(estimator: StateEstimator):
    """选择大跨度起终点"""
    min_x, max_x, min_y, max_y = estimator.get_bounds()
    # 起点设在左下角偏内陆
    start_xy = (min_x + 1000.0, min_y + 1000.0)
    # 终点设在右上角偏内陆
    goal_x = min_x + (max_x - min_x) * 0.8
    goal_y = min_y + (max_y - min_y) * 0.8
    goal_xy = (goal_x, goal_y)
    return start_xy, goal_xy

def run_dynamic_mission(config, estimator, physics, battery_manager, planner, start_xy, goal_xy):
    """执行完整的 4D 动态任务推演"""
    executor = MissionExecutor(
        config=config,
        estimator=estimator,
        physics=physics,
        battery_manager=battery_manager,
        planner=planner,
    )
    # 一气呵成，让 4D 时钟自然流淌！
    return executor.execute_mission(start_xy, goal_xy)

def print_mission_summary(mission_result):
    """输出结构化任务摘要"""
    summary = summarize_mission_result(mission_result)
    print("\n" + format_summary_text(summary) + "\n")

def main():
    print("--- 正在初始化环境与物理引擎 ---")
    config, map_manager, estimator, physics, battery_manager, planner, visualizer = build_system()

    start_xy, goal_xy = select_start_goal(estimator)

    print("\n--- 🚀 正在执行 4D 时空极值风险规避推演 (Spatio-Temporal Planning) ---")
    # 这一步可能会算得稍微久一点（比如 10-20 秒），因为它在脑海里推演未来 1000 秒的风暴走势
    mission_result = run_dynamic_mission(
        config=config,
        estimator=estimator,
        physics=physics,
        battery_manager=battery_manager,
        planner=planner,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )

    print_mission_summary(mission_result)

    print("--- 📊 正在绘制多重维度分析图表 ---")
    visualizer.plot_mission_comparison(
        mission_result=mission_result,
        start_xy=start_xy,
        goal_xy=goal_xy,
        wind_time_s=0.0,
        show_replanned_paths=True,
        save_path="mission_comparison.png",
    )

    # ==========================================
    # 🌟 调用新增的 3D Plotly 方法
    # ==========================================
    if mission_result.success:
        visualizer.plot_3d_trajectory_plotly(mission_result, start_xy, goal_xy)

    if mission_result.success:
        print("\n--- 🎬 正在渲染 4D 动态飞行过程，准备生成 GIF... ---")
        animator = MissionAnimator(config, estimator)
        animator.generate_gif(
            mission_result=mission_result, 
            start_xy=start_xy, 
            goal_xy=goal_xy, 
            filename="flight_with_storms.gif"
        )
    else:
        print("\n[⚠️ 提示] 任务未能成功抵达终点 (极端恶劣环境或被禁飞区完全封死)，跳过 GIF 动画生成。")

    print("\n✅ 所有计算均已完成！")
    # 这句话会卡住程序，保持 Matplotlib 窗口不关闭。此时浏览器应该已经弹出了 3D 图形。
    plt.show()

if __name__ == "__main__":
    main()