"""
无人机路径规划仿真系统主程序

此模块是传统仿真模式的主入口，执行完整的无人机任务仿真流程，
包括环境初始化、路径规划、任务执行、结果分析和可视化输出。

主要功能：
- 构建完整的仿真系统组件
- 执行动态任务推演（考虑风场、电池管理等）
- 生成多维度分析图表和动画
- 输出任务性能指标总结

作者：项目团队
版本：1.0.0
日期：2026-04-03
"""

import sys, os
import random  # 用于设置随机种子，确保仿真结果可重现
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np  # 数值计算库，用于向量和矩阵运算
import matplotlib.pyplot as plt  # 绘图库，用于生成静态图表
from utils.animation_builder import MissionAnimator  # 动画生成器，用于创建飞行过程GIF
from configs.config import SimulationConfig  # 仿真配置类，包含所有参数设置
from environment.map_manager import MapManager  # 地图管理器，处理障碍物和边界
from environment.wind_models import WindModelFactory  # 风场模型工厂，创建不同类型的风场
from core.physics import PhysicsEngine  # 物理引擎，模拟无人机运动学和动力学
from core.estimator import StateEstimator  # 状态估计器，实时估计位置、速度等状态
from core.planner import AStarPlanner  # A*路径规划器，计算最优路径
from core.battery_manager import BatteryManager  # 电池管理器，管理电量消耗
from simulation.mission_executor import MissionExecutor  # 任务执行器，执行完整的任务流程
from utils.visualizer_core import Visualizer  # 可视化核心，生成各种分析图表
from analysis.mission_metrics import summarize_mission_result, format_summary_text  # 任务指标计算和格式化

def build_system(config):
    """
    构建并初始化仿真系统的所有核心组件

    此函数是系统初始化的核心，创建所有必要的模块实例，
    确保各组件之间正确连接和配置。

    参数：
    config (SimulationConfig): 仿真配置对象，包含所有系统参数

    返回：
    tuple: 包含以下组件的元组：
        - map_manager (MapManager): 地图管理器实例
        - estimator (StateEstimator): 状态估计器实例
        - physics (PhysicsEngine): 物理引擎实例
        - battery_manager (BatteryManager): 电池管理器实例
        - planner (AStarPlanner): 路径规划器实例
        - visualizer (Visualizer): 可视化器实例

    组件依赖关系：
    - MapManager -> 提供地图边界给WindModelFactory
    - WindModelFactory -> 创建风场模型供StateEstimator使用
    - StateEstimator -> 依赖MapManager和风场模型
    - PhysicsEngine -> 独立组件
    - BatteryManager -> 独立组件
    - AStarPlanner -> 依赖estimator、physics
    - Visualizer -> 依赖config、estimator

    注意：
    组件的创建顺序很重要，后创建的组件可能依赖前面的组件。
    """
    map_manager = MapManager(config)  # 创建地图管理器，初始化障碍物和边界
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())  # 根据配置创建风场模型
    estimator = StateEstimator(map_manager, wind_model, config)  # 创建状态估计器，整合地图和风场信息
    physics = PhysicsEngine(config)  # 创建物理引擎，设置运动学参数
    battery_manager = BatteryManager(config)  # 创建电池管理器，设置电量参数
    planner = AStarPlanner(config, estimator, physics)  # 创建路径规划器，依赖估计器和物理引擎
    visualizer = Visualizer(config, estimator)  # 创建可视化器，用于生成图表

    return map_manager, estimator, physics, battery_manager, planner, visualizer

def select_start_goal(config, estimator: StateEstimator):
    """
    根据配置参数选择任务的起点和终点坐标

    此函数从配置文件中读取偏移参数，计算出在地图边界内的
    有效起点和终点位置。确保起点和终点都在有效区域内。

    参数：
    config (SimulationConfig): 仿真配置对象，包含起点和终点偏移参数
    estimator (StateEstimator): 状态估计器，用于获取地图边界信息

    返回：
    tuple: 包含起点和终点坐标的元组
        - start_xy (tuple): 起点坐标 (x, y)，单位为米
        - goal_xy (tuple): 终点坐标 (x, y)，单位为米

    计算逻辑：
    - 起点：从左下角偏移固定距离
    - 终点：从左下角按比例偏移到地图内部

    配置参数：
    - start_offset_x: 起点X方向偏移量
    - start_offset_y: 起点Y方向偏移量
    - goal_offset_factor_x: 终点X方向比例因子 (0-1)
    - goal_offset_factor_y: 终点Y方向比例因子 (0-1)

    注意：
    坐标系以地图左下角为原点，X轴向右，Y轴向上。
    """
    min_x, max_x, min_y, max_y = estimator.get_bounds()  # 获取地图的边界坐标
    start_xy = (min_x + config.start_offset_x, min_y + config.start_offset_y)  # 计算起点坐标
    goal_x = min_x + (max_x - min_x) * config.goal_offset_factor_x  # 计算终点X坐标
    goal_y = min_y + (max_y - min_y) * config.goal_offset_factor_y  # 计算终点Y坐标
    goal_xy = (goal_x, goal_y)  # 组合终点坐标
    return start_xy, goal_xy

def run_dynamic_mission(config, estimator, physics, battery_manager, planner, start_xy, goal_xy):
    """
    执行完整的4D动态任务推演（考虑时空维度）

    此函数是仿真核心，创建任务执行器并运行完整的任务流程，
    包括路径规划、动态执行、状态更新和风险规避。

    参数：
    config (SimulationConfig): 仿真配置对象
    estimator (StateEstimator): 状态估计器，实时更新环境状态
    physics (PhysicsEngine): 物理引擎，计算运动状态
    battery_manager (BatteryManager): 电池管理器，监控电量
    planner (AStarPlanner): 路径规划器，计算最优路径
    start_xy (tuple): 起点坐标 (x, y)
    goal_xy (tuple): 终点坐标 (x, y)

    返回：
    MissionResult: 任务执行结果对象，包含完整的状态历史和性能指标

    执行流程：
    1. 创建MissionExecutor实例
    2. 初始化任务参数
    3. 执行路径规划
    4. 实时仿真飞行过程
    5. 处理动态事件（风场变化、电池消耗等）
    6. 收集执行数据和指标

    4D含义：
    - X, Y: 空间位置
    - Z: 时间维度（任务执行时间）
    - 第四维: 风险规避（动态路径调整）

    注意：
    此函数会修改estimator的状态，因为任务执行过程中环境会动态变化。
    """
    executor = MissionExecutor(  # 创建任务执行器实例
        config=config,  # 传递配置参数
        estimator=estimator,  # 状态估计器
        physics=physics,  # 物理引擎
        battery_manager=battery_manager,  # 电池管理器
        planner=planner,  # 路径规划器
    )
    return executor.execute_mission(start_xy, goal_xy)  # 执行任务并返回结果

def print_mission_summary(mission_result):
    """
    打印任务执行结果的详细摘要

    此函数计算任务的关键性能指标，并以格式化的文本形式输出，
    方便用户快速了解任务执行情况。

    参数：
    mission_result (MissionResult): 任务执行结果对象

    输出内容：
    - 任务成功状态
    - 执行时间
    - 路径长度
    - 能耗统计
    - 平均速度
    - 风险规避指标

    注意：
    使用analysis.mission_metrics模块的函数来计算和格式化指标。
    """
    summary = summarize_mission_result(mission_result)  # 计算任务指标摘要
    print("\n" + format_summary_text(summary) + "\n")  # 格式化输出摘要文本

def main():
    """
    主函数：执行完整的无人机路径规划仿真流程

    此函数是程序的入口点， orchestrates 整个仿真过程：
    1. 设置随机种子确保结果可重现
    2. 初始化系统组件
    3. 执行任务仿真
    4. 重置环境状态用于可视化
    5. 生成分析图表和动画
    6. 输出结果摘要

    随机种子设置：
    - 使用固定种子确保每次运行结果一致
    - 种子同时设置Python random和NumPy random
    - 风场模型也使用相同种子

    可视化输出：
    - 纯地形图
    - 任务对比地图
    - 功率和能耗分析图
    - 3D轨迹图（如果任务成功）
    - 飞行过程动画GIF

    输出目录：
    所有图表和动画保存到mission_outputs目录

    注意：
    任务执行会修改estimator状态，因此需要重新创建风场模型
    用于干净的可视化状态。
    """
    # ===================================================================
    # [FIX 1] 全局锁死随机种子！确保宇宙的宿命是唯一的！
    # =====================================================================
    GLOBAL_SEED = 42  # 你可以换成任何数字，只要不改，每次雷暴的位置就绝对一样
    random.seed(GLOBAL_SEED)  # 设置Python随机种子
    np.random.seed(GLOBAL_SEED)  # 设置NumPy随机种子，确保数值计算可重现
    
    config = SimulationConfig()  # 创建仿真配置实例
    config.wind_seed = GLOBAL_SEED  # 设置风场随机种子

    print("--- 正在初始化环境与物理引擎 ---")
    map_manager, estimator, physics, battery_manager, planner, visualizer = build_system(config)  # 构建所有系统组件
    start_xy, goal_xy = select_start_goal(config, estimator)  # 根据配置选择起点和终点

    print("\n--- 正在执行 4D 时空极值风险规避推演 ---")
    # 这一步仿真会彻底改变 estimator 里面的雷暴坐标和状态
    mission_result = run_dynamic_mission(  # 执行完整的任务仿真
        config=config, estimator=estimator, physics=physics, 
        battery_manager=battery_manager, planner=planner, 
        start_xy=start_xy, goal_xy=goal_xy
    )

    # ===================================================================
    # [FIX 2] 推演结束后，用同一个种子直接"重建风场"！
    # 取代之前容易出错的手动"倒带坐标"逻辑，保证绝对干净的 t=0 状态
    # ====================================================================
    random.seed(GLOBAL_SEED)  # 重新设置随机种子
    np.random.seed(GLOBAL_SEED)  # 重新设置NumPy种子
    fresh_wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())  # 创建新的干净风场模型
    estimator.wind = fresh_wind_model  # 替换被污染的风场模型
    
    # 必须要重新实例化一次画图器，确保它拿到的是干净的风场
    visualizer = Visualizer(config, estimator)  # 重新创建可视化器

    print_mission_summary(mission_result)  # 打印任务摘要
    print("--- 正在绘制多重维度分析图表并保存本地 ---")
    
    output_dir = "mission_outputs"  # 输出目录
    if not os.path.exists(output_dir):  # 如果目录不存在则创建
        os.makedirs(output_dir)

    # 1. 绘制并保存纯地形图
    pure_terrain_path = os.path.join(output_dir, "00_pure_terrain.png")  # 纯地形图路径
    visualizer.plot_pure_terrain(save_filename=pure_terrain_path)  # 生成纯地形图

    # 2. 绘制 2D 对比地图
    visualizer.plot_mission_comparison(  # 生成任务对比地图
        mission_result=mission_result, start_xy=start_xy, goal_xy=goal_xy,
        wind_time_s=0.0, show_replanned_paths=True, save_dir=output_dir, physics_engine=physics 
    )

    # 3. 绘制功率对比图
    visualizer.plot_power_energy_comparison(  # 生成功率能耗对比图
        mission_result=mission_result, physics_engine=physics, save_dir=output_dir
    )

    if mission_result.success:  # 如果任务成功
        visualizer.plot_3d_trajectory_plotly(mission_result, start_xy, goal_xy)  # 生成3D轨迹图
        print("\n--- 正在渲染 4D 动态飞行过程，准备生成 GIF... ---")
        animator = MissionAnimator(config, estimator)  # 创建动画生成器
        gif_path = os.path.join(output_dir, "flight_with_storms.gif")  # GIF文件路径
        animator.generate_gif(  # 生成飞行过程动画
            mission_result=mission_result, start_xy=start_xy, goal_xy=goal_xy, 
            filename=gif_path, physics_engine=physics
        )

    print(f"\n所有计算均已完成！图片和动画已保存至目录: {output_dir}")  # 输出完成信息
    plt.show()  # 显示matplotlib图表

if __name__ == "__main__":
    main()