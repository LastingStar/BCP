
"""Legacy demonstration script and wrapper for the visualizer.

This file originally contained both plotting logic and a crude "crash
simulation" demo. 目前核心的可视化功能已经迁移到
``utils.visualizer_core.Visualizer``，建议在其它模块中直接导入并
使用该类，而无需运行本脚本。

保留此文件仅供快速手动调试或回归测试；未来可安全删除。
"""

from matplotlib import pyplot as plt
import numpy as np

from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.physics import PhysicsEngine
from core.estimator import StateEstimator
from core.planner import AStarPlanner
from utils.visualizer_core import Visualizer


def main():
    # 1. 初始化配置
    config = SimulationConfig()
    
    # 2. 构建环境模块
    map_manager = MapManager(config)
    wind_model = WindModelFactory.create('slope', config, bounds=map_manager.get_bounds())
    
    # 3. 构建代理层与物理引擎
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    
    # 4. 组装规划器
    planner = AStarPlanner(config, estimator, physics)
    
    # 5. 设置起终点
    min_x, max_x, min_y, max_y = estimator.get_bounds()
    start_point = (min_x * 0.8, min_y * 0.8)
    goal_point  = (max_x * 0.8, max_y * 0.8)
    
    paths = {}
    
    # --- 实验 A: 传统最短路径 ---
    print("\n--- 正在计算传统路径 (Shortest) ---")
    config.k_wind = 0.0
    paths['Shortest Distance'] = planner.search(start_point, goal_point)
    
    # 【新增：坠机模拟逻辑】
    # 如果传统算法返回 None (说明飞不过去)，我们就强行飞一次直线，看看在哪坠机
    if paths['Shortest Distance'] is None:
        print(">>> 传统路径无法通过，正在执行【坠机模拟】...")
        crashed_path = []
        
        # 1. 生成直线上的采样点
        steps = 100
        z_start = map_manager.get_altitude(*start_point) + 50
        z_end = map_manager.get_altitude(*goal_point) + 50
        
        for i in range(steps):
            # 线性插值
            r = i / steps
            cx = start_point[0] + (goal_point[0] - start_point[0]) * r
            cy = start_point[1] + (goal_point[1] - start_point[1]) * r
            cz = z_start + (z_end - z_start) * r
            
            crashed_path.append((cx, cy, cz))
            
            # 2. 检查物理限制
            if i > 0:
                prev = crashed_path[-1]
                # 计算向量
                move_vec = np.array([cx - prev[0], cy - prev[1]])
                norm = np.linalg.norm(move_vec)
                if norm > 0: move_vec = move_vec / norm * config.drone_speed
                
                # 获取风场 (注意高度转换)
                ground_h = map_manager.get_altitude(cx, cy)
                wind = estimator.get_wind(cx, cy, cz)
                
                # 计算功率
                power = physics.calculate_power(move_vec, wind)
                
                # 3. 判定坠机条件
                # 条件A: 撞山 (高度 < 地面)
                if cz < ground_h:
                    print(f"❌ 坠机警报：在 ({cx:.1f}, {cy:.1f}) 处撞山！")
                    paths['Crashed (Terrain)'] = crashed_path # 记录尸体位置
                    break
                
                # 条件B: 功率过载 (逆风太大)
                if power == float('inf') or power > config.max_power:
                    print(f"❌ 坠机警报：在 ({cx:.1f}, {cy:.1f}) 处电机过载烧毁！(Power > {config.max_power}W)")
                    paths['Crashed (Overload)'] = crashed_path
                    break
        
    # --- 实验 B: 能量与风场感知路径 ---
    print("\n--- 正在计算智能路径 (Energy Aware) ---")
    config.k_wind = 1.0
    paths['Energy Optimized'] = planner.search(start_point, goal_point)
    
    # 6. 可视化 (先画剖面图，关掉后再弹地图)
    # --- 绘制高度剖面图 (Side View) ---
    if 'Energy Optimized' in paths and paths['Energy Optimized']:
        path = paths['Energy Optimized']
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        path_z = [p[2] for p in path]
        
        dist = [0]
        for i in range(1, len(path)):
            d = np.sqrt((path_x[i]-path_x[i-1])**2 + (path_y[i]-path_y[i-1])**2)
            dist.append(dist[-1] + d)
            
        terrain_z = []
        for x, y in zip(path_x, path_y):
            terrain_z.append(map_manager.get_altitude(x, y))
            
        plt.figure(figsize=(10, 4))
        plt.fill_between(dist, 0, terrain_z, color='gray', alpha=0.5, label='Terrain')
        plt.plot(dist, path_z, 'g-', linewidth=2, label='Drone (3D)')
        plt.title("Flight Elevation Profile")
        plt.legend()
        plt.show() # <--- 这个窗口关掉后，才会显示下一张大图

    # 最后显示大地图
    vis = Visualizer(config, estimator)
    vis.plot_simulation(paths, start_point, goal_point)

if __name__ == "__main__":
    main()