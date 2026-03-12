import numpy as np
import matplotlib.pyplot as plt
from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.physics import PhysicsEngine
from core.estimator import StateEstimator
from core.planner import AStarPlanner
from utils.visualizer_core import Visualizer

def main():
    # 1. 初始化
    config = SimulationConfig()
    map_manager = MapManager(config)
    wind_model = WindModelFactory.create('slope', config)
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    planner = AStarPlanner(config, estimator, physics)
    
    # 2. 设置起终点
    min_x, max_x, min_y, max_y = estimator.get_bounds()
    start_point = (min_x * 0.8, min_y * 0.8)
    goal_point  = (max_x * 0.8, max_y * 0.8)
    
    paths = {}
    
    # ==========================================
    # 实验 A: 传统算法 (傻瓜式贴地直飞)
    # ==========================================
    print("\n--- 正在计算传统路径 (Terrain Following) ---")
    
    # 模拟逻辑：在地图平面上画一条直线，但是高度始终保持离地 50米
    # 看看它会在哪里因为功率过大而掉下来
    crashed_path = []
    steps = 300 # 采样点多一点，曲线更平滑
    
    for i in range(steps + 1):
        # 1. 计算平面坐标 (走直线)
        r = i / steps
        cx = start_point[0] + (goal_point[0] - start_point[0]) * r
        cy = start_point[1] + (goal_point[1] - start_point[1]) * r
        
        # 2. 计算高度 (傻瓜跟随：地面 + 50米)
        ground_h = map_manager.get_altitude(cx, cy)
        cz = ground_h + 50.0
        
        curr_pos = (cx, cy, cz)
        crashed_path.append(curr_pos)
        
        # 3. 物理检查 (从第2个点开始)
        if i > 0:
            prev = crashed_path[-2]
            # 【这里修正了】把 curr 改成了 curr_pos
            curr = curr_pos 
            
            # 计算位移和速度向量
            dist_xyz = np.linalg.norm(np.array(curr) - np.array(prev))
            dist_xy = np.linalg.norm(np.array(curr[:2]) - np.array(prev[:2]))
            
            if dist_xyz <= 0: continue
            
            # 估算垂直速度 vz (爬升率)
            time_step = dist_xy / config.drone_speed if dist_xy > 0 else 0.1
            
            # 这里的 move_vec 是单位向量 * 速度，用于算功率
            v_ground_vec = np.array([cx-prev[0], cy-prev[1]]) 
            if np.linalg.norm(v_ground_vec) > 0:
                v_ground_vec = v_ground_vec / np.linalg.norm(v_ground_vec) * config.drone_speed
                
            v_z = (cz - prev[2]) / time_step
            
            # 获取风场
            wind = estimator.get_wind(cx, cy, cz)
            
            # 计算空气动力功率
            power_aero = physics.calculate_power(v_ground_vec, wind)
            
            # 计算重力功率
            power_gravity = 10.0 * v_z # mg * vz
            
            total_power = power_aero + power_gravity
            
            # 4. 判定坠机：功率过载
            if total_power > config.max_power:
                print(f"❌ 传统路径在第 {i} 步坠机！爬坡功率 {total_power:.0f}W > 极限 {config.max_power}W")
                paths['Traditional (Crashed)'] = crashed_path
                break
                
    if 'Traditional (Crashed)' not in paths:
        # 如果居然飞通了，也把它画出来
        paths['Traditional (Success)'] = crashed_path

    # ==========================================
    # 实验 B: 智能规划 (Energy Aware)
    # ==========================================
    print("\n--- 正在计算智能路径 (Energy Aware) ---")
    config.k_wind = 1.0 
    paths['Energy Optimized'] = planner.search(start_point, goal_point)
    
    # ==========================================
    # 统一可视化 (所有图一起出)
    # ==========================================
    
    # 1. 弹出的剖面图
    target_path = paths.get('Energy Optimized')
    if target_path:
        path_x = [p[0] for p in target_path]
        path_y = [p[1] for p in target_path]
        path_z = [p[2] for p in target_path]
        
        dist = [0]
        terrain_z = []
        for i in range(len(target_path)):
            if i > 0:
                d = np.sqrt((path_x[i]-path_x[i-1])**2 + (path_y[i]-path_y[i-1])**2)
                dist.append(dist[-1] + d)
            terrain_z.append(map_manager.get_altitude(path_x[i], path_y[i]))
            
        plt.figure("Flight Elevation Profile", figsize=(10, 4))
        plt.fill_between(dist, 0, terrain_z, color='gray', alpha=0.5, label='Terrain')
        plt.plot(dist, path_z, 'g-', linewidth=2, label='Drone (3D)')
        plt.title("Flight Elevation Profile (Side View)")
        plt.xlabel("Distance (m)")
        plt.ylabel("Altitude (m)")
        plt.legend()

    # 2. 弹出的地图窗口
    vis = Visualizer(config, estimator)
    vis.plot_simulation(paths, start_point, goal_point)
    
    # 3. 最后一次性显示所有窗口
    print("正在显示所有图表...")
    plt.show()

if __name__ == "__main__":
    main()