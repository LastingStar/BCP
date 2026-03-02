from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.physics import PhysicsEngine
from core.estimator import StateEstimator
from core.planner import AStarPlanner
from utils.visualizer import Visualizer

def main():
    # 1. 初始化配置
    config = SimulationConfig()
    
    # 2. 构建环境模块
    map_manager = MapManager(config)
    wind_model = WindModelFactory.create('slope', config)
    
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
    
    # 实验 A: 传统最短路径 (不考虑风)
 #   config.k_wind = 0.0
 #   paths['Shortest Distance'] = planner.search(start_point, goal_point)
    
    # 实验 B: 能量与风场感知路径
 #   config.k_wind = 1.0
 #   paths['Energy Optimized'] = planner.search(start_point, goal_point)
    
    # 6. 可视化
    vis = Visualizer(config, estimator)
    vis.plot_simulation(paths, start_point, goal_point)

if __name__ == "__main__":
    main()