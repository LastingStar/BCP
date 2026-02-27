import numpy as np
import matplotlib.pyplot as plt
from configs.config import SimulationConfig
from core.estimator import StateEstimator
from typing import Dict, List, Tuple

class Visualizer:
    """可视化工具：负责渲染地形、风场和路径"""
    
    def __init__(self, config: SimulationConfig, estimator: StateEstimator):
        self.config = config
        self.estimator = estimator
        self.map = estimator.map

    def plot_simulation(self, paths: Dict[str, List[Tuple[float, float]]], start: Tuple[float, float], goal: Tuple[float, float]):
        plt.figure(figsize=(12, 10))
        
        # 1. 绘制地形
        plt.contourf(self.map.X, self.map.Y, self.map.dem, 40, cmap='gist_earth', alpha=0.8)
        plt.colorbar(label='Altitude (m)')
        
        # 2. 提取并绘制风场 (为了性能，直接使用 map 的梯度计算网格风场)
        skip = 15
        wind_u = np.zeros_like(self.map.X)
        wind_v = np.zeros_like(self.map.Y)
        
        # 批量获取风场用于绘图
        for i in range(0, self.map.size_y, skip):
            for j in range(0, self.map.size_x, skip):
                w = self.estimator.get_wind(self.map.x[j], self.map.y[i])
                wind_u[i, j] = w[0]
                wind_v[i, j] = w[1]

        plt.quiver(self.map.X[::skip, ::skip], self.map.Y[::skip, ::skip],
                   wind_u[::skip, ::skip], wind_v[::skip, ::skip],
                   color='white', alpha=0.5, scale=500, width=0.002)
        
        # 3. 绘制路径
        colors = ['r--', 'g-', 'b-.', 'y:']
        for idx, (label, path) in enumerate(paths.items()):
            if path:
                p = np.array(path)
                plt.plot(p[:, 0], p[:, 1], colors[idx % len(colors)], lw=3, label=label)
                
        # 4. 绘制起终点
        plt.scatter(*start, c='gold', s=200, marker='*', zorder=5, edgecolors='black', label='Start')
        plt.scatter(*goal, c='red', s=200, marker='X', zorder=5, edgecolors='black', label='Goal')
        
        plt.legend()
        plt.title(f"Drone Path Planning Simulation\nMap: {self.config.map_path}")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.show()