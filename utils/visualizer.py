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
        
        # 预计算风场数据 (通用数据)
        print("正在计算全图风场数据...")
        step = 2 
        X_sub = self.map.X[::step, ::step]
        Y_sub = self.map.Y[::step, ::step]
        speed_grid = np.zeros_like(X_sub)
        u_grid = np.zeros_like(X_sub)
        v_grid = np.zeros_like(X_sub)
        
        rows, cols = self.map.size_y, self.map.size_x
        for i in range(0, rows, step):
            for j in range(0, cols, step):
                w = self.estimator.get_wind(self.map.x[j], self.map.y[i])
                speed_grid[i//step, j//step] = np.linalg.norm(w)
                u_grid[i//step, j//step] = w[0]
                v_grid[i//step, j//step] = w[1]

        # ==========================================
        # 图 1: 纯地形图 (Terrain Elevation)
        # ==========================================
        plt.figure("Terrain Map", figsize=(10, 8)) # 新建独立窗口
        plt.contourf(self.map.X, self.map.Y, self.map.dem, 50, cmap='gist_earth')
        plt.colorbar(label='Altitude (m)')
        plt.title("Terrain Elevation Map", fontsize=14)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        self._plot_points(plt.gca(), start, goal)
        plt.tight_layout()

        # ==========================================
        # 图 2: 纯风场图 (Wind Field)
        # ==========================================
        plt.figure("Wind Field", figsize=(10, 8)) # 新建独立窗口
        # 背景风速
        plt.contourf(X_sub, Y_sub, speed_grid, 50, cmap='turbo')
        plt.colorbar(label='Wind Speed (m/s)')
        # 流线
        plt.streamplot(X_sub, Y_sub, u_grid, v_grid, density=1.2, color='white', linewidth=0.6, arrowsize=1.0)
        plt.title("Wind Field (Speed & Direction)", fontsize=14)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        self._plot_points(plt.gca(), start, goal)
        plt.tight_layout()

        # ==========================================
        # 图 3: 综合规划结果 (Path Planning)
        # ==========================================
        plt.figure("Path Planning Result", figsize=(10, 8)) # 新建独立窗口
        # 地形打底
        plt.contourf(self.map.X, self.map.Y, self.map.dem, 50, cmap='gist_earth', alpha=0.6)
        # 叠加流线 (半透明蓝色)
        plt.streamplot(X_sub, Y_sub, u_grid, v_grid, density=0.8, color=(0, 0, 1, 0.4), linewidth=0.4)
        
        # 绘制路径
        has_path = False
        for idx, (label, path) in enumerate(paths.items()):
            if path:
                has_path = True
                p = np.array(path)
                if 'Energy' in label:
                    c, s, lw = 'lime', '-', 3
                else:
                    c, s, lw = 'red', '--', 2
                
                plt.plot(p[:, 0], p[:, 1], linestyle=s, color=c, linewidth=lw, label=label)
        
        if has_path:
            plt.legend(loc='upper left', frameon=True, facecolor='black', framealpha=0.6, labelcolor='white')
            
        plt.title("Path Planning Result", fontsize=14)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        self._plot_points(plt.gca(), start, goal)
        plt.tight_layout()

        # 最后统一显示所有窗口
        plt.show()

    def _plot_points(self, ax, start, goal):
        """辅助函数：画起终点"""
        ax.scatter(*start, c='gold', s=200, marker='*', zorder=10, edgecolors='black', label='Start')
        ax.scatter(*goal, c='red', s=200, marker='X', zorder=10, edgecolors='black', label='Goal')