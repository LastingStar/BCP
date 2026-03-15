"""
可视化核心模块

此模块提供任务结果的可视化功能，包括地形图、风场图、
路径规划结果和剖面图的绘制。支持多种图表类型的组合输出。

主要组件：
- Visualizer: 可视化器主类
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go  # 🌟 新增 Plotly 库
from configs.config import SimulationConfig
from core.estimator import StateEstimator
from models.mission_models import MissionResult
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import splprep, splev


class Visualizer:
    """可视化工具：负责渲染地形、风场、路径、剖面图与禁飞区"""

    def __init__(self, config: SimulationConfig, estimator: StateEstimator):
        self.config = config
        self.estimator = estimator
        self.map = estimator.map

    def _compute_wind_grid(self, step: int = 2, t_s: float = 0.0):
        """预计算指定时刻的风场网格。"""
        X_sub = self.map.X[::step, ::step]
        Y_sub = self.map.Y[::step, ::step]
        speed_grid = np.zeros_like(X_sub, dtype=float)
        u_grid = np.zeros_like(X_sub, dtype=float)
        v_grid = np.zeros_like(X_sub, dtype=float)

        rows, cols = self.map.size_y, self.map.size_x
        for i in range(0, rows, step):
            for j in range(0, cols, step):
                w = self.estimator.get_wind(
                    self.map.x[j],
                    self.map.y[i],
                    z=-1.0,
                    t_s=t_s
                )
                speed_grid[i // step, j // step] = np.linalg.norm(w)
                u_grid[i // step, j // step] = w[0]
                v_grid[i // step, j // step] = w[1]

        return X_sub, Y_sub, speed_grid, u_grid, v_grid

    def _plot_points(self, ax, start, goal):
        """辅助函数：画起终点"""
        ax.scatter(*start, c='gold', s=180, marker='*', zorder=10,
                   edgecolors='black', label='Start')
        ax.scatter(*goal, c='red', s=180, marker='X', zorder=10,
                   edgecolors='black', label='Goal')

    def _smooth_path(self, arr: np.ndarray, s: float = 2.0) -> np.ndarray:
        """🌟 B样条曲线平滑滤波 (工业级防弹版)"""
        if len(arr) < 4:  # 点太少无法平滑，直接返回
            return arr
            
        # 1. 核心修复：移除相邻的完全重复点！
        diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        valid_idx = np.insert(diffs > 1e-5, 0, True)
        arr_clean = arr[valid_idx]
        
        # 清理后如果点不够了，直接返回原数组
        if len(arr_clean) < 4:
            return arr_clean
            
        x, y = arr_clean[:, 0], arr_clean[:, 1]
        
        # 2. 异常捕获
        try:
            tck, u = splprep([x, y], s=s)
            u_new = np.linspace(0, 1.0, max(300, len(arr_clean) * 3))
            x_new, y_new = splev(u_new, tck)
            return np.column_stack((x_new, y_new))
        except Exception as e:
            print(f"\n[⚠️ 视觉模块提示] 某段路径几何结构过于特殊，已自动跳过平滑处理。")
            return arr_clean
        
    def _path_xy_to_km(self, path_xy: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        if not path_xy: return None
        arr = np.array(path_xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2: return None
        return self._smooth_path(arr / 1000.0)

    def _path_xyz_to_xy_km(self, path_xyz: List[Tuple[float, float, float]]) -> Optional[np.ndarray]:
        if not path_xyz: return None
        arr = np.array(path_xyz, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2: return None
        return self._smooth_path(arr[:, :2] / 1000.0)

    def _draw_nfz(self, ax):
        if not self.config.enable_nfz: return
        has_nfz_legend = any('No-Fly Zone' in str(h.get_label()) for h in ax.get_legend_handles_labels()[0])
        for i, (cx_km, cy_km, r_km) in enumerate(self.config.nfz_list_km):
            label = 'No-Fly Zone' if not has_nfz_legend and i == 0 else None
            circle = patches.Circle((cx_km, cy_km), r_km, linewidth=2, edgecolor='red', facecolor='red', 
                                    alpha=0.3, hatch='//', zorder=5, label=label)
            ax.add_patch(circle)
            ax.text(cx_km, cy_km, 'NFZ', color='darkred', fontsize=12, fontweight='bold', ha='center', va='center', zorder=6)

    def _draw_storms_static(self, ax, duration_s=300.0):
        if not self.config.enable_storms or not hasattr(self.estimator.wind, 'storm_manager'): return
        has_storm_legend = any('Storm' in str(h.get_label()) for h in ax.get_legend_handles_labels()[0])
        for i, storm in enumerate(self.estimator.wind.storm_manager.storms):
            cx_km, cy_km = storm.center_xy[0] / 1000.0, storm.center_xy[1] / 1000.0
            r_km, vx_km, vy_km = storm.radius_m / 1000.0, storm.velocity_xy[0] / 1000.0, storm.velocity_xy[1] / 1000.0
            label = 'Storm (t=0)' if not has_storm_legend and i == 0 else None
            circle = patches.Circle((cx_km, cy_km), r_km, linewidth=1.5, edgecolor='navy', facecolor='navy', 
                                    alpha=0.25, zorder=4, label=label)
            ax.add_patch(circle)
            end_x, end_y = cx_km + vx_km * duration_s, cy_km + vy_km * duration_s
            ax.annotate('', xy=(end_x, end_y), xytext=(cx_km, cy_km),
                        arrowprops=dict(arrowstyle="->", color="navy", ls="dashed", lw=1.5, alpha=0.8), zorder=5)
            ax.text(cx_km, cy_km, 'Storm', color='midnightblue', fontsize=11, ha='center', va='center', zorder=6)

    def plot_mission_comparison(
        self,
        mission_result: MissionResult,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        wind_time_s: float = 0.0,
        show_replanned_paths: bool = True,
        save_path: Optional[str] = None,
    ):
        """M5：动态任务三路径对比图"""
        print("正在计算对比图风场数据...")
        X_sub, Y_sub, speed_grid, u_grid, v_grid = self._compute_wind_grid(step=2, t_s=wind_time_s)

        plt.figure("Terrain and Paths", figsize=(10, 8))
        ax1 = plt.gca()
        contour = ax1.contourf(self.map.X / 1000.0, self.map.Y / 1000.0, self.map.dem, 50, cmap='gist_earth', alpha=0.82)
        cbar = plt.colorbar(contour, ax=ax1, pad=0.02)
        cbar.set_label("Altitude (m)")
        self._draw_nfz(ax1)
        self._draw_storms_static(ax1, duration_s=400.0)

        paths_to_plot = [
            ("Initial Path (No Wind)", mission_result.initial_no_wind_path_xyz, 'red', '--'),
            ("Initial Path (With Wind)", mission_result.initial_wind_path_xyz, 'deepskyblue', '-.'),
            ("Executed Path (Dynamic Replanning)", mission_result.actual_flown_path_xyz, 'lime', '-')
        ]
        for label, path_xyz, color, linestyle in paths_to_plot:
            p = self._path_xyz_to_xy_km(path_xyz)
            if p is not None and len(p) >= 2:
                ax1.plot(p[:, 0], p[:, 1], color=color, linestyle=linestyle, linewidth=2.2, label=label)

        if show_replanned_paths and mission_result.replanned_paths_xyz:
            for idx, path_xyz in enumerate(mission_result.replanned_paths_xyz):
                p = self._path_xyz_to_xy_km(path_xyz)
                if p is not None and len(p) >= 2:
                    replan_label = "Intermediate Replans" if idx == 0 else None
                    ax1.plot(p[:, 0], p[:, 1], color='lightgray', linestyle='--', linewidth=1.0, alpha=0.6, label=replan_label)

        self._plot_points(ax1, (start_xy[0] / 1000.0, start_xy[1] / 1000.0), (goal_xy[0] / 1000.0, goal_xy[1] / 1000.0))
        ax1.set_title("Terrain and Paths", fontsize=14)
        ax1.set_xlabel("X (km)")
        ax1.set_ylabel("Y (km)")
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.2)
        plt.tight_layout()

        plt.figure("Wind Field RGB", figsize=(10, 8))
        ax2 = plt.gca()
        wind_speed = np.sqrt(u_grid**2 + v_grid**2)
        im = ax2.imshow(wind_speed, extent=[X_sub.min()/1000, X_sub.max()/1000, Y_sub.min()/1000, Y_sub.max()/1000], 
                        origin='lower', cmap='turbo', alpha=0.8)
        cbar2 = plt.colorbar(im, ax=ax2, pad=0.02)
        cbar2.set_label("Wind Speed (m/s)")
        ax2.streamplot(X_sub / 1000.0, Y_sub / 1000.0, u_grid, v_grid, density=1.0, color='white', linewidth=0.8, arrowsize=1.2)
        self._draw_nfz(ax2)
        self._draw_storms_static(ax2, duration_s=400.0)
        self._plot_points(ax2, (start_xy[0] / 1000.0, start_xy[1] / 1000.0), (goal_xy[0] / 1000.0, goal_xy[1] / 1000.0))
        ax2.set_title("Wind Field RGB", fontsize=14)
        ax2.set_xlabel("X (km)")
        ax2.set_ylabel("Y (km)")
        plt.tight_layout()

        plt.figure("Elevation Profile", figsize=(10, 6))
        ax3 = plt.gca()
        path_xyz = mission_result.actual_flown_path_xyz
        if path_xyz and len(path_xyz) > 1:
            path_x, path_y, path_z = [p[0] for p in path_xyz], [p[1] for p in path_xyz], [p[2] for p in path_xyz]
            dist = [0.0]
            for i in range(1, len(path_xyz)):
                dist.append(dist[-1] + np.sqrt((path_x[i]-path_x[i-1])**2 + (path_y[i]-path_y[i-1])**2))
            
            terrain_z = [self.map.get_altitude(x, y) for x, y in zip(path_x, path_y)]
            ax3.fill_between(dist, 0, terrain_z, color='gray', alpha=0.5, label='Terrain Altitude')
            ax3.plot(dist, path_z, 'g-', linewidth=2.5, label='Drone Trajectory')
            ax3.plot(dist, [z + self.config.takeoff_altitude_agl for z in terrain_z], 
                     'r--', linewidth=1.5, alpha=0.7, label=f'+{self.config.takeoff_altitude_agl}m AGL Clearance')
            
            ax3.set_title("Elevation Profile (Executed Path)", fontsize=14)
            ax3.set_xlabel("Cumulative Horizontal Distance (m)")
            ax3.set_ylabel("Absolute Altitude (m)")
            ax3.grid(True, linestyle=':', alpha=0.6)
            ax3.legend(loc='lower right')
        else:
            ax3.text(0.5, 0.5, "No path data available", ha='center', va='center', transform=ax3.transAxes)
        plt.tight_layout()

        if save_path:
            import os
            base, ext = os.path.splitext(save_path)
            plt.figure("Terrain and Paths")
            plt.savefig(f"{base}_terrain{ext}", dpi=300, bbox_inches='tight')

    # ==============================================================
    # 🌟 新增：生成浏览器 3D 交互图的方法
    # ==============================================================
    def plot_3d_trajectory_plotly(self, mission_result: MissionResult, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]):
        """在默认浏览器中打开并渲染无畸变的 3D 可交互轨迹图"""
        path = np.array(mission_result.actual_flown_path_xyz)
        if len(path) < 2:
            print("[⚠️ 3D可视化提示] 实际飞行轨迹为空，无法渲染3D图形。")
            return

        print("正在渲染 3D 交互地图并打开浏览器...")
        
        # 降采样地形以保证浏览器流畅度
        step = max(1, self.map.size_x // 60)
        X, Y = np.meshgrid(self.map.x[::step], self.map.y[::step])
        Z = self.map.dem[::step, ::step]

        fig = go.Figure()
        
        # 1. 绘制 3D 地形曲面
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z, 
            colorscale='Earth', opacity=0.8, showscale=False, name='地形'
        ))
        
        # 2. 绘制 3D 飞行轨迹
        fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2], 
            mode='lines', line=dict(color='#00FF00', width=6), name='实际飞行 3D 航迹'
        ))
        
        # 3. 绘制起终点
        fig.add_trace(go.Scatter3d(
            x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]], 
            mode='markers', marker=dict(size=8, color='yellow', line=dict(width=2, color='black')), name='起点'
        ))
        fig.add_trace(go.Scatter3d(
            x=[path[-1, 0]], y=[path[-1, 1]], z=[path[-1, 2]], 
            mode='markers', marker=dict(size=8, color='red', line=dict(width=2, color='black')), name='终点'
        ))

        # 4. 配置场景 (锁定物理比例，防止地形变成针尖)
        fig.update_layout(
            title="真 3D 航迹数字孪生 (滚轮缩放，左键旋转)",
            scene=dict(
                xaxis_title='X 坐标 (m)', 
                yaxis_title='Y 坐标 (m)', 
                zaxis_title='海拔高度 (m)', 
                aspectmode='data'  # 🌟 核心：强制 3D 比例对应真实物理尺寸
            ),
            margin=dict(l=0, r=0, b=0, t=40), 
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        # 自动在浏览器中弹出
        fig.show()