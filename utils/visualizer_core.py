import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from configs.config import SimulationConfig
from core.estimator import StateEstimator
from models.mission_models import MissionResult
from typing import Dict, List, Tuple, Optional


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

    def _path_xy_to_km(self, path_xy: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """将二维路径转为 km 单位数组。"""
        if not path_xy:
            return None
        arr = np.array(path_xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        return arr / 1000.0

    def _path_xyz_to_xy_km(self, path_xyz: List[Tuple[float, float, float]]) -> Optional[np.ndarray]:
        """将三维路径投影到 XY 平面，并转为 km 单位数组。"""
        if not path_xyz:
            return None
        arr = np.array(path_xyz, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return arr[:, :2] / 1000.0

    def _draw_nfz(self, ax):
        """辅助方法：在地图上绘制禁飞区 (No-Fly Zones)"""
        if not self.config.enable_nfz:
            return
            
        for (cx_km, cy_km, r_km) in self.config.nfz_list_km:
            circle = patches.Circle(
                (cx_km, cy_km), r_km, 
                linewidth=2, edgecolor='red', facecolor='red', 
                alpha=0.3, hatch='//', zorder=5, label='No-Fly Zone'
            )
            ax.add_patch(circle)
            ax.text(cx_km, cy_km, 'NFZ', color='darkred', 
                    fontsize=12, fontweight='bold', ha='center', va='center', zorder=6)


    def _draw_storms_static(self, ax, duration_s=300.0):
        """🌟 辅助方法：在静态地图上绘制初始风暴位置及其未来预测轨迹"""
        if not self.config.enable_storms or not hasattr(self.estimator.wind, 'storm_manager'):
            return
            
        for storm in self.estimator.wind.storm_manager.storms:
            cx_km, cy_km = storm.center_xy[0] / 1000.0, storm.center_xy[1] / 1000.0
            r_km = storm.radius_m / 1000.0
            vx_km = storm.velocity_xy[0] / 1000.0
            vy_km = storm.velocity_xy[1] / 1000.0
            
            # 1. 画初始位置 (t=0 的蓝色半透明圈)
            circle = patches.Circle(
                (cx_km, cy_km), r_km, 
                linewidth=1.5, edgecolor='navy', facecolor='navy', 
                alpha=0.25, zorder=4, label='Storm (t=0)'
            )
            ax.add_patch(circle)
            
            # 2. 画预测轨迹箭头 (虚线箭头指向未来位置)
            # 假设预测 duration_s 秒后的位置
            end_x = cx_km + vx_km * duration_s
            end_y = cy_km + vy_km * duration_s
            
            ax.annotate('', xy=(end_x, end_y), xytext=(cx_km, cy_km),
                        arrowprops=dict(arrowstyle="->", color="navy", ls="dashed", lw=1.5, alpha=0.8),
                        zorder=5)
            
            # 加个雷暴小图标/文字
            ax.text(cx_km, cy_km, '⛈️ Storm', color='midnightblue', fontsize=11, ha='center', va='center', zorder=6)

    def plot_simulation(
        self,
        paths: Dict[str, List[Tuple[float, float, float]]],
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ):
        """原有静态可视化接口，保留兼容，并新增高度剖面图。"""
        print("正在计算全图风场数据...")
        X_sub, Y_sub, speed_grid, u_grid, v_grid = self._compute_wind_grid(step=2, t_s=0.0)

        # ==========================================
        # 新增：图 0 - 高度剖面图 (Elevation Profile)
        # ==========================================
        target_path_3d = None
        target_label = ""
        for label, path in paths.items():
            if path is not None and len(path) > 1 and len(path[0]) >= 3:
                target_path_3d = path
                target_label = label
                if 'Energy' in label:  # 优先画能量优化路径的剖面
                    break

        if target_path_3d:
            path_x = [p[0] for p in target_path_3d]
            path_y = [p[1] for p in target_path_3d]
            path_z = [p[2] for p in target_path_3d]
            
            dist = [0.0]
            for i in range(1, len(target_path_3d)):
                d = np.sqrt((path_x[i]-path_x[i-1])**2 + (path_y[i]-path_y[i-1])**2)
                dist.append(dist[-1] + d)
                
            terrain_z = []
            for x, y in zip(path_x, path_y):
                terrain_z.append(self.map.get_altitude(x, y))
                
            plt.figure("Flight Elevation Profile", figsize=(10, 4))
            
            plt.fill_between(dist, 0, terrain_z, color='gray', alpha=0.5, label='Terrain Altitude')
            plt.plot(dist, path_z, 'g-', linewidth=2.5, label=f'Drone Trajectory ({target_label})')
            
            plt.plot(dist, [z + self.config.takeoff_altitude_agl for z in terrain_z], 
                     'r--', linewidth=1, alpha=0.5, label=f'+{self.config.takeoff_altitude_agl}m AGL Clearance')

            plt.title(f"3D Flight Elevation Profile ({target_label})", fontsize=14)
            plt.xlabel("Cumulative Horizontal Distance (m)")
            plt.ylabel("Absolute Altitude (m)")
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='lower right')
            plt.tight_layout()

        # 图 1: 地形图
        plt.figure("Terrain Map", figsize=(10, 8))
        ax1 = plt.gca()
        plt.contourf(self.map.X / 1000.0, self.map.Y / 1000.0, self.map.dem, 50, cmap='gist_earth')
        plt.colorbar(label='Altitude (m)')
        self._draw_nfz(ax1)
        plt.title("Terrain Elevation Map", fontsize=14)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        self._plot_points(ax1, (start[0] / 1000.0, start[1] / 1000.0),
                          (goal[0] / 1000.0, goal[1] / 1000.0))
        plt.tight_layout()

        # 图 2: 风场图
        plt.figure("Wind Field", figsize=(10, 8))
        ax2 = plt.gca()
        plt.contourf(X_sub / 1000.0, Y_sub / 1000.0, speed_grid, 50, cmap='turbo')
        plt.colorbar(label='Wind Speed (m/s)')
        plt.streamplot(X_sub / 1000.0, Y_sub / 1000.0, u_grid, v_grid,
                       density=1.2, color='white', linewidth=0.6, arrowsize=1.0)
        self._draw_nfz(ax2)
        plt.title("Wind Field (Speed & Direction)", fontsize=14)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        self._plot_points(ax2, (start[0] / 1000.0, start[1] / 1000.0),
                          (goal[0] / 1000.0, goal[1] / 1000.0))
        plt.tight_layout()

        # 图 3: 路径结果
        plt.figure("Path Planning Result", figsize=(10, 8))
        ax3 = plt.gca()
        plt.contourf(self.map.X / 1000.0, self.map.Y / 1000.0, self.map.dem, 50,
                     cmap='gist_earth', alpha=0.6)
        plt.streamplot(X_sub / 1000.0, Y_sub / 1000.0, u_grid, v_grid,
                       density=0.8, color=(0, 0, 1, 0.4), linewidth=0.4)
        
        self._draw_nfz(ax3)
        # 🌟 渲染禁飞区
        self._draw_nfz(ax3)      # (或者 ax1, ax2, ax3 根据上下文)
        # 🌟 渲染移动风暴及轨迹
        self._draw_storms_static(ax3, duration_s=400.0)  # 预测400秒后的走势

        has_path = False
        for label, path in paths.items():
            p = self._path_xy_to_km(path)
            if p is not None:
                has_path = True
                if 'Energy' in label:
                    c, s, lw = 'lime', '-', 3
                else:
                    c, s, lw = 'red', '--', 2
                plt.plot(p[:, 0], p[:, 1], linestyle=s, color=c, linewidth=lw, label=label)

        if has_path:
            handles, labels = ax3.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=True, facecolor='black',
                       framealpha=0.6, labelcolor='white')

        plt.title("Path Planning Result", fontsize=14)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        self._plot_points(ax3, (start[0] / 1000.0, start[1] / 1000.0),
                          (goal[0] / 1000.0, goal[1] / 1000.0))
        plt.tight_layout()
        
        plt.show()

    # =========================================================================
    # 🌟 核心修复：把你之前丢失的 plot_mission_comparison 完整补回到了这里！
    # =========================================================================
    def plot_mission_comparison(
        self,
        mission_result: MissionResult,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        wind_time_s: float = 0.0,
        show_replanned_paths: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        M5：动态任务三路径对比图
        """
        print("正在计算对比图风场数据...")
        X_sub, Y_sub, speed_grid, u_grid, v_grid = self._compute_wind_grid(step=2, t_s=wind_time_s)

        # 图1: 地形 + 路线图
        plt.figure("Terrain and Paths", figsize=(10, 8))
        ax1 = plt.gca()
        contour = ax1.contourf(
            self.map.X / 1000.0,
            self.map.Y / 1000.0,
            self.map.dem,
            50,
            cmap='gist_earth',
            alpha=0.82
        )
        cbar = plt.colorbar(contour, ax=ax1, pad=0.02)
        cbar.set_label("Altitude (m)")
        self._draw_nfz(ax1)
        # 🌟 渲染禁飞区
        self._draw_nfz(ax1)      # (或者 ax1, ax2, ax3 根据上下文)
        # 🌟 渲染移动风暴及轨迹
        self._draw_storms_static(ax1, duration_s=400.0)  # 预测400秒后的走势

        # 绘制所有路径
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

        # 图2: 风场RGB图
        plt.figure("Wind Field RGB", figsize=(10, 8))
        ax2 = plt.gca()
        # 使用RGB颜色表示风速和方向（简化版：速度用亮度，方向用色相）
        # 这里用速度的颜色图，流线表示方向
        wind_speed = np.sqrt(u_grid**2 + v_grid**2)
        im = ax2.imshow(wind_speed, extent=[X_sub.min()/1000, X_sub.max()/1000, Y_sub.min()/1000, Y_sub.max()/1000], 
                        origin='lower', cmap='turbo', alpha=0.8)
        cbar2 = plt.colorbar(im, ax=ax2, pad=0.02)
        cbar2.set_label("Wind Speed (m/s)")
        ax2.streamplot(X_sub / 1000.0, Y_sub / 1000.0, u_grid, v_grid, density=1.0, color='white', linewidth=0.8, arrowsize=1.2)
        self._draw_nfz(ax2)
        self._plot_points(ax2, (start_xy[0] / 1000.0, start_xy[1] / 1000.0), (goal_xy[0] / 1000.0, goal_xy[1] / 1000.0))
        ax2.set_title("Wind Field RGB", fontsize=14)
        ax2.set_xlabel("X (km)")
        ax2.set_ylabel("Y (km)")
        plt.tight_layout()

        # 图3: 剖面图（无人机升降）
        plt.figure("Elevation Profile", figsize=(10, 6))
        ax3 = plt.gca()
        # 选择实际飞行路径进行剖面
        path_xyz = mission_result.actual_flown_path_xyz
        if path_xyz and len(path_xyz) > 1:
            path_x = [p[0] for p in path_xyz]
            path_y = [p[1] for p in path_xyz]
            path_z = [p[2] for p in path_xyz]
            
            dist = [0.0]
            for i in range(1, len(path_xyz)):
                d = np.sqrt((path_x[i]-path_x[i-1])**2 + (path_y[i]-path_y[i-1])**2)
                dist.append(dist[-1] + d)
            
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
            # 保存为多页PDF或单独文件
            import os
            base, ext = os.path.splitext(save_path)
            plt.figure("Terrain and Paths")
            plt.savefig(f"{base}_terrain{ext}", dpi=300, bbox_inches='tight')
            plt.figure("Wind Field RGB")
            plt.savefig(f"{base}_wind{ext}", dpi=300, bbox_inches='tight')
            plt.figure("Elevation Profile")
            plt.savefig(f"{base}_profile{ext}", dpi=300, bbox_inches='tight')

        plt.show()