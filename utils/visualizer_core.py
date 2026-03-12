import numpy as np
import matplotlib.pyplot as plt
from configs.config import SimulationConfig
from core.estimator import StateEstimator
from models.mission_models import MissionResult
from typing import Dict, List, Tuple, Optional


class Visualizer:
    """可视化工具：负责渲染地形、风场和路径"""

    def __init__(self, config: SimulationConfig, estimator: StateEstimator):
        self.config = config
        self.estimator = estimator
        self.map = estimator.map

    def _compute_wind_grid(self, step: int = 2, t_s: float = 0.0):
        """
        预计算指定时刻的风场网格。
        """
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
        """
        将二维路径转为 km 单位数组。
        """
        if not path_xy:
            return None
        arr = np.array(path_xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        return arr / 1000.0

    def _path_xyz_to_xy_km(self, path_xyz: List[Tuple[float, float, float]]) -> Optional[np.ndarray]:
        """
        将三维路径投影到 XY 平面，并转为 km 单位数组。
        """
        if not path_xyz:
            return None
        arr = np.array(path_xyz, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return arr[:, :2] / 1000.0

    def plot_simulation(
        self,
        paths: Dict[str, List[Tuple[float, float]]],
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ):
        """
        原有静态可视化接口，保留兼容。
        """
        print("正在计算全图风场数据...")
        X_sub, Y_sub, speed_grid, u_grid, v_grid = self._compute_wind_grid(step=2, t_s=0.0)

        # 图 1: 地形图
        plt.figure("Terrain Map", figsize=(10, 8))
        plt.contourf(self.map.X / 1000.0, self.map.Y / 1000.0, self.map.dem, 50, cmap='gist_earth')
        plt.colorbar(label='Altitude (m)')
        plt.title("Terrain Elevation Map", fontsize=14)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        self._plot_points(plt.gca(), (start[0] / 1000.0, start[1] / 1000.0),
                          (goal[0] / 1000.0, goal[1] / 1000.0))
        plt.tight_layout()

        # 图 2: 风场图
        plt.figure("Wind Field", figsize=(10, 8))
        plt.contourf(X_sub / 1000.0, Y_sub / 1000.0, speed_grid, 50, cmap='turbo')
        plt.colorbar(label='Wind Speed (m/s)')
        plt.streamplot(X_sub / 1000.0, Y_sub / 1000.0, u_grid, v_grid,
                       density=1.2, color='white', linewidth=0.6, arrowsize=1.0)
        plt.title("Wind Field (Speed & Direction)", fontsize=14)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        self._plot_points(plt.gca(), (start[0] / 1000.0, start[1] / 1000.0),
                          (goal[0] / 1000.0, goal[1] / 1000.0))
        plt.tight_layout()

        # 图 3: 路径结果
        plt.figure("Path Planning Result", figsize=(10, 8))
        plt.contourf(self.map.X / 1000.0, self.map.Y / 1000.0, self.map.dem, 50,
                     cmap='gist_earth', alpha=0.6)
        plt.streamplot(X_sub / 1000.0, Y_sub / 1000.0, u_grid, v_grid,
                       density=0.8, color=(0, 0, 1, 0.4), linewidth=0.4)

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
            plt.legend(loc='upper left', frameon=True, facecolor='black',
                       framealpha=0.6, labelcolor='white')

        plt.title("Path Planning Result", fontsize=14)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        self._plot_points(plt.gca(), (start[0] / 1000.0, start[1] / 1000.0),
                          (goal[0] / 1000.0, goal[1] / 1000.0))
        plt.tight_layout()
        plt.show()

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

        对比内容：
        1. 无风初始路径
        2. 有风初始路径
        3. 动态重规划后的最终实际飞行轨迹

        可选：
        - 绘制每次重规划路径（淡灰色）
        """
        print("正在计算对比图风场数据...")
        X_sub, Y_sub, speed_grid, u_grid, v_grid = self._compute_wind_grid(step=2, t_s=wind_time_s)

        fig, ax = plt.subplots(figsize=(11, 8))

        # 地形底图
        contour = ax.contourf(
            self.map.X / 1000.0,
            self.map.Y / 1000.0,
            self.map.dem,
            50,
            cmap='gist_earth',
            alpha=0.82
        )
        cbar = plt.colorbar(contour, ax=ax, pad=0.02)
        cbar.set_label("Altitude (m)")

        # 风场流线（使用初始时刻或指定时刻）
        ax.streamplot(
            X_sub / 1000.0,
            Y_sub / 1000.0,
            u_grid,
            v_grid,
            density=0.9,
            color=(0.1, 0.2, 0.9, 0.35),
            linewidth=0.55,
            arrowsize=0.9
        )

        # 可选：中间每次重规划路径（淡灰）
        if show_replanned_paths and mission_result.replanned_paths_xyz:
            for idx, path_xyz in enumerate(mission_result.replanned_paths_xyz):
                p = self._path_xyz_to_xy_km(path_xyz)
                if p is not None and len(p) >= 2:
                    label = "Intermediate Replans" if idx == 0 else None
                    ax.plot(
                        p[:, 0], p[:, 1],
                        color='lightgray',
                        linestyle='--',
                        linewidth=1.0,
                        alpha=0.6,
                        label=label
                    )

        # 1) 无风初始路径
        p_nowind = self._path_xyz_to_xy_km(mission_result.initial_no_wind_path_xyz)
        if p_nowind is not None and len(p_nowind) >= 2:
            ax.plot(
                p_nowind[:, 0], p_nowind[:, 1],
                color='red',
                linestyle='--',
                linewidth=2.2,
                label='Initial Path (No Wind)'
            )

        # 2) 有风初始路径
        p_wind = self._path_xyz_to_xy_km(mission_result.initial_wind_path_xyz)
        if p_wind is not None and len(p_wind) >= 2:
            ax.plot(
                p_wind[:, 0], p_wind[:, 1],
                color='deepskyblue',
                linestyle='-.',
                linewidth=2.2,
                label='Initial Path (With Wind)'
            )

        # 3) 动态实际飞行轨迹
        p_actual = self._path_xyz_to_xy_km(mission_result.actual_flown_path_xyz)
        if p_actual is not None and len(p_actual) >= 2:
            ax.plot(
                p_actual[:, 0], p_actual[:, 1],
                color='lime',
                linestyle='-',
                linewidth=3.0,
                label='Executed Path (Dynamic Replanning)'
            )

        # 起终点
        self._plot_points(
            ax,
            (start_xy[0] / 1000.0, start_xy[1] / 1000.0),
            (goal_xy[0] / 1000.0, goal_xy[1] / 1000.0)
        )

        # 标题与统计摘要
        title = (
            "Mission Path Comparison\n"
            f"Success={mission_result.success}, "
            f"Replans={mission_result.total_replans}, "
            f"Time={mission_result.total_mission_time_s:.1f}s, "
            f"Energy={mission_result.total_energy_used_j:.1f}J"
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        ax.grid(alpha=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()