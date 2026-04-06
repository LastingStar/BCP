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
            print(f"\n[视觉模块提示] 某段路径几何结构过于特殊，已自动跳过平滑处理。")
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

    # --- 在 utils/visualizer_core.py 中替换 _draw_storms_static ---
    def _draw_storms_static(self, ax, duration_s=300.0, t_s=0.0): # 🌟 传入查询时间
        if not self.config.enable_storms or not hasattr(self.estimator.wind, 'storm_manager'): return
        has_storm_legend = any('Storm' in str(h.get_label()) for h in ax.get_legend_handles_labels()[0])
        
        active_storms = self.estimator.wind.storm_manager.get_active_storms(t_s)
        for i, storm in enumerate(active_storms):
            current_center = storm.center_at(t_s) # 获取该时刻的真实位置
            cx_km, cy_km = current_center[0] / 1000.0, current_center[1] / 1000.0
            r_km, vx_km, vy_km = storm.radius_m / 1000.0, storm.velocity_xy[0] / 1000.0, storm.velocity_xy[1] / 1000.0
            
            label = 'Storm (t=0)' if not has_storm_legend and i == 0 else None
            circle = patches.Circle((cx_km, cy_km), r_km, linewidth=1.5, edgecolor='navy', facecolor='navy', 
                                    alpha=0.25, zorder=4, label=label)
            ax.add_patch(circle)
            end_x, end_y = cx_km + vx_km * duration_s, cy_km + vy_km * duration_s
            ax.annotate('', xy=(end_x, end_y), xytext=(cx_km, cy_km),
                        arrowprops=dict(arrowstyle="->", color="navy", ls="dashed", lw=1.5, alpha=0.8), zorder=5)
            ax.text(cx_km, cy_km, 'Storm', color='midnightblue', fontsize=11, ha='center', va='center', zorder=6)

    def _draw_base_terrain(self, ax):
        """绘制基础地形轮廓的公用方法"""
        contour = ax.contourf(self.map.X / 1000.0, self.map.Y / 1000.0, self.map.dem, 50, cmap='gist_earth', alpha=0.82)
        return contour

    def _generate_straight_baseline(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> List[Tuple[float, float, float]]:
        """🌟 升级版基线：地形跟随 (Terrain-Following) 航线"""
        steps = 100
        path = []
        for i in range(steps + 1):
            r = i / steps
            x = start_xy[0] + (goal_xy[0] - start_xy[0]) * r
            y = start_xy[1] + (goal_xy[1] - start_xy[1]) * r
            
            # 核心改变：获取当前 (x, y) 的真实地形高度，加上固定的起飞离地间隙 (AGL)
            # 这样无人机就会像过山车一样，老老实实地贴着山体表面爬升和下降！
            ground_z = self.map.get_altitude(x, y)
            z = ground_z + self.config.takeoff_altitude_agl
            
            path.append((x, y, z))
        return path
    def plot_elevation_profile(self, mission_result: MissionResult, save_dir: str, method_name: str = ""):
        """独立的高度剖面图绘制函数 (复原版)"""
        fig_name = "analysis_03_elevation_profile"
        fig = plt.figure(fig_name, figsize=(10, 6))
        ax = plt.gca()
        
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
            
            # 1. 画地形阴影
            ax.fill_between(dist, 0, terrain_z, color='gray', alpha=0.5, label='Terrain Altitude')
            # 2. 画实际飞行轨迹
            ax.plot(dist, path_z, 'g-', linewidth=2.5, label=f'Trajectory ({method_name})')
            # 3. 画最低安全边界线
            safe_z = [z + self.config.takeoff_altitude_agl for z in terrain_z]
            ax.plot(dist, safe_z, 'r--', linewidth=1.5, alpha=0.7, label=f'+{self.config.takeoff_altitude_agl}m AGL Clearance')

            # 4. 如果坠机了，在最后标个骷髅/大叉
            if not mission_result.success:
                ax.scatter(dist[-1], path_z[-1], color='black', marker='X', s=150, zorder=5)
                ax.text(dist[-1], path_z[-1] + 30, "CRASH", color='red', fontweight='bold', ha='center')

            ax.set_title(f"Elevation Profile - {method_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Cumulative Horizontal Distance (m)", fontsize=12)
            ax.set_ylabel("Absolute Altitude (m)", fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower right')
        else:
            ax.text(0.5, 0.5, "No path data available", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
        plt.tight_layout()
        if save_dir:
            s_path = os.path.join(save_dir, f"{fig_name}.png")
            fig.savefig(s_path, dpi=300, bbox_inches='tight')
            print(f"已保存高度剖面图: {s_path}")
        plt.close(fig) # 防止内存泄漏

    def plot_swarm_elevation_profile(self, mission_result: MissionResult, save_dir: str):
        """集群专属：基于同步采样快照的多机时间-高度剖面图"""
        fig_name = "swarm_elevation_profile"
        fig = plt.figure(fig_name, figsize=(12, 6))
        ax = plt.gca()

        times = getattr(mission_result, "swarm_time_history", [])
        master_path = getattr(mission_result, "master_sync_path_xyz", [])
        scout_path = getattr(mission_result, "scout_flown_path_xyz", [])
        relay_path = getattr(mission_result, "relay_flown_path_xyz", [])
        support_path = getattr(mission_result, "support_flown_path_xyz", [])

        if not times or not master_path:
            plt.close(fig)
            return

        sample_count = min(len(times), len(master_path))
        if sample_count < 2:
            plt.close(fig)
            return

        times = times[:sample_count]
        master_path = master_path[:sample_count]

        master_z = [p[2] for p in master_path]
        terrain_z = [self.map.get_altitude(p[0], p[1]) for p in master_path]
        ax.fill_between(times, 0, terrain_z, color='gray', alpha=0.35, label='Terrain under Master')
        ax.plot(times, master_z, color='lime', linewidth=3.0, label='Master Altitude')

        if len(scout_path) >= sample_count:
            ax.plot(
                times,
                [p[2] for p in scout_path[:sample_count]],
                color='magenta',
                linestyle='--',
                linewidth=2.0,
                alpha=0.85,
                label='Scout Altitude',
            )

        if len(relay_path) >= sample_count:
            ax.plot(
                times,
                [p[2] for p in relay_path[:sample_count]],
                color='dodgerblue',
                linestyle='-.',
                linewidth=2.0,
                alpha=0.85,
                label='Relay Altitude',
            )

        if len(support_path) >= sample_count:
            ax.plot(
                times,
                [p[2] for p in support_path[:sample_count]],
                color='darkorange',
                linestyle=':',
                linewidth=2.5,
                alpha=0.9,
                label='Support Altitude',
            )

        if not mission_result.success:
            ax.scatter(times[-1], master_z[-1], color='black', marker='X', s=120, zorder=6)

        ax.set_title("FANET Swarm 4D Elevation Profile (Time vs Altitude)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Flight Time (Seconds)", fontsize=12)
        ax.set_ylabel("Absolute Altitude (m)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

        plt.tight_layout()
        if save_dir:
            s_path = os.path.join(save_dir, f"{fig_name}.png")
            fig.savefig(s_path, dpi=300, bbox_inches='tight')
            print(f"✅ 集群高度剖面图已保存至: {s_path}")
        plt.close(fig)

    # ==============================================================
    # 🌟 新增：生成并保存纯地形平面图的方法
    # ==============================================================
    def plot_pure_terrain(self, save_filename: str = "pure_terrain.png"):
        """
        绘制并保存纯净的地形平面图（不带路径、起点、终点、风暴、NFZ）。
        """
        print(f"--- 正在绘制纯地形图并保存为 {save_filename} ---")
        # 创建一个新的 Figure，避免污染其他图
        fig_pure = plt.figure("Pure Terrain Map", figsize=(10, 8))
        ax = fig_pure.gca()

        # 1. 仅绘制地形
        contour = self._draw_base_terrain(ax)
        
        # 2. 添加色柱
        cbar = fig_pure.colorbar(contour, ax=ax, pad=0.02)
        cbar.set_label("Altitude (m)")

        # 3. 设置坐标轴和标题
        ax.set_title("Pure Terrain Elevation Map", fontsize=14)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.grid(alpha=0.15) # 添加很淡的网格
        ax.set_aspect('equal') # 保持物理比例

        plt.tight_layout()

        # 4. 保存到本地
        fig_pure.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"纯地形图已保存。")
        # 注意：这里不显式调用 plt.close()，让它保持在内存中直到 main 调用的 plt.show()，
        # 或者如果你不想看弹窗只想保存，可以在这里 close。根据 main 的逻辑，这里保持原样。

    def plot_mission_comparison(
        self,
        mission_result: MissionResult,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        wind_time_s: float = 0.0,
        show_replanned_paths: bool = True,
        save_dir: Optional[str] = None, 
        physics_engine = None  # 🌟 新增参数：传入物理引擎以计算传统路径会在哪里坠机
    ):
        """M5：动态任务多维度对比图，并在地图上标出传统路径坠机点"""
        print("正在计算对比图风场数据...")
        X_sub, Y_sub, speed_grid, u_grid, v_grid = self._compute_wind_grid(step=2, t_s=wind_time_s)

        file_prefix = ""
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            file_prefix = save_dir

        # ---------------------------------------------------------
        # Figure 1: Terrain and Paths
        # ---------------------------------------------------------
        fig1_name = "analysis_01_terrain_and_paths"
        plt.figure(fig1_name, figsize=(10, 8))
        ax1 = plt.gca()
        contour = self._draw_base_terrain(ax1)
        cbar = plt.colorbar(contour, ax=ax1, pad=0.02)
        cbar.set_label("Altitude (m)")
        
        self._draw_nfz(ax1)
        self._draw_storms_static(ax1, duration_s=400.0)

        # 🌟 核心修改：如果传统路径为空（规划失败），我们强行给它一条纯直飞航线去送死！
        baseline_path = mission_result.initial_no_wind_path_xyz
        if not baseline_path or len(baseline_path) < 2:
            print("[视觉模块] 传统A*未找到路径，正在生成直线硬闯基线...")
            baseline_path = self._generate_straight_baseline(start_xy, goal_xy)

        baseline_crash_pt = None
        crash_reason = ""
        
        if physics_engine and baseline_path:
            survived_path = [baseline_path[0]]
            curr_time = 0.0
            for i in range(len(baseline_path) - 1):
                p0, p1 = np.array(baseline_path[i]), np.array(baseline_path[i+1])

                # 1. 检测是否撞山或进入禁飞区
                if self.map.is_collision(p1[0], p1[1], p1[2]):
                    baseline_crash_pt = p1
                    crash_reason = "💥 CRASH!\n(NFZ/Terrain)"
                    survived_path.append(p1)
                    break

                # 2. 检测风场阻力是否导致电机过载
                mid = (p0 + p1) / 2.0
                w2d = self.estimator.get_wind(mid[0], mid[1], mid[2], t_s=curr_time)
                _, seg_t, seg_p = physics_engine.estimate_segment_energy(
                    p0, p1, np.array([w2d[0], w2d[1], 0.0]), self.config.cruise_speed_mps
                )
                if seg_p > self.config.max_power or seg_p == float('inf'):
                    baseline_crash_pt = p1 # 记录坠机坐标
                    crash_reason = "CRASH!\n(Overload)"
                    survived_path.append(p1)
                    break # 截断路径
                
                survived_path.append(p1)
                curr_time += seg_t
            baseline_path = survived_path # 替换为截断后的真实物理路径

        # 绘制路径
        paths_to_plot = [
            ("Traditional Path (Baseline)", baseline_path, 'red', '--'),
            ("Initial 4D Plan (Pre-flight)", mission_result.initial_wind_path_xyz, 'deepskyblue', '-.'),
            ("Our Executed 4D Path", mission_result.actual_flown_path_xyz, 'lime', '-')
        ]

        for label, path_xyz, color, linestyle in paths_to_plot:
            p = self._path_xyz_to_xy_km(path_xyz)
            if p is not None and len(p) >= 2:
                ax1.plot(p[:, 0], p[:, 1], color=color, linestyle=linestyle, linewidth=2.5, label=label)

        # 🌟 在地图上绘制传统路径的坠机爆炸标志
        if baseline_crash_pt is not None:
            cx, cy = baseline_crash_pt[0]/1000.0, baseline_crash_pt[1]/1000.0
            ax1.scatter(cx, cy, color='red', marker='*', s=400, edgecolor='black', zorder=20)
            ax1.text(cx, cy + 0.3, crash_reason, color='red', fontsize=12, fontweight='bold', ha='center', va='bottom', zorder=21)

        if show_replanned_paths and mission_result.replanned_paths_xyz:
            for idx, path_xyz in enumerate(mission_result.replanned_paths_xyz):
                p = self._path_xyz_to_xy_km(path_xyz)
                if p is not None and len(p) >= 2:
                    replan_label = "Dynamic Replans" if idx == 0 else None
                    ax1.plot(p[:, 0], p[:, 1], color='lightgray', linestyle='--', linewidth=1.0, alpha=0.6, label=replan_label)

        self._plot_points(ax1, (start_xy[0] / 1000.0, start_xy[1] / 1000.0), (goal_xy[0] / 1000.0, goal_xy[1] / 1000.0))
        ax1.set_title("Flight Path Comparison: Traditional vs 4D Spatio-Temporal", fontsize=14)
        ax1.set_xlabel("X (km)")
        ax1.set_ylabel("Y (km)")
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.2)
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(file_prefix, f"{fig1_name}.png"), dpi=300, bbox_inches='tight')

        # ---------------------------------------------------------
        # Figure 2: Wind Field RGB
        # ---------------------------------------------------------
        fig2_name = "analysis_02_wind_field_rgb"
        plt.figure(fig2_name, figsize=(10, 8))
        ax2 = plt.gca()
        wind_speed = np.sqrt(u_grid**2 + v_grid**2)
        im = ax2.imshow(wind_speed, extent=[X_sub.min()/1000, X_sub.max()/1000, Y_sub.min()/1000, Y_sub.max()/1000],
                        origin='lower', cmap='turbo', alpha=0.8)
        cbar2 = plt.colorbar(im, ax=ax2, pad=0.02)
        cbar2.set_label("Wind Speed (m/s)")
        ax2.streamplot(X_sub / 1000.0, Y_sub / 1000.0, u_grid, v_grid, density=1.0, color='white', linewidth=0.8, arrowsize=1.2)

        # 风场图上也保留 NFZ 和风暴，增加参考性
        self._draw_nfz(ax2)
        self._draw_storms_static(ax2, duration_s=400.0)
        self._plot_points(ax2, (start_xy[0] / 1000.0, start_xy[1] / 1000.0), (goal_xy[0] / 1000.0, goal_xy[1] / 1000.0))

        ax2.set_title("Wind Field and Environment (t=0)", fontsize=14)
        ax2.set_xlabel("X (km)")
        ax2.set_ylabel("Y (km)")
        plt.tight_layout()

        # 🌟 自动保存 Fig 2
        if save_dir:
            s_path = os.path.join(file_prefix, f"{fig2_name}.png")
            plt.savefig(s_path, dpi=300, bbox_inches='tight')
            print(f"已保存动态风场图: {s_path}")


        # ---------------------------------------------------------
        # Figure 3: Elevation Profile
        # ---------------------------------------------------------
        fig3_name = "analysis_03_elevation_profile"
        plt.figure(fig3_name, figsize=(10, 6))
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

        # 🌟 自动保存 Fig 3
        if save_dir:
            s_path = os.path.join(file_prefix, f"{fig3_name}.png")
            plt.savefig(s_path, dpi=300, bbox_inches='tight')
            print(f"已保存飞行剖面图: {s_path}")
    
    def plot_single_mission_execution(
        self,
        mission_result: MissionResult,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        method_name: str,
        save_dir: str
    ):
        """🌟 纯净版：一图一路径，专门给 render_case_studies.py 使用"""
        fig_name = f"path_{method_name.replace(' ', '_')}"
        plt.figure(fig_name, figsize=(10, 8))
        ax = plt.gca()
        
        contour = self._draw_base_terrain(ax)
        cbar = plt.colorbar(contour, ax=ax, pad=0.02)
        cbar.set_label("Altitude (m)")
        
        self._draw_nfz(ax)
        self._draw_storms_static(ax, duration_s=400.0)

        # 核心：只画这一条实际执行的路线
        path_xyz = mission_result.actual_flown_path_xyz
        p = self._path_xyz_to_xy_km(path_xyz)
        
        if p is not None and len(p) >= 2:
            ax.plot(p[:, 0], p[:, 1], color='lime', linestyle='-', linewidth=3.0, label=method_name)

        # 失败坠机的标志
        if not mission_result.success and path_xyz:
            crash_pt = path_xyz[-1]
            cx, cy = crash_pt[0]/1000.0, crash_pt[1]/1000.0
            crash_reason = mission_result.failure_reason or "CRASH!"
            ax.scatter(cx, cy, color='red', marker='*', s=400, edgecolor='black', zorder=20)
            ax.text(cx, cy + 0.3, f"💥 {crash_reason}", color='red', fontsize=12, fontweight='bold', ha='center', va='bottom', zorder=21)

        self._plot_points(ax, (start_xy[0] / 1000.0, start_xy[1] / 1000.0), (goal_xy[0] / 1000.0, goal_xy[1] / 1000.0))
        
        ax.set_title(f"Flight Path Execution: {method_name}", fontsize=14)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.legend(loc='upper left')
        ax.grid(alpha=0.2)
        plt.tight_layout()
        
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, "analysis_01_terrain_and_paths.png"), dpi=300, bbox_inches='tight')
        
        plt.close() 

    # ==============================================================
    # 🌟 3D Plotly 方法保持不变
    # ==============================================================
    def plot_3d_trajectory_plotly(self, mission_result: MissionResult, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]):
        """在默认浏览器中打开并渲染无畸变的 3D 可交互轨迹图"""
        path = np.array(mission_result.actual_flown_path_xyz)
        if len(path) < 2:
            print("[3D可视化提示] 实际飞行轨迹为空，无法渲染3D图形。")
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
    def plot_power_energy_comparison(
        self, 
        mission_result: MissionResult, 
        physics_engine, 
        save_dir: Optional[str] = None, 
        method_name: str = "Our 4D Spatio-Temporal Path",
        show_baseline: bool = True  # 🌟 核心开关：True 为拉踩对比模式(UI用)，False 为纯净单线模式(出图用)
    ):
        """
        📊 功率与能耗分析图 (支持双轨制)
        """
        print("正在生成功率与能耗分析曲线...")

        def evaluate_path_metrics(path_xyz, is_baseline=False):
            times_s, powers_w, energies_j = [0.0], [0.0], [0.0]
            curr_time, curr_energy = 0.0, 0.0
            is_crashed = False
            crash_reason = ""

            if not path_xyz or len(path_xyz) < 2:
                return times_s, powers_w, energies_j, is_crashed, crash_reason

            for i in range(len(path_xyz) - 1):
                p0, p1 = np.array(path_xyz[i], dtype=float), np.array(path_xyz[i+1], dtype=float)

                # 基线要检测地形和禁飞区撞击
                if is_baseline and self.map.is_collision(p1[0], p1[1], p1[2]):
                    is_crashed = True
                    crash_reason = "NFZ/Terrain"
                    times_s.append(curr_time + 1.0)
                    powers_w.append(0.0) 
                    energies_j.append(curr_energy)
                    break

                midpoint = (p0 + p1) / 2.0
                wind_2d = self.estimator.get_wind(midpoint[0], midpoint[1], midpoint[2], t_s=curr_time)
                wind_3d = np.array([wind_2d[0], wind_2d[1], 0.0])

                seg_e, seg_t, seg_p = physics_engine.estimate_segment_energy(
                    p0, p1, wind_3d, self.config.cruise_speed_mps
                )

                # 过载坠机判定
                if seg_p > self.config.max_power or seg_p == float('inf'):
                    is_crashed = True
                    crash_reason = "Overload"
                    curr_time += seg_t
                    times_s.append(curr_time)
                    powers_w.append(self.config.max_power * 1.05)
                    energies_j.append(curr_energy)
                    break 

                curr_time += seg_t
                curr_energy += seg_e
                times_s.append(curr_time)
                powers_w.append(seg_p)
                energies_j.append(curr_energy)

            return times_s, powers_w, energies_j, is_crashed, crash_reason

        # --- 数据评估 ---
        path_ours = mission_result.actual_flown_path_xyz
        t_ours, p_ours, e_ours, ours_crashed, ours_reason = evaluate_path_metrics(path_ours, is_baseline=False)

        # 如果需要展示基线（红线）
        if show_baseline:
            path_baseline = mission_result.initial_no_wind_path_xyz
            if not path_baseline or len(path_baseline) < 2:
                if path_ours and len(path_ours) > 0:
                    start_pos = path_ours[0]
                    goal_pos = path_ours[-1]
                    path_baseline = self._generate_straight_baseline(start_pos[:2], goal_pos[:2])
                else:
                    path_baseline = []
            
            t_base, p_base, e_base, base_crashed, base_reason = evaluate_path_metrics(path_baseline, is_baseline=True)

        # --- 开始画图 ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

        # --------------------------------
        # 上半部分：瞬时功率 (Power)
        # --------------------------------
        ax1.axhline(y=self.config.max_power, color='red', linestyle=':', linewidth=2, label=f'Motor Limit ({self.config.max_power}W)')

        if show_baseline:
            ax1.plot(t_base, p_base, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Traditional Straight Flight')
            if base_crashed:
                ax1.scatter(t_base[-1], p_base[-1], color='black', marker='X', s=150, zorder=5)
                ax1.annotate(f' CRASH!\n({base_reason})', xy=(t_base[-1], p_base[-1]), xytext=(-40, 10),
                             textcoords='offset points', color='darkred', fontweight='bold', arrowprops=dict(arrowstyle="->", color='red'))

        ax1.plot(t_ours, p_ours, color='lime', linestyle='-', linewidth=2.5, label=method_name)
        if ours_crashed:
            ax1.scatter(t_ours[-1], p_ours[-1], color='black', marker='X', s=150, zorder=5)
            ax1.annotate(f' CRASH!\n({ours_reason})', xy=(t_ours[-1], p_ours[-1]), xytext=(-40, 10),
                         textcoords='offset points', color='darkred', fontweight='bold', arrowprops=dict(arrowstyle="->", color='red'))

        ax1.set_title("Instantaneous Flight Power vs Time", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Power (Watts)", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right')

        # --------------------------------
        # 下半部分：累积能耗 (Energy)
        # --------------------------------
        e_ours_kj = [e / 1000.0 for e in e_ours]

        if show_baseline:
            e_base_kj = [e / 1000.0 for e in e_base]
            ax2.plot(t_base, e_base_kj, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Traditional Energy')
            if base_crashed:
                ax2.scatter(t_base[-1], e_base_kj[-1], color='black', marker='X', s=150, zorder=5)
                ax2.text(t_base[-1], e_base_kj[-1] + 5, f"Crashed at {t_base[-1]:.0f}s", color='darkred', fontweight='bold')

        ax2.plot(t_ours, e_ours_kj, color='lime', linestyle='-', linewidth=2.5, label=f"{method_name} Energy")
        
        if ours_crashed:
            ax2.scatter(t_ours[-1], e_ours_kj[-1], color='black', marker='X', s=150, zorder=5)
            ax2.text(t_ours[-1], e_ours_kj[-1] + 5, f"Crashed at {t_ours[-1]:.0f}s", color='darkred', fontweight='bold')
        elif len(t_ours) > 1 and len(e_ours_kj) > 1:
            ax2.text(t_ours[-1], e_ours_kj[-1], f"Arrived: {e_ours_kj[-1]:.0f} kJ", color='green', fontsize=11, fontweight='bold', va='top')

        ax2.set_title("Cumulative Energy Consumption vs Time", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Flight Time (Seconds)", fontsize=12)
        ax2.set_ylabel("Energy (kJ)", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='lower right')

        plt.tight_layout()

        # --- 保存图片 ---
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "analysis_04_power_energy_comparison.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存功率对比图: {save_path}")
    def plot_swarm_execution(
        self,
        mission_result: MissionResult,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        save_dir: str
    ):
        """🌟 专属：生成集群协同 (Swarm) 的静态 2D 四机轨迹平面图"""
        fig_name = "swarm_static_trajectories"
        plt.figure(fig_name, figsize=(10, 8))
        ax = plt.gca()
        
        contour = self._draw_base_terrain(ax)
        cbar = plt.colorbar(contour, ax=ax, pad=0.02)
        cbar.set_label("Altitude (m)")
        
        self._draw_nfz(ax)
        self._draw_storms_static(ax, duration_s=400.0, t_s=0.0) # 画出初始风暴

        # 1. 画主航母轨迹 (加粗绿线)
        master_path = self._path_xyz_to_xy_km(mission_result.actual_flown_path_xyz)
        if master_path is not None and len(master_path) >= 2:
            ax.plot(master_path[:, 0], master_path[:, 1], color='lime', linestyle='-', linewidth=3.0, label="Master Drone Path")

        # 2. 画探路蜂轨迹 (品红虚线，稍细)
        scout_path_3d = getattr(mission_result, "scout_flown_path_xyz", [])
        if scout_path_3d:
            scout_path = self._path_xyz_to_xy_km(scout_path_3d)
            if scout_path is not None and len(scout_path) >= 2:
                ax.plot(scout_path[:, 0], scout_path[:, 1], color='magenta', linestyle='--', linewidth=1.5, alpha=0.8, label="Scout Drone Path")

        # 3. 画 Relay 轨迹（蓝色点划线）
        relay_path_3d = getattr(mission_result, "relay_flown_path_xyz", [])
        if relay_path_3d:
            relay_path = self._path_xyz_to_xy_km(relay_path_3d)
            if relay_path is not None and len(relay_path) >= 2:
                ax.plot(relay_path[:, 0], relay_path[:, 1], color='dodgerblue', linestyle='-.', linewidth=1.8, alpha=0.85, label="Relay Drone Path")

        support_path_3d = getattr(mission_result, "support_flown_path_xyz", [])
        if support_path_3d:
            support_path = self._path_xyz_to_xy_km(support_path_3d)
            if support_path is not None and len(support_path) >= 2:
                ax.plot(support_path[:, 0], support_path[:, 1], color='darkorange', linestyle=':', linewidth=2.0, alpha=0.9, label="Support Drone Path")

        # 4. 在终局位置画出链路快照，方便静态图快速看懂拓扑
        link_history = getattr(mission_result, "link_status_history", [])
        if mission_result.actual_flown_path_xyz and scout_path_3d:
            master_final = mission_result.actual_flown_path_xyz[-1]
            scout_final = scout_path_3d[-1]
            relay_final = relay_path_3d[-1] if relay_path_3d else None
            support_final = support_path_3d[-1] if support_path_3d else None
            link_snapshot = link_history[-1] if link_history else {}

            mx, my = master_final[0] / 1000.0, master_final[1] / 1000.0
            sx, sy = scout_final[0] / 1000.0, scout_final[1] / 1000.0

            if relay_final is not None:
                rx, ry = relay_final[0] / 1000.0, relay_final[1] / 1000.0
                ax.plot(
                    [mx, rx],
                    [my, ry],
                    color='lime' if link_snapshot.get("m_r", False) else 'red',
                    linestyle='-' if link_snapshot.get("m_r", False) else '--',
                    linewidth=1.8,
                    alpha=0.85,
                    label="Relay Link" if "Relay Link" not in ax.get_legend_handles_labels()[1] else None,
                )
                ax.plot(
                    [rx, sx],
                    [ry, sy],
                    color='lime' if link_snapshot.get("r_s", False) else 'red',
                    linestyle='-' if link_snapshot.get("r_s", False) else '--',
                    linewidth=1.8,
                    alpha=0.85,
                    label="Scout Link" if "Scout Link" not in ax.get_legend_handles_labels()[1] else None,
                )
                if support_final is not None:
                    ux, uy = support_final[0] / 1000.0, support_final[1] / 1000.0
                    ax.plot(
                        [mx, ux],
                        [my, uy],
                        color='lime' if link_snapshot.get("m_sup", False) else 'red',
                        linestyle='-' if link_snapshot.get("m_sup", False) else '--',
                        linewidth=1.6,
                        alpha=0.8,
                        label="Support Link" if "Support Link" not in ax.get_legend_handles_labels()[1] else None,
                    )
                    ax.plot(
                        [ux, rx],
                        [uy, ry],
                        color='lime' if link_snapshot.get("sup_r", False) else 'red',
                        linestyle='-' if link_snapshot.get("sup_r", False) else '--',
                        linewidth=1.6,
                        alpha=0.8,
                        label="Support-Relay Link" if "Support-Relay Link" not in ax.get_legend_handles_labels()[1] else None,
                    )
            else:
                comm_history = getattr(mission_result, "comm_status_history", [])
                is_connected = link_snapshot.get(
                    "path_direct",
                    comm_history[-1] if comm_history else False,
                )
                ax.plot(
                    [mx, sx],
                    [my, sy],
                    color='lime' if is_connected else 'red',
                    linestyle='-' if is_connected else '--',
                    linewidth=1.8,
                    alpha=0.85,
                    label="Data Link" if "Data Link" not in ax.get_legend_handles_labels()[1] else None,
                )

        # 失败坠机的标志
        if not mission_result.success and mission_result.actual_flown_path_xyz:
            crash_pt = mission_result.actual_flown_path_xyz[-1]
            cx, cy = crash_pt[0]/1000.0, crash_pt[1]/1000.0
            crash_reason = mission_result.failure_reason or "CRASH!"
            ax.scatter(cx, cy, color='red', marker='*', s=400, edgecolor='black', zorder=20)
            ax.text(cx, cy + 0.3, f"[!]{crash_reason}", color='red', fontsize=12, fontweight='bold', ha='center', va='bottom', zorder=21)

        self._plot_points(ax, (start_xy[0] / 1000.0, start_xy[1] / 1000.0), (goal_xy[0] / 1000.0, goal_xy[1] / 1000.0))
        
        ax.set_title("FANET Swarm Execution (Master + Scout + Relay + Support)", fontsize=14)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.legend(loc='upper left')
        ax.grid(alpha=0.2)
        plt.tight_layout()
        
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, "swarm_static_trajectories.png"), dpi=300, bbox_inches='tight')
            print(f"✅ 静态四机拓扑图已保存至 {save_dir}/swarm_static_trajectories.png")
        
        plt.close()
