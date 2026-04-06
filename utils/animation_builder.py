"""
任务动画构建模块

此模块提供任务执行过程的动画可视化功能，包括无人机轨迹、
风场演化和时间推进的动态展示。

主要组件：
- MissionAnimator: 任务动画器
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from configs.config import SimulationConfig
from core.estimator import StateEstimator
from models.mission_models import MissionResult

class MissionAnimator:
    """大创杀手锏：动态风场 + 风暴躲避 GIF 生成器"""
    def __init__(self, config: SimulationConfig, estimator: StateEstimator):
        self.config = config
        self.estimator = estimator
        self.map = estimator.map

    def generate_gif(self, mission_result: MissionResult, start_xy: tuple, goal_xy: tuple, filename="mission_dynamic.gif", physics_engine=None):
        print(f"\n🎬 正在准备包含动态风场渲染的 GIF 动画，请稍候...")
        
        path_3d = mission_result.actual_flown_path_xyz
        if not path_3d or len(path_3d) < 2: return
            
        path_xy_km = np.array([(p[0]/1000.0, p[1]/1000.0) for p in path_3d])
        
        # 1. 预计算每一帧的物理时间
        times_s = [0.0]
        curr_time = 0.0
        for i in range(len(path_3d) - 1):
            p0 = np.array(path_3d[i], dtype=float)
            p1 = np.array(path_3d[i+1], dtype=float)

            if physics_engine:
                mid = (p0 + p1) / 2.0
                w2d = self.estimator.get_wind(mid[0], mid[1], mid[2], t_s=curr_time)
                _, seg_t, _ = physics_engine.estimate_segment_energy(p0, p1, np.array([w2d[0], w2d[1], 0.0]), self.config.cruise_speed_mps)
            else:
                dist = np.linalg.norm(p1 - p0)
                seg_t = dist / self.config.cruise_speed_mps

            curr_time += seg_t
            times_s.append(curr_time)

        total_time = times_s[-1]
        
        # 2. 开始建立画布并渲染 GIF 第一帧
        fig, ax = plt.subplots(figsize=(11, 8))
        
        # 1. 静态地形底图
        ax.contourf(self.map.X / 1000.0, self.map.Y / 1000.0, self.map.dem, 50, cmap='gist_earth', alpha=0.7)
        
        # 2. 禁飞区与起终点
        if self.config.enable_nfz:
            for (cx, cy, r) in self.config.nfz_list_km:
                ax.add_patch(patches.Circle((cx, cy), r, lw=2, ec='red', fc='red', alpha=0.3, hatch='//'))
                ax.text(cx, cy, 'NFZ', color='darkred', fontsize=10, fontweight='bold', ha='center', va='center')

        ax.scatter([start_xy[0]/1000.0, goal_xy[0]/1000.0], [start_xy[1]/1000.0, goal_xy[1]/1000.0], 
                   c=['gold', 'red'], s=150, marker='*', ec='black', zorder=10)

        # 3. 动态风场初始化 (使用 Quiver 箭头，比流线图计算快)
        step = max(1, self.map.size_x // 25) 
        X_sub = self.map.X[::step, ::step]
        Y_sub = self.map.Y[::step, ::step]
        U_init = np.zeros_like(X_sub, dtype=float)
        V_init = np.zeros_like(X_sub, dtype=float)
        
        wind_quiver = ax.quiver(X_sub/1000.0, Y_sub/1000.0, U_init, V_init, 
                                color='cyan', alpha=0.5, scale=300, width=0.003)

        # 4. 飞行轨迹图层
        line_flown, = ax.plot([], [], 'lime', linewidth=3.5, label='Executed Path')
        drone_dot, = ax.plot([], [], 'o', color='white', markeredgecolor='black', markersize=10, zorder=11)
        
        # 5. 风暴图层 (预先分配足够的名额)
        storm_patches = []
        if self.config.enable_storms and hasattr(self.estimator.wind, 'storm_manager'):
            for _ in range(self.config.storm_count):
                sp = patches.Circle((0,0), 0, color='navy', alpha=0.4, lw=1.5, ec='blue')
                ax.add_patch(sp)
                storm_patches.append(sp)

        title_text = ax.set_title("Initializing...", fontsize=14)
        frames_count = 80 
        
        def update(frame):
            current_time_s = (frame / frames_count) * total_time
            
            # --- 更新无人机轨迹 ---
            idx = 0
            while idx < len(times_s) - 1 and times_s[idx+1] < current_time_s:
                idx += 1
            flown_x, flown_y = path_xy_km[:idx+1, 0], path_xy_km[:idx+1, 1]
            line_flown.set_data(flown_x, flown_y)
            if len(flown_x) > 0: drone_dot.set_data([flown_x[-1]], [flown_y[-1]])

            # --- 更新全图实时风场向量 ---
            U_new = np.zeros_like(X_sub, dtype=float)
            V_new = np.zeros_like(X_sub, dtype=float)
            for i in range(U_new.shape[0]):
                for j in range(U_new.shape[1]):
                    w = self.estimator.get_wind(self.map.x[j*step], self.map.y[i*step], z=-1.0, t_s=current_time_s)
                    U_new[i, j] = w[0]
                    V_new[i, j] = w[1]
            wind_quiver.set_UVC(U_new, V_new)

            # --- 🌟 重点：调用全新的无状态风场 API ---
            if storm_patches:
                # 隐藏所有圈圈（以防有风暴消散）
                for sp in storm_patches:
                    sp.set_visible(False)
                    
                # 拿取当前时间仍然存活的风暴
                active_storms = self.estimator.wind.storm_manager.get_active_storms(current_time_s)
                for i, storm in enumerate(active_storms):
                    if i < len(storm_patches):
                        # 纯数学计算当前时刻的位置
                        current_center = storm.center_at(current_time_s)
                        new_cx = current_center[0] / 1000.0
                        new_cy = current_center[1] / 1000.0
                        
                        storm_patches[i].set_center((new_cx, new_cy))
                        storm_patches[i].set_radius(storm.radius_m / 1000.0)
                        storm_patches[i].set_visible(True)

            title_text.set_text(f"4D Spatio-Temporal Flight | Time: {current_time_s:.1f} s")
            return [line_flown, drone_dot, title_text, wind_quiver] + storm_patches

        # 6. 生成动画
        print(f"⚙️ 开始逐帧渲染动态风场与轨迹 (共 {frames_count} 帧)，由于物理风场实时计算，预计需 20~40 秒...")
        ani = animation.FuncAnimation(fig, update, frames=frames_count, interval=100, blit=True)
        
        try:
            ani.save(filename, writer='pillow', fps=10)
            print(f"✅ 完美！带实时动态风场的动画已保存至 {filename}")
        except Exception as e:
            print(f"❌ 导出动画失败: {e}")
        plt.close(fig)

    def generate_swarm_gif(self, mission_result: MissionResult, start_xy: tuple, goal_xy: tuple, filename="swarm_dynamic.gif"):
        """🌟 专属：生成集群协同 (Swarm) 的动态通信与探测动图"""
        print(f"\n🎬 正在生成【FANET集群数据链】动态 GIF，请稍候...")

        master_path_3d = getattr(mission_result, "master_sync_path_xyz", []) or mission_result.actual_flown_path_xyz
        scout_path_3d = getattr(mission_result, "scout_flown_path_xyz", [])
        relay_path_3d = getattr(mission_result, "relay_flown_path_xyz", [])
        support_path_3d = getattr(mission_result, "support_flown_path_xyz", [])
        link_status_history = getattr(mission_result, "link_status_history", [])
        support_mode_history = getattr(mission_result, "support_mode_history", [])
        gust_active_history = getattr(mission_result, "gust_active_history", [])
        comm_status = getattr(mission_result, "comm_status_history", [])
        times_s = getattr(mission_result, "swarm_time_history", [])

        if not master_path_3d or len(master_path_3d) < 2:
            return

        if not times_s or len(times_s) != len(master_path_3d):
            times_s = [i for i in range(len(master_path_3d))]

        sample_count = min(
            len(master_path_3d),
            len(times_s),
            len(scout_path_3d) if scout_path_3d else len(master_path_3d),
            len(relay_path_3d) if relay_path_3d else len(master_path_3d),
            len(support_path_3d) if support_path_3d else len(master_path_3d),
            len(link_status_history) if link_status_history else len(master_path_3d),
            len(support_mode_history) if support_mode_history else len(master_path_3d),
            len(gust_active_history) if gust_active_history else len(master_path_3d),
        )
        if sample_count < 2:
            return

        master_path_3d = master_path_3d[:sample_count]
        times_s = times_s[:sample_count]
        scout_path_3d = scout_path_3d[:sample_count] if scout_path_3d else []
        relay_path_3d = relay_path_3d[:sample_count] if relay_path_3d else []
        support_path_3d = support_path_3d[:sample_count] if support_path_3d else []
        link_status_history = link_status_history[:sample_count] if link_status_history else []
        support_mode_history = support_mode_history[:sample_count] if support_mode_history else []
        gust_active_history = gust_active_history[:sample_count] if gust_active_history else []
        comm_status = comm_status[:sample_count] if comm_status else []

        path_xy_km = np.array([(p[0] / 1000.0, p[1] / 1000.0) for p in master_path_3d])
        scout_xy_km = np.array([(p[0] / 1000.0, p[1] / 1000.0) for p in scout_path_3d]) if scout_path_3d else None
        relay_xy_km = np.array([(p[0] / 1000.0, p[1] / 1000.0) for p in relay_path_3d]) if relay_path_3d else None
        support_xy_km = np.array([(p[0] / 1000.0, p[1] / 1000.0) for p in support_path_3d]) if support_path_3d else None

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.contourf(self.map.X / 1000.0, self.map.Y / 1000.0, self.map.dem, 50, cmap='gist_earth', alpha=0.7)
        if self.config.enable_nfz:
            for (cx, cy, r) in self.config.nfz_list_km:
                ax.add_patch(patches.Circle((cx, cy), r, lw=2, ec='red', fc='red', alpha=0.3, hatch='//'))
                ax.text(cx, cy, 'NFZ', color='darkred', fontsize=10, fontweight='bold', ha='center', va='center')
        ax.scatter(
            [start_xy[0] / 1000.0, goal_xy[0] / 1000.0],
            [start_xy[1] / 1000.0, goal_xy[1] / 1000.0],
            c=['gold', 'red'],
            s=150,
            marker='*',
            ec='black',
            zorder=10,
        )

        step = max(1, self.map.size_x // 25)
        X_sub, Y_sub = self.map.X[::step, ::step], self.map.Y[::step, ::step]
        wind_quiver = ax.quiver(
            X_sub / 1000.0,
            Y_sub / 1000.0,
            np.zeros_like(X_sub),
            np.zeros_like(X_sub),
            color='cyan',
            alpha=0.3,
            scale=300,
            width=0.003,
        )

        line_flown, = ax.plot([], [], 'lime', linewidth=3.5, label='Master Path')
        drone_dot, = ax.plot([], [], 'o', color='white', markeredgecolor='black', markersize=10, zorder=12, label='Master')
        scout_dot, = ax.plot([], [], '^', color='magenta', markeredgecolor='black', markersize=9, zorder=12, label='Scout')
        relay_dot, = ax.plot([], [], 's', color='dodgerblue', markeredgecolor='black', markersize=8, zorder=12, label='Relay')
        support_dot, = ax.plot([], [], 'd', color='darkorange', markeredgecolor='black', markersize=9, zorder=12, label='Support')
        line_m_r, = ax.plot([], [], color='lime', linestyle='-', linewidth=2, zorder=11, label='Master-Relay Link')
        line_r_s, = ax.plot([], [], color='lime', linestyle='-', linewidth=2, zorder=11, label='Relay-Scout Link')
        line_m_sup, = ax.plot([], [], color='lime', linestyle='-', linewidth=2, zorder=11, label='Master-Support Link')
        line_sup_r, = ax.plot([], [], color='lime', linestyle='-', linewidth=2, zorder=11, label='Support-Relay Link')

        ax.legend(loc='upper right')

        storm_patches = []
        if hasattr(self.estimator.wind, 'storm_manager'):
            for _ in range(self.config.storm_count):
                sp = patches.Circle((0, 0), 0, color='navy', alpha=0.4, lw=1.5, ec='blue')
                ax.add_patch(sp)
                storm_patches.append(sp)

        title_text = ax.set_title("Initializing FANET Swarm...", fontsize=14)
        frames_count = max(2, min(120, sample_count))

        def update(frame):
            idx = min(int((frame / max(frames_count - 1, 1)) * (sample_count - 1)), sample_count - 1)
            current_time_s = times_s[idx]

            flown_x, flown_y = path_xy_km[:idx + 1, 0], path_xy_km[:idx + 1, 1]
            line_flown.set_data(flown_x, flown_y)
            if len(flown_x) > 0:
                drone_dot.set_data([flown_x[-1]], [flown_y[-1]])

            sx = sy = None
            if scout_xy_km is not None and idx < len(scout_xy_km):
                sx, sy = scout_xy_km[idx, 0], scout_xy_km[idx, 1]
                scout_dot.set_data([sx], [sy])
            else:
                scout_dot.set_data([], [])

            rx = ry = None
            if relay_xy_km is not None and idx < len(relay_xy_km):
                rx, ry = relay_xy_km[idx, 0], relay_xy_km[idx, 1]
                relay_dot.set_data([rx], [ry])
            else:
                relay_dot.set_data([], [])

            ux = uy = None
            if support_xy_km is not None and idx < len(support_xy_km):
                ux, uy = support_xy_km[idx, 0], support_xy_km[idx, 1]
                support_dot.set_data([ux], [uy])
            else:
                support_dot.set_data([], [])

            link_snapshot = link_status_history[idx] if idx < len(link_status_history) else {}
            support_mode = support_mode_history[idx] if idx < len(support_mode_history) else "FORMATION"
            gust_active = gust_active_history[idx] if idx < len(gust_active_history) else False

            support_dot.set_color('darkorange')
            support_dot.set_markeredgecolor('black')
            support_dot.set_markersize(9)
            support_dot.set_alpha(0.95)
            if support_mode == "SHIELD":
                support_dot.set_color('gold')
                support_dot.set_markeredgecolor('crimson')
                support_dot.set_markersize(13)
                support_dot.set_alpha(1.0 if frame % 2 == 0 else 0.45)
            elif support_mode == "BRIDGE":
                support_dot.set_color('orange')
                support_dot.set_markeredgecolor('lime')
                support_dot.set_markersize(11)
            elif support_mode == "ESCAPE":
                support_dot.set_color('red')
                support_dot.set_markeredgecolor('black')
                support_dot.set_markersize(11)

            if rx is not None and ry is not None:
                line_m_r.set_data([flown_x[-1], rx], [flown_y[-1], ry])
                line_r_s.set_data([rx, sx] if sx is not None else [], [ry, sy] if sy is not None else [])
                line_m_r.set_color('lime' if link_snapshot.get("m_r", False) else 'red')
                line_m_r.set_linestyle('-' if link_snapshot.get("m_r", False) else '--')
                line_r_s.set_color('lime' if link_snapshot.get("r_s", False) else 'red')
                line_r_s.set_linestyle('-' if link_snapshot.get("r_s", False) else '--')
            elif sx is not None:
                direct_active = link_snapshot.get(
                    "path_direct",
                    comm_status[idx] if idx < len(comm_status) else False,
                )
                line_m_r.set_data([flown_x[-1], sx], [flown_y[-1], sy])
                line_m_r.set_color('lime' if direct_active else 'red')
                line_m_r.set_linestyle('-' if direct_active else '--')
                line_r_s.set_data([], [])
            else:
                line_m_r.set_data([], [])
                line_r_s.set_data([], [])

            line_m_sup.set_linewidth(2.0)
            line_sup_r.set_linewidth(2.0)
            if ux is not None and uy is not None:
                line_m_sup.set_data([flown_x[-1], ux], [flown_y[-1], uy])
                line_m_sup.set_color('lime' if link_snapshot.get("m_sup", False) else 'red')
                line_m_sup.set_linestyle('-' if link_snapshot.get("m_sup", False) else '--')
                if link_snapshot.get("path_support", False):
                    line_m_sup.set_color('gold')
                    line_m_sup.set_linewidth(3.5)
            else:
                line_m_sup.set_data([], [])

            if ux is not None and uy is not None and rx is not None and ry is not None:
                line_sup_r.set_data([ux, rx], [uy, ry])
                line_sup_r.set_color('lime' if link_snapshot.get("sup_r", False) else 'red')
                line_sup_r.set_linestyle('-' if link_snapshot.get("sup_r", False) else '--')
                if link_snapshot.get("path_support", False):
                    line_sup_r.set_color('gold')
                    line_sup_r.set_linewidth(3.5)
            else:
                line_sup_r.set_data([], [])

            if support_mode == "SHIELD":
                line_m_sup.set_linewidth(max(line_m_sup.get_linewidth(), 3.5))
                line_sup_r.set_linewidth(max(line_sup_r.get_linewidth(), 3.5))

            U_new, V_new = np.zeros_like(X_sub), np.zeros_like(X_sub)
            for i in range(U_new.shape[0]):
                for j in range(U_new.shape[1]):
                    w = self.estimator.get_wind(self.map.x[j * step], self.map.y[i * step], z=-1.0, t_s=current_time_s)
                    U_new[i, j], V_new[i, j] = w[0], w[1]
            wind_quiver.set_UVC(U_new, V_new)
            wind_quiver.set_alpha(0.55 if gust_active else 0.3)

            if storm_patches:
                for sp in storm_patches:
                    sp.set_visible(False)
                active_storms = self.estimator.wind.storm_manager.get_active_storms(current_time_s)
                for i, storm in enumerate(active_storms):
                    if i < len(storm_patches):
                        current_center = storm.center_at(current_time_s)
                        storm_patches[i].set_center((current_center[0] / 1000.0, current_center[1] / 1000.0))
                        storm_patches[i].set_radius(storm.radius_m / 1000.0)
                        storm_patches[i].set_visible(True)

            if link_snapshot:
                route_modes = []
                if link_snapshot.get("path_direct", False):
                    route_modes.append("Direct")
                if link_snapshot.get("path_relay", False):
                    route_modes.append("Relay")
                if link_snapshot.get("path_support", False):
                    route_modes.append("Support")
                route_str = "+".join(route_modes) if route_modes else "Isolated"
                status_str = "CONNECTED" if link_snapshot.get("network_active", False) else "ISOLATED"
            else:
                fallback_connected = comm_status[idx] if idx < len(comm_status) else False
                route_str = "Direct" if fallback_connected else "Isolated"
                status_str = "CONNECTED" if fallback_connected else "ISOLATED"

            title_text.set_text(
                f"FANET Swarm | T: {current_time_s:.1f}s | Link: {status_str}"
                f" | Route: {route_str} | Support: {support_mode}"
                f" | Gust: {'ON' if gust_active else 'OFF'}"
            )
            return [
                line_flown,
                drone_dot,
                scout_dot,
                relay_dot,
                support_dot,
                line_m_r,
                line_r_s,
                line_m_sup,
                line_sup_r,
                title_text,
                wind_quiver,
            ] + storm_patches

        ani = animation.FuncAnimation(fig, update, frames=frames_count, interval=100, blit=True)
        ani.save(filename, writer='pillow', fps=10)
        print(f"✅ 集群动态渲染完成！已保存至 {filename}")
        plt.close(fig)
