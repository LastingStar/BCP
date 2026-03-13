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

    def generate_gif(self, mission_result: MissionResult, start_xy: tuple, goal_xy: tuple, filename="mission_dynamic.gif"):
        print(f"\n🎬 正在准备包含动态风场渲染的 GIF 动画，请稍候...")
        
        path_3d = mission_result.actual_flown_path_xyz
        if not path_3d or len(path_3d) < 2: return
            
        path_xy_km = np.array([(p[0]/1000.0, p[1]/1000.0) for p in path_3d])
        
        # 估算时间轴
        times_s = [0.0]
        for i in range(1, len(path_3d)):
            dist = np.linalg.norm(np.array(path_3d[i]) - np.array(path_3d[i-1]))
            times_s.append(times_s[-1] + dist / self.config.cruise_speed_mps)
        total_time = times_s[-1]
        
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

        # ==========================================
        # 🌟 3. 动态风场初始化 (使用 Quiver 箭头，比流线图计算快)
        # 抽取稀疏网格以保证动画渲染速度
        # ==========================================
        step = max(1, self.map.size_x // 25)  # 25x25 的箭头网格
        X_sub = self.map.X[::step, ::step]
        Y_sub = self.map.Y[::step, ::step]
        U_init = np.zeros_like(X_sub, dtype=float)
        V_init = np.zeros_like(X_sub, dtype=float)
        
        # 初始化风向箭头，用浅蓝色表示
        wind_quiver = ax.quiver(X_sub/1000.0, Y_sub/1000.0, U_init, V_init, 
                                color='cyan', alpha=0.5, scale=300, width=0.003)

        # 4. 飞行轨迹图层
        line_flown, = ax.plot([], [], 'lime', linewidth=3.5, label='Executed Path')
        drone_dot, = ax.plot([], [], 'o', color='white', markeredgecolor='black', markersize=10, zorder=11)
        
        # 5. 风暴图层
        storm_patches = []
        if self.config.enable_storms and hasattr(self.estimator.wind, 'storm_manager'):
            for _ in self.estimator.wind.storm_manager.storms:
                sp = patches.Circle((0,0), 0, color='navy', alpha=0.4, lw=1.5, ec='blue')
                ax.add_patch(sp)
                storm_patches.append(sp)

        title_text = ax.set_title("Initializing...", fontsize=14)
        
        frames_count = 80 # 帧数
        
        def update(frame):
            current_time_s = (frame / frames_count) * total_time
            
            # --- 更新无人机轨迹 ---
            idx = 0
            while idx < len(times_s) - 1 and times_s[idx+1] < current_time_s:
                idx += 1
            flown_x, flown_y = path_xy_km[:idx+1, 0], path_xy_km[:idx+1, 1]
            line_flown.set_data(flown_x, flown_y)
            if len(flown_x) > 0: drone_dot.set_data([flown_x[-1]], [flown_y[-1]])

            # --- 🌟 更新全图实时风场向量 ---
            U_new = np.zeros_like(X_sub, dtype=float)
            V_new = np.zeros_like(X_sub, dtype=float)
            # 因为只有时变背景风和风暴在随时间变，地形风是静态的
            # 我们直接向 estimator 请求当前时刻 t_s 的网格风速
            for i in range(U_new.shape[0]):
                for j in range(U_new.shape[1]):
                    w = self.estimator.get_wind(self.map.x[j*step], self.map.y[i*step], z=-1.0, t_s=current_time_s)
                    U_new[i, j] = w[0]
                    V_new[i, j] = w[1]
            # 刷新风向标数据
            wind_quiver.set_UVC(U_new, V_new)

            # --- 更新动态风暴位置 ---
            if storm_patches:
                storms = self.estimator.wind.storm_manager.storms
                for i, storm in enumerate(storms):
                    new_cx = storm.center_xy[0] + storm.velocity_xy[0] * current_time_s
                    new_cy = storm.center_xy[1] + storm.velocity_xy[1] * current_time_s
                    storm_patches[i].set_center((new_cx / 1000.0, new_cy / 1000.0))
                    storm_patches[i].set_radius(storm.radius_m / 1000.0)

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