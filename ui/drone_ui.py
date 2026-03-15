import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
import time

from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.battery_manager import BatteryManager
from core.planner import AStarPlanner
from simulation.mission_executor import MissionExecutor
from utils.animation_builder import MissionAnimator
from analysis.mission_metrics import summarize_mission_result

class DronePlanningUI:
    def __init__(self):
        # 初始化全局状态
        if 'config' not in st.session_state:
            st.session_state.config = SimulationConfig()
        if 'map_manager' not in st.session_state:
            st.session_state.map_manager = None
        if 'map_loaded' not in st.session_state:
            st.session_state.map_loaded = False
            
        self.config = st.session_state.config
        
        # 初始化起终点 session state
        half_size_m = (self.config.map_size_km * 1000) / 2
        if 'start_x' not in st.session_state: st.session_state.start_x = -half_size_m * 0.6
        if 'start_y' not in st.session_state: st.session_state.start_y = -half_size_m * 0.6
        if 'goal_x' not in st.session_state: st.session_state.goal_x = half_size_m * 0.6
        if 'goal_y' not in st.session_state: st.session_state.goal_y = half_size_m * 0.6

    def create_sidebar_config(self):
        st.sidebar.title("⚙️ 系统配置中心")

        # --- 1. 地图上传与初始化 ---
        with st.sidebar.expander("🗺️ Step 1: 地形与环境构建", expanded=True):
            uploaded_file = st.file_uploader("上传自定义地形 (灰度PNG/JPG)", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    self.config.map_path = tmp_file.name
            
            self.config.map_size_km = st.slider("物理映射尺寸 (km)", 5.0, 50.0, self.config.map_size_km, 1.0)
            self.config.min_alt = st.number_input("地形最低海拔 (m)", value=self.config.min_alt, step=10.0)
            self.config.max_alt = st.number_input("地形最高海拔 (m)", value=self.config.max_alt, step=10.0)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                res_x = st.number_input("解析度 X", value=self.config.target_size[0], step=50)
            with col_res2:
                res_y = st.number_input("解析度 Y", value=self.config.target_size[1], step=50)
            self.config.target_size = (int(res_x), int(res_y))
            
            self.config.time_of_day = st.selectbox("环境时间", ["Day", "Night", "Dusk", "Dawn"], 
                                                   index=["Day", "Night", "Dusk", "Dawn"].index(self.config.time_of_day))
            
            if st.button("🌍 载入/刷新地形环境", type="primary", use_container_width=True):
                with st.spinner("正在解析地形高程数据..."):
                    st.session_state.map_manager = MapManager(self.config)
                    st.session_state.map_loaded = True
                st.success("地形解析完成！")

        # --- 2. 微气象与极端风险 ---
        with st.sidebar.expander("💨 Step 2: 气象与风险模型"):
            self.config.enable_storms = st.checkbox("启用移动雷暴区 (Storms)", self.config.enable_storms)
            self.config.env_wind_u = st.slider("基础东风 U (m/s)", -25.0, 25.0, self.config.env_wind_u, 1.0)
            self.config.env_wind_v = st.slider("基础北风 V (m/s)", -25.0, 25.0, self.config.env_wind_v, 1.0)
            st.markdown("<small>TKE 极值坠机概率映射参数</small>", unsafe_allow_html=True)
            self.config.drone_robustness_K = st.slider("无人机抗扰鲁棒性 (K)", 10.0, 500.0, self.config.drone_robustness_K, 10.0)
            self.config.fatal_crash_penalty_j = st.slider("坠机致命惩罚 (J)", 10000.0, 500000.0, self.config.fatal_crash_penalty_j, 10000.0)

        # --- 3. 🚁 无人机核心物理参数 ---
        with st.sidebar.expander("🚁 Step 3: 无人机物理参数"):
            self.config.drone_mass = st.number_input("无人机质量 (kg)", value=self.config.drone_mass, step=0.1)
            self.config.drone_speed = st.slider("巡航速度 (m/s)", 5.0, 40.0, self.config.drone_speed, 1.0)
            self.config.max_power = st.number_input("最大输出功率 (W)", value=self.config.max_power, step=100.0)
            self.config.battery_capacity_j = st.number_input("电池总容量 (J)", value=self.config.battery_capacity_j, step=10000.0)
            self.config.drag_coeff = st.number_input("风阻系数 (Cd)", value=self.config.drag_coeff, step=0.01)
            # 🌟 新增：允许飞行的最高高度
            self.config.max_ceiling = st.number_input("允许最高飞行海拔 (m)", value=self.config.max_ceiling, step=100.0)

        # --- 4. 🚫 静态禁飞区动态设置 ---
        with st.sidebar.expander("🚫 Step 4: 静态禁飞区 (NFZ)"):
            self.config.enable_nfz = st.checkbox("启用静态禁飞区", self.config.enable_nfz)
            if self.config.enable_nfz:
                df_nfz = pd.DataFrame(self.config.nfz_list_km, columns=['中心 X', '中心 Y', '半径 R'])
                edited_df = st.data_editor(df_nfz, num_rows="dynamic", use_container_width=True, hide_index=True)
                self.config.nfz_list_km = [tuple(x) for x in edited_df.to_numpy()]

        # --- 5. 🧠 A* 偏好控制 ---
        with st.sidebar.expander("🧠 Step 5: A* 算法 AI 偏好"):
            st.markdown("<small>决定无人机面对困难时的底层决策逻辑</small>", unsafe_allow_html=True)
            # 将 heuristic_safety_factor 映射为直观的描述
            self.config.heuristic_safety_factor = st.slider(
                "行为倾向 (自保 vs 突防)", 
                min_value=1.0, max_value=8.0, 
                value=self.config.heuristic_safety_factor, step=0.5,
                help="1.0 = 极度保守自保，绕远路保证绝对安全；数值越大 = 极度贪婪突防，像导弹一样不计代价直奔终点！"
            )

    def render_map_preview_and_selection(self):
        """渲染 2D 交互地图预览"""
        map_m = st.session_state.map_manager
        step = max(1, map_m.size_x // 100)
        x_mesh, y_mesh = np.meshgrid(map_m.x[::step], map_m.y[::step])
        z_mesh = map_m.dem[::step, ::step]

        fig = go.Figure(data=go.Contour(
            z=z_mesh, x=map_m.x[::step], y=map_m.y[::step],
            colorscale='Earth', contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))
        ))

        if self.config.enable_nfz:
            for cx_km, cy_km, r_km in self.config.nfz_list_km:
                fig.add_shape(type="circle",
                    x0=(cx_km-r_km)*1000, y0=(cy_km-r_km)*1000,
                    x1=(cx_km+r_km)*1000, y1=(cy_km+r_km)*1000,
                    fillcolor="red", opacity=0.3, line_color="red"
                )

        fig.add_trace(go.Scatter(
            x=[st.session_state.start_x], y=[st.session_state.start_y], mode='markers+text', text=["起飞点"], textposition="top center",
            marker=dict(size=18, color='yellow', symbol='star', line=dict(width=2, color='black'))
        ))
        fig.add_trace(go.Scatter(
            x=[st.session_state.goal_x], y=[st.session_state.goal_y], mode='markers+text', text=["目标点"], textposition="top center",
            marker=dict(size=18, color='red', symbol='x', line=dict(width=2, color='black'))
        ))

        fig.update_layout(
            title="实时地形与任务布点图", xaxis_title="X 坐标 (米)", yaxis_title="Y 坐标 (米)",
            yaxis=dict(scaleanchor="x", scaleratio=1), height=650, margin=dict(l=20, r=20, t=40, b=20), showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    def create_main_interface(self):
        st.title("🚁 基于 TKE 极值风险的 4D 轨迹规划系统")
        st.markdown("---")

        if not st.session_state.map_loaded:
            st.info("👈 请先在左侧边栏点击 **[载入/刷新地形环境]** 按钮初始化地图。")
            return

        st.subheader("📍 任务航点设置 (拖动滑块即可在下方地图预览位置)")
        half_size = float((self.config.map_size_km * 1000) / 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ⭐ 【起飞点】")
            st.session_state.start_x = st.slider("起点 X", min_value=-half_size, max_value=half_size, value=float(st.session_state.start_x), step=10.0)
            st.session_state.start_y = st.slider("起点 Y", min_value=-half_size, max_value=half_size, value=float(st.session_state.start_y), step=10.0)

        with col2:
            st.markdown("#### ❌ 【降落点】")
            st.session_state.goal_x = st.slider("终点 X", min_value=-half_size, max_value=half_size, value=float(st.session_state.goal_x), step=10.0)
            st.session_state.goal_y = st.slider("终点 Y", min_value=-half_size, max_value=half_size, value=float(st.session_state.goal_y), step=10.0)

        self.render_map_preview_and_selection()
        st.markdown("---")
        
        if st.button("🚀 开始 4D 时空推演与避障规划", type="primary", use_container_width=True):
            start_xy = (st.session_state.start_x, st.session_state.start_y)
            goal_xy = (st.session_state.goal_x, st.session_state.goal_y)
            self.run_simulation(start_xy, goal_xy)

    def run_simulation(self, start_xy, goal_xy):
        map_manager = st.session_state.map_manager
        wind_model = WindModelFactory.create(self.config.wind_model_type, self.config, bounds=map_manager.get_bounds())
        estimator = StateEstimator(map_manager, wind_model, self.config)
        physics = PhysicsEngine(self.config)
        battery_manager = BatteryManager(self.config)
        planner = AStarPlanner(self.config, estimator, physics)
        executor = MissionExecutor(self.config, estimator, physics, battery_manager, planner)

        with st.spinner("🤖 正在构建 TKE 微气象场与 4D 空间搜索树... (预计10-40秒)"):
            start_time = time.time()
            mission_result = executor.execute_mission(start_xy, goal_xy)
            calc_time = time.time() - start_time

        st.subheader("📊 航路综合评估报告")
        summary = summarize_mission_result(mission_result)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("执行状态", "✅ 突防成功" if summary['success'] else "❌ 强制禁飞")
        c2.metric("算法耗时", f"{calc_time:.2f} s")
        c3.metric("预估电耗", f"{summary['total_energy_used_j']/1000:.1f} kJ")
        c4.metric("航迹总长", f"{summary['executed_path_length_m']:.1f} m")

        if not mission_result.success:
            st.error(f"🛑 任务终止: {summary['failure_reason']}")
            return

        # ==========================================
        # 🌟 第一排：4D动画 与 3D轨迹
        # ==========================================
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            st.subheader("🎬 4D 气象风暴规避推演")
            with st.spinner("正在合成动图..."):
                gif_path = "web_dynamic_mission.gif"
                animator = MissionAnimator(self.config, estimator)
                animator.generate_gif(mission_result, start_xy, goal_xy, filename=gif_path)
                st.image(gif_path, use_container_width=True)

        with col_right:
            st.subheader("🌍 真 3D 航迹数字孪生")
            with st.spinner("渲染 3D 地形..."):
                fig_3d = self.create_3d_visualization(map_manager, mission_result)
                st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")

        # ==========================================
        # 🌟 第二排：风场RGB 与 高度剖面图
        # ==========================================
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.subheader("🌪️ 风场流线与风险区映射 (t=0)")
            with st.spinner("计算风场流线数据..."):
                fig_wind = self.create_matplotlib_wind_plot(map_manager, estimator, start_xy, goal_xy)
                st.pyplot(fig_wind)

        with col_plot2:
            st.subheader("📈 飞行高度智能剖面图")
            with st.spinner("绘制高度曲线..."):
                fig_profile = self.create_elevation_profile(map_manager, mission_result)
                st.plotly_chart(fig_profile, use_container_width=True)

    def create_3d_visualization(self, map_manager, mission_result):
        path = np.array(mission_result.actual_flown_path_xyz)
        step = max(1, map_manager.size_x // 60)
        X, Y = np.meshgrid(map_manager.x[::step], map_manager.y[::step])
        Z = map_manager.dem[::step, ::step]

        fig = go.Figure()
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Earth', opacity=0.8, showscale=False))
        fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines', line=dict(color='#00FF00', width=6), name='航迹'))
        fig.add_trace(go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]], mode='markers', marker=dict(size=8, color='yellow'), name='起点'))
        fig.add_trace(go.Scatter3d(x=[path[-1, 0]], y=[path[-1, 1]], z=[path[-1, 2]], mode='markers', marker=dict(size=8, color='red'), name='终点'))
        fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Alt (m)', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
        return fig

    def create_matplotlib_wind_plot(self, map_manager, estimator, start_xy, goal_xy):
        """🌟 核心图表集成：纯 Python/Matplotlib 绘制风场 RGB 与 流线"""
        step = max(1, map_manager.size_x // 60)
        X_sub = map_manager.X[::step, ::step]
        Y_sub = map_manager.Y[::step, ::step]
        rows, cols = X_sub.shape
        
        speed_grid = np.zeros((rows, cols))
        u_grid = np.zeros((rows, cols))
        v_grid = np.zeros((rows, cols))
        
        # 获取环境风
        for i in range(rows):
            for j in range(cols):
                w = estimator.get_wind(map_manager.x[j*step], map_manager.y[i*step], z=-1.0, t_s=0.0)
                speed_grid[i, j] = np.linalg.norm(w)
                u_grid[i, j] = w[0]
                v_grid[i, j] = w[1]

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 1. 风场底图 (Turbo色带)
        contour = ax.contourf(X_sub / 1000.0, Y_sub / 1000.0, speed_grid, 50, cmap='turbo')
        cbar = plt.colorbar(contour, ax=ax, pad=0.02)
        cbar.set_label("Wind Speed (m/s)")
        
        # 2. 气流流线
        ax.streamplot(X_sub / 1000.0, Y_sub / 1000.0, u_grid, v_grid, density=1.0, color='white', linewidth=0.8, arrowsize=1.2)
        
        # 3. 绘制静态禁飞区
        if self.config.enable_nfz:
            for cx_km, cy_km, r_km in self.config.nfz_list_km:
                circle = patches.Circle((cx_km, cy_km), r_km, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, hatch='//')
                ax.add_patch(circle)
                ax.text(cx_km, cy_km, 'NFZ', color='darkred', fontsize=12, fontweight='bold', ha='center', va='center')

        # 4. 绘制移动雷暴
        if self.config.enable_storms and hasattr(estimator.wind, 'storm_manager'):
            for storm in estimator.wind.storm_manager.storms:
                cx, cy = storm.center_xy[0]/1000.0, storm.center_xy[1]/1000.0
                r = storm.radius_m/1000.0
                vx, vy = storm.velocity_xy[0]/1000.0, storm.velocity_xy[1]/1000.0
                circle = patches.Circle((cx, cy), r, linewidth=1.5, edgecolor='navy', facecolor='navy', alpha=0.3)
                ax.add_patch(circle)
                
                # 画未来预测箭头
                dur = 400.0
                ax.annotate('', xy=(cx+vx*dur, cy+vy*dur), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color="navy", ls="dashed", lw=2, alpha=0.9))
                ax.text(cx, cy, 'Storm', color='white', fontsize=10, fontweight='bold', ha='center', va='center')

        # 5. 起点终点
        ax.scatter(start_xy[0]/1000.0, start_xy[1]/1000.0, c='gold', s=150, marker='*', edgecolors='black', label='Start')
        ax.scatter(goal_xy[0]/1000.0, goal_xy[1]/1000.0, c='red', s=150, marker='X', edgecolors='black', label='Goal')

        ax.set_title("Wind Field RGB & Extracted Features", fontsize=12)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.legend(loc='upper right')
        
        return fig

    def create_elevation_profile(self, map_manager, mission_result):
        """🌟 核心图表集成：交互式的高度剖面图 (Plotly版本体验更好)"""
        path = np.array(mission_result.actual_flown_path_xyz)
        if len(path) < 2: return go.Figure()

        # 计算累计水平距离
        xy_diff = np.diff(path[:, :2], axis=0)
        dists = np.insert(np.cumsum(np.linalg.norm(xy_diff, axis=1)), 0, 0)
        
        # 提取地形高度
        terrain_z = [map_manager.get_altitude(x, y) for x, y in path[:, :2]]
        safe_z = [z + self.config.takeoff_altitude_agl for z in terrain_z]
        
        fig = go.Figure()
        
        # 地形填充
        fig.add_trace(go.Scatter(x=dists, y=terrain_z, fill='tozeroy', mode='lines', 
                                 line=dict(color='gray', width=1), name='地形高度 (Terrain)', opacity=0.5))
        
        # 红色虚线安全高度
        fig.add_trace(go.Scatter(x=dists, y=safe_z, mode='lines', 
                                 line=dict(color='red', width=1.5, dash='dash'), name='最低安全边界 (+AGL)'))
        
        # 无人机实际飞行高度
        fig.add_trace(go.Scatter(x=dists, y=path[:, 2], mode='lines', 
                                 line=dict(color='#00FF00', width=3), name='无人机实际轨迹'))

        fig.update_layout(
            xaxis_title="水平飞行总里程 (米)", yaxis_title="绝对海拔 (米)",
            hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig

if __name__ == "__main__":
    st.set_page_config(page_title="无人机 4D 时空规划系统", page_icon="🚁", layout="wide")
    ui = DronePlanningUI()
    ui.create_sidebar_config()
    ui.create_main_interface()