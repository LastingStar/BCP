import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import tempfile
from PIL import Image

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
        # 使用 Streamlit 的 session_state 保持状态，防止组件刷新导致数据丢失
        if 'config' not in st.session_state:
            st.session_state.config = SimulationConfig()
        self.config = st.session_state.config

    def create_sidebar_config(self):
        """侧边栏：硬核科研参数配置面板"""
        st.sidebar.title("⚙️ 核心参数调优面板")
        st.sidebar.markdown("通过调整以下参数，观察算法的动态博弈行为。")

        # --- 1. 地图上传与设置 ---
        with st.sidebar.expander("🗺️ 地形与禁飞区", expanded=True):
            uploaded_file = st.file_uploader("上传自定义高程图 (灰度图 PNG/JPG)", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                # 将上传的文件保存为临时文件，供 MapManager 读取
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    self.config.map_path = tmp_file.name
                st.success("自定义地形加载成功！")

            self.config.map_size_km = st.slider("物理映射尺寸 (km)", 5.0, 50.0, self.config.map_size_km, 1.0)
            self.config.enable_nfz = st.checkbox("启用静态行政禁飞区 (NFZ)", self.config.enable_nfz)

        # --- 2. 微气象与风暴设置 ---
        with st.sidebar.expander("💨 微气象与风暴场", expanded=True):
            self.config.env_wind_u = st.slider("基础风速 U (m/s)", -25.0, 25.0, self.config.env_wind_u, 1.0)
            self.config.env_wind_v = st.slider("基础风速 V (m/s)", -25.0, 25.0, self.config.env_wind_v, 1.0)
            self.config.enable_storms = st.checkbox("启用动态移动风暴 (雷暴区)", self.config.enable_storms)
            if self.config.enable_storms:
                self.config.storm_count = st.slider("全局风暴数量", 1, 10, self.config.storm_count)
                self.config.storm_movement_speed_mps = st.slider("风暴移动速度 (m/s)", 1.0, 20.0, self.config.storm_movement_speed_mps, 1.0)

        # --- 3. TKE 极值风险模型 (大创核心) ---
        with st.sidebar.expander("⚠️ 极值风险模型 (核心创新)", expanded=True):
            st.markdown("<small>用于将 TKE 映射为坠机概率</small>", unsafe_allow_html=True)
            self.config.drone_robustness_K = st.slider("无人机抗扰鲁棒性 (K)", 10.0, 500.0, self.config.drone_robustness_K, 10.0)
            self.config.fatal_crash_penalty_j = st.slider("坠机致命惩罚 (J)", 10000.0, 500000.0, self.config.fatal_crash_penalty_j, 10000.0)
            self.config.k_wind = st.slider("风场代价整体权重", 0.0, 2.0, self.config.k_wind, 0.1)

        # --- 4. 无人机物理学 ---
        with st.sidebar.expander("🚁 无人机动力学", expanded=False):
            self.config.drone_mass = st.number_input("机体质量 (kg)", value=self.config.drone_mass)
            self.config.drone_speed = st.slider("巡航速度 (m/s)", 5.0, 30.0, self.config.drone_speed, 1.0)
            self.config.battery_capacity_j = st.slider("电池总容量 (kJ)", 100.0, 3000.0, self.config.battery_capacity_j/1000.0, 50.0) * 1000.0

    def create_main_interface(self):
        """主界面"""
        st.title("🚁 复杂气象下无人机 4D 轨迹规划系统")
        st.markdown("**核心算法**：基于 TKE 极值风险分布的 Spatio-Temporal A* 预测规划")
        st.markdown("---")

        # 动态计算坐标边界
        half_size = (self.config.map_size_km * 1000) / 2
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 起点坐标")
            start_x = st.number_input("X坐标 (米)", min_value=-half_size, max_value=half_size, value=-half_size*0.6, step=100.0)
            start_y = st.number_input("Y坐标 (米)", min_value=-half_size, max_value=half_size, value=-half_size*0.6, step=100.0)

        with col2:
            st.subheader("🎯 终点坐标")
            goal_x = st.number_input("X坐标 (米)", min_value=-half_size, max_value=half_size, value=half_size*0.6, step=100.0)
            goal_y = st.number_input("Y坐标 (米)", min_value=-half_size, max_value=half_size, value=half_size*0.6, step=100.0)

        if st.button("🚀 启动 4D 时空规划推演", type="primary", use_container_width=True):
            self.run_simulation((start_x, start_y), (goal_x, goal_y))

    def run_simulation(self, start_xy, goal_xy):
        """执行仿真并渲染图表"""
        # 1. 实例化组件 (每次运行重新实例化，确保读入最新 config)
        map_manager = MapManager(self.config)
        wind_model = WindModelFactory.create(self.config.wind_model_type, self.config, bounds=map_manager.get_bounds())
        estimator = StateEstimator(map_manager, wind_model, self.config)
        physics = PhysicsEngine(self.config)
        battery_manager = BatteryManager(self.config)
        planner = AStarPlanner(self.config, estimator, physics)
        executor = MissionExecutor(self.config, estimator, physics, battery_manager, planner)

        # 2. 运行核心算法
        with st.spinner("🤖 AI 正在进行 4D 时空前瞻推演与极值概率计算... (约需10-30秒)"):
            start_time = time.time()
            mission_result = executor.execute_mission(start_xy, goal_xy)
            calc_time = time.time() - start_time

        # 3. 展示结果指标
        st.subheader("📈 任务评估报告")
        summary = summarize_mission_result(mission_result)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("任务状态", "✅ 成功抵达" if summary['success'] else "❌ 规划失败")
        col2.metric("运算耗时", f"{calc_time:.2f} s")
        col3.metric("总耗电量", f"{summary['total_energy_used_j']/1000:.1f} kJ")
        col4.metric("实际飞行里程", f"{summary['executed_path_length_m']:.1f} m")

        if not mission_result.success:
            st.error(f"任务失败原因: {summary['failure_reason']}")
            return

        # 4. 展示 3D 交互地图
        st.subheader("🌍 3D 全景数字孪生")
        with st.spinner("正在渲染 3D 交互地图..."):
            fig_3d = self.create_3d_visualization(map_manager, mission_result)
            st.plotly_chart(fig_3d, use_container_width=True)

        # 5. 渲染动态 GIF (大创杀手锏)
        st.subheader("🎬 动态时空避障推演 (Spatio-Temporal Avoidance)")
        with st.spinner("正在合成高精度动态风场 GIF (生成较慢，请耐心等待)..."):
            gif_path = "web_temp_mission.gif"
            animator = MissionAnimator(self.config, estimator)
            animator.generate_gif(mission_result, start_xy, goal_xy, filename=gif_path)
            
            # 在网页中展示 GIF
            try:
                st.image(gif_path, caption="动态雷暴躲避推演图", use_container_width=True)
            except Exception as e:
                st.warning("GIF 生成存在问题，请检查后台日志。")

    def create_3d_visualization(self, map_manager, mission_result):
        """创建精美的 Plotly 3D 交互图"""
        path = np.array(mission_result.actual_flown_path_xyz)
        
        # 降采样地形以加速网页渲染
        step = max(1, map_manager.size_x // 60)
        X, Y = np.meshgrid(map_manager.x[::step], map_manager.y[::step])
        Z = map_manager.dem[::step, ::step]

        fig = go.Figure()

        # 绘制地形
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Earth',
            opacity=0.8,
            name='地形 (DEM)',
            showscale=False
        ))

        # 绘制 3D 轨迹
        fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode='lines',
            line=dict(color='#00FF00', width=6),
            name='无人机真实航迹'
        ))

        # 绘制起终点
        fig.add_trace(go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]], mode='markers', marker=dict(size=8, color='yellow'), name='起点'))
        fig.add_trace(go.Scatter3d(x=[path[-1, 0]], y=[path[-1, 1]], z=[path[-1, 2]], mode='markers', marker=dict(size=8, color='red'), name='终点'))

        fig.update_layout(
            scene=dict(
                xaxis_title='X (米)', yaxis_title='Y (米)', zaxis_title='海拔 (米)',
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3)
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig

if __name__ == "__main__":
    st.set_page_config(page_title="无人机 4D 路径规划系统", page_icon="🚁", layout="wide")
    ui = DronePlanningUI()
    ui.create_sidebar_config()
    ui.create_main_interface()