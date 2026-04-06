# --- START OF FILE drone_ui.py ---

import os
import sys
import time
from pathlib import Path

# 确保项目根目录在环境变量中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components  # 🌟 新增：用于加载本地 3D HTML

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from simulation.mission_executor import MissionExecutor
from simulation.swarm_mission_executor import SwarmMissionExecutor
from utils.animation_builder import MissionAnimator
from utils.visualizer_core import Visualizer

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

RESULTS_ROOT = Path(project_root) / "results"
DEFAULT_MODEL_PATH = Path(project_root) / "models" / "ppo_drone_stage3_obs31_run1_best" / "best_model.zip"


@st.cache_resource(show_spinner=False)
def load_rl_model_cached(model_path: str):
    if not PPO:
        return None
    model_file = Path(model_path)
    if not model_file.exists():
        return None
    return PPO.load(str(model_file), device="cpu")


def create_map_preview(map_manager: MapManager, start_xy, goal_xy):
    """创建 Plotly 交互式地图 2D 预览"""
    step = max(1, map_manager.size_x // 100)
    fig = go.Figure(
        data=go.Contour(
            z=map_manager.dem[::step, ::step],
            x=map_manager.x[::step],
            y=map_manager.y[::step],
            colorscale="Earth",
            contours_coloring="heatmap",
            colorbar=dict(title="海拔 (m)")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[start_xy[0]],
            y=[start_xy[1]],
            mode="markers+text",
            text=["起点"],
            textposition="top center",
            marker=dict(size=16, color="yellow", symbol="star", line=dict(width=2, color="black")),
            name="起点",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[goal_xy[0]],
            y=[goal_xy[1]],
            mode="markers+text",
            text=["终点"],
            textposition="top center",
            marker=dict(size=16, color="red", symbol="x", line=dict(width=2, color="black")),
            name="终点",
        )
    )
    fig.update_layout(
        height=500, 
        margin=dict(l=20, r=20, t=40, b=20), 
        title="当前配置任务地图预览",
        yaxis=dict(scaleanchor="x", scaleratio=1) # 强制正方形比例
    )
    return fig


def save_3d_interactive_html(map_manager: MapManager, mission_result, save_path: Path):
    """🌟 核心新增：生成并在本地保存真 3D 可交互地形与航迹图 (HTML格式)"""
    path_xyz = np.array(mission_result.actual_flown_path_xyz)
    if len(path_xyz) < 2:
        return

    # 降采样地形以保证浏览器流畅度 (网格控制在 60x60 左右)
    step = max(1, map_manager.size_x // 60)
    X, Y = np.meshgrid(map_manager.x[::step], map_manager.y[::step])
    Z = map_manager.dem[::step, ::step]

    fig = go.Figure()

    # 1. 绘制 3D 地形曲面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Earth', opacity=0.85, showscale=False, name='地形'
    ))

    # 2. 绘制 3D 飞行轨迹
    fig.add_trace(go.Scatter3d(
        x=path_xyz[:, 0], y=path_xyz[:, 1], z=path_xyz[:, 2],
        mode='lines', line=dict(color='#00FF00', width=6), name='实际飞行 3D 航迹'
    ))

    # 3. 绘制起终点
    fig.add_trace(go.Scatter3d(
        x=[path_xyz[0, 0]], y=[path_xyz[0, 1]], z=[path_xyz[0, 2]],
        mode='markers', marker=dict(size=8, color='yellow', line=dict(width=2, color='black')), name='起点'
    ))
    fig.add_trace(go.Scatter3d(
        x=[path_xyz[-1, 0]], y=[path_xyz[-1, 1]], z=[path_xyz[-1, 2]],
        mode='markers', marker=dict(size=8, color='red', line=dict(width=2, color='black')), name='终点/坠机点'
    ))

    # 4. 配置场景比例
    fig.update_layout(
        title="真 3D 航迹数字孪生 (鼠标左键拖拽旋转、滚轮缩放)",
        scene=dict(
            xaxis_title='X 坐标 (m)',
            yaxis_title='Y 坐标 (m)',
            zaxis_title='海拔高度 (m)',
            aspectmode='data'  # 强制 3D 比例对应真实物理尺寸
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # 写入到 HTML 文件
    fig.write_html(str(save_path), include_plotlyjs="cdn")


def apply_ui_config(ui_state: dict, disturbance_enabled: bool = False) -> SimulationConfig:
    """将 UI 的状态全面注入到底层 config 中"""
    config = SimulationConfig()
    config.curriculum_stage = 3
    config.enable_storms = True

    # 1. 地图与环境
    config.map_path = ui_state["map_path"]
    config.map_size_km = ui_state["map_size_km"]
    config.min_alt = ui_state["min_alt"]
    config.max_alt = ui_state["max_alt"]
    config.target_size = (ui_state["map_resolution"], ui_state["map_resolution"])
    config.time_of_day = ui_state["time_of_day"]
    config.env_wind_u = ui_state["env_wind_u"]
    config.env_wind_v = ui_state["env_wind_v"]
    config.enable_nfz = ui_state["enable_nfz"]

    # 2. 风暴与阵风
    config.wind_seed = ui_state["wind_seed"]
    config.storm_count = ui_state["storm_count"]
    config.enable_random_gusts = disturbance_enabled
    config.gust_trigger_prob = ui_state["gust_trigger_prob"]
    config.gust_duration_s = ui_state["gust_duration_s"]
    config.gust_min_speed_mps = ui_state["gust_min_speed_mps"]
    config.gust_max_speed_mps = ui_state["gust_max_speed_mps"]

    # 3. 物理与 A*
    config.cruise_speed_mps = ui_state["cruise_speed_mps"]
    config.drone_speed = ui_state["cruise_speed_mps"]
    config.max_power = ui_state["max_power"]
    config.heuristic_safety_factor = ui_state["heuristic_safety_factor"]
    config.max_replans = ui_state["max_replans"]

    # 4. 集群与护盾
    config.enable_support_shield_mode = ui_state["enable_support_shield_mode"]
    config.support_shield_master_radius_m = ui_state["support_shield_master_radius_m"]
    config.support_shield_offset_m = ui_state["support_shield_offset_m"]

    # 5. RL 参数
    config.gust_obs_noise_std = ui_state["gust_obs_noise_std"]

    return config


def run_mission_case(case_name: str, mode_name: str, disturbance_enabled: bool, ui_state: dict, rl_model, output_dir: Path):
    """运行测评单个案例（支持单机与编队）"""
    config = apply_ui_config(ui_state, disturbance_enabled)
    if mode_name != "rl":
        config.gust_obs_noise_std = 0.0

    start_xy = ui_state["matrix_start"]
    goal_xy = ui_state["matrix_goal"]
    is_swarm = ui_state["fleet_mode"] == "Swarm"

    map_manager = MapManager(config)
    wind_model = WindModelFactory.create(config.wind_model_type, config, bounds=map_manager.get_bounds())
    estimator = StateEstimator(map_manager, wind_model, config)
    physics = PhysicsEngine(config)
    battery = BatteryManager(config)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    vis = Visualizer(config, estimator)
    animator = MissionAnimator(config, estimator)

    start_t = time.time()

    if is_swarm:
        # 四机编队模式
        executor = SwarmMissionExecutor(config, estimator, physics, battery, master_mode=mode_name, rl_model=rl_model)
        mission_result = executor.execute_mission(start_xy, goal_xy)
        
        vis.plot_swarm_execution(mission_result, start_xy, goal_xy, save_dir=str(output_dir))
        vis.plot_swarm_elevation_profile(mission_result, save_dir=str(output_dir))
        animator.generate_swarm_gif(mission_result, start_xy, goal_xy, filename=str(output_dir / f"{case_name}.gif"))
    else:
        # 单机模式
        planner = AStarPlanner(config, estimator, physics)
        executor = MissionExecutor(config, estimator, physics, battery, planner)
        mission_result = executor.execute_mission(start_xy, goal_xy)
        
        vis.plot_single_mission_execution(mission_result, start_xy, goal_xy, method_name=mode_name.upper(), save_dir=str(output_dir))
        vis.plot_elevation_profile(mission_result, save_dir=str(output_dir), method_name=mode_name.upper())
        animator.generate_gif(mission_result, start_xy, goal_xy, filename=str(output_dir / f"{case_name}.gif"), physics_engine=physics)

    elapsed = time.time() - start_t

    # 🌟 无论单机还是多机，都生成一份真 3D 的 HTML 文件保存下来
    save_3d_interactive_html(map_manager, mission_result, output_dir / "3d_trajectory.html")

    return {
        "场景": case_name,
        "控制算法": mode_name.upper(),
        "微观扰动": "开启 (GUST)" if disturbance_enabled else "无 (CLEAN)",
        "任务成功": "✅" if mission_result.success else "❌",
        "终止原因": mission_result.failure_reason or "安全抵达",
        "飞行耗时(s)": round(mission_result.total_mission_time_s, 1),
        "消耗能量(kJ)": round(mission_result.total_energy_used_j / 1000.0, 1),
        "重规划次数": mission_result.total_replans,
        "计算耗时(s)": round(elapsed, 2),
        "output_dir": str(output_dir),
    }


def display_artifacts(output_dir: Path, gif_name: str | None = None):
    """端正美观地渲染生成的文件 (加入 3D 视图内嵌)"""
    # 查找静态图纸
    swarm_static = output_dir / "swarm_static_trajectories.png"
    single_static = output_dir / "analysis_01_terrain_and_paths.png"
    static_png = swarm_static if swarm_static.exists() else single_static

    swarm_prof = output_dir / "swarm_elevation_profile.png"
    single_prof = output_dir / "analysis_03_elevation_profile.png"
    profile_png = swarm_prof if swarm_prof.exists() else single_prof

    gif_path = output_dir / gif_name if gif_name else None
    if gif_path is None:
        gifs = list(output_dir.glob("*.gif"))
        gif_path = gifs[0] if gifs else None

    html_3d_path = output_dir / "3d_trajectory.html"

    # 第一排：居中显示大尺寸 4D 动图
    if gif_path and gif_path.exists():
        col_g1, col_g2, col_g3 = st.columns([1, 4, 1]) # 中间占比大，实现居中
        with col_g2:
            st.image(str(gif_path), caption=f"4D 时空动态飞行追踪 ({gif_path.name})", use_container_width=True)
            
    st.divider()

    # 第二排：左右对齐显示 2D 静态结果
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if static_png.exists():
            st.image(str(static_png), caption="最终静态拓扑/平面轨迹图", use_container_width=True)
    with col_s2:
        if profile_png.exists():
            st.image(str(profile_png), caption="飞行高度与地形时间剖面图", use_container_width=True)

    # 🌟 第三排：满宽展示可交互 3D 渲染图
    if html_3d_path.exists():
        st.divider()
        st.markdown("#### 🌍 真 3D 航迹数字孪生 (Interactive 3D View)")
        st.caption("提示：你可以使用鼠标左键旋转视角，滚轮放大缩小，悬停查看具体的空间坐标与海拔。")
        # 读取本地 HTML 并通过 iframe 嵌入 Streamlit
        with open(html_3d_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=600)
            
    st.caption(f"📂 成果文件保存路径: {output_dir}")


def list_result_dirs():
    """获取之前所有的结果文件夹"""
    if not RESULTS_ROOT.exists():
        return []
    return sorted([path for path in RESULTS_ROOT.iterdir() if path.is_dir()], key=lambda p: p.name)


def main():
    st.set_page_config(page_title="无人机风暴环境突防测控中心", page_icon="🛰️", layout="wide")
    st.title("🛰️ 4D 时空极值风险规避与协同控制中心")
    st.caption("支持自定义高程地图导入、单/多机编队切换，以及不同算法抗环境扰动的矩阵测评与 3D 可视化。")

    # ==========================
    # 侧边栏：多维度配置区
    # ==========================
    st.sidebar.header("🗺️ 地形与地图参数设定")
    
    # 1. 自定义地图上传
    uploaded_map = st.sidebar.file_uploader("1. 上传自定义高程图 (PNG/JPG)", type=["png", "jpg", "jpeg"], help="不上传则使用默认的瑞士雪山地图。建议上传正方形图片。")
    default_config = SimulationConfig()
    
    if uploaded_map is not None:
        temp_map_path = os.path.join(project_root, "temp_uploaded_map.png")
        with open(temp_map_path, "wb") as f:
            f.write(uploaded_map.getbuffer())
        map_path = temp_map_path
    else:
        map_path = default_config.map_path

    # 地图物理属性 (开放设置)
    map_size_km = st.sidebar.number_input("2. 地图物理边长 (km)", value=17.28, step=1.0, help="定义这张图片在真实世界里代表多宽。")
    col_alt1, col_alt2 = st.sidebar.columns(2)
    min_alt = col_alt1.number_input("最低海拔(m)", value=563.0, step=50.0)
    max_alt = col_alt2.number_input("最高海拔(m)", value=3985.4, step=50.0)
    map_res = st.sidebar.slider("3. 内部解析分辨率 (像素)", 100, 600, 300, 50, help="值越大网格越精细，但 A* 寻路会变慢。")
    
    st.sidebar.header("🌤️ 气象与环境")
    time_of_day = st.sidebar.selectbox("昼夜模式 (影响地形风向)", ["Day (白昼)", "Night (夜晚)"], index=1)
    time_val = "Day" if "Day" in time_of_day else "Night"
    
    col_wind1, col_wind2 = st.sidebar.columns(2)
    env_wind_u = col_wind1.number_input("恒定背景风向 X (m/s)", value=-3.0, step=1.0)
    env_wind_v = col_wind2.number_input("恒定背景风向 Y (m/s)", value=5.0, step=1.0)
    enable_nfz = st.sidebar.checkbox("启用静态禁飞区 (NFZ)", value=True)

    with st.sidebar.expander("🌪️ 动态风暴与微观阵风设定", expanded=False):
        wind_seed = st.number_input("随机风暴生成种子", min_value=0, value=37, step=1)
        storm_count = st.slider("动态风暴数量", 1, 8, 3)
        gust_trigger_prob = st.slider("随机阵风触发概率", 0.0, 0.1, 0.02, 0.01)
        gust_duration_s = st.slider("单次阵风持续时间 (s)", 2.0, 30.0, 8.0, 2.0)
        gust_min_speed_mps = st.number_input("阵风最低风速 (m/s)", value=4.0)
        gust_max_speed_mps = st.number_input("阵风最高风速 (m/s)", value=10.0)

    with st.sidebar.expander("🚁 无人机物理与 A* 参数", expanded=False):
        cruise_speed_mps = st.slider("无人机巡航速度 (m/s)", 5.0, 30.0, 15.0, 1.0)
        max_power = st.number_input("电机最大抗风功率限制 (W)", value=4000.0, step=100.0)
        heuristic_safety_factor = st.slider("A* 启发式贪婪加速因子", 1.0, 5.0, 2.0, 0.5)
        max_replans = st.number_input("遇到死胡同时全局最大重规划次数", value=100, step=10)

    with st.sidebar.expander("🛡️ 异构集群护盾设定", expanded=False):
        enable_support_shield_mode = st.checkbox("启用 Support (支援蜂) 物理抗风护盾", value=True)
        support_shield_master_radius_m = st.slider("威胁风暴进入多少米触发护盾", 800, 2500, 1400, 100)
        support_shield_offset_m = st.slider("支援蜂超前掩护距离 (m)", 200, 1000, 450, 50)

    with st.sidebar.expander("🧠 强化学习 (RL) 模型与传感器", expanded=False):
        model_path = st.text_input("PPO 神经网络权重路径", value=str(DEFAULT_MODEL_PATH))
        gust_obs_noise_std = st.slider("RL 雷达传感器环境噪声标准差", 0.0, 0.1, 0.01, 0.01)
        rl_model = load_rl_model_cached(model_path)
        if rl_model is None:
            st.error("❌ RL 模型未加载或不存在。")
        else:
            st.success("✅ PPO RL 模型已就绪！")

    # ==========================
    # 顶部控制：模式与起终点
    # ==========================
    st.subheader("🛠️ 任务编队与坐标设定")
    
    col_top1, col_top2 = st.columns([1, 2])
    with col_top1:
        fleet_mode_str = st.radio("选择出击机群规模：", ["四机编队 (FANET Swarm)", "单机模式 (Single Drone)"])
        fleet_mode = "Swarm" if "四机" in fleet_mode_str else "Single"
        
        if fleet_mode == "Single":
            st.warning("⚠️ 警告：单机模式视野严重受限！失去预警与护盾掩护后，迎面撞上风暴大概率坠机！此外，本系统单机回退使用传统 A* 算法。")
            
    with col_top2:
        # 🌟 根据用户设定的地图物理尺寸自动限制滑块的范围，防止越界
        half_m = float((map_size_km * 1000) / 2)
        def clamp(v): return max(-half_m, min(v, half_m))
        
        st.write("请在滑块上拖动或直接点击数字修改（单位：米）：")
        col_coord1, col_coord2 = st.columns(2)
        # 默认值优先取限制后的合理值
        start_x = col_coord1.slider("起点 X (m)", -half_m, half_m, clamp(-8000.0), 100.0)
        start_y = col_coord2.slider("起点 Y (m)", -half_m, half_m, clamp(-8000.0), 100.0)
        goal_x = col_coord1.slider("终点 X (m)", -half_m, half_m, clamp(6000.0), 100.0)
        goal_y = col_coord2.slider("终点 Y (m)", -half_m, half_m, clamp(7500.0), 100.0)

    # 汇总 UI 状态
    ui_state = {
        "map_path": map_path,
        "map_size_km": map_size_km,
        "min_alt": min_alt,
        "max_alt": max_alt,
        "map_resolution": map_res,
        "time_of_day": time_val,
        "env_wind_u": env_wind_u,
        "env_wind_v": env_wind_v,
        "enable_nfz": enable_nfz,
        "wind_seed": wind_seed,
        "storm_count": storm_count,
        "gust_trigger_prob": gust_trigger_prob,
        "gust_duration_s": gust_duration_s,
        "gust_min_speed_mps": gust_min_speed_mps,
        "gust_max_speed_mps": gust_max_speed_mps,
        "cruise_speed_mps": cruise_speed_mps,
        "max_power": max_power,
        "heuristic_safety_factor": heuristic_safety_factor,
        "max_replans": max_replans,
        "enable_support_shield_mode": locals().get("enable_support_shield_mode", False),
        "support_shield_master_radius_m": locals().get("support_shield_master_radius_m", 1400.0),
        "support_shield_offset_m": locals().get("support_shield_offset_m", 450.0),
        "gust_obs_noise_std": gust_obs_noise_std,
        "fleet_mode": fleet_mode,
        "matrix_start": (start_x, start_y),
        "matrix_goal": (goal_x, goal_y),
    }

    # ==========================
    # 主体界面 Tabs
    # ==========================
    preview_config = apply_ui_config(ui_state, disturbance_enabled=False)
    preview_map = MapManager(preview_config)

    tab_matrix, tab_artifacts = st.tabs(["🚀 任务推演大厅", "📂 历史成果画廊"])

    # ---------------------------
    # Tab 1: 算法矩阵对比
    # ---------------------------
    with tab_matrix:
        col_m1, col_m2 = st.columns([1.5, 1])
        with col_m1:
            st.plotly_chart(create_map_preview(preview_map, (start_x, start_y), (goal_x, goal_y)), use_container_width=True)
        with col_m2:
            st.info("💡 **测试说明**\n\n通过对强化学习(RL)与传统路径规划(A*)分别施加不可见微观阵风，测试抗扰动能力与生存率。单机模式强制仅运行 A*。")
            
            run_astar = st.checkbox("☑️ 对比项：传统 A* (A* Replanning)", value=True)
            run_rl = st.checkbox("☑️ 对比项：强化学习微观控制 (RL Agent)", value=True, disabled=(fleet_mode == "Single"))
            
            st.divider()
            run_clean = st.checkbox("☑️ 环境：无随机阵风基准环境 (Clean)", value=True)
            run_gust = st.checkbox("☑️ 环境：注入强微观随机阵风干扰 (Gust)", value=True)
            
            if st.button("▶️ 一键启动选中推演矩阵", type="primary", use_container_width=True):
                results = []
                cases = []
                
                if run_astar:
                    if run_clean: cases.append(("astar_clean (无阵风)", "astar", False, None))
                    if run_gust:  cases.append(("astar_gust (阵风干扰)", "astar", True, None))
                    
                if run_rl and fleet_mode == "Swarm" and rl_model is not None:
                    if run_clean: cases.append(("rl_clean (无阵风)", "rl", False, rl_model))
                    if run_gust:  cases.append(("rl_gust (阵风干扰)", "rl", True, rl_model))
                
                if not cases:
                    st.warning("⚠️ 请至少组合一种要测试的算法和环境！如果选择 RL 需确保在四机编队模式且模型已加载。")
                else:
                    progress_text = "正在执行 4D 物理仿真与图像/3D渲染，请耐心等待..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    for idx, (case_name, mode_name, disturbance_enabled, model) in enumerate(cases):
                        folder_prefix = "ui_swarm" if fleet_mode == "Swarm" else "ui_single"
                        output_dir = RESULTS_ROOT / f"{folder_prefix}_{case_name.split()[0]}"
                        
                        res = run_mission_case(case_name, mode_name, disturbance_enabled, ui_state, model, output_dir)
                        results.append(res)
                        my_bar.progress((idx + 1) / len(cases), text=f"✅ 已完成: {case_name}")
                        
                    st.session_state["matrix_results"] = results
                    st.success("🎉 所有推演任务全部完成，结果已加载！")

        matrix_results = st.session_state.get("matrix_results", [])
        if matrix_results:
            st.divider()
            st.write("### 📊 推演结果与性能比对")
            summary_df = pd.DataFrame(matrix_results)
            st.dataframe(
                summary_df[["场景", "控制算法", "微观扰动", "任务成功", "终止原因", "飞行耗时(s)", "消耗能量(kJ)", "重规划次数"]],
                use_container_width=True,
            )
            st.write("### 🎞️ 动态复盘与 3D 可视化图集")
            for row in matrix_results:
                with st.expander(f"📍 {row['场景']} | 算法: {row['控制算法']} | 环境: {row['微观扰动']}"):
                    display_artifacts(Path(row["output_dir"]), gif_name=f"{row['场景']}.gif")

    # ---------------------------
    # Tab 2: 历史图库
    # ---------------------------
    with tab_artifacts:
        result_dirs = list_result_dirs()
        if not result_dirs:
            st.info("目前尚未生成任何实验数据。")
        else:
            selected_dir = st.selectbox("选择要回放的历史实验归档：", result_dirs, format_func=lambda p: p.name)
            if selected_dir:
                gifs = list(selected_dir.glob("*.gif"))
                gif_name = gifs[0].name if gifs else None
                display_artifacts(selected_dir, gif_name=gif_name)


if __name__ == "__main__":
    main()
