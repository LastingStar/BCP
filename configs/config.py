from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class SimulationConfig:
    # --- 地图与环境参数 ---
    map_path: str = r'C:\Users\20340\Desktop\project1\Bernese_Oberland_46.6241_8.0413.png'
    map_size_km: float = 17.28
    min_alt: float = 563.0
    max_alt: float = 3985.4
    target_size: Tuple[int, int] = (300, 300)
    time_of_day: str = 'Night' # 'Day' 或 'Night'
    
    # --- 风场参数 (为了匹配 Mavic 4 的抗风能力，必须降低风速) ---
    env_wind_u: float = -12.0   
    env_wind_v: float = -5.0   
    k_slope: float = 8.0       # 降低地形风系数，防止局部风速爆表
    max_wind_speed: float = 30.0 # 强制风速硬顶，防止出现非物理的极端风

    # --- 风暴 / 强风区 (移动风暴) ---
    enable_storms: bool = True
    storm_count: int = 3
    storm_radius_range_m: Tuple[float, float] = (300.0, 1000.0)
    storm_max_speed_mps: float = 15.0
    storm_movement_speed_mps: float = 2.0
    storm_lifetime_s: float = 600.0
    storm_strength_scale: float = 1.5
    wind_seed: int = 0

     # =========================
    # 🌟 新增：基于 TKE 的极值概率风险模型参数
    # =========================
    # 无人机响应特征频率 (Hz)，决定对阵风的敏感度
    drone_response_freq_N0: float = 3.0  
    # 抗扰鲁棒性系数 K_robust，越大说明无人机越抗风 (经验值)
    drone_robustness_K: float = 200      
    # 坠机致命惩罚权重 (W_fatal)，转化为等效焦耳能量惩罚
    fatal_crash_penalty_j: float = 150000.0
    # TKE 计算系数
    tke_shear_coeff: float = 0.2   # 从 0.5 降到 0.2
    tke_wake_coeff: float = 0.002  # 从 0.02 降到 0.002 (非常关键)
    tke_slope_coeff: float = 0.05  # 从 0.1 降到 0.05

    # =========================
    # 🌟 新增：禁飞区 (No-Fly Zones) 配置
    # =========================
    enable_nfz: bool = True
    # 手动定义禁飞区列表: [(x_km, y_km, radius_km), ...]
    # 坐标相对于地图中心(0,0)，这里我们放两个有代表性的禁飞区
    nfz_list_km: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (2.0, 2.0, 1.5),    # 在起点终点路线上放一个半径1.5km的禁飞区
        (-3.0, 1.0, 1.2)    # 另一个禁飞区
    ])

    # --- 无人机物理参数 (基于 DJI Mavic 4 Pro 数据) ---
    # 巡航速度：数据说是 32.4km/h(9m/s)最省电，54km/h(15m/s)是长续航
    drone_speed: float = 15.0       # m/s

    # 机体质量 (kg)，用于计算重力功率
    drone_mass: float = 0.895       # Mavic 3 实测约 0.895kg

    # 空气密度 (kg/m^3)
    air_density: float = 1.225      # 标准海平面

    # 正面迎风面积 (m^2)
    frontal_area: float = 0.057     # 估计值

    # 悬停功率：95.3Wh / 45min ≈ 127W
    base_power: float = 130.0       # W

    # 风阻系数（无量纲 Cd），实际组合在 PhysicsEngine 中
    drag_coeff: float = 0.03        

    # 速度搜索参数
    min_speed: float = 3.0          # m/s，下限巡航速度
    speed_decrement: float = 2.0    # m/s，降速步长

    # 最大功率：给足余量，防止仿真中稍微一顶风就报错
    max_power: float = 2500.0       # W
    
    # --- 规划器参数 ---
    k_wind: float = 1.0             # 保持默认，让物理引擎决定
    max_steps: int = 1000000         
    risk_factor: float = 20.0       # 重新把风险加回来，但别太大(50太大了)
    
    # --- 状态估计器参数 ---
    noise_level: float = 0.0

     # --- 3D 规划参数 ---
    z_step: float = 50.0       # 垂直步长 (每层高度差 50米)
    max_ceiling: float = 2000.0 # 最大飞行高度 (相对于起飞点/地面)
    
    # 垂直距离的权重 (爬升很难，所以垂直距离更"贵")
    z_weight: float = 2.0      # 爬 1米 等效于 平飞 5米

    # 重力加速度 (m/s^2)
    gravity: float = 9.81

    # 启发函数安全系数
    heuristic_safety_factor: float = 1.5

    # 起飞离地高度 (m AGL)
    takeoff_altitude_agl: float = 50.0

    # =========================
    # Dynamic mission settings
    # =========================

    # Replanning interval in seconds
    mission_update_interval_s: float = 30.0

    # Maximum number of replanning attempts
    max_replans: int = 50

    # Maximum mission duration in seconds
    max_mission_time_s: float = 3600.0

    # Cruise speed used during mission execution (m/s)
    cruise_speed_mps: float = 12.0

    # =========================
    # Battery / energy settings
    # =========================

    # Total available battery energy (J)
    battery_capacity_j: float = 1200000.0

    # Minimum reserve energy ratio, e.g. 0.15 means keep 15% unused
    reserve_energy_ratio: float = 0.15

    # =========================
    # Goal check settings
    # =========================

    # Goal check mode:
    # - "3d_distance"
    # - "xy_z_tolerance"
    goal_check_mode: str = "3d_distance"

    # 3D distance tolerance to goal (m)
    goal_tolerance_3d_m: float = 25.0

    # Horizontal distance tolerance to goal (m)
    goal_tolerance_xy_m: float = 20.0

    # Vertical distance tolerance to goal (m)
    goal_tolerance_z_m: float = 10.0

    # =========================
    # Time-varying wind settings
    # =========================

    # 风场模型类型: "slope" (默认) / "storm" (带移动风暴)
    wind_model_type: str = "slope"

    # Time-varying wind model type
    wind_time_model_type: str = "smooth_periodic"

    # Characteristic time scale for wind variation (s)
    wind_time_scale_s: float = 300.0

    # Maximum background wind direction variation (degrees)
    wind_direction_variation_deg: float = 20.0

    # Relative background wind speed variation ratio
    wind_speed_variation_ratio: float = 0.25

    planner_verbose: bool = False