from dataclasses import dataclass
from typing import Tuple

@dataclass
class SimulationConfig:
    # --- 地图与环境参数 ---
    map_path: str = r'D:\BCP\BCP-main\BCP-main\Bernese_Oberland_46.6241_8.0413.png'
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
    
    # 启发函数中，垂直距离的权重 (爬升很难，所以垂直距离更"贵")
    z_weight: float = 2.0      # 爬 1米 等效于 平飞 5米

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

    # Time-varying wind model type
    wind_time_model_type: str = "smooth_periodic"

    # Characteristic time scale for wind variation (s)
    wind_time_scale_s: float = 300.0

    # Maximum background wind direction variation (degrees)
    wind_direction_variation_deg: float = 20.0

    # Relative background wind speed variation ratio
    wind_speed_variation_ratio: float = 0.25