"""
无人机路径规划仿真配置模块

此模块定义了仿真系统的所有配置参数，包括地图、风场、无人机物理参数、
规划器设置、电池管理和任务参数等。所有面向用户的数值设置都集中在此处，
便于调整和维护。

配置参数分类：
- 地图与环境参数：DEM地图路径、尺寸、高程范围等
- 风场参数：基础风速、地形风系数、风暴参数等
- 无人机物理参数：质量、空气密度、功率参数等
- 规划器参数：搜索步长、风险权重等
- 状态估计器参数：噪声水平等
- 3D规划参数：垂直步长、最大高度等
- 动态任务设置：重新规划间隔、最大任务时间等
- 电池/能量设置：电池容量、安全余量等
- 目标检查设置：目标到达判定条件
- 时变风场设置：风场模型选择、时变参数
- 任务起终点设置：起点终点位置偏移
- 演示参数设置：演示模式下的便捷配置

使用说明：
- 修改参数值时请注意单位和物理意义
- 布尔参数控制功能启用/禁用
- 列表参数如nfz_list_km支持多个禁飞区定义

作者：[你的名字]
版本：1.0
日期：2024年12月
"""

from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class SimulationConfig:
    """
    仿真系统主配置类

    包含所有仿真参数的配置，使用dataclass自动生成初始化方法。
    参数分组清晰，便于管理和修改。

    属性：
        所有参数都有默认值，可以通过实例化时覆盖
        使用field(default_factory=...)处理可变默认值
    """

    # --- 地图与环境参数 ---
    # DEM数字高程模型文件路径
    map_path: str = r'C:\Users\20340\Desktop\project1\Bernese_Oberland_46.6241_8.0413.png'

    # 地图尺寸（千米），影响坐标系范围
    map_size_km: float = 17.28

    # DEM数据的最小和最大高程值（米）
    min_alt: float = 563.0
    max_alt: float = 3985.4

    # 目标分辨率尺寸（像素），影响计算精度和内存使用
    target_size: Tuple[int, int] = (300, 300)

    # 时间段设置：'Day' 或 'Night'，影响坡度风方向
    time_of_day: str = 'Night' # 'Day' 或 'Night'
    
    # --- 风场参数 (为了匹配 Mavic 4 的抗风能力，必须降低风速) ---
    # 环境基础风速U分量（东向风，正东为正，米/秒）
    env_wind_u: float = -12.0

    # 环境基础风速V分量（北向风，正北为正，米/秒）
    env_wind_v: float = -5.0

    # 地形坡度风系数，控制山地风增强程度
    k_slope: float = 8.0       # 降低地形风系数，防止局部风速爆表

    # 风速最大限制（米/秒），防止物理不合理的高风速
    max_wind_speed: float = 30.0 # 强制风速硬顶，防止出现非物理的极端风

    # --- 风暴 / 强风区 (移动风暴) ---
    # 是否启用移动风暴功能
    enable_storms: bool = True

    # 风暴数量，影响计算复杂度和视觉效果
    storm_count: int = 3

    # 风暴影响半径范围（米），随机选择
    storm_radius_range_m: Tuple[float, float] = (300.0, 1000.0)

    # 风暴最大风速（米/秒）
    storm_max_speed_mps: float = 15.0

    # 风暴移动速度（米/秒）
    storm_movement_speed_mps: float = 2.0

    # 风暴存活时间（秒）
    storm_lifetime_s: float = 600.0

    # 风暴强度缩放系数
    storm_strength_scale: float = 1.5

    # 随机种子，保证结果可重现
    wind_seed: int = 0

     # =========================
    # 🌟 新增：基于 TKE 的极值概率风险模型参数
    # =========================
    # 无人机响应特征频率 (Hz)，决定对阵风的敏感度
    drone_response_freq_N0: float = 3.0

    # 抗扰鲁棒性系数 K_robust，越大说明无人机越抗风 (经验值)
    drone_robustness_K: float = 200.0

    # 坠机致命惩罚权重 (W_fatal)，转化为等效焦耳能量惩罚
    fatal_crash_penalty_j: float = 150000.0

    # TKE 计算系数
    tke_shear_coeff: float = 0.2   # 从 0.5 降到 0.2
    tke_wake_coeff: float = 0.002  # 从 0.02 降到 0.002 (非常关键)
    tke_slope_coeff: float = 0.05  # 从 0.1 降到 0.05

    # =========================
    # 🌟 新增：禁飞区 (No-Fly Zones) 配置
    # =========================
    # 是否启用禁飞区功能
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
    # 风场权重系数，平衡风阻和距离
    k_wind: float = 1.0             # 保持默认，让物理引擎决定

    # 最大搜索步数，防止无限循环
    max_steps: int = 1000000

    # 风险权重因子，平衡安全性和效率
    risk_factor: float = 20.0       # 重新把风险加回来，但别太大(50太大了)

    # --- 状态估计器参数 ---
    # 状态估计噪声水平，0表示无噪声
    noise_level: float = 0.0

     # --- 3D 规划参数 ---
    # 垂直步长 (每层高度差 50米)
    z_step: float = 50.0       # 垂直步长 (每层高度差 50米)

    # 最大飞行高度 (相对于起飞点/地面)
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
    # 动态任务设置
    # =========================

    # 重新规划间隔时间（秒）
    mission_update_interval_s: float = 30.0

    # 最大重新规划尝试次数
    max_replans: int = 50

    # 最大任务持续时间（秒）
    max_mission_time_s: float = 3600.0

    # 任务执行期间使用的巡航速度（m/s）
    cruise_speed_mps: float = 12.0

    # =========================
    # 电池/能量设置
    # =========================

    # 总可用电池能量（J）
    battery_capacity_j: float = 1200000.0

    # 最小储备能量比例，例如 0.15 表示保留 15% 未使用
    reserve_energy_ratio: float = 0.15

    # =========================
    # 目标检查设置
    # =========================

    # 目标检查模式：
    # - "3d_distance"
    # - "xy_z_tolerance"
    goal_check_mode: str = "3d_distance"

    # 到目标的 3D 距离容差（m）
    goal_tolerance_3d_m: float = 25.0

    # 到目标的水平距离容差（m）
    goal_tolerance_xy_m: float = 20.0

    # 到目标的垂直距离容差（m）
    goal_tolerance_z_m: float = 10.0

    # =========================
    # 时变风场设置
    # =========================

    # 风场模型类型: "slope" (基于地形坡度的风场模型)
    wind_model_type: str = "slope"

    # 时变风模型类型
    wind_time_model_type: str = "smooth_periodic"

    # 风变化的特征时间尺度（秒）
    wind_time_scale_s: float = 300.0

    # 背景风方向变化的最大幅度（度）
    wind_direction_variation_deg: float = 20.0

    # 背景风速度变化的相对比例
    wind_speed_variation_ratio: float = 0.25

    planner_verbose: bool = False

    # =========================
    # 任务起终点设置
    # =========================

    # 起点相对于地图左下角的X偏移（米）
    start_offset_x_m: float = 300.0

    # 起点相对于地图左下角的Y偏移（米）
    start_offset_y_m: float = 300.0

    # 终点相对于地图左下角的X偏移（米）
    goal_offset_x_m: float = 1200.0

    # 终点相对于地图左下角的Y偏移（米）
    goal_offset_y_m: float = 900.0

    # =========================
    # 演示参数设置（面向用户的便捷配置）
    # =========================

    # 演示模式下的最大重新规划次数
    demo_max_replans: int = 4

    # 演示模式下的最大任务时间（秒）
    demo_max_mission_time_s: float = 240.0

    # 演示模式下的重新规划间隔（秒）
    demo_mission_update_interval_s: float = 20.0

    # 演示模式下的巡航速度（m/s）
    demo_cruise_speed_mps: float = 18.0

    # 演示模式下的最大规划步骤
    demo_max_steps: int = 12000

    # 演示模式下是否启用规划器详细输出
    demo_planner_verbose: bool = False

    # 演示模式下的风场权重调整
    demo_k_wind: float = 0.8

    # 演示模式下的风险因子调整
    demo_risk_factor: float = 0.3

    # 演示模式下的电池容量调整（焦耳）
    demo_battery_capacity_j: float = 800000.0