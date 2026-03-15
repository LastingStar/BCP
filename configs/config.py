"""
无人机路径规划仿真配置模块 (黄金参数版)
专为大创项目 4D TKE 极值风险规避演示调优
"""

from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class SimulationConfig:
    # =========================
    # 🗺️ 地图与环境参数
    # =========================
    map_path: str = r'C:\Users\20340\Desktop\project1\Bernese_Oberland_46.6241_8.0413.png'
    map_size_km: float = 17.28
    min_alt: float = 563.0
    max_alt: float = 3985.4
    target_size: Tuple[int, int] = (300, 300)
    time_of_day: str = 'Night' 
    
    # =========================
    # 💨 基础风场参数
    # =========================
    env_wind_u: float = -12.0
    env_wind_v: float = -5.0
    k_slope: float = 8.0       
    max_wind_speed: float = 30.0 

    # =========================
    # ⛈️ 动态移动风暴 (大创杀手锏)
    # =========================
    enable_storms: bool = True   # 🌟 必须开启，展示动态避障
    storm_count: int = 3
    storm_radius_range_m: Tuple[float, float] = (500.0, 1500.0) # 调大了风暴，视觉更震撼
    storm_max_speed_mps: float = 15.0
    storm_movement_speed_mps: float = 5.0 # 让风暴跑快点，动画更好看
    storm_lifetime_s: float = 800.0
    storm_strength_scale: float = 1.5
    wind_seed: int = 2024

    # =========================
    # ⚠️ TKE 极值概率风险模型 (核心创新)
    # =========================
    drone_response_freq_N0: float = 3.0
    drone_robustness_K: float = 200.0
    # 🌟 稍微调低致命惩罚，允许无人机在绝境中进行“高风险突防”，防止无路可走
    fatal_crash_penalty_j: float = 80000.0 

    tke_shear_coeff: float = 0.2   
    tke_wake_coeff: float = 0.002  
    tke_slope_coeff: float = 0.05  

    # =========================
    # 🚫 静态禁飞区 (NFZ)
    # =========================
    enable_nfz: bool = True  # 🌟 重新开启禁飞区
    nfz_list_km: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (2.0, 2.0, 1.5),    
        (-3.0, 1.0, 1.2)    
    ])

    # =========================
    # 🚁 无人机物理参数
    # =========================
    drone_speed: float = 15.0       
    drone_mass: float = 0.895       
    air_density: float = 1.225      
    frontal_area: float = 0.057     
    base_power: float = 130.0       
    drag_coeff: float = 0.03
    # 🌟 放开功率限制：让无人机在强逆风时能“大力出奇迹”，避免方向被风锁死
    max_power: float = 4000.0       

    min_speed: float = 3.0          
    speed_decrement: float = 2.0    
    
    # =========================
    # 🧠 4D 规划器参数
    # =========================
    # 🌟 恢复灵魂参数：1.0代表完全融入物理与概率计算
    k_wind: float = 1.0             
    risk_factor: float = 20.0       
    max_steps: int = 500000

    noise_level: float = 0.0
    z_step: float = 50.0       
    max_ceiling: float = 5000.0 
    
    # 🌟 恢复垂直权重：体现爬山的艰难
    z_weight: float = 1.2      

    # 🚀 救命参数：加权 A* (Weighted A*) 核心因子
    # 从 1.0 提高到 3.5。强迫 A* 算法变得贪婪，像导弹一样直奔终点，极大提升长距离搜索速度！
    heuristic_safety_factor: float = 3.5  

    gravity: float = 9.81
    takeoff_altitude_agl: float = 50.0

    # =========================
    # 🔋 动态任务与电池设置
    # =========================
    mission_update_interval_s: float = 150.0
    max_replans: int = 100
    max_mission_time_s: float = 3600.0
    cruise_speed_mps: float = 15.0

    battery_capacity_j: float = 1500000.0 # 稍微增加点电池，应对长途跋涉
    reserve_energy_ratio: float = 0.15

    # =========================
    # 🎯 目标检查与时变风场
    # =========================
    goal_check_mode: str = "3d_distance"
    goal_tolerance_3d_m: float = 50.0 # 放宽到达判定，接触即算成功
    goal_tolerance_xy_m: float = 40.0
    goal_tolerance_z_m: float = 20.0

    wind_model_type: str = "slope"
    wind_time_model_type: str = "smooth_periodic"
    wind_time_scale_s: float = 300.0
    wind_direction_variation_deg: float = 20.0
    wind_speed_variation_ratio: float = 0.25

    planner_verbose: bool = False