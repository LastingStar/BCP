from dataclasses import dataclass
from typing import Tuple

@dataclass
class SimulationConfig:
    # --- 地图与环境参数 ---
    map_path: str = 'Bernese_Oberland_46.6241_8.0413.png'
    map_size_km: float = 17.28
    min_alt: float = 536.0
    max_alt: float = 3985.4
    target_size: Tuple[int, int] = (300, 300)
    
    # --- 风场参数 ---
    env_wind_u: float = -6.0
    env_wind_v: float = -0
    k_slope: float = 5.0
    max_wind_speed: float = 30.0
    
    # --- 无人机物理参数 ---
    drone_speed: float = 25.0       # 无人机对地巡航速度 (m/s)
    base_power: float = 100.0       # 基础悬停/电子设备功率 (W)
    drag_coeff: float = 0.05        # 风阻系数
    max_power: float = 3000.0       # 电机最大输出功率 (W)
    
    # --- 规划器参数 ---
    k_wind: float = 1.0             # 风场代价权重 (0表示传统最短路径)
    max_steps: int = 200000      # A* 最大搜索步数
    risk_factor: float = 50.0       # 风险惩罚系数
    
    # --- 状态估计器参数 ---
    noise_level: float = 0.0        # 传感器高斯噪声标准差