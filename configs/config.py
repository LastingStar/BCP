from dataclasses import dataclass
from typing import Tuple

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
    # 既然抗风上限是 12m/s，我们把环境风设为 6-8m/s，加上地形风刚好在极限边缘
    env_wind_u: float = 0.0   # 逆风 6m/s
    env_wind_v: float = 0.0   # 稍微加点侧风
    k_slope: float = 8.0       # 降低地形风系数，防止局部风速爆表
    max_wind_speed: float = 30.0 # 强制风速硬顶，防止出现非物理的极端风
    
    # --- 无人机物理参数 (基于 DJI Mavic 4 Pro 数据) ---
    # 巡航速度：数据说是 32.4km/h(9m/s)最省电，54km/h(15m/s)是长续航
    drone_speed: float = 15.0       # 选用 15m/s 作为任务速度
    
    # 悬停功率：95.3Wh / 45min ≈ 127W
    base_power: float = 130.0       
    
    # 风阻系数：反推所得
    drag_coeff: float = 0.03        
    
    # 最大功率：给足余量，防止仿真中稍微一顶风就报错
    max_power: float = 1200.0       
    
    # --- 规划器参数 ---
    k_wind: float = 1.0             # 保持默认，让物理引擎决定
    max_steps: int = 200000         
    risk_factor: float = 20.0       # 重新把风险加回来，但别太大(50太大了)
    
    # --- 状态估计器参数 ---
    noise_level: float = 0.0