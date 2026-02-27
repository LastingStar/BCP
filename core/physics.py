import numpy as np
from configs.config import SimulationConfig

class PhysicsEngine:
    """无人机物理引擎：负责计算空气动力学、功率与能耗"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        # 计算无风状态下的最低功率和每米能耗 (用于启发式函数)
        self.p_min = self.config.drag_coeff * (self.config.drone_speed ** 3) + self.config.base_power
        self.energy_per_meter = self.p_min / self.config.drone_speed

    def calculate_power(self, v_ground: np.ndarray, v_wind: np.ndarray) -> float:
        """
        根据地速和风速计算所需电机功率
        :param v_ground: 无人机对地速度向量 [vx, vy]
        :param v_wind: 当前位置风速向量 [ux, vy]
        :return: 所需功率 (W)，若超出最大功率则返回 inf
        """
        v_air = v_ground - v_wind
        v_air_mag = np.linalg.norm(v_air)
        
        power = self.config.drag_coeff * (v_air_mag ** 3) + self.config.base_power
        
        if power > self.config.max_power:
            return float('inf')
        return power

    def calculate_energy(self, power: float, distance_m: float) -> float:
        """
        计算在特定功率下飞行指定距离消耗的能量
        """
        if power == float('inf'):
            return float('inf')
        time_s = distance_m / self.config.drone_speed
        return power * time_s