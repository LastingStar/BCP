import numpy as np
from typing import Tuple
from configs.config import SimulationConfig

class PhysicsEngine:
    """无人机物理引擎：负责计算空气动力学、功率与能耗

    所有输入输出单位均采用 SI，速度 m/s，功率 W，距离 m，能量 J。"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        # 最低功率估算（平飞，无风）
        # 使用简单模型 P = 0.5 * rho * Cd * A * v^3 + base_power
        self.p_min = 0.5 * self.config.air_density * self.config.drag_coeff * self.config.frontal_area * (self.config.drone_speed ** 3) + self.config.base_power
        # 每米能耗 (J/m)
        self.energy_per_meter = self.p_min / self.config.drone_speed

    def power_for_speed(self, v_air_mag: float) -> float:
        """返回给定迎风速度的空气阻力功率 (W)。

        P_drag = 0.5 * rho * Cd * A * v_air_mag^3
        """
        return 0.5 * self.config.air_density * self.config.drag_coeff * self.config.frontal_area * (v_air_mag ** 3) + self.config.base_power

    def find_feasible_speed(self, v_ground: np.ndarray, v_wind: np.ndarray) -> Tuple[bool, float, float]:
        """在给定的风场下查找一个可行的地速与对应功率。

        返回 (feasible, power_W, used_speed_m_s)。
        如果不可行，feasible=False, power=inf, used_speed=0
        """
        target_speed = np.linalg.norm(v_ground)
        if target_speed <= 0:
            return True, self.config.base_power, 0.0
        direction = v_ground / target_speed

        current_speed = target_speed
        while current_speed >= self.config.min_speed:
            v_ground_vec = direction * current_speed
            v_air = v_ground_vec - v_wind
            v_air_mag = np.linalg.norm(v_air)

            power = self.power_for_speed(v_air_mag)
            if power <= self.config.max_power:
                return True, power, current_speed

            current_speed -= self.config.speed_decrement
        return False, float('inf'), 0.0

    def calculate_power(self, v_ground: np.ndarray, v_wind: np.ndarray) -> float:
        """兼容旧接口，返回查找到的功率 (W)，不可达返回 inf。"""
        feasible, power, _ = self.find_feasible_speed(v_ground, v_wind)
        return power if feasible else float('inf')

    def calculate_energy(self, power: float, distance_m: float) -> float:
        """
        计算在特定功率下飞行指定距离消耗的能量
        """
        if power == float('inf'):
            return float('inf')
        time_s = distance_m / self.config.drone_speed
        return power * time_s