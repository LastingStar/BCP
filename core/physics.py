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
        智能功率计算：如果功率不够，自动尝试降速
        """
        # 1. 获取目标地速的大小和方向
        target_speed = np.linalg.norm(v_ground)
        if target_speed == 0: return self.config.base_power
        direction = v_ground / target_speed
        
        # 2. 尝试全速飞行
        current_speed = target_speed
        
        # 循环降速尝试 (比如从 15m/s -> 10m/s -> 5m/s)
        # 最低允许速度：3 m/s
        while current_speed >= 3.0:
            v_ground_vec = direction * current_speed
            v_air = v_ground_vec - v_wind
            v_air_mag = np.linalg.norm(v_air)
            
            power = self.config.drag_coeff * (v_air_mag ** 3) + self.config.base_power
            
            if power <= self.config.max_power:
                # 成功！虽然速度慢了点，但能飞过去
                # 注意：这里我们返回功率，但在 Planner 里算能量时
                # 实际上应该用这个新的 current_speed 来算时间。
                # 为了不改动 Planner 的接口，我们这里做一个近似处理：
                # 我们返回一个“等效的惩罚功率”。
                # 因为速度慢了，时间变长了 (t' = t * target/current)
                # Energy = P * t' = P * t * (target/current)
                # 等效 P_equiv = P * (target/current)
                return power * (target_speed / current_speed)
            
            # 功率还是太大，减速 2m/s 再试
            current_speed -= 2.0
            
        # 实在飞不动了
        return float('inf')

    def calculate_energy(self, power: float, distance_m: float) -> float:
        """
        计算在特定功率下飞行指定距离消耗的能量
        """
        if power == float('inf'):
            return float('inf')
        time_s = distance_m / self.config.drone_speed
        return power * time_s