import numpy as np
from typing import Callable, List, Tuple
from configs.config import SimulationConfig


class PhysicsEngine:
    """
    无人机物理引擎：负责计算空气动力学、功率与能耗。

    所有输入输出单位均采用 SI：
    - 速度: m/s
    - 功率: W
    - 距离: m
    - 能量: J
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

        # 基于默认巡航速度估算的最低平飞功率
        self.p_min = (
            0.5
            * self.config.air_density
            * self.config.drag_coeff
            * self.config.frontal_area
            * (self.config.drone_speed ** 3)
            + self.config.base_power
        )

        # 仅供兼容旧逻辑参考使用
        self.energy_per_meter = self.p_min / self.config.drone_speed

    def power_for_speed(self, v_air_mag: float) -> float:
        """
        计算给定相对空气速度下的总功率（W）。

        P_total = P_base + P_drag
        P_drag = 0.5 * rho * Cd * A * v_air^3
        """
        return (
            0.5
            * self.config.air_density
            * self.config.drag_coeff
            * self.config.frontal_area
            * (v_air_mag ** 3)
            + self.config.base_power
        )

    def estimate_power_from_vectors(
        self,
        ground_velocity_xyz: np.ndarray,
        wind_velocity_xyz: np.ndarray,
    ) -> float:
        """
        根据地速向量与风速向量估算总功率（W）。

        参数:
        - ground_velocity_xyz: 无人机相对地面的速度向量 [vx, vy, vz]
        - wind_velocity_xyz: 风速向量 [wx, wy, wz]

        返回:
        - total_power_w: 总功率（W）
        """
        v_air_vec = ground_velocity_xyz - wind_velocity_xyz
        v_air_mag = np.linalg.norm(v_air_vec)

        power_drag_and_base = self.power_for_speed(v_air_mag)

        # 爬升功率，仅在上升时计入
        vertical_speed = ground_velocity_xyz[2]
        power_climb = max(0.0, self.config.drone_mass * 9.81 * vertical_speed)

        return power_drag_and_base + power_climb

    def find_feasible_speed(
        self,
        v_ground: np.ndarray,
        v_wind: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        在给定风场下查找一个可行地速及对应功率。

        返回:
        - feasible: 是否可行
        - power_w: 对应功率
        - used_speed_mps: 实际采用的速度大小
        """
        target_speed = np.linalg.norm(v_ground)
        if target_speed <= 0:
            return True, self.config.base_power, 0.0

        direction = v_ground / target_speed
        current_speed = target_speed

        while current_speed >= self.config.min_speed:
            v_ground_vec = direction * current_speed

            # 若输入是二维向量，补成三维
            if v_ground_vec.shape[0] == 2:
                v_ground_vec = np.array([v_ground_vec[0], v_ground_vec[1], 0.0])

            if v_wind.shape[0] == 2:
                v_wind_3d = np.array([v_wind[0], v_wind[1], 0.0])
            else:
                v_wind_3d = v_wind

            power = self.estimate_power_from_vectors(v_ground_vec, v_wind_3d)

            if power <= self.config.max_power:
                return True, power, current_speed

            current_speed -= self.config.speed_decrement

        return False, float("inf"), 0.0

    def calculate_power(self, v_ground: np.ndarray, v_wind: np.ndarray) -> float:
        """
        兼容旧接口，返回可行功率；不可行则返回 inf。
        """
        feasible, power, _ = self.find_feasible_speed(v_ground, v_wind)
        return power if feasible else float("inf")

    def calculate_energy(self, power: float, distance_m: float) -> float:
        """
        兼容旧接口：按默认 drone_speed 估算飞行指定距离的能量消耗。
        """
        if power == float("inf"):
            return float("inf")

        time_s = distance_m / self.config.drone_speed
        return power * time_s

    def estimate_segment_energy(
        self,
        p0_xyz: np.ndarray,
        p1_xyz: np.ndarray,
        wind_velocity_xyz: np.ndarray,
        cruise_speed_mps: float,
    ) -> Tuple[float, float, float]:
        """
        对一段路径估计能耗、时间和平均功率。

        返回:
        - segment_energy_j
        - segment_time_s
        - segment_power_w
        """
        delta = p1_xyz - p0_xyz
        segment_distance_m = np.linalg.norm(delta)

        if segment_distance_m <= 1e-9:
            return 0.0, 0.0, 0.0

        direction = delta / segment_distance_m
        ground_velocity_xyz = direction * cruise_speed_mps

        segment_time_s = segment_distance_m / cruise_speed_mps
        segment_power_w = self.estimate_power_from_vectors(
            ground_velocity_xyz, wind_velocity_xyz
        )
        segment_energy_j = segment_power_w * segment_time_s

        return segment_energy_j, segment_time_s, segment_power_w

    def estimate_path_energy(
        self,
        path_xyz: List[Tuple[float, float, float]],
        wind_sampler: Callable[[float, float, float], np.ndarray],
        cruise_speed_mps: float,
    ) -> float:
        """
        估计整条路径的总能耗。

        参数:
        - path_xyz: 路径点列表 [(x, y, z), ...]
        - wind_sampler: 风采样函数，输入 (x, y, z)，输出 np.ndarray([wx, wy, wz])
        - cruise_speed_mps: 执行速度

        返回:
        - total_energy_j
        """
        if len(path_xyz) < 2:
            return 0.0

        total_energy_j = 0.0

        for i in range(len(path_xyz) - 1):
            p0 = np.array(path_xyz[i], dtype=float)
            p1 = np.array(path_xyz[i + 1], dtype=float)

            midpoint = 0.5 * (p0 + p1)
            wind_velocity_xyz = wind_sampler(midpoint[0], midpoint[1], midpoint[2])

            if wind_velocity_xyz.shape[0] == 2:
                wind_velocity_xyz = np.array(
                    [wind_velocity_xyz[0], wind_velocity_xyz[1], 0.0]
                )

            segment_energy_j, _, _ = self.estimate_segment_energy(
                p0, p1, wind_velocity_xyz, cruise_speed_mps
            )

            total_energy_j += segment_energy_j

        return total_energy_j