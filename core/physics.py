"""
无人机物理引擎模块

此模块实现了无人机的空气动力学模型，包括功率计算、能耗估算和运动学模拟。
基于经典的空气动力学原理，考虑阻力、质量、风速等因素。

主要功能：
- 功率计算：根据速度计算所需功率
- 能耗估算：计算飞行过程中的能量消耗
- 运动学模拟：速度、加速度、位置更新
- 风力影响：考虑环境风对运动的影响

物理模型：
- 阻力功率：P_drag = 0.5 * ρ * Cd * A * v^3
- 总功率：P_total = P_drag + P_base
- 能耗：E = P * t
- 运动：F = ma, v = v0 + at, x = x0 + vt + 0.5at^2

单位：
- 长度：米 (m)
- 速度：米/秒 (m/s)
- 加速度：米/秒² (m/s²)
- 功率：瓦特 (W)
- 能量：焦耳 (J)
- 质量：千克 (kg)
- 密度：千克/立方米 (kg/m³)
- 面积：平方米 (m²)

作者：项目团队
版本：1.0.0
更新日期：2026-04-03
"""

import numpy as np  # 数值计算库，用于向量运算和数学函数
from typing import Callable, List, Tuple  # 类型提示，用于函数签名
from configs.config import SimulationConfig  # 仿真配置类


class PhysicsEngine:
    """
    无人机物理引擎：负责计算空气动力学、功率与能耗

    此类实现了完整的无人机物理模型，包括空气阻力、功率计算、
    能耗估算和运动状态更新。所有计算都基于经典物理学原理。

    属性：
    config (SimulationConfig): 仿真配置对象，包含物理参数
    p_min (float): 最低平飞功率，基于巡航速度计算
    energy_per_meter (float): 每米能耗，仅供兼容性使用

    物理参数（来自config）：
    - air_density: 空气密度 (kg/m³)
    - drag_coeff: 阻力系数 (无量纲)
    - frontal_area: 正面面积 (m²)
    - drone_mass: 无人机质量 (kg)
    - base_power: 基础功率 (W)
    - drone_speed: 巡航速度 (m/s)

    计算方法：
    - 功率 = 基础功率 + 阻力功率
    - 阻力功率 = 0.5 * ρ * Cd * A * v³
    - 能耗 = 功率 * 时间

    注意：
    所有输入输出都使用SI单位制，确保计算一致性。
    """

    def __init__(self, config: SimulationConfig):
        """
        初始化物理引擎

        根据配置参数计算基础物理常数，如最低平飞功率。

        参数：
        config (SimulationConfig): 包含所有物理参数的配置对象
        """
        self.config = config  # 保存配置引用

        # 基于默认巡航速度估算的最低平飞功率
        # 公式：P_min = 0.5 * ρ * Cd * A * v³ + P_base
        self.p_min = (
            0.5
            * self.config.air_density  # 空气密度
            * self.config.drag_coeff   # 阻力系数
            * self.config.frontal_area # 正面面积
            * (self.config.drone_speed ** 3)  # 速度的三次方
            + self.config.base_power   # 基础功率
        )

        # 每米能耗估算，仅供兼容旧逻辑参考使用
        # 公式：E/m = P_min / v_cruise
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
        P_total = P_base + P_drag + P_climb (新增重力爬升做功)
        """
        v_air_vec = ground_velocity_xyz - wind_velocity_xyz
        v_air_mag = np.linalg.norm(v_air_vec)

        # 1. 基础悬停功率与空气阻力功率
        power_drag_and_base = self.power_for_speed(v_air_mag)

        # 2. 🌟 新增：重力爬升功率 P = m * g * v_z
        vertical_speed = ground_velocity_xyz[2]  # 获取 Z 轴速度 (m/s)
        power_climb = 0.0
        
        if vertical_speed > 0:
            # 只有上升时才消耗额外功率克服重力
            power_climb = self.config.drone_mass * self.config.gravity * vertical_speed
        elif vertical_speed < 0:
            # 下降时，保守估计势能转化为废热（旋翼机不具备高效势能回收）
            # 或者如果你想模拟略微省电，可以减去一小部分： power_climb = 0.2 * m * g * v_z
            power_climb = 0.0

        # 总功率
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