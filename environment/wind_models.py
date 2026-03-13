import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from configs.config import SimulationConfig
from typing import List, Optional, Tuple


class BaseWindModel(ABC):
    """风场模型抽象基类。"""

    @abstractmethod
    def get_wind(
        self,
        x: float,
        y: float,
        z: float,
        terrain_gradient: Tuple[float, float],
        z0: float,
        t_s: float = 0.0,
    ) -> np.ndarray:
        """
        计算指定位置、指定时刻的风矢量。

        Parameters
        ----------
        x, y : float
            地面坐标（米）。
        z : float
            相对于地面的高度（AGL），单位米。
        terrain_gradient : Tuple[float, float]
            地形梯度 (gx, gy)。
        z0 : float
            地表粗糙度长度，单位米。
        t_s : float
            时间，单位秒。默认 0.0，兼容静态调用。
        """
        pass


@dataclass
class StormCell:
    """风暴单元（移动高风区）。"""

    center_xy: np.ndarray  # 当前中心 (x, y)
    velocity_xy: np.ndarray  # 运动速度向量 (m/s)
    radius_m: float  # 影响半径
    strength_mps: float  # 风暴引入的最大风速幅度
    birth_time_s: float  # 生成时间
    lifetime_s: float  # 存活时长
    last_t_s: float  # 上次更新时间

    def age(self, t_s: float) -> float:
        return t_s - self.birth_time_s

    def is_alive(self, t_s: float) -> bool:
        return self.age(t_s) < self.lifetime_s

    def update(self, t_s: float, bounds: Tuple[float, float, float, float]):
        """按时间推进风暴位置，并对边界做简单反弹处理。"""
        dt = t_s - self.last_t_s
        if dt <= 0:
            self.last_t_s = t_s
            return

        self.center_xy = self.center_xy + self.velocity_xy * dt

        min_x, max_x, min_y, max_y = bounds
        # 简单反弹边界
        if self.center_xy[0] < min_x or self.center_xy[0] > max_x:
            self.velocity_xy[0] *= -1.0
            self.center_xy[0] = np.clip(self.center_xy[0], min_x, max_x)
        if self.center_xy[1] < min_y or self.center_xy[1] > max_y:
            self.velocity_xy[1] *= -1.0
            self.center_xy[1] = np.clip(self.center_xy[1], min_y, max_y)

        self.last_t_s = t_s

    def wind_at(self, x: float, y: float) -> np.ndarray:
        """计算该点受该风暴影响的风速矢量"""
        dist = np.linalg.norm(np.array([x, y]) - self.center_xy)
        if dist >= self.radius_m:
            return np.array([0.0, 0.0], dtype=float)

        # 风暴影响权重（中心最大 -> 0）
        weight = (1.0 - (dist / self.radius_m)) ** 2
        direction = self.velocity_xy
        speed = np.linalg.norm(direction)
        if speed < 1e-6:
            # 若风暴静止，随机给个方向避免0除
            direction = np.array([1.0, 0.0])
            speed = 1.0
        direction_unit = direction / speed

        storm_speed = self.strength_mps * weight
        return direction_unit * storm_speed


class StormWindManager:
    """管理一组移动风暴单元（用于叠加到基础风场）"""

    def __init__(self, config: SimulationConfig, bounds: Tuple[float, float, float, float]):
        self.config = config
        self.bounds = bounds
        self.rng = np.random.default_rng(self.config.wind_seed)
        self.storms: List[StormCell] = []
        self._init_storms(0.0)

    def _init_storms(self, t_s: float):
        self.storms = [self._create_storm(t_s) for _ in range(self.config.storm_count)]

    def _create_storm(self, t_s: float) -> StormCell:
        min_x, max_x, min_y, max_y = self.bounds
        center_x = self.rng.uniform(min_x, max_x)
        center_y = self.rng.uniform(min_y, max_y)

        theta = self.rng.uniform(0, 2 * math.pi)
        speed = self.rng.uniform(0.0, self.config.storm_movement_speed_mps)
        velocity = np.array([math.cos(theta), math.sin(theta)]) * speed

        radius = self.rng.uniform(self.config.storm_radius_range_m[0], self.config.storm_radius_range_m[1])
        strength = self.rng.uniform(0.5, 1.0) * self.config.storm_strength_scale
        lifetime = self.rng.uniform(0.5, 1.5) * self.config.storm_lifetime_s

        return StormCell(
            center_xy=np.array([center_x, center_y], dtype=float),
            velocity_xy=velocity,
            radius_m=radius,
            strength_mps=strength,
            birth_time_s=t_s,
            lifetime_s=lifetime,
            last_t_s=t_s,
        )

    def sample(self, x: float, y: float, t_s: float) -> np.ndarray:
        if not self.config.enable_storms:
            return np.array([0.0, 0.0], dtype=float)

        # 更新风暴位置/生命周期
        for i, storm in enumerate(self.storms):
            if not storm.is_alive(t_s):
                self.storms[i] = self._create_storm(t_s)
            else:
                storm.update(t_s, self.bounds)

        # 叠加所有风暴的影响
        total = np.array([0.0, 0.0], dtype=float)
        for storm in self.storms:
            total += storm.wind_at(x, y)
        return total


class SlopeWindModel(BaseWindModel):
    """基于地形坡度的风场模型（包含时变背景风 + 昼夜坡度风 + 对数廓线 + 可选移动风暴）。"""

    def __init__(self, config: SimulationConfig, bounds: Optional[Tuple[float, float, float, float]] = None):
        self.config = config
        self.bounds = bounds
        if bounds is not None:
            self.storm_manager = StormWindManager(config, bounds)
        else:
            self.storm_manager = None

    def get_wind(
        self,
        x: float,
        y: float,
        z: float,
        terrain_gradient: Tuple[float, float],
        z0: float,
        t_s: float = 0.0,
    ) -> np.ndarray:
        gx, gy = terrain_gradient

        # 1. 时变背景环境风
        u_env, v_env = self._get_time_varying_background_wind(t_s)

        # 2. 坡度风
        k = self.config.k_slope
        if self.config.time_of_day == "Day":
            u_slope = k * gx
            v_slope = k * gy
        else:
            u_slope = -k * gx
            v_slope = -k * gy

        u_raw = u_env + u_slope
        v_raw = v_env + v_slope

        # 3. 高度对数风廓线修正
        factor = self._log_profile_factor(z, z0)
        u = u_raw * factor
        v = v_raw * factor

        # 4. 叠加移动风暴
        if self.storm_manager is not None and self.config.enable_storms:
            storm_uv = self.storm_manager.sample(x, y, t_s)
            u += storm_uv[0]
            v += storm_uv[1]

        # 5. 限幅
        u = np.clip(u, -self.config.max_wind_speed, self.config.max_wind_speed)
        v = np.clip(v, -self.config.max_wind_speed, self.config.max_wind_speed)

        return np.array([u, v], dtype=float)

    def _get_time_varying_background_wind(self, t_s: float) -> Tuple[float, float]:
        """
        当前阶段的时变背景风模型：
        - 风速做平滑周期波动
        - 风向做小幅周期摆动
        """
        u0 = self.config.env_wind_u
        v0 = self.config.env_wind_v

        base_speed = math.hypot(u0, v0)
        base_dir = math.atan2(v0, u0)

        # 若背景风为零，则给一个稳定零风输出
        if base_speed < 1e-9:
            return 0.0, 0.0

        time_scale_s = max(self.config.wind_time_scale_s, 1.0)
        omega = 2.0 * math.pi / time_scale_s

        speed_ratio = self.config.wind_speed_variation_ratio
        dir_variation_rad = math.radians(self.config.wind_direction_variation_deg)

        # 风速平滑波动
        speed_t = base_speed * (1.0 + speed_ratio * math.sin(omega * t_s))

        # 风向平滑摆动
        dir_t = base_dir + dir_variation_rad * math.sin(0.5 * omega * t_s)

        u_t = speed_t * math.cos(dir_t)
        v_t = speed_t * math.sin(dir_t)
        return u_t, v_t

    def _log_profile_factor(self, z: float, z0: float) -> float:
        """
        计算对数风廓线修正系数。
        引入严格的数值安全保护，防止 z <= z0 导致对数为负，或 z0=0 导致除零。
        """
        h_ref = 200.0  # 参考高度 (m)
        
        # 🌟 极小值保护：防止除以0或算数异常
        safe_z0 = max(z0, 0.001)  # 粗糙度最小为 1 毫米
        
        # 🌟 贴地飞行保护：如果无人机飞得比草还低，风速极小但不为负
        if z <= safe_z0:
            return 0.05  # 给予一个非常小的摩擦层底层风速系数

        # 核心对数公式
        try:
            factor = math.log(z / safe_z0) / math.log(h_ref / safe_z0)
            # 限制系数范围，防止极高空风速无限放大
            return float(np.clip(factor, 0.05, 1.5))
        except ValueError:
            # 兜底：万一出现数学域错误，返回底层风速
            return 0.05


class WindModelFactory:
    """风场模型工厂"""

    @staticmethod
    def create(
        model_type: str,
        config: SimulationConfig,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> BaseWindModel:
        if model_type == "slope":
            return SlopeWindModel(config, bounds)

        if model_type == "storm":
            # “storm” 模式会自动启用风暴叠加
            config.enable_storms = True
            return SlopeWindModel(config, bounds)

        raise ValueError(f"未知的风场模型类型: {model_type}")