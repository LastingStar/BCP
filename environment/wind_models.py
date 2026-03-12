import math
import numpy as np
from abc import ABC, abstractmethod
from configs.config import SimulationConfig
from typing import Tuple


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


class SlopeWindModel(BaseWindModel):
    """基于地形坡度的风场模型（包含时变背景风 + 昼夜坡度风 + 对数廓线）"""

    def __init__(self, config: SimulationConfig):
        self.config = config

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

        # 4. 限幅
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
        """计算对数风廓线修正系数。"""
        h_ref = 200.0  # 参考高度 (m)
        if z <= z0:
            return 0.1
        if z0 <= 0:
            return 1.0

        factor = math.log(z / z0) / math.log(h_ref / z0)
        return float(np.clip(factor, 0.2, 1.2))


class WindModelFactory:
    """风场模型工厂"""

    @staticmethod
    def create(model_type: str, config: SimulationConfig) -> BaseWindModel:
        if model_type == "slope":
            return SlopeWindModel(config)
        raise ValueError(f"未知的风场模型类型: {model_type}")