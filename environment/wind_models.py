import math
import numpy as np
from abc import ABC, abstractmethod
from configs.config import SimulationConfig
from typing import Tuple

class BaseWindModel(ABC):
    """风场模型抽象基类

    子类实现水平风计算之后，基类将负责通过地表粗糙度 z0 和高度 z
    应用对数廓线公式进行风速修正，因此 :meth:`get_wind` 的返回值应当是
    最终的二维风速向量 (u,v)。
    """

    @abstractmethod
    def get_wind(
        self,
        x: float,
        y: float,
        z: float,
        terrain_gradient: Tuple[float, float],
        z0: float,
    ) -> np.ndarray:
        """计算指定位置的风矢量。

        Parameters
        ----------
        x, y : float
            地面坐标（米）。
        z : float
            相对于地面的高度，单位米。
        terrain_gradient : Tuple[float, float]
            (gx, gy) 地形梯度。
        z0 : float
            地表粗糙度长度，单位米。
        """
        pass

class SlopeWindModel(BaseWindModel):
    """基于地形坡度的风场模型 (包含昼夜山谷风效应)"""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def get_wind(
        self,
        x: float,
        y: float,
        z: float,
        terrain_gradient: Tuple[float, float],
        z0: float,
    ) -> np.ndarray:
        gx, gy = terrain_gradient

        # 1. 基础环境风
        u_env = self.config.env_wind_u
        v_env = self.config.env_wind_v

        # 2. 爬坡风逻辑
        k = self.config.k_slope
        if self.config.time_of_day == 'Day':
            u_slope = k * gx
            v_slope = k * gy
        else:
            u_slope = -k * gx
            v_slope = -k * gy

        u_raw = u_env + u_slope
        v_raw = v_env + v_slope

        # 对数风廓线修正因子
        factor = self._log_profile_factor(z, z0)
        u = u_raw * factor
        v = v_raw * factor

        # 限幅
        u = np.clip(u, -self.config.max_wind_speed, self.config.max_wind_speed)
        v = np.clip(v, -self.config.max_wind_speed, self.config.max_wind_speed)

        return np.array([u, v])

    def _log_profile_factor(self, z: float, z0: float) -> float:
        """计算对数风廓线修正系数。"""
        h_ref = 200.0  # 参考高度 (m)
        if z <= z0:
            return 0.1
        # 防止分母为零
        if z0 <= 0:
            return 1.0
        factor = math.log(z / z0) / math.log(h_ref / z0)
        return float(np.clip(factor, 0.2, 1.2))

class WindModelFactory:
    """风场模型工厂"""
    @staticmethod
    def create(model_type: str, config: SimulationConfig) -> BaseWindModel:
        if model_type == 'slope':
            return SlopeWindModel(config)
        raise ValueError(f"未知的风场模型类型: {model_type}")