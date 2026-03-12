import numpy as np
from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import BaseWindModel
from typing import Tuple


class StateEstimator:
    """状态估计器：作为 Planner 和 Environment 之间的代理层，可注入噪声"""

    def __init__(self, map_manager: MapManager, wind_model: BaseWindModel, config: SimulationConfig):
        self.map = map_manager
        self.wind = wind_model
        self.config = config

    def get_altitude(self, x: float, y: float) -> float:
        alt = self.map.get_altitude(x, y)
        if self.config.noise_level > 0:
            alt += np.random.normal(0, self.config.noise_level)
        return alt

    def get_wind(
            self,
            x: float,
            y: float,
            z: float = -1.0,
            t_s: float = 0.0,
    ) -> np.ndarray:
        """返回指定位置的二维风速向量 (u,v)，包含高度和粗糙度的影响。

        Parameters
        ----------
        x, y : float
            地面坐标（米）。
        z : float
            绝对海拔（m），若传递负值则使用默认 50m AGL。
        """
        # 1. 计算离地高度 AGL
        ground_alt = self.map.get_altitude(x, y)
        if z < 0:
            height_agl = 50.0
        else:
            height_agl = z - ground_alt
        height_agl = max(height_agl, 0.1)  # 最低 0.1 m，防止负值

        # 2. 获取地形梯度与粗糙度
        grad = self.map.get_gradient(x, y)
        z0 = self.map.get_roughness(x, y)

        # 3. 交给风模型计算最终风速
        wind_vec = self.wind.get_wind(x, y, height_agl, grad, z0, t_s=t_s)

        # 4. 添加可选噪声
        if self.config.noise_level > 0:
            wind_vec = wind_vec + np.random.normal(0, self.config.noise_level, 2)
        return wind_vec

    def get_risk(self, x: float, y: float, t_s: float = 0.0) -> float:
        """获取该坐标点的环境风险值 (如：风切变、湍流强度)"""
        # 计算风险时，可以默认用离地 50m 的风速作为参考
        wind_vec = self.get_wind(x, y, z=-1.0, t_s=t_s)
        turbulence = np.linalg.norm(wind_vec)
        return float(turbulence)

    def get_resolution(self) -> float:
        return self.map.resolution

    def get_bounds(self) -> Tuple[float, float, float, float]:
        return self.map.get_bounds()