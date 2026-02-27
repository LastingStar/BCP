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

    def get_wind(self, x: float, y: float, z: float = 0.0) -> np.ndarray:
        grad = self.map.get_gradient(x, y)
        wind_vec = self.wind.get_wind(x, y, z, grad)
        
        if self.config.noise_level > 0:
            wind_vec += np.random.normal(0, self.config.noise_level, 2)
            
        return wind_vec

    def get_risk(self, x: float, y: float) -> float:
        """获取该坐标点的环境风险值 (如：风切变、湍流强度)"""
        wind_vec = self.get_wind(x, y)
        turbulence = np.linalg.norm(wind_vec)
        return float(turbulence)

    def get_resolution(self) -> float:
        return self.map.resolution
        
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return self.map.get_bounds()