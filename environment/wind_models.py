import numpy as np
from abc import ABC, abstractmethod
from configs.config import SimulationConfig
from typing import Tuple

class BaseWindModel(ABC):
    """风场模型抽象基类"""
    @abstractmethod
    def get_wind(self, x: float, y: float, z: float, terrain_gradient: Tuple[float, float]) -> np.ndarray:
        pass

class SlopeWindModel(BaseWindModel):
    """基于地形坡度的风场模型"""
    def __init__(self, config: SimulationConfig):
        self.config = config

    def get_wind(self, x: float, y: float, z: float, terrain_gradient: Tuple[float, float]) -> np.ndarray:
        gx, gy = terrain_gradient
        
        u = self.config.env_wind_u + self.config.k_slope * gx
        v = self.config.env_wind_v + self.config.k_slope * gy
        
        u = np.clip(u, -self.config.max_wind_speed, self.config.max_wind_speed)
        v = np.clip(v, -self.config.max_wind_speed, self.config.max_wind_speed)
        
        return np.array([u, v])

class WindModelFactory:
    """风场模型工厂"""
    @staticmethod
    def create(model_type: str, config: SimulationConfig) -> BaseWindModel:
        if model_type == 'slope':
            return SlopeWindModel(config)
        # 预留接口：elif model_type == 'netcdf': return NetCDFWindModel(config)
        raise ValueError(f"未知的风场模型类型: {model_type}")