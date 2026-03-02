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
    """基于地形坡度的风场模型 (包含昼夜山谷风效应)"""
    def __init__(self, config: SimulationConfig):
        self.config = config

    def get_wind(self, x: float, y: float, z: float, terrain_gradient: Tuple[float, float]) -> np.ndarray:
        gx, gy = terrain_gradient
        
        # 1. 基础环境风
        u_env = self.config.env_wind_u
        v_env = self.config.env_wind_v
        
        # 2. 爬坡风逻辑 (核心修改点！)
        k = self.config.k_slope
        
        if self.config.time_of_day == 'Day':
            # 白天：空气受热上升 -> 风顺着坡度往上吹 (Anabatic)
            # 梯度方向就是上坡方向，所以系数为正
            u_slope = k * gx
            v_slope = k * gy
        else:
            # 夜间：空气冷却下沉 -> 风顺着坡度往下吹 (Katabatic)
            # 也就是梯度的反方向，所以系数为负
            u_slope = -k * gx 
            v_slope = -k * gy
            
        # 3. 合成总风速
        u = u_env + u_slope
        v = v_env + v_slope
        
        # 4. 限幅 (防止数值爆炸)
        u = np.clip(u, -self.config.max_wind_speed, self.config.max_wind_speed)
        v = np.clip(v, -self.config.max_wind_speed, self.config.max_wind_speed)
        
        return np.array([u, v])

class WindModelFactory:
    """风场模型工厂"""
    @staticmethod
    def create(model_type: str, config: SimulationConfig) -> BaseWindModel:
        if model_type == 'slope':
            return SlopeWindModel(config)
        raise ValueError(f"未知的风场模型类型: {model_type}")