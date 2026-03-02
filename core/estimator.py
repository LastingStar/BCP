import numpy as np
import math
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

    def get_wind(self, x: float, y: float, z: float = 50.0) -> np.ndarray:
        # z: 无人机离地高度，默认为 50米
        
        # 1. 获取基准梯度风 (假设这是离地 200米处的风速)
        grad = self.map.get_gradient(x, y)
        base_wind = self.wind.get_wind(x, y, z, grad)
        
        # 2. 获取该点的地表粗糙度 z0 (森林=1.0, 雪地=0.005)
        z0 = self.map.get_roughness(x, y)
        
        # 3. 应用对数风廓线公式进行修正
        # 公式: U(z) = U_ref * [ln(z/z0) / ln(z_ref/z0)]
        h_ref = 200.0  # 参考高度
        
        # 防止 z0 太大导致除零错误 (z 必须大于 z0)
        if z <= z0: 
            factor = 0.1 # 贴地/树丛里，风速极小
        else:
            factor = math.log(z / z0) / math.log(h_ref / z0)
            
        # 限制一下系数范围 (0.2 ~ 1.2)，防止数学异常
        factor = np.clip(factor, 0.2, 1.2)
        
        final_wind = base_wind * factor
        
        if self.config.noise_level > 0:
            final_wind += np.random.normal(0, self.config.noise_level, 2)
            
        return final_wind

    def get_risk(self, x: float, y: float) -> float:
        """获取该坐标点的环境风险值 (如：风切变、湍流强度)"""
        wind_vec = self.get_wind(x, y)
        turbulence = np.linalg.norm(wind_vec)
        return float(turbulence)

    def get_resolution(self) -> float:
        return self.map.resolution
        
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return self.map.get_bounds()