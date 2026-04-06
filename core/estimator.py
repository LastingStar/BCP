import numpy as np
from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import BaseWindModel
from typing import Tuple
import math

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

    def _get_single_agent_gust(self, t_s: float) -> np.ndarray:
        if not self.config.enable_single_agent_gusts or self.config.gust_max_speed_mps <= 0.0:
            return np.zeros(2, dtype=float)

        slot_duration = max(self.config.gust_duration_s, 1.0)
        slot_idx = int(math.floor(max(t_s, 0.0) / slot_duration))
        slot_seed = int(self.config.wind_seed) * 1000003 + slot_idx * 97 + 17
        rng = np.random.default_rng(slot_seed)
        if rng.random() >= self.config.gust_trigger_prob:
            return np.zeros(2, dtype=float)

        magnitude = float(rng.uniform(self.config.gust_min_speed_mps, self.config.gust_max_speed_mps))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        return np.array([math.cos(angle), math.sin(angle)], dtype=float) * magnitude

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
        wind_vec = wind_vec + self._get_single_agent_gust(t_s)

        # 4. 添加可选噪声
        if self.config.noise_level > 0:
            wind_vec = wind_vec + np.random.normal(0, self.config.noise_level, 2)
        return wind_vec

    def get_resolution(self) -> float:
        return self.map.resolution

    def get_bounds(self) -> Tuple[float, float, float, float]:
        return self.map.get_bounds()
    
    def get_tke(self, x: float, y: float, z: float, t_s: float = 0.0) -> float:
        """
        核心模型层 1：基于参数化的湍流动能 (TKE) 预估
        结合风切变、粗糙度尾流与地形坡度
        """
        # 1. 获取当前高度和稍高处的风速，近似计算垂直风切变 (Wind Shear)
        dz = 5.0
        w1 = self.get_wind(x, y, z, t_s)
        w2 = self.get_wind(x, y, z + dz, t_s)
        shear_magnitude = np.linalg.norm(w2 - w1) / dz

        # 2. 获取地表粗糙度 z0 和地形坡度
        z0 = self.map.get_roughness(x, y)
        gx, gy = self.map.get_gradient(x, y)
        slope_magnitude = math.hypot(gx, gy)
        
        wind_speed = np.linalg.norm(w1)

        # TKE 预算方程降维化：
        # 项1: 机械切变 (Shear)
        term_shear = self.config.tke_shear_coeff * (shear_magnitude ** 2)
        # 项2: 粗糙度尾流 (Wake) -> 与风速的三次方和 z0 正相关
        term_wake = self.config.tke_wake_coeff * z0 * (wind_speed ** 3)
        # 项3: 地形坡度扰动 (Slope) -> 风速与坡度的乘积
        term_slope = self.config.tke_slope_coeff * slope_magnitude * wind_speed

        tke = term_shear + term_wake + term_slope
        return max(tke, 0.01) # 避免除0错误，给予一个极小的环境底噪

    def get_risk(self, x: float, y: float, z: float, v_ground: float, t_s: float = 0.0) -> Tuple[float, float]:
        """
        核心模型层 2：将 TKE 映射为坠机概率 P_crash
        返回: (坠机概率 P_crash, 当前TKE值)
        """
        # 获取该点 TKE
        tke = self.get_tke(x, y, z, t_s)
        
        # 限制最小地速避免暴露时间无限大
        v_eff = max(v_ground, 1.0) 
        grid_size = self.map.resolution
        
        # 暴露时间 (Exposure Time) = 穿越当前网格所需时间
        exposure_time = grid_size / v_eff

        # 极值响应分布公式 (Gust Exceedance Model)
        # P_safe = exp( -N0 * t * exp( -K / TKE ) )
        N0 = self.config.drone_response_freq_N0
        K_robust = self.config.drone_robustness_K

        # 核心算式：
        exponent = -N0 * exposure_time * math.exp(-K_robust / tke)
        p_safe = math.exp(exponent)
        p_crash = 1.0 - p_safe

        return p_crash, tke
