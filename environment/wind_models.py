import sys, os
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from configs.config import SimulationConfig
from typing import List, Optional, Tuple


class BaseWindModel(ABC):
    """风场模型抽象基类"""
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
        pass


@dataclass
class StormCell:
    """风暴单元（移动高风区）。"""
    center_xy: np.ndarray  
    velocity_xy: np.ndarray  
    radius_m: float  
    strength_mps: float  
    birth_time_s: float  
    lifetime_s: float  
    last_t_s: float  

    def age(self, t_s: float) -> float:
        return t_s - self.birth_time_s

    def is_alive(self, t_s: float, bounds: Tuple[float, float, float, float]) -> bool:
        """🌟 死亡机制 1：寿命耗尽 或 移出地图边界则判定死亡"""
        if self.age(t_s) >= self.lifetime_s:
            return False
            
        min_x, max_x, min_y, max_y = bounds
        cx, cy = self.center_xy
        # 宽限一点：风暴中心超出地图边界一个半径的距离，才算彻底消散
        if cx < min_x - self.radius_m or cx > max_x + self.radius_m or \
           cy < min_y - self.radius_m or cy > max_y + self.radius_m:
            return False
            
        return True

    def update(self, t_s: float):
        """🌟 去除反弹逻辑：风暴现在一往无前地移动，直到越界死亡"""
        dt = t_s - self.last_t_s
        if dt <= 0:
            self.last_t_s = t_s
            return
        # 直线移动
        self.center_xy = self.center_xy + self.velocity_xy * dt
        self.last_t_s = t_s

    def wind_at(self, x: float, y: float) -> np.ndarray:
        """计算该点受该风暴影响的风速矢量"""
        dist = np.linalg.norm(np.array([x, y]) - self.center_xy)
        
        # 🌟 PPO 核心优化：使用高斯衰减 (Gaussian Decay)
        sigma = self.radius_m / 2.5 
        weight = math.exp(-0.5 * (dist / sigma) ** 2)
        
        if dist > self.radius_m * 1.5:
            return np.array([0.0, 0.0], dtype=float)

        direction = self.velocity_xy
        speed = np.linalg.norm(direction)
        if speed < 1e-6:
            direction = np.array([1.0, 0.0])
            speed = 1.0
        direction_unit = direction / speed

        storm_speed = self.strength_mps * weight
        return direction_unit * storm_speed


class StormWindManager:
    """管理一组移动风暴单元"""
    def __init__(self, config: SimulationConfig, bounds: Tuple[float, float, float, float]):
        self.config = config
        self.bounds = bounds
        self.rng = np.random.default_rng(self.config.wind_seed)
        self.storms: List[StormCell] = []
        self._init_storms(0.0)

    def _init_storms(self, t_s: float):
        self.storms = [self._create_storm(t_s) for _ in range(self.config.storm_count)]

    def _create_storm(self, t_s: float) -> StormCell:
        """🌟 出生机制 2：新风暴总是在地图边缘随机生成，并向内陆吹"""
        min_x, max_x, min_y, max_y = self.bounds
        
        edge = self.rng.integers(0, 4)
        if edge == 0:
            cx, cy = min_x, self.rng.uniform(min_y, max_y)
            theta = self.rng.uniform(-math.pi/4, math.pi/4) 
        elif edge == 1:
            cx, cy = max_x, self.rng.uniform(min_y, max_y)
            theta = self.rng.uniform(3*math.pi/4, 5*math.pi/4) 
        elif edge == 2:
            cx, cy = self.rng.uniform(min_x, max_x), min_y
            theta = self.rng.uniform(math.pi/4, 3*math.pi/4) 
        else:
            cx, cy = self.rng.uniform(min_x, max_x), max_y
            theta = self.rng.uniform(5*math.pi/4, 7*math.pi/4) 

        speed = self.rng.uniform(0.5 * self.config.storm_movement_speed_mps, self.config.storm_movement_speed_mps)
        velocity = np.array([math.cos(theta), math.sin(theta)]) * speed

        radius = self.rng.uniform(self.config.storm_radius_range_m[0], self.config.storm_radius_range_m[1])
        strength = self.rng.uniform(0.7, 1.2) * self.config.storm_strength_scale
        lifetime = self.rng.uniform(0.8, 1.5) * self.config.storm_lifetime_s

        return StormCell(
            center_xy=np.array([cx, cy], dtype=float),
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

        # 🌟 核心轮回逻辑：检查死亡，触发新生
        for i, storm in enumerate(self.storms):
            if not storm.is_alive(t_s, self.bounds):
                self.storms[i] = self._create_storm(t_s)
            else:
                storm.update(t_s)

        total = np.array([0.0, 0.0], dtype=float)
        for storm in self.storms:
            total += storm.wind_at(x, y)
        return total


class SlopeWindModel(BaseWindModel):
    """基于地形坡度的风场模型（包含时变背景风 + 昼夜坡度风 + 对数廓线 + 移动风暴）"""
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
        """时变背景风模型：风速平滑波动 + 风向摆动"""
        u0 = self.config.env_wind_u
        v0 = self.config.env_wind_v

        base_speed = math.hypot(u0, v0)
        base_dir = math.atan2(v0, u0)

        if base_speed < 1e-9:
            return 0.0, 0.0

        time_scale_s = max(self.config.wind_time_scale_s, 1.0)
        omega = 2.0 * math.pi / time_scale_s

        speed_ratio = self.config.wind_speed_variation_ratio
        dir_variation_rad = math.radians(self.config.wind_direction_variation_deg)

        speed_t = base_speed * (1.0 + speed_ratio * math.sin(omega * t_s))
        dir_t = base_dir + dir_variation_rad * math.sin(0.5 * omega * t_s)

        u_t = speed_t * math.cos(dir_t)
        v_t = speed_t * math.sin(dir_t)
        return u_t, v_t

    def _log_profile_factor(self, z: float, z0: float) -> float:
        """
        计算对数风廓线修正系数。
        🌟 引入严格的数值安全保护，防止 z <= z0 导致对数为负，或 z0=0 导致除零。
        """
        h_ref = 200.0  
        safe_z0 = max(z0, 0.001)  
        
        if z <= safe_z0:
            return 0.05  

        try:
            factor = math.log(z / safe_z0) / math.log(h_ref / safe_z0)
            return float(np.clip(factor, 0.05, 1.5))
        except ValueError:
            return 0.05


class WindModelFactory:
    """风场模型工厂"""
    @staticmethod
    def create(
        model_type: str,
        config: SimulationConfig,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> BaseWindModel:
        if model_type == "slope" or model_type == "storm":
            # 兼容 storm 和 slope 模式，统一交由 SlopeWindModel 处理
            if model_type == "storm":
                config.enable_storms = True
            return SlopeWindModel(config, bounds)

        raise ValueError(f"未知的风场模型类型: {model_type}")