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

# 🌟 这里是你刚才不小心删掉的基类！
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

# ==========================================
# 无状态风暴模块 (全新修复版)
# ==========================================
@dataclass
class StatelessStormCell:
    """无状态风暴单元：位置仅由绝对时间 t_s 决定，解决时间穿梭探测导致的错位Bug"""
    start_center_xy: np.ndarray  
    velocity_xy: np.ndarray  
    radius_m: float  
    strength_mps: float  
    birth_time_s: float  
    actual_lifetime_s: float  # 精确寿命（含飞出边界的时间）

    def is_active(self, t_s: float) -> bool:
        return self.birth_time_s <= t_s < (self.birth_time_s + self.actual_lifetime_s)

    def center_at(self, t_s: float) -> np.ndarray:
        age = t_s - self.birth_time_s
        return self.start_center_xy + self.velocity_xy * age

    def wind_at(self, x: float, y: float, t_s: float) -> np.ndarray:
        if not self.is_active(t_s):
            return np.array([0.0, 0.0], dtype=float)
            
        current_center = self.center_at(t_s)
        dist = np.linalg.norm(np.array([x, y]) - current_center)
        
        # 高斯衰减
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
    """管理一组无状态风暴：在任务伊始预生成整场战役的风暴队列"""
    def __init__(self, config: SimulationConfig, bounds: Tuple[float, float, float, float]):
        self.config = config
        self.bounds = bounds
        self.storm_slots = []
        self._pregenerate_storms()

    def _pregenerate_storms(self):
        rng = np.random.default_rng(self.config.wind_seed)
        self.storm_slots = []
        
        # 为每个名额 (storm_count) 预生成足以覆盖整个任务时长 (比如 7200 秒) 的接力风暴
        for _ in range(self.config.storm_count):
            slot_storms = []
            current_t = 0.0
            while current_t < 7200.0:  # 预生成 2 小时的风暴队列
                storm = self._create_stateless_storm(rng, current_t)
                slot_storms.append(storm)
                current_t += storm.actual_lifetime_s
            self.storm_slots.append(slot_storms)

    def _create_stateless_storm(self, rng, birth_time_s: float) -> StatelessStormCell:
        min_x, max_x, min_y, max_y = self.bounds
        
        edge = rng.integers(0, 4)
        if edge == 0:
            cx, cy = min_x, rng.uniform(min_y, max_y)
            theta = rng.uniform(-math.pi/4, math.pi/4) 
        elif edge == 1:
            cx, cy = max_x, rng.uniform(min_y, max_y)
            theta = rng.uniform(3*math.pi/4, 5*math.pi/4) 
        elif edge == 2:
            cx, cy = rng.uniform(min_x, max_x), min_y
            theta = rng.uniform(math.pi/4, 3*math.pi/4) 
        else:
            cx, cy = rng.uniform(min_x, max_x), max_y
            theta = rng.uniform(5*math.pi/4, 7*math.pi/4) 

        speed = rng.uniform(0.5 * self.config.storm_movement_speed_mps, self.config.storm_movement_speed_mps)
        velocity = np.array([math.cos(theta), math.sin(theta)]) * speed
        radius = rng.uniform(self.config.storm_radius_range_m[0], self.config.storm_radius_range_m[1])
        strength = rng.uniform(0.7, 1.2) * self.config.storm_strength_scale
        nominal_lifetime = rng.uniform(0.8, 1.5) * self.config.storm_lifetime_s

        actual_lifetime = 0.0
        test_cx, test_cy = cx, cy
        while actual_lifetime < nominal_lifetime:
            if test_cx < min_x - radius or test_cx > max_x + radius or \
               test_cy < min_y - radius or test_cy > max_y + radius:
                break
            test_cx += velocity[0]
            test_cy += velocity[1]
            actual_lifetime += 1.0
            
        if actual_lifetime < 1.0: actual_lifetime = 1.0

        return StatelessStormCell(
            start_center_xy=np.array([cx, cy], dtype=float),
            velocity_xy=velocity,
            radius_m=radius,
            strength_mps=strength,
            birth_time_s=birth_time_s,
            actual_lifetime_s=actual_lifetime
        )

    def get_active_storms(self, t_s: float) -> List[StatelessStormCell]:
        """查询在绝对时间 t_s 存活的风暴"""
        active = []
        for slot in self.storm_slots:
            for storm in slot:
                if storm.is_active(t_s):
                    active.append(storm)
                    break 
        return active

    def sample(self, x: float, y: float, t_s: float) -> np.ndarray:
        if not self.config.enable_storms:
            return np.array([0.0, 0.0], dtype=float)

        total = np.array([0.0, 0.0], dtype=float)
        for storm in self.get_active_storms(t_s):
            total += storm.wind_at(x, y, t_s)
        return total

# ==========================================
# 基础坡度风模型
# ==========================================
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
        """计算对数风廓线修正系数。"""
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
            if model_type == "storm":
                config.enable_storms = True
            return SlopeWindModel(config, bounds)

        raise ValueError(f"未知的风场模型类型: {model_type}")