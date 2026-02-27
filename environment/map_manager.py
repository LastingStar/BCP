import numpy as np
import cv2
from scipy.interpolate import RegularGridInterpolator
from configs.config import SimulationConfig
from typing import Tuple

class MapManager:
    """地图管理器：仅负责地形高程(DEM)、梯度计算与坐标映射"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._load_map()
        self._calculate_gradients()
        self._build_interpolators()

    def _load_map(self):
        img = cv2.imread(self.config.map_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"警告: 无法读取 {self.config.map_path}，将生成虚拟高斯地形。")
            self._generate_fake_map()
            return
            
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, self.config.target_size, interpolation=cv2.INTER_AREA)
        
        self.dem = self.config.min_alt + (img.astype(float) / 255.0) * (self.config.max_alt - self.config.min_alt)
        self.size_y, self.size_x = self.dem.shape
        
        real_world_size_m = self.config.map_size_km * 1000.0
        self.resolution = real_world_size_m / self.size_x
        
        half_size = self.config.map_size_km / 2
        self.x = np.linspace(-half_size, half_size, self.size_x)
        self.y = np.linspace(-half_size, half_size, self.size_y)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def _generate_fake_map(self):
        """Fallback: 生成虚拟地形"""
        self.size_x, self.size_y = self.config.target_size
        self.resolution = (self.config.map_size_km * 1000.0) / self.size_x
        self.x = np.linspace(-self.config.map_size_km/2, self.config.map_size_km/2, self.size_x)
        self.y = np.linspace(-self.config.map_size_km/2, self.config.map_size_km/2, self.size_y)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dem = 500 * np.exp(-(self.X**2 + self.Y**2) / 20) + self.config.min_alt

    def _calculate_gradients(self):
        """计算真实的物理坡度 (dh/dx, dh/dy)"""
        self.grad_y, self.grad_x = np.gradient(self.dem)
        self.grad_y /= self.resolution
        self.grad_x /= self.resolution

    def _build_interpolators(self):
        self.interp_h = RegularGridInterpolator((self.y, self.x), self.dem, bounds_error=False, fill_value=0)
        self.interp_gx = RegularGridInterpolator((self.y, self.x), self.grad_x, bounds_error=False, fill_value=0)
        self.interp_gy = RegularGridInterpolator((self.y, self.x), self.grad_y, bounds_error=False, fill_value=0)

    def get_altitude(self, x: float, y: float) -> float:
        return float(self.interp_h((y, x)))

    def get_gradient(self, x: float, y: float) -> Tuple[float, float]:
        return float(self.interp_gx((y, x))), float(self.interp_gy((y, x)))
        
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return self.x[0], self.x[-1], self.y[0], self.y[-1]