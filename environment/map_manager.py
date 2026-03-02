import numpy as np
import cv2
from scipy.interpolate import RegularGridInterpolator
from configs.config import SimulationConfig
from typing import Tuple

class MapManager:
    """地图管理器：仅负责地形高程(DEM)、梯度计算与坐标映射"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # 1. 先加载地图 (必须第一步！)
        self._load_map()
        
        # 2. 有了地图才能算梯度
        self._calculate_gradients()
        
        # 3. 有了地图才能算粗糙度 (新增)
        self._generate_roughness_map()
        
        # 4. 最后构建插值器
        self._build_interpolators()

    # 新增这个方法
    def _generate_roughness_map(self):
        """
        基于海拔生成地表粗糙度 (z0)
        物理意义：
        - 森林 (海拔 < 2000m): z0 = 1.0m (阻力大，但能挡风)
        - 草甸 (2000-2800m): z0 = 0.1m
        - 雪地/岩石 (海拔 > 2800m): z0 = 0.005m (非常光滑，风很大)
        """
        # 1. 初始化 z0 矩阵
        self.z0_map = np.ones_like(self.dem) * 0.1 # 默认草地
        
        # 2. 森林层
        self.z0_map[self.dem < 2000.0] = 1.0 
        
        # 3. 雪地层
        self.z0_map[self.dem > 2800.0] = 0.005
        
        # 4. 构建插值器
        self.interp_z0 = RegularGridInterpolator((self.y, self.x), self.z0_map, bounds_error=False, fill_value=0.1)

    # 新增接口
    def get_roughness(self, x, y):
        return float(self.interp_z0((y, x)))

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