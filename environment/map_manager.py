import numpy as np
import cv2
import math
import os
from scipy.interpolate import RegularGridInterpolator
from configs.config import SimulationConfig
from typing import Tuple
from scipy.ndimage import gaussian_filter
try:
    from PIL import Image
except ImportError:  # Pillow is usually available via Streamlit dependency tree
    Image = None

class MapManager:
    """地图管理器：仅负责地形高程(DEM)、梯度计算与坐标映射"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.map_loaded_from_file = False
        self.map_load_error = ""
        self.map_source_path = ""
        
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
        path = str(self.config.map_path)
        img = self._read_image_any(path)
        if img is not None:
            img = self._to_grayscale_uint8(img)

        if img is None:
            self.map_loaded_from_file = False
            self.map_source_path = path
            self.map_load_error = (
                f"无法读取地图文件: {path}。"
                "可能是图片格式不受支持或文件损坏。"
            )
            print(f"警告: {self.map_load_error}，将生成虚拟高斯地形。")
            self._generate_fake_map()
            return

        self.map_loaded_from_file = True
        self.map_source_path = path
        self.map_load_error = ""
            
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, self.config.target_size, interpolation=cv2.INTER_AREA)
        
        self.dem = self.config.min_alt + (img.astype(float) / 255.0) * (self.config.max_alt - self.config.min_alt)
        self.size_y, self.size_x = self.dem.shape

        # 地图的真实尺寸（米）与分辨率（m/pixel）
        real_world_size_m = self.config.map_size_km * 1000.0
        self.resolution = real_world_size_m / self.size_x

        # 将坐标统一为米（m），避免 km/m 混用
        half_size_m = real_world_size_m / 2.0
        self.x = np.linspace(-half_size_m, half_size_m, self.size_x)
        self.y = np.linspace(-half_size_m, half_size_m, self.size_y)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def _read_image_any(self, path: str):
        """
        Robust image loader:
        1) cv2.imread
        2) np.fromfile + cv2.imdecode (better Windows non-ASCII compatibility)
        3) PIL fallback (better 16-bit PNG/TIFF compatibility)
        """
        if not path or not os.path.exists(path):
            return None

        # 1) OpenCV decode from bytes (best for Windows non-ASCII path)
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            if raw.size > 0:
                img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    return img
        except Exception:
            pass

        # 2) OpenCV direct read
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
        except Exception:
            pass

        # 3) PIL fallback
        if Image is not None:
            try:
                with Image.open(path) as pil_img:
                    return np.array(pil_img)
            except Exception:
                return None

        return None

    def _to_grayscale_uint8(self, img: np.ndarray) -> np.ndarray:
        """Normalize any decoded image to grayscale uint8 for DEM mapping."""
        arr = img
        if arr.ndim == 3:
            # Handle BGRA/RGBA/RGB/BGR conservatively by channel-averaging fallback.
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            # For both RGB and BGR this gives a valid grayscale fallback.
            arr = arr.astype(np.float32).mean(axis=2)
        elif arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            min_v = float(np.nanmin(arr))
            max_v = float(np.nanmax(arr))
            if max_v > min_v:
                arr = (arr - min_v) / (max_v - min_v)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        return arr

    def _generate_fake_map(self):
        """Fallback: 生成虚拟地形"""
        self.size_x, self.size_y = self.config.target_size
        real_world_size_m = self.config.map_size_km * 1000.0
        self.resolution = real_world_size_m / self.size_x
        half_size_m = real_world_size_m / 2.0
        self.x = np.linspace(-half_size_m, half_size_m, self.size_x)
        self.y = np.linspace(-half_size_m, half_size_m, self.size_y)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        # 以米为单位构造一个高程场
        self.dem = 500 * np.exp(-((self.X/1000.0)**2 + (self.Y/1000.0)**2) / 20) + self.config.min_alt

    def _calculate_gradients(self):
        """计算真实的物理坡度 (dh/dx, dh/dy)，并应用高斯平滑滤波"""
        self.grad_y, self.grad_x = np.gradient(self.dem)
        self.grad_y /= self.resolution
        self.grad_x /= self.resolution
        
        # 🌟 PPO 核心优化：对地形梯度进行二维高斯平滑滤波
        # sigma=2.0 意味着在几个网格的范围内把突变的坡度风抹平，保证导数连续
        self.grad_x = gaussian_filter(self.grad_x, sigma=2.0)
        self.grad_y = gaussian_filter(self.grad_y, sigma=2.0)

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
    
    
    # 替换 environment/map_manager.py 中的这两个函数
    def is_in_nfz(self, x: float, y: float, nfz_list_km=None, inflation_m: float = 0.0) -> bool:
        """判断坐标 (x, y) 是否在禁飞区内，支持安全半径膨胀"""
        if not self.config.enable_nfz:
            return False

        zones = self.config.nfz_list_km if nfz_list_km is None else nfz_list_km

        for (cx_km, cy_km, r_km) in zones:
            cx, cy = cx_km * 1000.0, cy_km * 1000.0
            r = (r_km * 1000.0) + inflation_m  # 🌟 核心：膨胀禁飞区半径
            dist = math.hypot(x - cx, y - cy)
            if dist <= r:
                return True
        return False

    def is_collision(self, x: float, y: float, z: float, nfz_list_km=None, inflation_m: float = 0.0) -> bool:
        """判断是否撞山或进入禁飞区"""
        if self.is_in_nfz(x, y, nfz_list_km=nfz_list_km, inflation_m=inflation_m):
            return True

        ground_alt = self.get_altitude(x, y)
        safety_margin = 10.0
        if z < (ground_alt + safety_margin):
            return True

        return False
