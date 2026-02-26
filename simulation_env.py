import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class MapManager:
    def __init__(self, size=100, scale=1.0):
        """
        初始化地形与风场管理器
        :param size: 地图网格大小 (100x100)
        :param scale: 地图物理尺度 (比如每个格子代表10米)
        """
        self.size = size
        self.scale = scale
        
        # 1. 生成地形网格 (Meshgrid)
        self.x = np.linspace(-10, 10, size)
        self.y = np.linspace(-10, 10, size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # 2. 生成高斯山包地形 (Terrain Generation)
        # 高度 H = 500 * exp(...)
        self.dem = 500 * np.exp(-(self.X**2 + self.Y**2)/20)
        
        # 3. 计算地形梯度 (Terrain Gradient)
        # gradient 返回的是 (dy, dx)，注意顺序！
        self.grad_y, self.grad_x = np.gradient(self.dem)
        
        # 4. 生成风场 (Wind Field Generation)
        # 简单物理模型：白昼谷风 (Anabatic Wind)
        # 风速向量正比于地形坡度向量
        self.k_wind = 5.0 
        self.wind_u = self.k_wind * self.grad_x
        self.wind_v = self.k_wind * self.grad_y
        
        # 5. 构建插值器 (Interpolators)
        # 这样我们就能查询任意浮点坐标 (x, y) 的高度和风速了
        # bounds_error=False, fill_value=None 允许在边界外查询（返回最近值）
        self.interp_h = RegularGridInterpolator((self.y, self.x), self.dem, bounds_error=False, fill_value=None)
        self.interp_u = RegularGridInterpolator((self.y, self.x), self.wind_u, bounds_error=False, fill_value=None)
        self.interp_v = RegularGridInterpolator((self.y, self.x), self.wind_v, bounds_error=False, fill_value=None)

    def get_altitude(self, x, y):
        """获取指定坐标 (x, y) 的地形高度"""
        # 注意：interpolate 需要输入 (y, x) 顺序的坐标点
        return self.interp_h((y, x))

    def get_wind(self, x, y):
        """获取指定坐标 (x, y) 的风速向量 (u, v)"""
        u = self.interp_u((y, x))
        v = self.interp_v((y, x))
        return np.array([u, v])

    def plot_environment(self):
        """可视化地形与风场"""
        plt.figure(figsize=(10, 8))
        # 画地形等高线
        plt.contourf(self.X, self.Y, self.dem, 20, cmap='terrain')
        plt.colorbar(label='Altitude (m)')
        
        # 画风场箭头 (降采样，每隔5个点画一个，否则太密)
        skip = 5
        plt.quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip],
                   self.wind_u[::skip, ::skip], self.wind_v[::skip, ::skip],
                   color='white', alpha=0.8)
        
        plt.title(f"Terrain & Anabatic Wind Field ({self.size}x{self.size})")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.show()

# --- 测试代码 ---
if __name__ == "__main__":
    # 1. 实例化地图管理器
    env = MapManager(size=100)
    
    # 2. 测试查询一个点的风速
    test_x, test_y = 2.5, -1.5
    h = env.get_altitude(test_x, test_y)
    w = env.get_wind(test_x, test_y)
    
    print(f"坐标 ({test_x}, {test_y}) 的信息：")
    print(f"  - 海拔高度: {h:.2f} m")
    print(f"  - 风速向量: u={w[0]:.2f}, v={w[1]:.2f} m/s")
    print(f"  - 风速大小: {np.linalg.norm(w):.2f} m/s")
    
    # 3. 画图验证
    env.plot_environment()