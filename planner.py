import numpy as np
import matplotlib.pyplot as plt
from simulation_env import MapManager
import heapq
import math

class Node:
    """
    A* 算法的搜索节点，用于记录路径搜索的状态
    """
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.g = g  # g(n): 从起点到当前节点的实际能量代价 (Actual Cost)
        self.h = h  # h(n): 启发式函数，预估当前节点到终点的剩余代价 (Heuristic)
        self.f = g + h  # f(n) = g(n) + h(n): 优先级评估函数，值越小搜索优先级越高
        self.parent = parent  # 用于最终回溯找到完整的路径

    def __lt__(self, other):
        # 让 heapq 能够根据 f 值进行自动排序（小顶堆）
        return self.f < other.f
    
    def get_pos(self):
        return (self.x, self.y)

class AStarPlanner:
    def __init__(self, map_manager, k_wind=0.5):
        """
        初始化规划器
        :param map_manager: 地图管理器，提供风场和地形数据
        :param k_wind: 风场影响系数。k=0表示不考虑风（传统最短距离）；k越大越倾向于利用顺风。
        """
        self.map = map_manager
        self.step_size = map_manager.scale  # 搜索步长，与地图分辨率保持一致
        self.k_wind = k_wind # 【关键修改点】：将系数存入类中，方便后续对比

    def heuristic(self, node_pos, goal_pos):
        """
        启发式函数：使用欧几里得距离 $ \sqrt{\Delta x^2 + \Delta y^2} $
        """
        return math.sqrt((node_pos[0]-goal_pos[0])**2 + (node_pos[1]-goal_pos[1])**2)

    def calculate_cost(self, current_node, next_x, next_y):
        """
        修正后的核心物理模型
        """
        # 1. 基础距离
        dist = math.sqrt((next_x - current_node.x)**2 + (next_y - current_node.y)**2)
        
        # 【关键修复】：如果 k_wind 为 0，模拟传统算法，只考虑距离，无视风场
        if self.k_wind == 0:
            return dist

        # --- 以下是智能模式 (考虑风和物理能耗) ---
        
        # 2. 获取风速
        wind_vec = self.map.get_wind(next_x, next_y) 
        
        # 3. 计算移动向量
        move_vec = np.array([next_x - current_node.x, next_y - current_node.y])
        norm = np.linalg.norm(move_vec)
        if norm > 0:
            move_vec = move_vec / norm
        
        # 4. 物理动力学计算
        # 假设地速恒定 10m/s
        v_ground_mag = 10.0 
        v_ground_vec = move_vec * v_ground_mag
        
        # 计算空速向量 V_air = V_ground - V_wind
        v_air_vec = v_ground_vec - wind_vec
        v_air_mag = np.linalg.norm(v_air_vec)
        
        # 功率公式 P = c1 * v^3 + c2
        # c1=0.05, c2=100
        power = 0.05 * (v_air_mag ** 3) + 100.0
        
        # 能量 Energy = Power * Time
        time = dist / v_ground_mag
        energy_joules = power * time
        
        # 5. 加上风险项 (Risk)
        # 这里把 self.k_wind 用作风险/风场权重的放大系数
        # k 越大，越在意风的影响
        turbulence_risk = np.linalg.norm(wind_vec) 
        total_cost = energy_joules + (turbulence_risk * self.k_wind * 10.0)
        
        return total_cost
        

    def search(self, start_pos, goal_pos):
        """执行 A* 路径搜索"""
        start_node = Node(start_pos[0], start_pos[1], g=0, h=0)
        start_node.h = self.heuristic(start_pos, goal_pos)
        
        open_list = []  # 待探索的节点 (Priority Queue)
        closed_set = set()  # 已访问过的网格
        
        heapq.heappush(open_list, start_node)
        steps = 0
        max_steps = 10000 
        
        while open_list and steps < max_steps:
            steps += 1
            current_node = heapq.heappop(open_list)
            
            # 判断是否接近终点（允许一定误差）
            if self.heuristic(current_node.get_pos(), goal_pos) < 1.0:
                print(f"成功找到路径！风力系数 k={self.k_wind}, 搜索步数: {steps}")
                return self._reconstruct_path(current_node)
            
            # 网格化锁定：防止算法在同一个微小区域无限打转
            grid_pos = (int(round(current_node.x)), int(round(current_node.y)))
            if grid_pos in closed_set:
                continue
            closed_set.add(grid_pos)
            
            # 扩展 8 个方向的邻居节点
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    
                    next_x = current_node.x + dx * self.step_size
                    next_y = current_node.y + dy * self.step_size
                    
                    # 边界范围检查
                    if not (-10 <= next_x <= 10 and -10 <= next_y <= 10):
                        continue
                        
                    # 计算到这个邻居的能量代价
                    move_cost = self.calculate_cost(current_node, next_x, next_y)
                    
                    # 创建并入队新节点
                    new_g = current_node.g + move_cost
                    new_h = self.heuristic((next_x, next_y), goal_pos)
                    neighbor = Node(next_x, next_y, g=new_g, h=new_h, parent=current_node)
                    
                    heapq.heappush(open_list, neighbor)
                    
        print("未找到有效路径！")
        return None

    def _reconstruct_path(self, node):
        """回溯路径：从终点沿 parent 指针找回起点"""
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

# --- 测试与可视化对比代码 ---

if __name__ == "__main__":
    # 1. 环境初始化
    env = MapManager(size=100) 
    planner = AStarPlanner(env) 
    
    start_point = (-8.0, -8.0)
    goal_point = (8.0, 8.0)

    # 2. 实验 A: 模拟传统算法 (k_wind = 0)
    # 这时无人机完全无视风场，只会走直线（欧几里得距离最短）
    planner.k_wind = 0.0
    path_traditional = planner.search(start_point, goal_point)
    
    # 3. 实验 B: 模拟风场感知算法 (k_wind 调大)
    # 这时无人机非常“聪明”，宁愿多走两步路也要去蹭顺风
    planner.k_wind = 3.0  # 增大系数让效果在图上更明显
    path_energy_aware = planner.search(start_point, goal_point)
    
    # 4. 可视化绘图
    plt.figure(figsize=(10, 8))
    
    # 绘制背景：地形高度 (DEM)
    plt.contourf(env.X, env.Y, env.dem, 20, cmap='terrain', alpha=0.3)
    
    # 绘制风场箭头 (Quiver)
    plt.quiver(env.X[::5, ::5], env.Y[::5, ::5],
               env.wind_u[::5, ::5], env.wind_v[::5, ::5],
               color='blue', alpha=0.3, label='Wind Direction')

    # 绘制两条路径对比
    if path_traditional:
        path_t = np.array(path_traditional)
        plt.plot(path_t[:,0], path_t[:,1], 'r--', linewidth=2, label='Traditional (Shortest)')

    if path_energy_aware:
        path_e = np.array(path_energy_aware)
        plt.plot(path_e[:,0], path_e[:,1], 'g-', linewidth=3, label='Energy-Aware (Wind Assisted)')

    # 标注起点终点
    plt.scatter(*start_point, c='gold', s=200, marker='*', label='Start', zorder=5)
    plt.scatter(*goal_point, c='red', s=200, marker='X', label='Goal', zorder=5)

    plt.title("A* Planning Comparison: Traditional vs Energy-Aware", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()