import heapq
import math
import numpy as np
from typing import List, Tuple, Optional
from configs.config import SimulationConfig
from core.estimator import StateEstimator
from core.physics import PhysicsEngine

class Node:
    def __init__(self, x: float, y: float, g: float = 0, h: float = 0, parent=None):
        self.x, self.y = x, y
        self.g, self.h = g, h
        self.f = g + h
        self.parent = parent
        
    def __lt__(self, other):
        return self.f < other.f
        
    def get_pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

class AStarPlanner:
    """A* 路径规划器 (依赖注入架构)"""
    
    def __init__(self, config: SimulationConfig, estimator: StateEstimator, physics: PhysicsEngine):
        self.config = config
        self.estimator = estimator
        self.physics = physics
        
        self.step_size = self.estimator.get_resolution() / 1000.0  # 转为 km
        self.min_x, self.max_x, self.min_y, self.max_y = self.estimator.get_bounds()

    def heuristic(self, node_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> float:
        dist_km = math.hypot(node_pos[0] - goal_pos[0], node_pos[1] - goal_pos[1])
        if self.config.k_wind == 0:
            return dist_km
        # 启发式：直线距离 * 每米最低能耗
        return dist_km * 1000.0 * self.physics.energy_per_meter * 1.50  # 1.5倍安全系数

    def calculate_cost(self, current_node: Node, next_x: float, next_y: float) -> float:
        dist_km = math.hypot(next_x - current_node.x, next_y - current_node.y)
        if self.config.k_wind == 0:
            return dist_km

        # 1. 运动学向量
        move_vec = np.array([next_x - current_node.x, next_y - current_node.y])
        norm = np.linalg.norm(move_vec)
        if norm > 0:
            move_vec = move_vec / norm
        v_ground = move_vec * self.config.drone_speed
        
        # 2. 从状态估计器获取风场
        wind_vec = self.estimator.get_wind(next_x, next_y)
        
        # 3. 从物理引擎计算能耗
        power = self.physics.calculate_power(v_ground, wind_vec)
        if power == float('inf'):
            return float('inf')
            
        energy_joules = self.physics.calculate_energy(power, dist_km * 1000.0)
        
        # 4. 从状态估计器获取风险代价
        risk_cost = self.estimator.get_risk(next_x, next_y) * self.config.risk_factor * self.config.k_wind
        
        return energy_joules + risk_cost

    def search(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        start_node = Node(start_pos[0], start_pos[1], g=0, h=self.heuristic(start_pos, goal_pos))
        open_list = []
        closed_set = set()
        heapq.heappush(open_list, start_node)
        
        steps = 0
        arrival_threshold = self.step_size
        
        print(f"开始搜索... 考虑风场: {'是' if self.config.k_wind > 0 else '否'}")
        
        while open_list and steps < self.config.max_steps:
            steps += 1
            current_node = heapq.heappop(open_list)
            
            # 到达判断
            h_val = self.heuristic(current_node.get_pos(), goal_pos)
            threshold = arrival_threshold * self.physics.energy_per_meter if self.config.k_wind > 0 else arrival_threshold
            if h_val < threshold:
                print(f"寻路成功！耗时步数: {steps}")
                return self._reconstruct_path(current_node)
            
            # 离散化网格索引用于去重
            grid_idx = (
                int(round((current_node.x - self.min_x) / self.step_size)),
                int(round((current_node.y - self.min_y) / self.step_size))
            )
            if grid_idx in closed_set:
                continue
            closed_set.add(grid_idx)
            
            # 8邻域扩展
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    
                    next_x = current_node.x + dx * self.step_size
                    next_y = current_node.y + dy * self.step_size
                    
                    if not (self.min_x <= next_x <= self.max_x and self.min_y <= next_y <= self.max_y):
                        continue
                        
                    move_cost = self.calculate_cost(current_node, next_x, next_y)
                    if move_cost == float('inf'):
                        continue

                    new_g = current_node.g + move_cost
                    new_h = self.heuristic((next_x, next_y), goal_pos)
                    heapq.heappush(open_list, Node(next_x, next_y, g=new_g, h=new_h, parent=current_node))
                    
        print("寻路失败：超出最大步数或无可行路径。")
        return None

    def _reconstruct_path(self, node: Node) -> List[Tuple[float, float]]:
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]