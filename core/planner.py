import heapq
import math
import numpy as np
from typing import List, Tuple, Optional
from configs.config import SimulationConfig
from core.estimator import StateEstimator
from core.physics import PhysicsEngine


class Node:
    def __init__(self, x: float, y: float, z: float, g: float = 0.0, h: float = 0.0, parent=None):
        self.x = x
        self.y = y
        self.z = z  # 绝对海拔（m）
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def get_pos(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class AStarPlanner:
    """3D A* 路径规划器（内部单位：米）"""

    def __init__(self, config: SimulationConfig, estimator: StateEstimator, physics: PhysicsEngine):
        self.config = config
        self.estimator = estimator
        self.physics = physics

        # 水平步长（m）: 使用 MapManager 的分辨率（m/pixel）
        self.step_size = float(self.estimator.get_resolution())
        # 垂直步长（m）
        self.z_step = float(self.config.z_step)

        # 地图边界（x,y 单位：米）
        self.min_x, self.max_x, self.min_y, self.max_y = self.estimator.get_bounds()

    def heuristic(self, node_pos: Tuple[float, float, float], goal_pos: Tuple[float, float, float]) -> float:
        """返回启发值（单位：若 k_wind==0 则为米，否则为焦耳）"""
        dx = node_pos[0] - goal_pos[0]
        dy = node_pos[1] - goal_pos[1]
        dz = node_pos[2] - goal_pos[2]

        dist_xy_m = math.hypot(dx, dy)
        dist_z_m = abs(dz)

        weighted_dist_m = math.sqrt(dist_xy_m ** 2 + (dist_z_m * self.config.z_weight) ** 2)

        if self.config.k_wind == 0:
            return weighted_dist_m

        # 估算能量为 距离 * 每米能耗（J/m） * 安全系数
        return weighted_dist_m * self.physics.energy_per_meter * 1.5

    def calculate_cost(self, current_node: Node, next_x: float, next_y: float, next_z: float) -> float:
        """计算从 current_node 到 (next_x,next_y,next_z) 的代价（单位：焦耳 + 风险惩罚）"""
        # 使用米为单位
        dist_xy_m = math.hypot(next_x - current_node.x, next_y - current_node.y)
        dist_z_m = next_z - current_node.z
        total_dist_m = math.sqrt(dist_xy_m ** 2 + dist_z_m ** 2)

        # k_wind==0 时退化为距离代价（米）
        if self.config.k_wind == 0:
            return total_dist_m

        # 时间估算（秒）
        v_total = float(self.config.drone_speed)
        if v_total <= 0:
            return float('inf')
        time_s = total_dist_m / v_total
        if time_s <= 0:
            return float('inf')

        # 垂直速度
        v_z = dist_z_m / time_s

        # 水平地速分量 (m/s)
        v_xy = dist_xy_m / time_s
        move_vec_xy = np.array([next_x - current_node.x, next_y - current_node.y], dtype=float)
        if np.linalg.norm(move_vec_xy) > 0:
            move_vec_xy = move_vec_xy / np.linalg.norm(move_vec_xy)
        v_ground_xy = move_vec_xy * v_xy

        # 风速：传入 AGL（米）
        agl = next_z - self.estimator.get_altitude(next_x, next_y)
        if agl < 0:
            # 在地形以下，不可行
            return float('inf')
        wind_vec = self.estimator.get_wind(next_x, next_y, agl)

        # 空气动力功率（W）
        power_aero = self.physics.calculate_power(v_ground_xy, wind_vec)
        if power_aero == float('inf'):
            return float('inf')

        # 重力功率近似（W）: m*g*v_z （这里 m 和 g 可在 config 中扩展）
        power_gravity = 10.0 * v_z

        total_power = power_aero + power_gravity
        if total_power < self.config.base_power:
            total_power = self.config.base_power

        energy_joules = total_power * time_s

        # 风险代价
        risk_cost = self.estimator.get_risk(next_x, next_y) * self.config.risk_factor * self.config.k_wind

        return energy_joules + risk_cost

    def search(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> Optional[List[Tuple[float, float, float]]]:
        # 初始化起点/终点高度（海拔，m）
        start_z = self.estimator.get_altitude(start_pos[0], start_pos[1]) + 50.0
        goal_z = self.estimator.get_altitude(goal_pos[0], goal_pos[1]) + 50.0

        start_node = Node(start_pos[0], start_pos[1], start_z, g=0.0, h=0.0)
        start_node.h = self.heuristic(start_node.get_pos(), (goal_pos[0], goal_pos[1], goal_z))

        open_list: List[Node] = []
        closed_set = set()
        heapq.heappush(open_list, start_node)

        steps = 0
        arrival_dist_xy = self.step_size
        arrival_dist_z = self.z_step

        print(f"🚀 3D 搜索开始... 起点Z:{start_z:.1f}m -> 终点Z:{goal_z:.1f}m")

        while open_list and steps < self.config.max_steps:
            steps += 1
            current_node = heapq.heappop(open_list)

            # 到达判定（米）
            d_xy = math.hypot(current_node.x - goal_pos[0], current_node.y - goal_pos[1])
            d_z = abs(current_node.z - goal_z)
            if d_xy < arrival_dist_xy and d_z < arrival_dist_z * 2:
                print(f"✅ 3D寻路成功！耗时步数: {steps}, 总代价: {current_node.g:.2f}")
                return self._reconstruct_path(current_node)

            # 网格索引（floor 更稳定）
            grid_idx = (
                int(math.floor((current_node.x - self.min_x) / self.step_size)),
                int(math.floor((current_node.y - self.min_y) / self.step_size)),
                int(math.floor(current_node.z / self.z_step)),
            )
            if grid_idx in closed_set:
                continue
            closed_set.add(grid_idx)

            # 26 邻域扩展
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        next_x = current_node.x + dx * self.step_size
                        next_y = current_node.y + dy * self.step_size
                        next_z = current_node.z + dz * self.z_step

                        # 边界检查（米）
                        if not (self.min_x <= next_x <= self.max_x and self.min_y <= next_y <= self.max_y):
                            continue

                        terrain_alt = self.estimator.get_altitude(next_x, next_y)
                        # 限制相对于地形的最大高度（m）
                        if next_z > terrain_alt + self.config.max_ceiling:
                            continue
                        # 离地至少 10 m
                        if next_z < terrain_alt + 10.0:
                            continue

                        move_cost = self.calculate_cost(current_node, next_x, next_y, next_z)
                        if move_cost == float('inf'):
                            continue

                        new_g = current_node.g + move_cost
                        new_h = self.heuristic((next_x, next_y, next_z), (goal_pos[0], goal_pos[1], goal_z))
                        heapq.heappush(open_list, Node(next_x, next_y, next_z, g=new_g, h=new_h, parent=current_node))

        print("❌ 3D 搜索失败：步数耗尽。")
        return None

    def _reconstruct_path(self, node: Node) -> List[Tuple[float, float, float]]:
        path: List[Tuple[float, float, float]] = []
        while node:
            path.append((node.x, node.y, node.z))
            node = node.parent
        return path[::-1]
