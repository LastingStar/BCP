"""
无人机路径规划强化学习环境

基于OpenAI Gym标准，将当前路径规划系统封装为RL环境，
支持PPO等算法的训练和评估。

状态空间：
- 无人机位置 (x, y, z)
- 目标位置 (x, y, z)
- 当前风场向量 (u, v)
- 剩余能量比例
- 时间步

动作空间：
- 连续动作：航向角(0-360°)、俯仰角(-90°-90°)、速度比例(0-1)

奖励函数：
- 距离奖励：向目标前进获得正奖励
- 能量惩罚：消耗能量获得负奖励
- 风险惩罚：进入高风险区域获得负奖励
- 成功奖励：到达目标获得大正奖励
- 失败惩罚：碰撞或能量耗尽获得大负奖励
"""

import gym
import numpy as np
from typing import Tuple, Dict, Any
from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.battery_manager import BatteryManager


class DronePathPlanningEnv(gym.Env):
    """
    无人机路径规划强化学习环境

    状态空间: 12维连续向量
    [drone_x, drone_y, drone_z, goal_x, goal_y, goal_z,
     wind_u, wind_v, energy_ratio, time_step, dist_to_goal, heading_error]

    动作空间: 3维连续向量
    [heading_delta(-180°-180°), pitch_delta(-45°-45°), speed_ratio(0.1-1.0)]
    """

    def __init__(self, config: SimulationConfig = None):
        super().__init__()

        # 初始化系统组件
        self.config = config or SimulationConfig()
        self.map_manager = MapManager(self.config)
        self.wind_model = WindModelFactory.create(
            self.config.wind_model_type,
            self.config,
            bounds=self.map_manager.get_bounds()
        )
        self.estimator = StateEstimator(self.map_manager, self.wind_model, self.config)
        self.physics = PhysicsEngine(self.config)
        self.battery_manager = BatteryManager(self.config)

        # 定义状态空间 (12维)
        self.observation_space = gym.spaces.Box(
            low=np.array([
                self.map_manager.x[0], self.map_manager.y[0], 0,  # drone pos
                self.map_manager.x[0], self.map_manager.y[0], 0,  # goal pos
                -50, -50,  # wind components
                0, 0, 0, 0  # energy, time, dist, heading
            ]),
            high=np.array([
                self.map_manager.x[-1], self.map_manager.y[-1], 5000,  # drone pos
                self.map_manager.x[-1], self.map_manager.y[-1], 5000,  # goal pos
                50, 50,  # wind components
                1, 1000, 20000, 180  # energy, time, dist, heading
            ]),
            dtype=np.float32
        )

        # 定义动作空间 (3维连续)
        self.action_space = gym.spaces.Box(
            low=np.array([-180, -45, 0.1]),   # heading_delta, pitch_delta, speed_ratio
            high=np.array([180, 45, 1.0]),
            dtype=np.float32
        )

        # 环境状态
        self.current_state = None
        self.goal_position = None
        self.episode_step = 0
        self.max_episode_steps = 1000

        # 奖励参数
        self.distance_reward_scale = 1.0
        self.energy_penalty_scale = 0.01
        self.risk_penalty_scale = 10.0
        self.success_reward = 1000.0
        self.collision_penalty = -500.0
        self.timeout_penalty = -100.0

    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        # 随机生成起终点
        bounds = self.map_manager.get_bounds()
        margin = 500  # 边界余量

        start_x = np.random.uniform(bounds[0] + margin, bounds[1] - margin)
        start_y = np.random.uniform(bounds[2] + margin, bounds[3] - margin)
        goal_x = np.random.uniform(bounds[0] + margin, bounds[1] - margin)
        goal_y = np.random.uniform(bounds[2] + margin, bounds[3] - margin)

        # 确保起终点距离适中
        min_dist = 2000
        max_dist = 8000
        dist = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        while dist < min_dist or dist > max_dist:
            goal_x = np.random.uniform(bounds[0] + margin, bounds[1] - margin)
            goal_y = np.random.uniform(bounds[2] + margin, bounds[3] - margin)
            dist = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)

        start_z = self.estimator.get_altitude(start_x, start_y) + self.config.takeoff_altitude_agl
        goal_z = self.estimator.get_altitude(goal_x, goal_y) + self.config.takeoff_altitude_agl

        self.current_state = {
            'position': np.array([start_x, start_y, start_z]),
            'goal': np.array([goal_x, goal_y, goal_z]),
            'energy': self.config.battery_capacity_j,
            'time': 0.0,
            'prev_distance': dist
        }

        self.goal_position = np.array([goal_x, goal_y, goal_z])
        self.episode_step = 0

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一步动作"""
        self.episode_step += 1

        # 解析动作
        heading_delta, pitch_delta, speed_ratio = action

        # 计算新位置
        new_position, energy_used = self._execute_action(action)

        # 更新状态
        self.current_state['position'] = new_position
        self.current_state['energy'] -= energy_used
        self.current_state['time'] += 1.0  # 简化时间步

        # 计算奖励
        reward, done, info = self._calculate_reward()

        # 获取观测
        observation = self._get_observation()

        return observation, reward, done, info

    def _execute_action(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """执行动作并返回新位置和能量消耗"""
        heading_delta, pitch_delta, speed_ratio = action

        # 当前位置和朝向
        pos = self.current_state['position']
        current_heading = getattr(self, 'current_heading', 0.0)

        # 更新朝向
        new_heading = current_heading + heading_delta
        new_heading = np.clip(new_heading, -180, 180)
        self.current_heading = new_heading

        # 计算移动向量
        speed = self.config.drone_speed * speed_ratio
        dx = speed * np.cos(np.radians(new_heading))
        dy = speed * np.sin(np.radians(new_heading))
        dz = speed * np.sin(np.radians(pitch_delta))

        new_pos = pos + np.array([dx, dy, dz])

        # 边界检查
        bounds = self.map_manager.get_bounds()
        new_pos[0] = np.clip(new_pos[0], bounds[0], bounds[1])
        new_pos[1] = np.clip(new_pos[1], bounds[2], bounds[3])

        # 高度检查
        ground_alt = self.estimator.get_altitude(new_pos[0], new_pos[1])
        min_alt = ground_alt + 10.0
        max_alt = ground_alt + self.config.max_ceiling
        new_pos[2] = np.clip(new_pos[2], min_alt, max_alt)

        # 计算能量消耗
        wind = self.estimator.get_wind(new_pos[0], new_pos[1], new_pos[2])
        velocity_vector = np.array([dx, dy, dz])
        power = self.physics.estimate_power_from_vectors(velocity_vector, wind)
        energy_used = power * 1.0  # 1秒时间步

        return new_pos, energy_used

    def _calculate_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """计算奖励和终止条件"""
        pos = self.current_state['position']
        goal = self.current_state['goal']
        energy = self.current_state['energy']

        # 计算到目标距离
        current_distance = np.linalg.norm(pos - goal)
        prev_distance = self.current_state['prev_distance']

        # 距离奖励
        distance_reward = (prev_distance - current_distance) * self.distance_reward_scale
        self.current_state['prev_distance'] = current_distance

        # 能量惩罚
        energy_penalty = self.energy_penalty_scale

        # 风险惩罚
        risk_penalty = 0.0
        if hasattr(self.estimator, 'get_risk'):
            _, risk = self.estimator.get_risk(pos[0], pos[1], pos[2], 15.0)  # 假设速度15m/s
            risk_penalty = risk * self.risk_penalty_scale

        # 碰撞检查
        collision = self.map_manager.is_collision(pos[0], pos[1], pos[2])

        # 终止条件
        done = False
        info = {}

        if current_distance < 50.0:  # 到达目标
            reward = self.success_reward
            done = True
            info['success'] = True
        elif collision:
            reward = self.collision_penalty
            done = True
            info['collision'] = True
        elif energy <= self.battery_manager.get_min_reserve_energy_j():
            reward = self.collision_penalty
            done = True
            info['battery_depleted'] = True
        elif self.episode_step >= self.max_episode_steps:
            reward = self.timeout_penalty
            done = True
            info['timeout'] = True
        else:
            reward = distance_reward - energy_penalty - risk_penalty

        return reward, done, info

    def _get_observation(self) -> np.ndarray:
        """获取当前观测状态"""
        pos = self.current_state['position']
        goal = self.current_state['goal']

        # 风场信息
        wind = self.estimator.get_wind(pos[0], pos[1], pos[2])

        # 能量比例
        energy_ratio = self.current_state['energy'] / self.config.battery_capacity_j

        # 到目标距离
        dist_to_goal = np.linalg.norm(pos - goal)

        # 朝向误差 (简化计算)
        heading_error = 0.0  # 可以计算当前朝向与目标方向的差值

        observation = np.array([
            pos[0], pos[1], pos[2],           # 无人机位置
            goal[0], goal[1], goal[2],         # 目标位置
            wind[0], wind[1],                  # 风场
            energy_ratio,                      # 能量比例
            self.current_state['time'],        # 时间
            dist_to_goal,                      # 到目标距离
            heading_error                      # 朝向误差
        ], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        """渲染环境状态"""
        # 可以实现简单的matplotlib可视化
        pass

    def close(self):
        """清理环境"""
        pass