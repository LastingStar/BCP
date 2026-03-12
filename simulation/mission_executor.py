import math
import numpy as np
from typing import List, Tuple, Optional

from configs.config import SimulationConfig
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.battery_manager import BatteryManager
from core.planner import AStarPlanner
from models.mission_models import SimulationState, MissionResult, PathPlanResult


Point3D = Tuple[float, float, float]


class MissionExecutor:
    """
    动态任务执行器：
    - 初始规划
    - 沿路径飞行固定时间
    - 更新时间 / 位置 / 电量
    - 周期重规划
    - 直到到达或失败
    """

    def __init__(
        self,
        config: SimulationConfig,
        estimator: StateEstimator,
        physics: PhysicsEngine,
        battery_manager: BatteryManager,
        planner: AStarPlanner,
    ):
        self.config = config
        self.estimator = estimator
        self.physics = physics
        self.battery_manager = battery_manager
        self.planner = planner

    def execute_mission(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> MissionResult:
        """
        执行完整动态任务。
        """
        start_z = self.estimator.get_altitude(start_xy[0], start_xy[1]) + 50.0
        initial_state = SimulationState(
            current_time_s=0.0,
            position_xyz=(start_xy[0], start_xy[1], start_z),
            remaining_energy_j=self.config.battery_capacity_j,
            traveled_path_xyz=[(start_xy[0], start_xy[1], start_z)],
            replans_count=0,
            total_energy_used_j=0.0,
            is_goal_reached=False,
            is_mission_failed=False,
            failure_reason=None,
        )

        # 1) 记录三类路径中的前两类
        initial_no_wind_path = self._plan_once(start_xy, goal_xy, use_wind=False)
        initial_wind_path = self._plan_once(start_xy, goal_xy, use_wind=True)

        replanned_paths_xyz: List[List[Point3D]] = []

        state = initial_state

        # 2) 动态执行主循环
        while True:
            # 2.1 终止条件检查
            if self._check_goal_reached(state.position_xyz, goal_xy):
                state.is_goal_reached = True
                break

            if state.replans_count >= self.config.max_replans:
                state.is_mission_failed = True
                state.failure_reason = "maximum replans exceeded"
                break

            if state.current_time_s >= self.config.max_mission_time_s:
                state.is_mission_failed = True
                state.failure_reason = "maximum mission time exceeded"
                break

            if state.remaining_energy_j <= self.battery_manager.get_min_reserve_energy_j():
                state.is_mission_failed = True
                state.failure_reason = "battery below reserve threshold"
                break

            # 2.2 以当前位置重新规划
            current_xy = (state.position_xyz[0], state.position_xyz[1])
            planned_path = self._plan_once(
                current_xy,
                goal_xy,
                use_wind=True,
            )

            if planned_path is None or len(planned_path) < 2:
                state.is_mission_failed = True
                state.failure_reason = "planner failed to find a path"
                break

            replanned_paths_xyz.append(planned_path)

            # 2.3 估计这条新路径理论能耗是否可行
            estimated_path_energy_j = self.physics.estimate_path_energy(
                planned_path,
                wind_sampler=lambda x, y, z: self._sample_wind_3d(x, y, z, state.current_time_s),
                cruise_speed_mps=self.config.cruise_speed_mps,
            )

            if not self.battery_manager.is_path_feasible(
                state.remaining_energy_j,
                estimated_path_energy_j,
            ):
                state.is_mission_failed = True
                state.failure_reason = "replanned path is not battery feasible"
                break

            # 2.4 沿路径飞行一个更新周期
            state = self._advance_along_path(
                state=state,
                path_xyz=planned_path,
                delta_t_s=self.config.mission_update_interval_s,
            )

            state.replans_count += 1

            if state.is_mission_failed:
                break

        return MissionResult(
            success=state.is_goal_reached and not state.is_mission_failed,
            final_state=state,
            initial_no_wind_path_xyz=initial_no_wind_path if initial_no_wind_path else [],
            initial_wind_path_xyz=initial_wind_path if initial_wind_path else [],
            replanned_paths_xyz=replanned_paths_xyz,
            actual_flown_path_xyz=state.traveled_path_xyz,
            total_replans=state.replans_count,
            total_mission_time_s=state.current_time_s,
            total_energy_used_j=state.total_energy_used_j,
            failure_reason=state.failure_reason,
        )

    def _plan_once(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        use_wind: bool,
    ) -> Optional[List[Point3D]]:
        """
        单次规划。通过临时切换 k_wind 控制是否启用风代价。
        """
        original_k_wind = self.config.k_wind
        self.config.k_wind = 1.0 if use_wind else 0.0
        try:
            path = self.planner.search(start_xy, goal_xy)
        finally:
            self.config.k_wind = original_k_wind
        return path

    def _advance_along_path(
        self,
        state: SimulationState,
        path_xyz: List[Point3D],
        delta_t_s: float,
    ) -> SimulationState:
        """
        沿当前路径前进 delta_t_s 秒，更新：
        - 当前位置
        - 时间
        - 剩余电量
        - 已飞行轨迹
        """
        remaining_distance_m = self.config.cruise_speed_mps * delta_t_s
        if remaining_distance_m <= 0:
            return state

        current_pos = np.array(state.position_xyz, dtype=float)
        new_traveled_path = list(state.traveled_path_xyz)
        total_energy_used_j = state.total_energy_used_j
        remaining_energy_j = state.remaining_energy_j

        # 将路径首点替换为当前状态，避免因重规划起点与当前点不完全重合带来跳变
        execution_path = [tuple(current_pos)] + list(path_xyz[1:])

        for i in range(len(execution_path) - 1):
            p0 = np.array(execution_path[i], dtype=float)
            p1 = np.array(execution_path[i + 1], dtype=float)

            seg_vec = p1 - p0
            seg_len = np.linalg.norm(seg_vec)
            if seg_len <= 1e-9:
                continue

            # 整段可飞完
            if remaining_distance_m >= seg_len:
                midpoint = 0.5 * (p0 + p1)
                wind_3d = self._sample_wind_3d(
                    midpoint[0], midpoint[1], midpoint[2], state.current_time_s
                )

                seg_energy_j, seg_time_s, _ = self.physics.estimate_segment_energy(
                    p0_xyz=p0,
                    p1_xyz=p1,
                    wind_velocity_xyz=wind_3d,
                    cruise_speed_mps=self.config.cruise_speed_mps,
                )

                if not self.battery_manager.can_consume(remaining_energy_j, seg_energy_j):
                    return SimulationState(
                        current_time_s=state.current_time_s,
                        position_xyz=tuple(current_pos),
                        remaining_energy_j=remaining_energy_j,
                        traveled_path_xyz=new_traveled_path,
                        replans_count=state.replans_count,
                        total_energy_used_j=total_energy_used_j,
                        is_goal_reached=False,
                        is_mission_failed=True,
                        failure_reason="battery depleted during path execution",
                    )

                remaining_energy_j = self.battery_manager.consume_energy(
                    remaining_energy_j, seg_energy_j
                )
                total_energy_used_j += seg_energy_j
                state.current_time_s += seg_time_s

                current_pos = p1
                new_traveled_path.append(tuple(current_pos))
                remaining_distance_m -= seg_len

            else:
                # 只能飞一部分
                ratio = remaining_distance_m / seg_len
                p_partial = p0 + ratio * seg_vec

                midpoint = 0.5 * (p0 + p_partial)
                wind_3d = self._sample_wind_3d(
                    midpoint[0], midpoint[1], midpoint[2], state.current_time_s
                )

                seg_energy_j, seg_time_s, _ = self.physics.estimate_segment_energy(
                    p0_xyz=p0,
                    p1_xyz=p_partial,
                    wind_velocity_xyz=wind_3d,
                    cruise_speed_mps=self.config.cruise_speed_mps,
                )

                if not self.battery_manager.can_consume(remaining_energy_j, seg_energy_j):
                    return SimulationState(
                        current_time_s=state.current_time_s,
                        position_xyz=tuple(current_pos),
                        remaining_energy_j=remaining_energy_j,
                        traveled_path_xyz=new_traveled_path,
                        replans_count=state.replans_count,
                        total_energy_used_j=total_energy_used_j,
                        is_goal_reached=False,
                        is_mission_failed=True,
                        failure_reason="battery depleted during partial path execution",
                    )

                remaining_energy_j = self.battery_manager.consume_energy(
                    remaining_energy_j, seg_energy_j
                )
                total_energy_used_j += seg_energy_j
                state.current_time_s += seg_time_s

                current_pos = p_partial
                new_traveled_path.append(tuple(current_pos))
                remaining_distance_m = 0.0
                break

        return SimulationState(
            current_time_s=state.current_time_s,
            position_xyz=tuple(current_pos),
            remaining_energy_j=remaining_energy_j,
            traveled_path_xyz=new_traveled_path,
            replans_count=state.replans_count,
            total_energy_used_j=total_energy_used_j,
            is_goal_reached=self._check_goal_reached(tuple(current_pos), (path_xyz[-1][0], path_xyz[-1][1])),
            is_mission_failed=False,
            failure_reason=None,
        )

    def _check_goal_reached(
        self,
        current_xyz: Point3D,
        goal_xy: Tuple[float, float],
    ) -> bool:
        """
        按配置检查是否到达终点。
        """
        goal_z = self.estimator.get_altitude(goal_xy[0], goal_xy[1]) + 50.0

        dx = current_xyz[0] - goal_xy[0]
        dy = current_xyz[1] - goal_xy[1]
        dz = current_xyz[2] - goal_z

        if self.config.goal_check_mode == "3d_distance":
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            return dist_3d <= self.config.goal_tolerance_3d_m

        if self.config.goal_check_mode == "xy_z_tolerance":
            dist_xy = math.hypot(dx, dy)
            return (
                dist_xy <= self.config.goal_tolerance_xy_m
                and abs(dz) <= self.config.goal_tolerance_z_m
            )

        raise ValueError(f"Unknown goal check mode: {self.config.goal_check_mode}")

    def _sample_wind_3d(
        self,
        x: float,
        y: float,
        z: float,
        t_s: float,
    ) -> np.ndarray:
        """
        将 estimator 的二维风扩展成三维风向量 [wx, wy, wz]。
        当前阶段 wz = 0。
        """
        wind_2d = self.estimator.get_wind(x, y, z=z, t_s=t_s)
        return np.array([wind_2d[0], wind_2d[1], 0.0], dtype=float)