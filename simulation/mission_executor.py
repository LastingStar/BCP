"""
Single-drone mission executor used by planning and RL ablation experiments.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from configs.config import SimulationConfig
from core.battery_manager import BatteryManager
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.planner import AStarPlanner
from models.mission_models import MissionResult, PathPlanResult, SimulationState


Point3D = Tuple[float, float, float]


class MissionExecutor:
    """
    Dynamic mission executor for the legacy single-drone pipeline.
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
        start_z = self.estimator.get_altitude(start_xy[0], start_xy[1]) + self.config.takeoff_altitude_agl
        state = SimulationState(
            current_time_s=0.0,
            position_xyz=(start_xy[0], start_xy[1], start_z),
            remaining_energy_j=self.config.battery_capacity_j,
            traveled_path_xyz=[(start_xy[0], start_xy[1], start_z)],
        )

        initial_no_wind_path = self._plan_once(start_xy, goal_xy, use_wind=False)
        initial_wind_path = self._plan_once(start_xy, goal_xy, use_wind=True)
        replanned_paths_xyz: List[List[Point3D]] = []
        time_history_s: List[float] = []
        power_history_w: List[float] = []
        risk_history: List[float] = []

        if self.config.disable_periodic_replan:
            if not initial_wind_path or len(initial_wind_path) < 2:
                state.is_mission_failed = True
                state.failure_reason = "planner failed to find a path"
            else:
                state = self._advance_along_path(
                    state=state,
                    path_xyz=initial_wind_path,
                    delta_t_s=self.config.max_mission_time_s,
                    time_history_s=time_history_s,
                    power_history_w=power_history_w,
                    risk_history=risk_history,
                )
                state.is_goal_reached = self._check_goal_reached(state.position_xyz, goal_xy)
                if not state.is_goal_reached and not state.is_mission_failed:
                    state.is_mission_failed = True
                    state.failure_reason = "goal_not_reached_without_replan"
            return self._build_result(
                state=state,
                initial_no_wind_path=initial_no_wind_path,
                initial_wind_path=initial_wind_path,
                replanned_paths_xyz=replanned_paths_xyz,
                time_history_s=time_history_s,
                power_history_w=power_history_w,
                risk_history=risk_history,
            )

        while True:
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

            current_xy = (state.position_xyz[0], state.position_xyz[1])
            planned_path = self._plan_once(
                current_xy,
                goal_xy,
                use_wind=True,
                start_time_s=state.current_time_s,
            )
            if not planned_path or len(planned_path) < 2:
                state.is_mission_failed = True
                state.failure_reason = "planner failed to find a path"
                break

            replanned_paths_xyz.append(planned_path)
            estimated_path_energy_j = self.physics.estimate_path_energy(
                planned_path,
                wind_sampler=lambda x, y, z: self._sample_wind_3d(x, y, z, state.current_time_s),
                cruise_speed_mps=self.config.cruise_speed_mps,
            )
            if not self.battery_manager.is_path_feasible(state.remaining_energy_j, estimated_path_energy_j):
                state.is_mission_failed = True
                state.failure_reason = "replanned path is not battery feasible"
                break

            state = self._advance_along_path(
                state=state,
                path_xyz=planned_path,
                delta_t_s=self.config.mission_update_interval_s,
                time_history_s=time_history_s,
                power_history_w=power_history_w,
                risk_history=risk_history,
            )
            state.replans_count += 1
            if state.is_mission_failed:
                break

        return self._build_result(
            state=state,
            initial_no_wind_path=initial_no_wind_path,
            initial_wind_path=initial_wind_path,
            replanned_paths_xyz=replanned_paths_xyz,
            time_history_s=time_history_s,
            power_history_w=power_history_w,
            risk_history=risk_history,
        )

    def _build_result(
        self,
        state: SimulationState,
        initial_no_wind_path: Optional[List[Point3D]],
        initial_wind_path: Optional[List[Point3D]],
        replanned_paths_xyz: List[List[Point3D]],
        time_history_s: List[float],
        power_history_w: List[float],
        risk_history: List[float],
    ) -> MissionResult:
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
            time_history_s=time_history_s,
            power_history_w=power_history_w,
            risk_history=risk_history,
        )

    def _plan_once(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        use_wind: bool,
        start_time_s: float = 0.0,
    ) -> Optional[List[Point3D]]:
        original_k_wind = self.config.k_wind
        original_penalty = self.config.fatal_crash_penalty_j
        self.config.k_wind = 1.0 if use_wind else 0.0

        try:
            path = self.planner.search(start_xy, goal_xy, start_time_s=start_time_s)
            if path is None and use_wind:
                print("\n[System Warning] Safe path not found. Retrying with softer risk penalty...")
                self.config.fatal_crash_penalty_j = original_penalty * 0.33
                path = self.planner.search(start_xy, goal_xy, start_time_s=start_time_s)
        finally:
            self.config.k_wind = original_k_wind
            self.config.fatal_crash_penalty_j = original_penalty

        return path

    def _advance_along_path(
        self,
        state: SimulationState,
        path_xyz: List[Point3D],
        delta_t_s: float,
        time_history_s: List[float],
        power_history_w: List[float],
        risk_history: List[float],
    ) -> SimulationState:
        remaining_distance_m = self.config.cruise_speed_mps * delta_t_s
        if remaining_distance_m <= 0:
            return state

        current_pos = np.array(state.position_xyz, dtype=float)
        new_traveled_path = list(state.traveled_path_xyz)
        total_energy_used_j = state.total_energy_used_j
        remaining_energy_j = state.remaining_energy_j
        current_time_s = state.current_time_s
        execution_path = [tuple(current_pos)] + list(path_xyz[1:])

        for i in range(len(execution_path) - 1):
            p0 = np.array(execution_path[i], dtype=float)
            p1_full = np.array(execution_path[i + 1], dtype=float)
            seg_vec = p1_full - p0
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len <= 1e-9:
                continue

            if remaining_distance_m >= seg_len:
                p1 = p1_full
            else:
                ratio = remaining_distance_m / seg_len
                p1 = p0 + ratio * seg_vec

            midpoint = 0.5 * (p0 + p1)
            wind_3d = self._sample_wind_3d(midpoint[0], midpoint[1], midpoint[2], current_time_s)
            seg_energy_j, seg_time_s, seg_power_w = self.physics.estimate_segment_energy(
                p0_xyz=p0,
                p1_xyz=p1,
                wind_velocity_xyz=wind_3d,
                cruise_speed_mps=self.config.cruise_speed_mps,
            )
            next_time_s = current_time_s + seg_time_s
            v_ground = max(np.linalg.norm(p1 - p0) / max(seg_time_s, 1e-6), 1.0)
            p_crash, _ = self.estimator.get_risk(p1[0], p1[1], p1[2], v_ground, next_time_s)

            time_history_s.append(next_time_s)
            power_history_w.append(float(seg_power_w))
            risk_history.append(float(p_crash))

            if seg_power_w > self.config.max_power * self.config.rl_overload_power_ratio:
                return SimulationState(
                    current_time_s=current_time_s,
                    position_xyz=tuple(current_pos),
                    remaining_energy_j=remaining_energy_j,
                    traveled_path_xyz=new_traveled_path,
                    replans_count=state.replans_count,
                    total_energy_used_j=total_energy_used_j,
                    is_mission_failed=True,
                    failure_reason="overload",
                )
            if p_crash > self.config.rl_terminate_risk_threshold:
                return SimulationState(
                    current_time_s=current_time_s,
                    position_xyz=tuple(current_pos),
                    remaining_energy_j=remaining_energy_j,
                    traveled_path_xyz=new_traveled_path,
                    replans_count=state.replans_count,
                    total_energy_used_j=total_energy_used_j,
                    is_mission_failed=True,
                    failure_reason="storm_risk_too_high",
                )
            if not self.battery_manager.can_consume(remaining_energy_j, seg_energy_j):
                return SimulationState(
                    current_time_s=current_time_s,
                    position_xyz=tuple(current_pos),
                    remaining_energy_j=remaining_energy_j,
                    traveled_path_xyz=new_traveled_path,
                    replans_count=state.replans_count,
                    total_energy_used_j=total_energy_used_j,
                    is_mission_failed=True,
                    failure_reason="battery_depleted",
                )

            remaining_energy_j = self.battery_manager.consume_energy(remaining_energy_j, seg_energy_j)
            total_energy_used_j += seg_energy_j
            current_time_s = next_time_s
            current_pos = p1
            new_traveled_path.append(tuple(current_pos))

            if remaining_distance_m < seg_len:
                remaining_distance_m = 0.0
                break
            remaining_distance_m -= seg_len

        return SimulationState(
            current_time_s=current_time_s,
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
        goal_z = self.estimator.get_altitude(goal_xy[0], goal_xy[1]) + self.config.takeoff_altitude_agl
        dx = current_xyz[0] - goal_xy[0]
        dy = current_xyz[1] - goal_xy[1]
        dz = current_xyz[2] - goal_z

        if self.config.goal_check_mode == "3d_distance":
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            return dist_3d <= self.config.goal_tolerance_3d_m
        if self.config.goal_check_mode == "xy_z_tolerance":
            dist_xy = math.hypot(dx, dy)
            return dist_xy <= self.config.goal_tolerance_xy_m and abs(dz) <= self.config.goal_tolerance_z_m
        raise ValueError(f"Unknown goal check mode: {self.config.goal_check_mode}")

    def _sample_wind_3d(
        self,
        x: float,
        y: float,
        z: float,
        t_s: float,
    ) -> np.ndarray:
        wind_2d = self.estimator.get_wind(x, y, z=z, t_s=t_s)
        return np.array([wind_2d[0], wind_2d[1], 0.0], dtype=float)
