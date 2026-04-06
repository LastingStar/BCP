# --- START OF FILE simulation/swarm_mission_executor.py ---
import math
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple

from configs.config import SimulationConfig
from core.estimator import StateEstimator
from core.physics import PhysicsEngine
from core.battery_manager import BatteryManager
from core.planner import AStarPlanner
from models.mission_models import SimulationState, MissionResult
from simulation.swarm_disturbance import GustEvent, SwarmDisturbanceManager

Point3D = Tuple[float, float, float]

class SwarmMissionExecutor:
    """
    大创核心创新 V3：异构四机编队 (Master + Scout + Relay + Support)
    支持双主机模式切换：纯 A* 几何追踪 vs RL 微观抗扰控制
    """
    def __init__(
        self,
        config: SimulationConfig,
        true_estimator: StateEstimator,
        physics: PhysicsEngine,
        battery_manager: BatteryManager,
        master_mode: str = "astar",   
        rl_model = None               
    ):
        self.config = config
        self.true_estimator = true_estimator 
        self.physics = physics
        self.battery = battery_manager
        
        self.master_mode = master_mode
        self.rl_model = rl_model
        
        if self.master_mode == "rl":
            if self.rl_model is None:
                raise ValueError("启用 rl 模式必须传入 rl_model!")
            from rl_env.drone_env import GuidedDroneEnv
            self.rl_perception_env = GuidedDroneEnv(self.config)
            self.rl_perception_env.estimator = self.true_estimator 
            self.rl_perception_env.physics = self.physics
            self.master_heading = 0.0
        
        self.belief_config = copy.deepcopy(config)
        isolated_belief_wind = copy.deepcopy(true_estimator.wind)
        self.belief_estimator = StateEstimator(true_estimator.map, isolated_belief_wind, self.belief_config)
        self.planner = AStarPlanner(self.belief_config, self.belief_estimator, self.physics)
        
        self.swarm_dt_s = 5.0                  
        self.scout_speed_mps = 25.0            
        self.relay_speed_mps = 30.0
        self.support_speed_mps = 35.0
        
        self.scout_radar_radius_m = 1000.0     
        self.support_radar_radius_m = 800.0
        self.comm_range_m = 1000.0             
        self.relay_lateral_offset_m = 350.0
        self.support_trail_offset_m = 100.0
        self.support_flank_offset_m = 300.0
        self.safe_risk_threshold = getattr(
            self.config,
            "rl_safe_spawn_risk_threshold",
            min(0.25, getattr(self.config, "rl_terminate_risk_threshold", 0.55) * 0.5),
        )
        
        self.scout_patrol_target_m = 1800.0    
        self.scout_return_trigger_m = 2200.0   
        self.scout_resume_trigger_m = 1200.0   
        
        self.scout_state = "FORWARD"           
        
        self.shared_buffered_storms = set()    
        self.master_known_storms = set()       
        self.is_link_active = True
        self.disturbance = SwarmDisturbanceManager(self.config)
        self.support_local_threat_xy: Optional[np.ndarray] = None
        self.support_mode = "FORMATION"
        self.first_warning_time_s: Optional[float] = None
        self.first_warning_distance_m: Optional[float] = None
        self.first_warning_route: Optional[str] = None
        self.warning_count = 0

    def execute_mission(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> MissionResult:
        print("\n" + "="*60)
        print(f"🚀 [FANET 异构集群协同启动] 控制模式: {self.master_mode.upper()}")
        print("="*60 + "\n")
        self.disturbance.reset(seed=self.config.wind_seed)
        self.support_local_threat_xy = None
        self.support_mode = "FORMATION"
        self.first_warning_time_s = None
        self.first_warning_distance_m = None
        self.first_warning_route = None
        self.warning_count = 0
        
        start_z = self.true_estimator.get_altitude(start_xy[0], start_xy[1]) + self.config.takeoff_altitude_agl
        
        state = SimulationState(
            current_time_s=0.0,
            position_xyz=(start_xy[0], start_xy[1], start_z),
            remaining_energy_j=self.config.battery_capacity_j,
            traveled_path_xyz=[(start_xy[0], start_xy[1], start_z)],
        )

        scout_pos = np.array(start_xy, dtype=float)
        relay_pos = np.array(start_xy, dtype=float)
        support_pos = np.array(start_xy, dtype=float)
        master_sync_path_history = [state.position_xyz]
        scout_path_history = [(scout_pos[0], scout_pos[1], start_z)]
        relay_path_history = [(relay_pos[0], relay_pos[1], start_z)]
        support_path_history = [(support_pos[0], support_pos[1], start_z)]
        swarm_time_history = [state.current_time_s]
        comm_status_history = [True]
        support_mode_history = [self.support_mode]
        gust_active_history = [False]
        master_power_history_w = []
        master_risk_history = []
        nearest_threat_distance_history_m = []
        link_status_history = [
            self._evaluate_link_status(np.array(start_xy, dtype=float), scout_pos, relay_pos, support_pos)
        ]

        if self.master_mode == "rl":
            vec = np.array(goal_xy) - np.array(start_xy)
            self.master_heading = math.degrees(math.atan2(vec[1], vec[0]))

        self._sync_belief_world(state.current_time_s)
        initial_path = self._plan_in_belief_world(start_xy, goal_xy, state.current_time_s)
        current_path = initial_path
        replanned_paths = []
        
        while not state.is_goal_reached and not state.is_mission_failed:
            if not current_path:
                state.is_mission_failed, state.failure_reason = True, "no_path"
                break

            master_pos_np = np.array(state.position_xyz[:2])
            master_z = state.position_xyz[2]
            
            # --- 1. Scout 持续前出探测 ---
            self.scout_state = "FORWARD"
            scout_pos = self._move_scout(scout_pos, master_pos_np, current_path, self.swarm_dt_s)
            scout_z = self._get_formation_safe_z(scout_pos[0], scout_pos[1], master_z)

            # --- 2. Relay 动态搭桥 ---
            relay_active = self.config.swarm_topology_mode != "no_relay"
            if relay_active:
                relay_target = self._compute_relay_target(
                    master_xy=master_pos_np,
                    scout_xy=scout_pos,
                    current_t=state.current_time_s,
                )
                relay_pos = self._move_vehicle_towards(
                    current_pos=relay_pos,
                    target_pos=relay_target,
                    speed_mps=self.relay_speed_mps,
                    dt=self.swarm_dt_s,
                )
            else:
                relay_pos = master_pos_np.copy()
            relay_z = self._get_formation_safe_z(relay_pos[0], relay_pos[1], master_z)

            # --- 3. Support 弹性补位与护航 ---
            self.support_local_threat_xy = self._detect_support_shield_threat(
                master_xy=master_pos_np,
                support_xy=support_pos,
                current_t=state.current_time_s,
            )
            support_target = self._compute_support_target(
                master_xy=master_pos_np,
                relay_xy=relay_pos if relay_active else None,
                support_xy=support_pos,
                current_path=current_path,
                current_t=state.current_time_s,
            )
            support_pos = self._move_vehicle_towards(
                current_pos=support_pos,
                target_pos=support_target,
                speed_mps=self.support_speed_mps,
                dt=self.swarm_dt_s,
            )
            support_z = self._get_formation_safe_z(support_pos[0], support_pos[1], master_z)
            
            # --- 4. 探路与补盲雷达 ---
            new_finds_by_scout = self._scout_radar_scan(scout_pos, state.current_time_s)
            if new_finds_by_scout:
                self.shared_buffered_storms.update(new_finds_by_scout)

            new_finds_by_support = self._support_radar_scan(support_pos, state.current_time_s)
            if new_finds_by_support:
                self.shared_buffered_storms.update(new_finds_by_support)
            
            # --- 5. 四机链路状态 ---
            link_snapshot = self._evaluate_link_status(
                master_pos_np,
                scout_pos,
                relay_pos if relay_active else None,
                support_pos,
            )
            link_was_active = self.is_link_active
            self.is_link_active = link_snapshot["network_active"]

            if self.is_link_active and not link_was_active:
                route_modes = []
                if link_snapshot["path_direct"]:
                    route_modes.append("Direct")
                if link_snapshot["path_relay"]:
                    route_modes.append("Relay")
                if link_snapshot["path_support"]:
                    route_modes.append("Support")
                route_text = " + ".join(route_modes) if route_modes else "Unknown"
                print(f"  🟢 [T={state.current_time_s:.0f}s] FANET 数据链重连！路由: {route_text}")
            elif not self.is_link_active and link_was_active:
                print(f"  🔴 [T={state.current_time_s:.0f}s] 四机网络断链！进入【信息孤岛】")

            # --- 6. 协同重规划 ---
            if self.is_link_active and len(self.shared_buffered_storms) > 0:
                self.warning_count += 1
                if self.first_warning_time_s is None:
                    self.first_warning_time_s = state.current_time_s
                    self.first_warning_distance_m = self._compute_nearest_threat_distance(master_pos_np, state.current_time_s)
                    self.first_warning_route = self._route_name_from_snapshot(link_snapshot)
                print(f"  📥 同步 {len(self.shared_buffered_storms)} 个盲区威胁！触发重规划...")
                self.master_known_storms.update(self.shared_buffered_storms)
                self.shared_buffered_storms.clear()
                
                current_xy = (state.position_xyz[0], state.position_xyz[1])
                self._sync_belief_world(state.current_time_s)
                new_path = self._plan_in_belief_world(current_xy, goal_xy, state.current_time_s)
                
                if new_path:
                    current_path = new_path
                    replanned_paths.append(new_path)
                    state.replans_count += 1
                else:
                    print("  ⚠️ [系统告警] 主航母前方无路可走！")
            
            # --- 7. 双轨接口分流：主机推演 ---
            # 🌟 【修复】删掉 len(current_path) < 2 的限制！允许只剩终点！
            master_prev_xyz = np.array(state.position_xyz, dtype=float)
            master_prev_time_s = state.current_time_s
            master_prev_energy_j = state.total_energy_used_j
            if self.master_mode == "astar":
                state, current_path = self._advance_master_step_astar(state, current_path, self.swarm_dt_s)
            elif self.master_mode == "rl":
                state, current_path = self._advance_master_step_rl(state, current_path, self.swarm_dt_s)

            master_pos_end = np.array(state.position_xyz[:2], dtype=float)
            record_link_snapshot = self._evaluate_link_status(
                master_pos_end,
                scout_pos,
                relay_pos if relay_active else None,
                support_pos,
            )
            self.is_link_active = record_link_snapshot["network_active"]
            master_power_w, master_risk = self._estimate_master_step_telemetry(
                previous_xyz=master_prev_xyz,
                current_xyz=np.array(state.position_xyz, dtype=float),
                previous_time_s=master_prev_time_s,
                current_time_s=state.current_time_s,
                previous_energy_j=master_prev_energy_j,
                current_energy_j=state.total_energy_used_j,
            )
            master_sync_path_history.append(state.position_xyz)
            scout_path_history.append((scout_pos[0], scout_pos[1], scout_z))
            relay_path_history.append((relay_pos[0], relay_pos[1], relay_z))
            support_path_history.append((support_pos[0], support_pos[1], support_z))
            swarm_time_history.append(state.current_time_s)
            link_status_history.append(record_link_snapshot)
            support_mode_history.append(self.support_mode)
            gust_active_history.append(self._is_gust_active(state.current_time_s))
            master_power_history_w.append(master_power_w)
            master_risk_history.append(master_risk)
            nearest_threat_distance_history_m.append(
                self._compute_nearest_threat_distance(master_pos_end, state.current_time_s)
            )
            comm_status_history.append(record_link_snapshot["network_active"])

            # --- 8. 任务状态判定 ---
            if self._check_goal_reached(state.position_xyz, goal_xy):
                state.is_goal_reached = True
                print(f"\n✅ 任务圆满完成！主机安全抵达终点。")
                break
            if state.current_time_s >= self.config.max_mission_time_s:
                state.is_mission_failed, state.failure_reason = True, "timeout"
                break

        return MissionResult(
            success=state.is_goal_reached and not state.is_mission_failed,
            final_state=state,
            initial_no_wind_path_xyz=initial_path if initial_path else [],
            initial_wind_path_xyz=initial_path if initial_path else [],
            replanned_paths_xyz=replanned_paths,
            actual_flown_path_xyz=state.traveled_path_xyz,
            total_replans=state.replans_count,
            total_mission_time_s=state.current_time_s,
            total_energy_used_j=state.total_energy_used_j,
            failure_reason=state.failure_reason,
            master_sync_path_xyz=master_sync_path_history,
            scout_flown_path_xyz=scout_path_history,      
            relay_flown_path_xyz=relay_path_history,
            support_flown_path_xyz=support_path_history,
            swarm_time_history=swarm_time_history,
            link_status_history=link_status_history,
            support_mode_history=support_mode_history,
            gust_active_history=gust_active_history,
            master_power_history_w=master_power_history_w,
            master_risk_history=master_risk_history,
            nearest_threat_distance_history_m=nearest_threat_distance_history_m,
            first_warning_time_s=self.first_warning_time_s,
            first_warning_distance_m=self.first_warning_distance_m,
            first_warning_route=self.first_warning_route,
            warning_count=self.warning_count,
            time_history_s=swarm_time_history,
            power_history_w=master_power_history_w,
            risk_history=master_risk_history,
            comm_status_history=comm_status_history       
        )

    # ==============================================================
    # 模式 A: A* 纯几何追踪器
    # ==============================================================
    def _advance_master_step_astar(self, state: SimulationState, path_xyz: List[Point3D], dt: float) -> Tuple[SimulationState, List[Point3D]]:
        if not path_xyz or len(path_xyz) < 2:
            state.current_time_s += dt
            return state, path_xyz

        new_path = list(path_xyz)
        current_pos = np.array(state.position_xyz, dtype=float)
        gust_xy, gust_event, gust_started = self.disturbance.begin_step(state.current_time_s, dt)
        if gust_started:
            self._log_gust_event(gust_event)

        if np.linalg.norm(np.array(new_path[0]) - current_pos) > 1.0:
            new_path.insert(0, tuple(current_pos))

        remaining_dist = self.config.cruise_speed_mps * dt

        while remaining_dist > 1e-6 and len(new_path) >= 2:
            p_curr = np.array(state.position_xyz, dtype=float)
            new_path[0] = tuple(p_curr)
            p_next = np.array(new_path[1], dtype=float)
            full_seg_len = np.linalg.norm(p_next - p_curr)

            if full_seg_len <= 1e-9:
                new_path.pop(0)
                continue

            move_dist = min(remaining_dist, full_seg_len)
            ratio = move_dist / full_seg_len
            p1 = p_curr + ratio * (p_next - p_curr)

            w2d = self._get_execution_wind(p1[0], p1[1], p1[2], state.current_time_s, gust_xy)
            w3d = np.array([w2d[0], w2d[1], 0.0])

            seg_e, seg_t, seg_p = self.physics.estimate_segment_energy(p_curr, p1, w3d, self.config.cruise_speed_mps)

            if seg_p > self.config.max_power * self.config.rl_overload_power_ratio:
                state.is_mission_failed, state.failure_reason = True, "overload"
                return state, new_path

            p_crash, _ = self.true_estimator.get_risk(p1[0], p1[1], p1[2], self.config.cruise_speed_mps, state.current_time_s)
            if p_crash > self.config.rl_terminate_risk_threshold:
                state.is_mission_failed, state.failure_reason = True, "storm_risk_too_high"
                return state, new_path

            if not self.battery.can_consume(state.remaining_energy_j, seg_e):
                state.is_mission_failed, state.failure_reason = True, "battery_depleted"
                return state, new_path

            state.remaining_energy_j -= seg_e
            state.total_energy_used_j += seg_e
            state.current_time_s += seg_t
            state.position_xyz = tuple(p1)
            state.traveled_path_xyz.append(tuple(p1))

            remaining_dist -= move_dist

            if move_dist >= full_seg_len - 1e-6:
                new_path.pop(0)
            else:
                new_path[0] = tuple(p1)

        return state, new_path

    # ==============================================================
    # 模式 B: RL 柔性微观控制 (加入四大修复的严谨对齐版)
    # ==============================================================
    def _advance_master_step_rl(self, state: SimulationState, path_xyz: List[Point3D], dt: float) -> Tuple[SimulationState, List[Point3D]]:
        # 🌟 【修复】删掉 len(path_xyz) < 2 的限制！
        if not path_xyz: 
            state.current_time_s += dt
            return state, path_xyz

        new_path = list(path_xyz)
        remaining_t = dt

        while remaining_t > 1e-6:
            step_dt = min(remaining_t, self.config.rl_dt)
            gust_xy, gust_event, gust_started = self.disturbance.begin_step(state.current_time_s, step_dt)
            if gust_started:
                self._log_gust_event(gust_event)

            # 1. 向“视觉代理”同步当前状态
            self.rl_perception_env.current_pos = np.array(state.position_xyz)
            self.rl_perception_env.current_time = state.current_time_s
            self.rl_perception_env.energy_remaining = state.remaining_energy_j
            self.rl_perception_env.current_heading = self.master_heading
            self.rl_perception_env.global_astar_path = new_path
            self.rl_perception_env.goal_pos = np.array([new_path[-1][0], new_path[-1][1], new_path[-1][2]])
            self.rl_perception_env.current_wp_idx = 1 if len(new_path) > 1 else 0
            self.rl_perception_env.episode_nfz_list_km = self.config.nfz_list_km
            self.rl_perception_env.dt = step_dt

            # 2. 提取 Obs 并推理
            obs = self.rl_perception_env._get_obs()
            obs = self.disturbance.apply_obs_noise(obs)
            action, _ = self.rl_model.predict(obs, deterministic=True)
            action = np.clip(action, -1.0, 1.0)

            speed_mid = 0.5 * (self.config.rl_speed_min + self.config.rl_speed_max)
            speed_half = 0.5 * (self.config.rl_speed_max - self.config.rl_speed_min)

            delta_heading = float(action[0] * self.config.rl_heading_delta_max_deg)
            target_speed = float(np.clip(speed_mid + action[1] * speed_half, self.config.rl_speed_min, self.config.rl_speed_max))
            delta_agl = float(action[2] * self.config.rl_agl_delta_max_m)

            # 🌟【核心修复 1】必须先把 RL 预测的方向盘转角加上！
            desired_heading = (self.master_heading + delta_heading + 180.0) % 360.0 - 180.0

            # 4. 🌟 底层飞控干预层 (Autopilot APAS - 真·物理级包络线保护 + 极限求生模式)
            final_x, final_y, final_z = state.position_xyz[0], state.position_xyz[1], state.position_xyz[2]
            final_power = float("inf")
            action_is_fatal = True
            fatal_reason = "unknown"
            applied_heading_offset = 0.0

            # 🌟 【求生升级 1】探索方向扩充到 90 度！
            # 原定航向优先，若遇陡坡撞山则向两侧打方向盘，直到完全侧飞（贴着山脊飞）
            heading_candidates = [0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 90.0, -90.0]

            for h_off in heading_candidates:
                if not action_is_fatal: break
                
                test_rad = math.radians((desired_heading + h_off + 360.0) % 360.0)
                test_speed = target_speed

                # 🌟 【求生升级 2】允许突破 rl_speed_min！
                # 在生死关头，允许水平速度降到 1.0 m/s（相当于原地悬停拔高），以换取安全的爬升率
                emergency_min_speed = 1.0 
                
                while test_speed >= emergency_min_speed:
                    dx = test_speed * math.cos(test_rad) * step_dt
                    dy = test_speed * math.sin(test_rad) * step_dt
                    new_x, new_y = self.rl_perception_env._clamp_position(
                        state.position_xyz[0] + dx, state.position_xyz[1] + dy
                    )

                    # 测算前方地形高度
                    current_ground_alt = self.true_estimator.get_altitude(state.position_xyz[0], state.position_xyz[1])
                    min_agl = self.rl_perception_env.min_clearance_agl
                    max_agl = self.rl_perception_env.max_clearance_agl
                    current_agl = max(min_agl, state.position_xyz[2] - current_ground_alt)
                    desired_agl = float(np.clip(current_agl + delta_agl, min_agl, max_agl))
                    
                    terrain_alt_new = self.true_estimator.get_altitude(new_x, new_y)
                    theoretical_z = terrain_alt_new + desired_agl

                    # 计算物理上需要的真实垂直爬升率
                    vz_req = (theoretical_z - state.position_xyz[2]) / step_dt

                    # 拦截网 1：如果前方山太陡，要求爬升率超过无人机物理极限
                    if vz_req > 8.0:
                        fatal_reason = f"climb_rate_exceeded_{vz_req:.1f}m/s"
                        test_speed -= 2.0  # 猛踩刹车！降低水平推进速度，把爬升压力降下来
                        continue 

                    if vz_req < -12.0:
                        vz_req = -12.0 # 下降率截断，避免自由落体

                    actual_new_z = state.position_xyz[2] + vz_req * step_dt

                    # 拦截网 2：如果该航向前方直接切入禁飞区（因为现在高度判定安全了，主要是防NFZ和微小地形突出）
                    if self.true_estimator.map.is_collision(new_x, new_y, actual_new_z, nfz_list_km=self.config.nfz_list_km):
                        fatal_reason = "terrain_or_nfz_crash"
                        break # 撞山/禁飞区说明这条路彻底堵死，减速没用，直接 break 换下一个方向盘角度

                    # 拦截网 3：功率是否过载
                    actual_vx = (new_x - state.position_xyz[0]) / step_dt
                    actual_vy = (new_y - state.position_xyz[1]) / step_dt
                    ground_velocity = np.array([actual_vx, actual_vy, vz_req])
                    w2d = self._get_execution_wind(
                        new_x,
                        new_y,
                        actual_new_z,
                        state.current_time_s + step_dt,
                        gust_xy,
                    )
                    w3d = np.array([w2d[0], w2d[1], 0.0])

                    power = self.physics.estimate_power_from_vectors(ground_velocity, w3d)

                    if power <= self.config.max_power * self.config.rl_overload_power_ratio:
                        # 检查全部通过！安全放行！
                        final_x, final_y, final_z, final_power = new_x, new_y, actual_new_z, power
                        action_is_fatal = False
                        applied_heading_offset = h_off
                        break

                    # 如果还是过载（可能逆风极大或刚好压在线上），继续踩刹车减速！
                    fatal_reason = f"overload_{power:.0f}W"
                    test_speed -= 2.0

            # 🌟【核心修复 3】同步生效的绝对航向
            if not action_is_fatal:
                self.master_heading = (desired_heading + applied_heading_offset + 180.0) % 360.0 - 180.0

            if action_is_fatal:
                state.is_mission_failed = True
                state.failure_reason = fatal_reason
                return state, new_path

            v_ground = np.linalg.norm([
                (final_x - state.position_xyz[0]) / step_dt,
                (final_y - state.position_xyz[1]) / step_dt,
                (final_z - state.position_xyz[2]) / step_dt
            ])
            p_crash, _ = self.true_estimator.get_risk(
                final_x, final_y, final_z, max(v_ground, 1.0), state.current_time_s + step_dt
            )
            if p_crash > self.config.rl_terminate_risk_threshold:
                state.is_mission_failed = True
                state.failure_reason = "storm_risk_too_high"
                return state, new_path

            seg_e = final_power * step_dt
            if not self.battery.can_consume(state.remaining_energy_j, seg_e):
                state.is_mission_failed = True
                state.failure_reason = "battery_depleted"
                return state, new_path

            # 6. 更新宏观状态
            state.remaining_energy_j -= seg_e
            state.total_energy_used_j += seg_e
            state.current_time_s += step_dt
            state.position_xyz = (final_x, final_y, final_z)
            state.traveled_path_xyz.append(state.position_xyz)

            # 🌟【核心修复 4】绝不可污染 A* 路径节点！采用单纯的丢弃逻辑防绕圈
            p_curr_xy = np.array([final_x, final_y])
            while len(new_path) >= 2:
                wp0 = np.array(new_path[0][:2])
                dist0 = np.linalg.norm(wp0 - p_curr_xy)
                
                # 碰到就吃掉
                if dist0 < self.config.rl_waypoint_refresh_radius_m:
                    new_path.pop(0)
                    continue
                
                # 防绕圈机制：如果离第二个点比离第一个点更近，说明已经飞过头了，绝不回头！
                wp1 = np.array(new_path[1][:2])
                dist1 = np.linalg.norm(wp1 - p_curr_xy)
                if dist1 < dist0:
                    new_path.pop(0)
                    continue
                    
                break

            remaining_t -= step_dt

        return state, new_path
    # ==============================================================
    # FSM 辅助函数 (保持不变)
    # ==============================================================
    def _move_scout(self, scout_pos: np.ndarray, master_xy: np.ndarray, path: List[Point3D], dt: float) -> np.ndarray:
        if self.scout_state == "RETREAT":
            target_pt = master_xy  
        else:
            target_pt = np.array(path[-1][:2]) 
            accumulated = 0.0
            
            # 🌟【核心修复 5】解耦探路蜂与主机的横向绑定！
            # 直接沿着纯净的 A* 原生航点测量，不再拼接 master_xy，防止主机漂移导致标尺剧烈拉伸形变！
            measure_path_xy = [np.array(p[:2]) for p in path]

            for i in range(len(measure_path_xy) - 1):
                p0, p1 = measure_path_xy[i], measure_path_xy[i+1]
                dist = np.linalg.norm(p1 - p0)
                if accumulated + dist >= self.scout_patrol_target_m:
                    ratio = (self.scout_patrol_target_m - accumulated) / (dist + 1e-6)
                    target_pt = p0 + ratio * (p1 - p0)
                    break
                accumulated += dist
        
        move_vec = target_pt - scout_pos
        dist_to_target = np.linalg.norm(move_vec)
        if dist_to_target < 1.0: return scout_pos
        direction = move_vec / dist_to_target
        step_dist = min(self.scout_speed_mps * dt, dist_to_target)
        next_pos = scout_pos + direction * step_dist
        return self._project_vehicle_to_safe_xy(next_pos, fallback_xy=scout_pos)

    def _scout_radar_scan(self, scout_pos: np.ndarray, current_t: float) -> List[int]:
        new_finds = []
        if not hasattr(self.true_estimator.wind, 'storm_manager'): return new_finds
        true_storms = self.true_estimator.wind.storm_manager.get_active_storms(current_t)
        for storm in true_storms:
            storm_id = id(storm)
            if storm_id in self.master_known_storms or storm_id in self.shared_buffered_storms:
                continue
            dist_to_storm = np.linalg.norm(scout_pos - storm.center_at(current_t))
            if dist_to_storm - storm.radius_m <= self.scout_radar_radius_m:
                new_finds.append(storm_id)
        return new_finds

    def _support_radar_scan(self, support_pos: np.ndarray, current_t: float) -> List[int]:
        new_finds = []
        if not hasattr(self.true_estimator.wind, 'storm_manager'):
            return new_finds
        true_storms = self.true_estimator.wind.storm_manager.get_active_storms(current_t)
        for storm in true_storms:
            storm_id = id(storm)
            if storm_id in self.master_known_storms or storm_id in self.shared_buffered_storms:
                continue
            dist_to_storm = np.linalg.norm(support_pos - storm.center_at(current_t))
            if dist_to_storm - storm.radius_m <= self.support_radar_radius_m:
                new_finds.append(storm_id)
        return new_finds

    def _move_vehicle_towards(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        speed_mps: float,
        dt: float,
    ) -> np.ndarray:
        safe_target = self._project_vehicle_to_safe_xy(target_pos, fallback_xy=current_pos)
        move_vec = safe_target - current_pos
        dist_to_target = np.linalg.norm(move_vec)
        if dist_to_target < 1.0:
            return current_pos
        direction = move_vec / dist_to_target
        step_dist = min(speed_mps * dt, dist_to_target)
        next_pos = current_pos + direction * step_dist
        return self._project_vehicle_to_safe_xy(next_pos, fallback_xy=current_pos)

    def _get_formation_safe_z(self, x: float, y: float, ref_z: float) -> float:
        terrain_z = self.true_estimator.get_altitude(x, y)
        min_safe_z = terrain_z + self.config.takeoff_altitude_agl
        return max(ref_z, min_safe_z)

    def _compute_relay_target(
        self,
        master_xy: np.ndarray,
        scout_xy: np.ndarray,
        current_t: float,
    ) -> np.ndarray:
        ideal_midpoint = self._project_vehicle_to_safe_xy((master_xy + scout_xy) / 2.0, fallback_xy=master_xy)
        vec_ms = scout_xy - master_xy
        candidate_targets = [ideal_midpoint]

        if np.linalg.norm(vec_ms) > 1e-6:
            perp_vec = np.array([-vec_ms[1], vec_ms[0]], dtype=float)
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            candidate_targets.append(
                self._project_vehicle_to_safe_xy(ideal_midpoint + perp_vec * self.relay_lateral_offset_m, fallback_xy=master_xy)
            )
            candidate_targets.append(
                self._project_vehicle_to_safe_xy(ideal_midpoint - perp_vec * self.relay_lateral_offset_m, fallback_xy=master_xy)
            )

        safest_target = ideal_midpoint
        safest_risk = float("inf")
        for candidate in candidate_targets:
            candidate_z = self.true_estimator.get_altitude(candidate[0], candidate[1]) + self.config.takeoff_altitude_agl
            p_crash, _ = self.true_estimator.get_risk(candidate[0], candidate[1], candidate_z, 10.0, current_t)
            if p_crash < safest_risk:
                safest_risk = p_crash
                safest_target = candidate
            if p_crash <= self.safe_risk_threshold:
                return candidate
        return safest_target

    def _compute_support_target(
        self,
        master_xy: np.ndarray,
        relay_xy: Optional[np.ndarray],
        support_xy: np.ndarray,
        current_path: List[Point3D],
        current_t: float,
    ) -> np.ndarray:
        support_z = self.true_estimator.get_altitude(support_xy[0], support_xy[1]) + self.config.takeoff_altitude_agl
        p_crash_sup, _ = self.true_estimator.get_risk(
            support_xy[0], support_xy[1], support_z, 10.0, current_t
        )
        if p_crash_sup > self.safe_risk_threshold:
            self.support_mode = "ESCAPE"
            escape_vec = support_xy - master_xy
            if np.linalg.norm(escape_vec) < 1e-6:
                escape_vec = np.array([1.0, 0.0], dtype=float)
            escape_vec = escape_vec / np.linalg.norm(escape_vec)
            return self._project_vehicle_to_safe_xy(
                support_xy + escape_vec * 0.5 * self.comm_range_m,
                fallback_xy=support_xy,
            )

        if self.support_local_threat_xy is not None and self.config.enable_support_shield_mode:
            self.support_mode = "SHIELD"
            vec_m_to_threat = self.support_local_threat_xy - master_xy
            if np.linalg.norm(vec_m_to_threat) > 1e-6:
                vec_m_to_threat = vec_m_to_threat / np.linalg.norm(vec_m_to_threat)
                shield_pos = master_xy + vec_m_to_threat * self.config.support_shield_offset_m
                return self._project_vehicle_to_safe_xy(shield_pos, fallback_xy=support_xy)

        dist_m_r = np.linalg.norm(master_xy - relay_xy) if relay_xy is not None else 0.0
        if relay_xy is not None and dist_m_r > self.comm_range_m:
            self.support_mode = "BRIDGE"
            return self._project_vehicle_to_safe_xy(
                master_xy + (relay_xy - master_xy) * 0.5,
                fallback_xy=support_xy,
            )

        self.support_mode = "FORMATION"
        forward_vec = self._get_path_forward_vector(master_xy, current_path)
        right_flank = np.array([forward_vec[1], -forward_vec[0]], dtype=float)
        target = (
            master_xy
            - forward_vec * self.support_trail_offset_m
            + right_flank * self.support_flank_offset_m
        )
        return self._project_vehicle_to_safe_xy(target, fallback_xy=support_xy)

    def _get_path_forward_vector(self, master_xy: np.ndarray, current_path: List[Point3D]) -> np.ndarray:
        for point in current_path:
            move_vec = np.array(point[:2], dtype=float) - master_xy
            if np.linalg.norm(move_vec) > 1.0:
                return move_vec / np.linalg.norm(move_vec)
        return np.array([1.0, 0.0], dtype=float)

    def _detect_support_shield_threat(
        self,
        master_xy: np.ndarray,
        support_xy: np.ndarray,
        current_t: float,
    ) -> Optional[np.ndarray]:
        if not self.config.enable_support_shield_mode:
            return None
        if not hasattr(self.true_estimator.wind, "storm_manager"):
            return None

        candidate_threat = None
        candidate_dist = float("inf")
        storms = self.true_estimator.wind.storm_manager.get_active_storms(current_t)
        for storm in storms:
            storm_center = storm.center_at(current_t)
            dist_to_support = np.linalg.norm(support_xy - storm_center)
            dist_to_master = np.linalg.norm(master_xy - storm_center)
            if dist_to_support <= self.support_radar_radius_m and dist_to_master <= self.config.support_shield_master_radius_m:
                if dist_to_master < candidate_dist:
                    candidate_dist = dist_to_master
                    candidate_threat = storm_center
        return candidate_threat

    def _evaluate_link_status(
        self,
        master_xy: np.ndarray,
        scout_xy: np.ndarray,
        relay_xy: Optional[np.ndarray] = None,
        support_xy: Optional[np.ndarray] = None,
    ) -> Dict[str, bool]:
        if self.config.swarm_topology_mode == "no_relay":
            relay_xy = None
        status = {
            "m_s": False,
            "m_r": False,
            "r_s": False,
            "m_sup": False,
            "sup_r": False,
            "path_direct": False,
            "path_relay": False,
            "path_support": False,
            "network_active": False,
        }

        status["m_s"] = np.linalg.norm(master_xy - scout_xy) <= self.comm_range_m
        if relay_xy is not None:
            status["m_r"] = np.linalg.norm(master_xy - relay_xy) <= self.comm_range_m
            status["r_s"] = np.linalg.norm(relay_xy - scout_xy) <= self.comm_range_m
        if support_xy is not None:
            status["m_sup"] = np.linalg.norm(master_xy - support_xy) <= self.comm_range_m
            if relay_xy is not None:
                status["sup_r"] = np.linalg.norm(support_xy - relay_xy) <= self.comm_range_m

        status["path_direct"] = status["m_s"]
        status["path_relay"] = status["m_r"] and status["r_s"]
        status["path_support"] = status["m_sup"] and status["sup_r"] and status["r_s"]
        status["network_active"] = (
            status["path_direct"] or status["path_relay"] or status["path_support"]
        )
        return status

    def _route_name_from_snapshot(self, link_snapshot: Dict[str, bool]) -> str:
        if link_snapshot.get("path_direct", False):
            return "direct"
        if link_snapshot.get("path_relay", False):
            return "relay"
        if link_snapshot.get("path_support", False):
            return "support"
        return "unknown"

    def _compute_nearest_threat_distance(self, master_xy: np.ndarray, current_t: float) -> float:
        if not hasattr(self.true_estimator.wind, "storm_manager"):
            return self.config.warning_distance_default_m
        storms = self.true_estimator.wind.storm_manager.get_active_storms(current_t)
        if not storms:
            return self.config.warning_distance_default_m
        nearest = float("inf")
        for storm in storms:
            center_xy = storm.center_at(current_t)
            dist = np.linalg.norm(master_xy - center_xy) - getattr(storm, "radius_m", 0.0)
            nearest = min(nearest, dist)
        return float(nearest if nearest < float("inf") else self.config.warning_distance_default_m)

    def _estimate_master_step_telemetry(
        self,
        previous_xyz: np.ndarray,
        current_xyz: np.ndarray,
        previous_time_s: float,
        current_time_s: float,
        previous_energy_j: float,
        current_energy_j: float,
    ) -> Tuple[float, float]:
        dt_s = max(current_time_s - previous_time_s, 1e-6)
        avg_power_w = max(0.0, (current_energy_j - previous_energy_j) / dt_s)
        v_ground = max(np.linalg.norm(current_xyz - previous_xyz) / dt_s, 1.0)
        p_crash, _ = self.true_estimator.get_risk(
            current_xyz[0], current_xyz[1], current_xyz[2], v_ground, current_time_s
        )
        return float(avg_power_w), float(p_crash)

    def _clamp_xy_to_bounds(self, pos_xy: np.ndarray) -> np.ndarray:
        min_x, max_x, min_y, max_y = self.true_estimator.get_bounds()
        return np.array(
            [
                np.clip(pos_xy[0], min_x, max_x),
                np.clip(pos_xy[1], min_y, max_y),
            ],
            dtype=float,
        )

    def _is_vehicle_xy_safe(self, pos_xy: np.ndarray) -> bool:
        pos_xy = self._clamp_xy_to_bounds(pos_xy)
        test_z = self.true_estimator.get_altitude(pos_xy[0], pos_xy[1]) + self.config.takeoff_altitude_agl
        return not self.true_estimator.map.is_collision(
            pos_xy[0],
            pos_xy[1],
            test_z,
            nfz_list_km=self.config.nfz_list_km,
            inflation_m=self.config.swarm_nfz_inflation_m,
        )

    def _project_vehicle_to_safe_xy(self, candidate_xy: np.ndarray, fallback_xy: np.ndarray) -> np.ndarray:
        candidate_xy = self._clamp_xy_to_bounds(np.asarray(candidate_xy, dtype=float))
        fallback_xy = self._clamp_xy_to_bounds(np.asarray(fallback_xy, dtype=float))
        if self._is_vehicle_xy_safe(candidate_xy):
            return candidate_xy
        if self._is_vehicle_xy_safe(fallback_xy):
            base_xy = fallback_xy
        else:
            base_xy = candidate_xy

        search_radii = [80.0, 160.0, 260.0, 400.0, 600.0]
        search_angles = np.linspace(0.0, 2.0 * np.pi, num=16, endpoint=False)
        best_xy = fallback_xy
        best_score = float("inf")
        for radius_m in search_radii:
            for angle_rad in search_angles:
                offset = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float) * radius_m
                probe_xy = self._clamp_xy_to_bounds(base_xy + offset)
                if not self._is_vehicle_xy_safe(probe_xy):
                    continue
                score = np.linalg.norm(probe_xy - candidate_xy)
                if score < best_score:
                    best_score = score
                    best_xy = probe_xy
            if best_score < float("inf"):
                return best_xy
        return fallback_xy if self._is_vehicle_xy_safe(fallback_xy) else candidate_xy

    def _get_execution_wind(
        self,
        x: float,
        y: float,
        z: float,
        t_s: float,
        gust_xy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        wind_xy = self.true_estimator.get_wind(x, y, z, t_s)
        if gust_xy is None:
            return wind_xy
        return wind_xy + gust_xy

    def _is_gust_active(self, current_time_s: float) -> bool:
        return (
            self.disturbance.active_event is not None
            and current_time_s < self.disturbance.active_event.end_time_s
        )

    def _log_gust_event(self, gust_event: Optional[GustEvent]) -> None:
        if gust_event is None:
            return
        gust_speed = np.linalg.norm(gust_event.vector_xy)
        gust_heading_deg = math.degrees(math.atan2(gust_event.vector_xy[1], gust_event.vector_xy[0]))
        print(
            f"  🌪️ [T={gust_event.start_time_s:.0f}s] 突发阵风注入！"
            f" 持续 {gust_event.end_time_s - gust_event.start_time_s:.0f}s"
            f" | 强度 {gust_speed:.1f} m/s | 航向 {gust_heading_deg:.0f}°"
        )

    def _sync_belief_world(self, t_s: float):
        if not hasattr(self.belief_estimator.wind, 'storm_manager'): return
        belief_storm_mgr = self.belief_estimator.wind.storm_manager
        true_storm_mgr = self.true_estimator.wind.storm_manager
        visible_storms = [s for s in true_storm_mgr.get_active_storms(t_s) if id(s) in self.master_known_storms]
        belief_storm_mgr.get_active_storms = lambda t: [s for s in visible_storms if s.is_active(t)]

    def _plan_in_belief_world(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float], t_s: float):
        return self.planner.search(start_xy, goal_xy, start_time_s=t_s)

    def _check_goal_reached(self, current_xyz: Point3D, goal_xy: Tuple[float, float]) -> bool:
        goal_z = self.true_estimator.get_altitude(goal_xy[0], goal_xy[1]) + self.config.takeoff_altitude_agl
        dist = math.sqrt((current_xyz[0]-goal_xy[0])**2 + (current_xyz[1]-goal_xy[1])**2 + (current_xyz[2]-goal_z)**2)
        return dist <= self.config.goal_tolerance_3d_m
# --- END OF FILE simulation/swarm_mission_executor.py ---
