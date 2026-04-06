# --- START OF FILE adapters/rl_adapter.py ---
import numpy as np
from typing import Any, Dict, Tuple
import logging

from rl_env.drone_env import GuidedDroneEnv
from models.mission_models import SimulationState, MissionResult

logger = logging.getLogger(__name__)

def run_env_episode_to_mission_result(
    env: GuidedDroneEnv, 
    policy: Any, 
    method_kind: str, 
    seed: int,
    options: Dict[str, Any] = None  # 🌟 新增
) -> Tuple[MissionResult, Dict[str, Any]]:
    """
    桥接器：运行一次 Gym 环境 Episode，并将其转化为标准的 MissionResult。
    这样 RL 的结果就能完美复用现有的 Visualizer 和 MissionAnimator！
    
    参数:
        env: GuidedDroneEnv 实例
        policy: RL 策略（需要有 predict() 方法）
        method_kind: "rl" 或 "teacher"
        seed: 随机种子
    
    返回:
        MissionResult: 统一的任务结果对象
        Dict: 额外指标 (reward, ep_length 等)
    
    异常:
        TypeError: 如果 env 或 policy 类型不正确
        ValueError: 如果 method_kind 无效
    """
    # 类型检查 - 防守性编程
    if not isinstance(env, GuidedDroneEnv):
        raise TypeError(f"env must be GuidedDroneEnv, got {type(env).__name__}")
    
    if not hasattr(policy, 'predict'):
        raise TypeError(f"policy must have predict() method, got {type(policy).__name__}")
    
    if method_kind not in ("rl", "teacher"):
        raise ValueError(f"method_kind must be 'rl' or 'teacher', got '{method_kind}'")
    
    logger.debug(f"Starting episode: method={method_kind}, seed={seed}")
    
    # 环境重置
    try:
        obs, info = env.reset(seed=seed, options=options)
    except Exception as e:
        logger.error(f"Failed to reset environment with seed {seed}: {e}")
        raise
    
    # 初始化轨迹记录
    path_xyz = [tuple(env.current_pos)]
    total_reward = 0.0
    step_count = 0
    
    terminated = False
    truncated = False
    final_info = info

    # 运行 Episode
    while not (terminated or truncated):
        # 兼容 SB3 policy 和 自定义 TeacherPolicy 的入参差异
        try:
            if method_kind == "teacher":
                action = policy.predict(env)
            else:
                action, _ = policy.predict(obs, deterministic=True)
        except Exception as e:
            logger.error(f"Policy prediction failed at step {step_count}: {e}")
            raise
        
        try:
            obs, reward, terminated, truncated, final_info = env.step(action)
        except Exception as e:
            logger.error(f"Environment step failed at step {step_count}: {e}")
            raise
        
        path_xyz.append(tuple(env.current_pos))
        total_reward += float(reward)
        step_count += 1

    # ==========================================
    # 🌟 核心组装：将 RL 状态打包为 MissionResult
    # ==========================================
    # 我们把 Teacher 的 A* 路径作为 baseline 存入 initial_wind_path_xyz，
    # 这样可视化画对比图时，就能自动画出 A* 参考线和 RL 的实际飞行线！
    
    # 检查 global_astar_path 属性
    if not hasattr(env, 'global_astar_path'):
        logger.warning("env.global_astar_path not found, using empty baseline")
    
    teacher_path = getattr(env, 'global_astar_path', [])
    if not isinstance(teacher_path, list):
        logger.warning(f"global_astar_path is not a list, got {type(teacher_path).__name__}, converting...")
        teacher_path = list(teacher_path) if hasattr(teacher_path, '__iter__') else []
    
    # 验证 final_info 字典的完整性
    if not isinstance(final_info, dict):
        logger.warning(f"final_info is not dict, got {type(final_info).__name__}, using empty dict")
        final_info = {}
    
    # 检查关键字段
    if "is_success" not in final_info:
        logger.warning("final_info missing 'is_success' key, defaulting to False")
        final_info["is_success"] = False
    
    if "terminated_reason" not in final_info:
        logger.warning("final_info missing 'terminated_reason' key, defaulting to 'unknown'")
        final_info["terminated_reason"] = "unknown"
    
    final_state = SimulationState(
        current_time_s=env.current_time,
        position_xyz=tuple(env.current_pos),
        remaining_energy_j=env.energy_remaining,
        traveled_path_xyz=path_xyz,
        replans_count=0, # RL 是高频控制，没有传统意义的重规划
        total_energy_used_j=env.config.battery_capacity_j - env.energy_remaining,
        is_goal_reached=final_info.get("is_success", False),
        is_mission_failed=not final_info.get("is_success", False),
        failure_reason=final_info.get("terminated_reason", "unknown")
    )

    mission_result = MissionResult(
        success=final_info.get("is_success", False),
        final_state=final_state,
        initial_no_wind_path_xyz=[], # 留空
        initial_wind_path_xyz=teacher_path, # 🌟 骗过 Visualizer，让它画出 Teacher 参考线
        replanned_paths_xyz=[],
        actual_flown_path_xyz=path_xyz, # 🌟 RL 实际飞出的 3D 轨迹
        total_replans=0,
        total_mission_time_s=env.current_time,
        total_energy_used_j=env.config.battery_capacity_j - env.energy_remaining,
        failure_reason=final_info.get("terminated_reason", "unknown"),
        time_history_s=getattr(env, "telemetry_time_s", []),
        power_history_w=getattr(env, "telemetry_power_w", []),
        risk_history=getattr(env, "telemetry_risk", []),
    )
    
    # 额外收集用于统计的 RL 专属指标
    extra_metrics = {
        "reward": total_reward,
        "ep_length": step_count,
        "teacher_path_len": len(teacher_path),
        "max_p_crash": getattr(env, "telemetry_max_p_crash", 0.0),
        "terminated_reason": final_info.get("terminated_reason", "unknown"),
    }
    
    # 记录完成情况
    logger.info(f"✓ Episode completed: success={final_info.get('is_success', False)}, "
                f"steps={step_count}, reward={total_reward:.2f}, "
                f"teacher_path_len={len(teacher_path)}")

    return mission_result, extra_metrics
# --- END OF FILE adapters/rl_adapter.py ---
