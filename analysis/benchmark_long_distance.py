# --- START OF FILE analysis/benchmark_long_distance.py ---
import sys, os, copy, json, logging, math
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import SimulationConfig
from environment.map_manager import MapManager
from environment.wind_models import WindModelFactory
from core.physics import PhysicsEngine
from core.estimator import StateEstimator
from core.planner import AStarPlanner
from core.battery_manager import BatteryManager
from simulation.mission_executor import MissionExecutor
from rl_env.drone_env import GuidedDroneEnv
from adapters.rl_adapter import run_env_episode_to_mission_result
from analysis.benchmark_fixed_suite import TeacherPolicy

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

LOG_FORMAT = '[%(asctime)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ==========================================
# 实验配置区
# ==========================================
DISTANCE_BANDS = [6000, 8000, 10000, 12000]  
TOLERANCE = 200                             
TASKS_PER_BAND = 20                         
METHODS = ['astar', 'teacher', 'rl']

STAGE_CONFIG = 3                            
RL_MODEL_PATH = "models/ppo_drone_stage3_obs31_run1_best/best_model.zip"
OUTPUT_DIR = Path('results/long_distance_eval')

# 🌟 修复 2：动态任务文件名，防止改了配置还在读老文件
bands_str = f"b{len(DISTANCE_BANDS)}_max{max(DISTANCE_BANDS)}"
TASK_FILE = OUTPUT_DIR / f'tasks_stage{STAGE_CONFIG}_{bands_str}_n{TASKS_PER_BAND}.json'

MAX_WORKERS = min(8, max(1, os.cpu_count() - 2))

_GLOBAL_RL_MODEL = None

def get_rl_model():
    global _GLOBAL_RL_MODEL
    if _GLOBAL_RL_MODEL is None:
        _GLOBAL_RL_MODEL = PPO.load(RL_MODEL_PATH, device='cpu')
    return _GLOBAL_RL_MODEL

def generate_and_save_tasks():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if TASK_FILE.exists():
        logger.info(f"✅ 发现匹配当前配置的任务集: {TASK_FILE.name}，直接加载！")
        with open(TASK_FILE, 'r') as f:
            return json.load(f)

    logger.info(f"⚙️ 开始生成全新长距离任务集: {TASK_FILE.name} (请稍候)...")
    config = SimulationConfig()
    config.curriculum_stage = STAGE_CONFIG
    env = GuidedDroneEnv(config)
    min_x, max_x, min_y, max_y = env.estimator.get_bounds()
    
    MARGIN = 1000.0  
    
    tasks = []
    global_task_id = 0
    rng = np.random.default_rng(2024) 
    
    for target_dist in DISTANCE_BANDS:
        logger.info(f"  -> 正在生成 {target_dist/1000:.0f} km 档位的任务...")
        count = 0
        attempts = 0
        while count < TASKS_PER_BAND:
            attempts += 1
            if attempts > 5000:
                logger.error(f"❌ 警告：在 {target_dist}m 档位极难找到合法坐标！请检查地图大小。")
                break
                
            sx = float(rng.uniform(min_x + MARGIN, max_x - MARGIN))
            sy = float(rng.uniform(min_y + MARGIN, max_y - MARGIN))
            if not env._is_safe_location(sx, sy, 0.0): continue
            
            angle = float(rng.uniform(0, 2 * math.pi))
            raw_dist = target_dist + float(rng.uniform(-TOLERANCE, TOLERANCE))
            
            gx = float(np.clip(sx + raw_dist * math.cos(angle), min_x + MARGIN, max_x - MARGIN))
            gy = float(np.clip(sy + raw_dist * math.sin(angle), min_y + MARGIN, max_y - MARGIN))
            
            actual_dist = math.hypot(gx - sx, gy - sy)
            if abs(actual_dist - target_dist) > TOLERANCE:
                continue
                
            if not env._is_safe_location(gx, gy, 0.0): continue
            
            wind_seed = int(rng.integers(10000, 99999))
            
            task = {
                "task_id": global_task_id,
                "distance_band": target_dist,
                "start_xy": (sx, sy),
                "goal_xy": (gx, gy),
                "wind_seed": wind_seed
            }
            tasks.append(task)
            count += 1
            global_task_id += 1

    with open(TASK_FILE, 'w') as f:
        json.dump(tasks, f, indent=4)
    logger.info(f"✅ 成功生成并保存 {len(tasks)} 个任务到 {TASK_FILE}！")
    return tasks


def build_fixed_task_config(task: dict) -> SimulationConfig:
    config = SimulationConfig()
    config.curriculum_stage = STAGE_CONFIG
    env = GuidedDroneEnv(config)
    env_options = {"start_xy": tuple(task["start_xy"]), "goal_xy": tuple(task["goal_xy"])}
    
    env.reset(seed=task["wind_seed"], options=env_options)
    
    fixed_config = copy.deepcopy(env.config)
    fixed_config.wind_seed = task["wind_seed"]
    return fixed_config

def worker_evaluate_single_run(args) -> Dict:
    task, method = args
    tid = task["task_id"]
    band = task["distance_band"]
    s_xy = tuple(task["start_xy"])
    g_xy = tuple(task["goal_xy"])
    w_seed = task["wind_seed"]
    
    metrics = {
        'task_id': tid, 'distance_band': band, 'method': method, 
        'success': False, 'time_s': 0.0, 'energy_j': 0.0, 'path_length_m': 0.0, 'failure_reason': 'unknown'
    }
    
    try:
        fixed_config = build_fixed_task_config(task)
        
        if method == 'astar':
            map_manager = MapManager(fixed_config)
            wind_model = WindModelFactory.create(fixed_config.wind_model_type, fixed_config, bounds=map_manager.get_bounds())
            estimator = StateEstimator(map_manager, wind_model, fixed_config)
            physics = PhysicsEngine(fixed_config)
            battery = BatteryManager(fixed_config)
            planner = AStarPlanner(fixed_config, estimator, physics)
            executor = MissionExecutor(fixed_config, estimator, physics, battery, planner)
            
            res = executor.execute_mission(s_xy, g_xy)
            path_xyz = res.actual_flown_path_xyz
            
        else:
            env = GuidedDroneEnv(fixed_config)
            env_options = {"start_xy": s_xy, "goal_xy": g_xy}
            
            if method == 'teacher':
                policy = TeacherPolicy()
            else:
                policy = get_rl_model()
                
            res, _ = run_env_episode_to_mission_result(env, policy, method, w_seed, options=env_options)
            path_xyz = res.actual_flown_path_xyz

        def calc_len(path):
            if not path or len(path) < 2: return 0.0
            return float(np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1)))
            
        metrics.update({
            'success': res.success,
            'time_s': res.total_mission_time_s,
            'energy_j': res.total_energy_used_j,
            'path_length_m': calc_len(path_xyz),
            'failure_reason': res.failure_reason or 'goal_reached'
        })
    except Exception as e:
        metrics['failure_reason'] = f"Error: {str(e)}"

    return metrics


def run_long_distance_benchmark():
    tasks = generate_and_save_tasks()
    
    run_args = []
    for task in tasks:
        for method in METHODS:
            run_args.append((task, method))
            
    logger.info(f"🚀 开始执行多进程测评，共 {len(run_args)} 个跑测任务，启用 {MAX_WORKERS} 个核心...")
    
    all_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker_evaluate_single_run, arg): arg for arg in run_args}
        
        for future in tqdm(as_completed(futures), total=len(run_args), desc="Evaluating"):
            res = future.result()
            all_results.append(res)
            
    df = pd.DataFrame(all_results)
    raw_csv = OUTPUT_DIR / f'results_raw_stage{STAGE_CONFIG}_{bands_str}_n{TASKS_PER_BAND}.csv'
    df.to_csv(raw_csv, index=False)
    
    # ==========================
    # 统计与生成折线趋势图
    # ==========================
    logger.info("📊 正在生成长距离趋势图表...")
    
    summary = []
    for band in DISTANCE_BANDS:
        for method in METHODS:
            subset = df[(df['distance_band'] == band) & (df['method'] == method)]
            if len(subset) == 0: continue
            
            successes = subset[subset['success'] == True]
            sr = len(successes) / len(subset) * 100
            
            # 🌟 修复 3：补全所有的核心统计指标
            avg_time = successes['time_s'].mean() if len(successes) > 0 else 0
            avg_len_m = successes['path_length_m'].mean() if len(successes) > 0 else 0
            avg_eng_j = successes['energy_j'].mean() if len(successes) > 0 else 0
            
            avg_eng_kj = avg_eng_j / 1000.0
            kj_per_km = avg_eng_kj / (avg_len_m / 1000.0) if avg_len_m > 0 else 0
            
            failures = subset[subset['success'] == False]
            fail_counts = failures['failure_reason'].value_counts().to_dict()
            fail_str = " | ".join([f"{k}:{v}" for k, v in fail_counts.items()]) if fail_counts else "None"
            
            summary.append({
                'Distance Band (km)': band / 1000.0,
                'Method': method,
                'Success Rate (%)': sr,
                'Avg Time (s)': avg_time,             # 🌟 新增
                'Avg Path (m)': avg_len_m,            # 🌟 新增
                'Avg Energy (kJ)': avg_eng_kj,        # 🌟 新增
                'Energy/km (kJ/km)': kj_per_km,
                'Failure Breakdown': fail_str
            })
            
    df_summary = pd.DataFrame(summary)
    sum_csv = OUTPUT_DIR / f'summary_stage{STAGE_CONFIG}_{bands_str}_n{TASKS_PER_BAND}.csv'
    df_summary.to_csv(sum_csv, index=False)
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'astar': '#FF6B6B', 'teacher': '#4ECDC4', 'rl': '#45B7D1'}
    markers = {'astar': 's-', 'teacher': '^-', 'rl': 'o-'}
    
    for method in METHODS:
        method_data = df_summary[df_summary['Method'] == method].sort_values('Distance Band (km)')
        if len(method_data) == 0: continue
        
        x = method_data['Distance Band (km)']
        ax1.plot(x, method_data['Success Rate (%)'], markers[method], color=colors[method], label=method.upper(), linewidth=2.5, markersize=8)
        ax2.plot(x, method_data['Energy/km (kJ/km)'], markers[method], color=colors[method], label=method.upper(), linewidth=2.5, markersize=8)

    ax1.set_title("Success Rate vs. Mission Distance", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Target Distance (km)", fontsize=12)
    ax1.set_ylabel("Success Rate (%)", fontsize=12)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    ax2.set_title("Energy Efficiency (Success Cases) vs. Distance", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Target Distance (km)", fontsize=12)
    ax2.set_ylabel("Energy per km (kJ/km)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plot_file = OUTPUT_DIR / f'trends_stage{STAGE_CONFIG}_{bands_str}_n{TASKS_PER_BAND}.png'
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)
    logger.info(f"✅ 长距离跑测全部完成！报告与趋势图已保存至 {OUTPUT_DIR}")

if __name__ == '__main__':
    run_long_distance_benchmark()