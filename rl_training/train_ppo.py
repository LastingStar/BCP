"""
PPO算法训练脚本

使用Stable Baselines3的PPO算法训练无人机路径规划策略。
支持多进程训练、模型保存、评估和可视化。
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
import matplotlib.pyplot as plt

from rl_env.drone_env import DronePathPlanningEnv
from configs.config import SimulationConfig


def create_envs(config: SimulationConfig, n_envs: int = 4):
    """创建并行环境"""
    def make_env():
        env = DronePathPlanningEnv(config)
        env = Monitor(env)
        return env

    env = make_vec_env(make_env, n_envs=n_envs)
    return env


def train_ppo(
    config: SimulationConfig,
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    model_save_path: str = "models/ppo_drone",
    log_dir: str = "logs/"
):
    """
    训练PPO模型

    参数:
    - total_timesteps: 总训练步数
    - n_envs: 并行环境数量
    - model_save_path: 模型保存路径
    - log_dir: 日志目录
    """

    # 创建环境
    env = create_envs(config, n_envs)
    eval_env = DronePathPlanningEnv(config)
    eval_env = Monitor(eval_env)

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # PPO模型配置
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,  # 每次更新的轨迹长度
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # 折扣因子
        gae_lambda=0.95,
        clip_range=0.2,  # PPO裁剪参数
        ent_coef=0.01,  # 熵系数，鼓励探索
        vf_coef=0.5,  # 价值函数系数
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        verbose=1,
        device="auto"
    )

    # 回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path + "_best",
        log_path=log_dir,
        eval_freq=10000,  # 每10000步评估一次
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # 每50000步保存一次
        save_path=model_save_path + "_checkpoints",
        name_prefix="ppo_drone"
    )

    # 开始训练
    print("🚀 开始PPO训练...")
    print(f"总训练步数: {total_timesteps}")
    print(f"并行环境数: {n_envs}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # 保存最终模型
    model.save(model_save_path + "_final")
    print(f"✅ 模型已保存到: {model_save_path}_final")

    return model


def evaluate_model(
    model_path: str,
    config: SimulationConfig,
    n_episodes: int = 10
):
    """评估训练好的模型"""
    print(f"📊 评估模型: {model_path}")

    # 加载模型
    model = PPO.load(model_path)

    # 创建评估环境
    env = DronePathPlanningEnv(config)

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if info.get('success', False):
            success_count += 1

        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Length={episode_length}, Success={info.get('success', False)}")

    print(f"
📈 评估结果:"    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"成功率: {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes
    }


def plot_training_results(log_dir: str):
    """绘制训练结果"""
    plot_results([log_dir], num_timesteps=None, results_plotter.X_TIMESTEPS, "PPO Drone Training")
    plt.show()


if __name__ == "__main__":
    # 配置
    config = SimulationConfig()

    # 训练参数
    TRAIN_CONFIG = {
        'total_timesteps': 500_000,  # 可以根据需要调整
        'n_envs': 4,
        'model_save_path': 'models/ppo_drone',
        'log_dir': 'logs/'
    }

    # 训练模型
    model = train_ppo(config, **TRAIN_CONFIG)

    # 评估模型
    eval_results = evaluate_model(
        TRAIN_CONFIG['model_save_path'] + '_final',
        config,
        n_episodes=20
    )

    # 可视化训练结果
    # plot_training_results(TRAIN_CONFIG['log_dir'])</content>
    <parameter name="filePath">c:\Users\20340\Desktop\project1\rl_training\train_ppo.py