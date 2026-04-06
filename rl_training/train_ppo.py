import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import SimulationConfig
from rl_env.drone_env import GuidedDroneEnv


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{run_name}.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class SuccessFirstEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_path: str,
        eval_csv_path: str,
        eval_seed_base: int = 42,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_path = best_model_path
        self.eval_csv_path = eval_csv_path
        self.eval_seed_base = eval_seed_base
        self.best_success_rate = -1.0
        self.best_mean_length = float("inf")
        self.best_mean_reward = -float("inf")

        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(eval_csv_path), exist_ok=True)
        if not os.path.exists(self.eval_csv_path):
            with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timesteps",
                        "mean_reward",
                        "mean_ep_length",
                        "success_rate",
                        "best_success_rate",
                        "best_mean_length",
                        "best_mean_reward",
                    ]
                )

    def _evaluate_once(self):
        rewards = []
        lengths = []
        successes = []
        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset(seed=self.eval_seed_base + ep)
            terminated = False
            truncated = False
            ep_reward = 0.0
            ep_len = 0
            final_info = info
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, final_info = self.eval_env.step(action)
                ep_reward += float(reward)
                ep_len += 1
            rewards.append(ep_reward)
            lengths.append(ep_len)
            successes.append(1.0 if final_info.get("is_success", False) else 0.0)
        return float(np.mean(rewards)), float(np.mean(lengths)), float(np.mean(successes))

    def _append_eval_csv(self, mean_reward, mean_length, success_rate):
        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.num_timesteps,
                    mean_reward,
                    mean_length,
                    success_rate,
                    self.best_success_rate,
                    self.best_mean_length,
                    self.best_mean_reward,
                ]
            )

    def _is_better(self, success_rate, mean_length, mean_reward):
        if success_rate > self.best_success_rate:
            return True
        if success_rate == self.best_success_rate and mean_length < self.best_mean_length:
            return True
        if (
            success_rate == self.best_success_rate
            and mean_length == self.best_mean_length
            and mean_reward > self.best_mean_reward
        ):
            return True
        return False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            mean_reward, mean_length, success_rate = self._evaluate_once()
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/success_rate", success_rate)
            if self._is_better(success_rate, mean_length, mean_reward):
                self.best_success_rate = success_rate
                self.best_mean_length = mean_length
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_path)
            self._append_eval_csv(mean_reward, mean_length, success_rate)
        return True


def create_envs(config: SimulationConfig, n_envs: int = 1):
    def make_env():
        return Monitor(GuidedDroneEnv(config))

    return make_vec_env(make_env, n_envs=n_envs)


def build_new_model(env, log_dir: str, seed: int):
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.6,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        verbose=1,
        device="auto",
        seed=seed,
    )


def evaluate_model(model_path: str, config: SimulationConfig, n_episodes: int, eval_seed: int, raw_csv_path: Optional[str] = None):
    model = PPO.load(model_path, device="cpu")
    env = GuidedDroneEnv(config)
    raw_rows = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=eval_seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_length = 0
        final_info = info
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
        raw_rows.append(
            {
                "episode": episode,
                "seed": eval_seed + episode,
                "success": final_info.get("is_success", False),
                "terminated_reason": final_info.get("terminated_reason", "unknown"),
                "max_p_crash": getattr(env, "telemetry_max_p_crash", 0.0),
                "ep_length": episode_length,
                "total_reward": episode_reward,
            }
        )

    raw_df = pd.DataFrame(raw_rows)
    if raw_csv_path:
        Path(raw_csv_path).parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "mean_reward": float(raw_df["total_reward"].mean()) if not raw_df.empty else 0.0,
        "std_reward": float(raw_df["total_reward"].std()) if len(raw_df) > 1 else 0.0,
        "mean_length": float(raw_df["ep_length"].mean()) if not raw_df.empty else 0.0,
        "success_rate": float(raw_df["success"].mean()) if not raw_df.empty else 0.0,
        "storm_risk_fail_rate": float((raw_df["terminated_reason"] == "storm_risk_too_high").mean()) if not raw_df.empty else 0.0,
        "max_p_crash_mean": float(raw_df["max_p_crash"].mean()) if not raw_df.empty else 0.0,
        "max_p_crash_max": float(raw_df["max_p_crash"].max()) if not raw_df.empty else 0.0,
    }
    return summary, raw_df


def build_config_from_args(args) -> SimulationConfig:
    config = SimulationConfig()
    config.curriculum_stage = args.curriculum_stage
    config.wind_seed = args.seed
    config.obs_ablation_mode = args.obs_mode
    config.enable_single_agent_gusts = True
    config.collect_ablation_telemetry = True
    return config


def train_ppo(
    config: SimulationConfig,
    total_timesteps: int,
    n_envs: int,
    model_save_root: str,
    log_root: str,
    run_name: str,
    seed: int,
    load_model_path: Optional[str] = None,
    from_scratch: bool = False,
):
    env = create_envs(config, n_envs=n_envs)
    eval_env = Monitor(GuidedDroneEnv(config))

    model_root = Path(model_save_root)
    log_dir = Path(log_root)
    model_root.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if not from_scratch and load_model_path and os.path.exists(load_model_path + ".zip"):
        try:
            model = PPO.load(load_model_path, env=env, device="auto")
            new_lr = 5e-5
            model.learning_rate = new_lr
            model.lr_schedule = get_schedule_fn(new_lr)
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = new_lr
        except Exception:
            model = None

    if model is None:
        model = build_new_model(env, str(log_dir), seed=seed)

    eval_callback = SuccessFirstEvalCallback(
        eval_env=eval_env,
        eval_freq=20_000,
        n_eval_episodes=min(config.ablation_eval_episodes, 10),
        best_model_path=str(model_root) + "_best/best_model",
        eval_csv_path=str(model_root) + "_eval/eval_metrics.csv",
        eval_seed_base=seed,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=str(model_root) + "_checkpoints",
        name_prefix=run_name,
    )

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback], progress_bar=True)
    final_model_path = str(model_root) + "_final"
    model.save(final_model_path)
    return final_model_path, str(model_root) + "_best/best_model"


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO with ablation-ready CLI")
    parser.add_argument("--run-name", default="stage3_obs31_run1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-mode", choices=["full", "no_future", "no_radar"], default="full")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--curriculum-stage", type=int, default=3)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--model-save-root", default="models/ppo_drone")
    parser.add_argument("--log-root", default="logs/")
    parser.add_argument("--load-model-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    logger = setup_logger(args.log_root, args.run_name)
    logger.info("Starting PPO training run %s", args.run_name)

    config = build_config_from_args(args)
    final_model_path, best_model_path = train_ppo(
        config=config,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        model_save_root=args.model_save_root,
        log_root=args.log_root,
        run_name=args.run_name,
        seed=args.seed,
        load_model_path=args.load_model_path,
        from_scratch=args.from_scratch,
    )

    eval_raw_path = str(Path(args.model_save_root).parent / f"{args.run_name}_final_eval_raw.csv")
    summary, _ = evaluate_model(
        best_model_path,
        config,
        n_episodes=config.ablation_eval_episodes,
        eval_seed=args.seed,
        raw_csv_path=eval_raw_path,
    )
    summary_path = str(Path(args.model_save_root).parent / f"{args.run_name}_final_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Training finished. Final model: %s", final_model_path)
    logger.info("Best model: %s", best_model_path)
    logger.info("Final eval raw: %s", eval_raw_path)
    logger.info("Final eval summary: %s", summary_path)
