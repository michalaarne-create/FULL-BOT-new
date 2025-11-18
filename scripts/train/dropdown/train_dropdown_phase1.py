"""
Train PPO on simulated dropdown (bounding-box only) with discrete actions.

Environment: DropdownBBoxEnv (no images)
Actions: 0=click, 1=scroll_down, 2=scroll_up
Goal: open -> scroll to bottom -> click (lower option = higher reward)
"""

import sys
from pathlib import Path

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

from envs.dropdown_env import DropdownBBoxEnv


def make_env(seed=42):
    def _init():
        env = DropdownBBoxEnv()
        env = Monitor(env)
        env.action_space.seed(seed)
        return env
    return _init


def train():
    save_root = Path('models/saved/phase2_bbox')
    (save_root / 'best_model').mkdir(parents=True, exist_ok=True)
    (save_root / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (save_root / 'eval_logs').mkdir(parents=True, exist_ok=True)

    # Single-env setup is enough for this simple simulation
    env = DummyVecEnv([make_env(42)])
    eval_env = DummyVecEnv([make_env(1337)])

    # MLP-only policy (no images)
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
        tensorboard_log=str(save_root / 'tensorboard'),
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=None),
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_root / 'best_model'),
        log_path=str(save_root / 'eval_logs'),
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(save_root / 'checkpoints'),
        name_prefix='ppo_dropdown_bbox',
    )

    total_timesteps = 300_000
    model.learn(total_timesteps=total_timesteps, callback=[eval_cb, ckpt_cb], progress_bar=True)

    model.save(str(save_root / 'final_model'))


if __name__ == '__main__':
    train()
