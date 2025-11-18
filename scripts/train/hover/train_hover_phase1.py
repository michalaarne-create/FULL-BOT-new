"""
PPO Phase 1 Training - Multi-line Hover Behavior V2.
"""

import gymnasium as gym
import sys
from pathlib import Path

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import torch
import json
import numpy as np
from datetime import datetime

# ✅ POPRAWIONY IMPORT:
from envs.hover_env import HoverEnvMultiLineV2


class CustomLoggingCallback(BaseCallback):
    """Custom callback do logowania dodatkowych metryk."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_dots = []
        self.episode_lines_completed = []
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
            
            if 'total_dots' in info:
                self.episode_dots.append(info['total_dots'])
            if 'lines_completed' in info:
                self.episode_lines_completed.append(info['lines_completed'])
        
        if self.n_calls % 1000 == 0 and len(self.episode_rewards) > 0:
            self.logger.record('custom/mean_dots', np.mean(self.episode_dots[-100:]) if self.episode_dots else 0)
            self.logger.record('custom/mean_lines_completed', np.mean(self.episode_lines_completed[-100:]) if self.episode_lines_completed else 0)
        
        return True


class ProgressCallback(BaseCallback):
    """Callback do pokazywania postępu treningu"""
    
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_print = 0
        
    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        
        if progress - self.last_print >= 0.05:
            self.last_print = progress
            
            ep_rew = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            ep_len = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            
            print(f"\n{'='*70}")
            print(f"Progress: {progress*100:.1f}% ({self.num_timesteps:,} / {self.total_timesteps:,} steps)")
            print(f"Mean Episode Reward: {ep_rew:+.1f}")
            print(f"Mean Episode Length: {ep_len:.1f}")
            print(f"{'='*70}\n")
        
        return True


def train_ppo_phase1(
    total_timesteps=500_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.15,  # ✅ Zwiększone exploration (było 0.01)
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_envs=128,
    seed=42,
    output_dir='models/saved/phase1',
    device='auto',
):
    """Trenuj PPO na multi-line hover behavior V2."""
    
    print("=" * 70)
    print("🚀 PPO PHASE 1 TRAINING - MULTI-LINE HOVER V2")
    print("=" * 70)
    print()

    vecnorm_saved = False

    def _maybe_save_vecnorm():
        nonlocal vecnorm_saved
        if vecnorm_saved:
            return
        if isinstance(env, VecNormalize):
            env.save(str(vecnorm_path))
            vecnorm_saved = True
            print(f"   VecNormalize stats saved to: {vecnorm_path}")

    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔧 Configuration:")
    print(f"   Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Parallel envs: {n_envs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Entropy coef: {ent_coef} (exploration)")
    print()
    
    print("🌍 Creating environments...")
    
    # ✅ Funkcja make_env z parametrami dopasowanymi do fazy 1:
    def make_env(*, randomize_layout: bool):
        def _init():
            env = HoverEnvMultiLineV2(
                lines_file='data/text_lines.json',
                line_randomization=randomize_layout,
                line_jitter_y=30.0 if randomize_layout else 0.0,
                line_jitter_x=50.0 if randomize_layout else 0.0,
                line_scale_jitter=0.1 if randomize_layout else 0.0,
                line_shuffle_prob=0.0,
            )
            return Monitor(env)
        return _init
    
    env = DummyVecEnv([make_env(randomize_layout=True) for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env(randomize_layout=False)])

    vecnorm_path = output_path / 'vecnormalize.pkl'
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = True
        env.norm_obs = False
        env.norm_reward = True

        eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        eval_env.training = False
        eval_env.norm_obs = False
        eval_env.norm_reward = False
        eval_env.ret_rms = env.ret_rms
        print(f"   Loaded VecNormalize stats from {vecnorm_path}")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=50.0)
        eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=True, clip_reward=50.0)
        eval_env.norm_reward = False
        eval_env.ret_rms = env.ret_rms
        print("   Initialised new VecNormalize statistics.")
    
    print(f"   Training envs: {n_envs}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print()
    
    # Reszta bez zmian...
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        ),
        activation_fn=torch.nn.ReLU,
    )
    
    print("🧠 Creating PPO model...")
    
    resume_candidates = [
        output_path / 'best_model' / 'best_model.zip',
        output_path / 'final_model.zip',
        output_path / 'interrupted_model.zip',
    ]
    resume_path = next((p for p in resume_candidates if p.exists()), None)
    
    model_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        device=device,
        tensorboard_log=str(output_path / 'tensorboard'),
        seed=seed,
        policy_kwargs=policy_kwargs,
    )
    
    if resume_path is not None:
        print(f"   Loading pretrained weights from {resume_path}")
        loaded_model = PPO.load(resume_path, env=env, device=device)
        model = PPO(**model_kwargs)
        model.policy.load_state_dict(loaded_model.policy.state_dict())
        del loaded_model
    else:
        model = PPO(**model_kwargs)
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"   Policy: MlpPolicy")
    print(f"   Architecture: {policy_kwargs['net_arch']}")
    print(f"   Total parameters: {total_params:,}")
    print()
    
    print("📊 Setting up callbacks...")
    
    eval_freq = 20_000 // n_envs
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / 'best_model'),
        log_path=str(output_path / 'eval_logs'),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    checkpoint_freq = 50_000 // n_envs
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(output_path / 'checkpoints'),
        name_prefix='ppo_hover_v2',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    custom_logging_callback = CustomLoggingCallback(verbose=1)
    progress_callback = ProgressCallback(total_timesteps=total_timesteps, verbose=1)
    
    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        custom_logging_callback,
        progress_callback,
    ])
    
    print(f"   ✓ Eval frequency: every {eval_freq * n_envs:,} steps")
    print(f"   ✓ Checkpoint frequency: every {checkpoint_freq * n_envs:,} steps")
    print()
    
    config = {
        'phase': 1,
        'version': 'V2',
        'description': 'Multi-line hover V2 - easier rewards, better observation',
        'environment': 'HoverEnvMultiLineV2',
        'total_timesteps': total_timesteps,
        'n_envs': n_envs,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'max_grad_norm': max_grad_norm,
        },
        'started_at': datetime.now().isoformat(),
        'device': str(device),
        'seed': seed,
    }
    
    config_path = output_path / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved")
    print()
    
    print("=" * 70)
    print("🏋️  STARTING TRAINING")
    print("=" * 70)
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        print()
        print("=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        
        final_model_path = output_path / 'final_model'
        model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")
        
        _maybe_save_vecnorm()
        config['completed_at'] = datetime.now().isoformat()
        config['status'] = 'completed'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        interrupted_path = output_path / 'interrupted_model'
        model.save(interrupted_path)
        print(f"💾 Saved to: {interrupted_path}")
        
        _maybe_save_vecnorm()
        
    finally:
        _maybe_save_vecnorm()
        env.close()
        eval_env.close()
    
    return model


if __name__ == "__main__":
    model = train_ppo_phase1(
        total_timesteps=5000_000,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # ✅ Więcej exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_envs=4,
        seed=42,
        output_dir='models/saved/phase1',
        device='auto',
    )
