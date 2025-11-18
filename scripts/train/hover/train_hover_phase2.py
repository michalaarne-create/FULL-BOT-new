"""
PPO Phase 2 - TRANSFER LEARNING + LIGHTWEIGHT CNN FEATURES
Complete training script with Phase 1 weights transfer
"""

import sys
from pathlib import Path

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "envs").exists()),
    Path(__file__).resolve().parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecCheckNan
from tqdm.auto import tqdm

try:
    import torch.backends.cudnn as cudnn
except ImportError:
    cudnn = None

try:
    import torch_directml
except ImportError:
    torch_directml = None

from envs.hover_env import HoverEnvMultiLineV2
from models.tiny_hover_extractor import TinyHoverCNNExtractor
from utils.ocr_features import (
    STATE_DIM,
    ZERO_STATE,
)
from utils.feature_prefetch import FeaturePrefetchManager
from utils.gpu_monitor import GPUMonitor
from utils.training_callbacks import DetailedStatsCallback, GPUProgressBarCallback

# Enable backend optimisations when CUDA is available.
if torch.cuda.is_available():
    if cudnn is not None:
        cudnn.benchmark = True
        if hasattr(cudnn, "allow_tf32"):
            cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass


def _aggregate_polygons(polygons: List[np.ndarray], frame_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Aggregate synthetic OCR polygons into the shared 12-D state representation.
    """
    h, w = frame_shape[:2]

    boxes: List[Tuple[float, float, float, float]] = []
    heights: List[float] = []
    bottoms: List[float] = []
    area_sum = 0.0

    for poly in polygons:
        if poly.shape != (4, 2):
            poly = poly.reshape(-1, 2)
            if poly.shape[0] < 4:
                continue

        xs = np.clip(poly[:, 0], 0, w)
        ys = np.clip(poly[:, 1], 0, h)

        x_min = float(xs.min())
        y_min = float(ys.min())
        x_max = float(xs.max())
        y_max = float(ys.max())

        if x_max <= x_min or y_max <= y_min:
            continue

        boxes.append((x_min, y_min, x_max, y_max))
        height = y_max - y_min
        heights.append(height)
        bottoms.append(y_max)
        area_sum += (x_max - x_min) * height

    if not boxes:
        return ZERO_STATE.copy()

    x_union_min = min(b[0] for b in boxes)
    y_union_min = min(b[1] for b in boxes)
    x_union_max = max(b[2] for b in boxes)
    y_union_max = max(b[3] for b in boxes)

    center_x = (x_union_min + x_union_max) / 2.0
    center_y = (y_union_min + y_union_max) / 2.0
    width = x_union_max - x_union_min
    height = y_union_max - y_union_min

    coverage = np.clip(area_sum / float(w * h), 0.0, 1.0)
    count_norm = min(len(boxes) / 10.0, 1.0)
    avg_height = np.clip(np.mean(heights) / h, 0.0, 1.0)
    height_std = np.clip(np.std(heights) / h, 0.0, 1.0)
    bottom_norm = np.clip(max(bottoms) / h, 0.0, 1.0)

    def _norm(value: float, max_value: float) -> float:
        if max_value <= 0:
            return -1.0
        return float(np.clip(value / max_value, 0.0, 1.0) * 2.0 - 1.0)

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:8] = [
        _norm(center_x, w),
        _norm(center_y, h),
        _norm(width, w),
        _norm(height, h),
        coverage * 2.0 - 1.0,
        count_norm * 2.0 - 1.0,
        avg_height * 2.0 - 1.0,
        bottom_norm * 2.0 - 1.0,
    ]
    state[8] = _norm(x_union_min, w)
    state[9] = _norm(y_union_min, h)
    state[10] = height_std * 2.0 - 1.0
    state[11] = (1.0 if len(boxes) > 1 else 0.0) * 2.0 - 1.0
    return state


class HoverEnvWithScreenshots(HoverEnvMultiLineV2):
    """
    Hover environment wrapper that produces training observations via a scripted OCR surrogate.
    """

    def __init__(
        self,
        *args,
        training: bool = True,
        state_dropout_prob: float = 0.25,
        state_block_dropout_prob: float = 0.05,
        env_hint_weight: Union[float, Tuple[float, float]] = 0.1,
        **kwargs,
    ):
        line_randomization = kwargs.pop("line_randomization", training)
        line_jitter_y = kwargs.pop("line_jitter_y", 32.0 if training else 0.0)
        line_jitter_x = kwargs.pop("line_jitter_x", 60.0 if training else 0.0)
        line_scale_jitter = kwargs.pop("line_scale_jitter", 0.15 if training else 0.0)
        line_shuffle_prob = kwargs.pop("line_shuffle_prob", 0.05 if training else 0.0)

        super().__init__(
            *args,
            line_randomization=line_randomization,
            line_jitter_y=line_jitter_y,
            line_jitter_x=line_jitter_x,
            line_scale_jitter=line_scale_jitter,
            line_shuffle_prob=line_shuffle_prob,
            **kwargs,
        )
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32),
            }
        )
        self.training_mode = training
        self.state_dropout_prob = float(np.clip(state_dropout_prob if training else 0.0, 0.0, 1.0))
        self.state_block_dropout_prob = float(np.clip(state_block_dropout_prob if training else 0.0, 0.0, 1.0))
        if isinstance(env_hint_weight, tuple):
            hint_min, hint_max = env_hint_weight
        else:
            hint_min = hint_max = float(env_hint_weight)
        hint_min = float(np.clip(hint_min, 0.0, 1.0))
        hint_max = float(np.clip(hint_max, 0.0, 1.0))
        if hint_max < hint_min:
            hint_min, hint_max = hint_max, hint_min
        self.env_hint_range = (hint_min, hint_max) if training else (0.0, 0.0)
        self._last_hint_weight = 0.0
        self._rng = np.random.default_rng()
        self._grid_size = 16
        self._grid_cell_w = self.screen_w / float(self._grid_size)
        self._grid_cell_h = self.screen_h / float(self._grid_size)
        self._line_render_data: List[Tuple[int, int, int, str]] = []
        self._line_mask = np.zeros((self._grid_size, self._grid_size), dtype=bool)
        self._base_frame: Optional[np.ndarray] = None
        self._latest_boxes: List[Dict[str, Any]] = []
        self._script_polygons: Optional[List[np.ndarray]] = None
        self._script_boxes_template: List[Dict[str, Any]] = []
        self._script_frame_shape: Optional[Tuple[int, int, int]] = None
        self._prefetch_manager = FeaturePrefetchManager.instance() if training else None
        self._refresh_render_primitives()

    def _refresh_render_primitives(self) -> None:
        self._line_render_data = [
            (int(line["y1"]), int(line["x1"]), int(line["x2"]), line["id"]) for line in self.lines
        ]
        self._line_mask = self._compute_line_mask()
        self._base_frame = None
        self._script_polygons = None
        self._script_boxes_template = []
        self._script_frame_shape = None

    def _compute_line_mask(self) -> np.ndarray:
        mask = np.zeros((self._grid_size, self._grid_size), dtype=bool)
        cell_w = self._grid_cell_w
        cell_h = self._grid_cell_h
        line_thickness = 28.0
        half_thickness = line_thickness / 2.0

        for y, x1, x2, _ in self._line_render_data:
            if x2 <= x1:
                continue

            center_row = int(np.round(float(y) / cell_h))
            center_row = max(0, min(self._grid_size - 1, center_row))

            col_start = int(np.floor(float(x1) / cell_w))
            col_end = int(np.ceil(float(x2) / cell_w) - 1)

            col_start = max(0, min(self._grid_size - 1, col_start))
            col_end = max(col_start, min(self._grid_size - 1, col_end))

            mask[center_row, col_start : col_end + 1] = True

        return mask

    def _render_screenshot(self):
        if self._base_frame is None:
            base = np.where(self._line_mask[..., None], 0, 255).astype(np.uint8)
            if base.shape[2] == 1:
                base = np.repeat(base, 3, axis=2)
            self._base_frame = np.ascontiguousarray(base)
        return self._base_frame

    def _sample_env_hint_weight(self) -> float:
        low, high = self.env_hint_range
        if high <= 0.0:
            return 0.0
        if not self.training_mode:
            return high
        if high <= low:
            weight = high
        else:
            weight = float(self._rng.uniform(low, high))
        return float(np.clip(weight, 0.0, 1.0))

    def _blend_state(self, scripted_state: np.ndarray, env_state: np.ndarray) -> np.ndarray:
        weight = self._sample_env_hint_weight()
        self._last_hint_weight = weight
        if weight <= 0:
            return scripted_state.astype(np.float32, copy=False)
        env_state = np.clip(env_state.astype(np.float32), -1.0, 1.0)
        return np.clip(
            (1.0 - weight) * scripted_state + weight * env_state,
            -1.0,
            1.0,
        ).astype(np.float32)

    def _apply_dropout(self, state: np.ndarray) -> np.ndarray:
        if self.state_block_dropout_prob > 0.0 and self._rng.random() < self.state_block_dropout_prob:
            noise = self._rng.normal(0.0, 0.05, size=state.shape).astype(np.float32)
            return np.clip(noise, -1.0, 1.0)
        if self.state_dropout_prob <= 0.0:
            return state
        mask = (self._rng.random(state.shape) > self.state_dropout_prob).astype(np.float32)
        if not mask.any():
            mask[self._rng.integers(0, state.size)] = 1.0
        noisy = state * mask
        noise = self._rng.normal(0.0, 0.05, size=state.shape).astype(np.float32)
        return np.clip(noisy + noise, -1.0, 1.0)

    def _ensure_script_geometry(self, geometry_shape: Tuple[int, int, int]) -> None:
        if self._script_frame_shape == geometry_shape and self._script_polygons is not None:
            return

        h, w = geometry_shape[:2]
        line_height = 28

        polygons: List[np.ndarray] = []
        boxes: List[Dict[str, Any]] = []

        for y, x1, x2, line_id in self._line_render_data:
            top = int(np.clip(y - line_height // 2, 0, h - 1))
            bottom = int(np.clip(y + line_height // 2, 0, h - 1))
            left = int(np.clip(x1, 0, w - 1))
            right = int(np.clip(x2, 0, w - 1))

            if right <= left or bottom <= top:
                continue

            poly = np.array(
                [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ],
                dtype=np.float32,
            )
            polygons.append(poly)
            boxes.append(
                {
                    "id": line_id,
                    "box": poly.tolist(),
                    "text": "",
                    "confidence": 1.0,
                }
            )

        self._script_polygons = polygons
        self._script_boxes_template = boxes
        self._script_frame_shape = geometry_shape

    def _compute_script_state(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        channels = int(frame.shape[2]) if frame.ndim == 3 else 1
        geometry_shape = (self.screen_h, self.screen_w, channels)
        self._ensure_script_geometry(geometry_shape)

        if not self._script_polygons:
            return ZERO_STATE.copy(), []

        state = _aggregate_polygons(self._script_polygons, geometry_shape)
        boxes = [
            {
                "id": entry["id"],
                "box": [list(pt) for pt in entry["box"]],
                "text": entry["text"],
                "confidence": entry["confidence"],
            }
            for entry in self._script_boxes_template
        ]
        return state.astype(np.float32, copy=False), boxes

    def _build_observation(self, env_state: np.ndarray) -> dict:
        frame = self._render_screenshot()
        scripted_state, boxes = self._compute_script_state(frame)
        if self.training_mode:
            scripted_state = self._apply_dropout(scripted_state)
            if self._prefetch_manager is not None:
                self._prefetch_manager.submit(frame)
        fused_state = self._blend_state(scripted_state, env_state)
        self._latest_boxes = boxes
        return {"screen": frame, "state": fused_state}

    def reset(self, seed=None, options=None):
        self._latest_boxes = []
        env_state, info = super().reset(seed=seed, options=options)
        self._refresh_render_primitives()
        reseed_val = int(self.np_random.integers(0, 2**31 - 1)) if hasattr(self, "np_random") else None
        if reseed_val is not None:
            self._rng = np.random.default_rng(reseed_val)
        obs = self._build_observation(env_state)
        info["ocr_boxes"] = self._latest_boxes
        info["hint_weight"] = self._last_hint_weight
        info["state_dropout_prob"] = self.state_dropout_prob
        info["state_block_dropout_prob"] = self.state_block_dropout_prob
        return obs, info

    def step(self, action):
        env_state, reward, terminated, truncated, info = super().step(action)
        obs = self._build_observation(env_state)
        info["ocr_boxes"] = self._latest_boxes
        info["hint_weight"] = self._last_hint_weight
        info["state_dropout_prob"] = self.state_dropout_prob
        info["state_block_dropout_prob"] = self.state_block_dropout_prob
        return obs, reward, terminated, truncated, info

    def set_modality_params(
        self,
        *,
        env_hint_range: Optional[Tuple[float, float]] = None,
        state_dropout: Optional[float] = None,
        state_block_dropout: Optional[float] = None,
    ) -> None:
        if not self.training_mode:
            return
        if env_hint_range is not None:
            hint_min, hint_max = env_hint_range
            hint_min = float(np.clip(hint_min, 0.0, 1.0))
            hint_max = float(np.clip(hint_max, 0.0, 1.0))
            if hint_max < hint_min:
                hint_min, hint_max = hint_max, hint_min
            self.env_hint_range = (hint_min, hint_max)
        if state_dropout is not None:
            self.state_dropout_prob = float(np.clip(state_dropout, 0.0, 1.0))
        if state_block_dropout is not None:
            self.state_block_dropout_prob = float(np.clip(state_block_dropout, 0.0, 1.0))

    def close(self):
        super().close()


class AsyncDictRolloutBuffer(DictRolloutBuffer):
    """
    DictRolloutBuffer with optional CUDA stream prefetching.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
            print("[INFO] Dict rollout buffer initialised with CUDA stream prefetching.")
        else:
            self.stream = None


class PPOAsyncTrainer(PPO):
    """
    PPO with async rollout buffer and gradient accumulation.
    """

    def __init__(self, *args, gradient_accumulation_steps=4, **kwargs):
        self.gradient_accumulation_steps = gradient_accumulation_steps

        super().__init__(*args, **kwargs)

        # Replace with async version
        self.rollout_buffer = AsyncDictRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        print("[INFO] Async PPO trainer initialised:")
        print(f"    Gradient accumulation: {gradient_accumulation_steps}x")
        print("    Async buffer: Enabled")

    def train(self) -> None:
        """
        Training loop with gradient accumulation.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_fractions, pg_losses, value_losses, entropy_losses = [], [], [], []

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            clip_range_value = self.clip_range(self._current_progress_remaining) if callable(self.clip_range) else self.clip_range
            clip_range_vf_value = None
            if self.clip_range_vf is not None:
                clip_range_vf_value = self.clip_range_vf(self._current_progress_remaining) if callable(self.clip_range_vf) else self.clip_range_vf

            self.policy.optimizer.zero_grad()
            minibatch_count = 0

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - clip_range_value, 1 + clip_range_value),
                ).mean()

                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range_value).float()).item())

                values_pred = (
                    values
                    if clip_range_vf_value is None
                    else rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf_value, clip_range_vf_value
                    )
                )
                value_loss = nn.functional.mse_loss(rollout_data.returns, values_pred)
                entropy_loss = -torch.mean(entropy) if entropy is not None else -torch.mean(-log_prob)

                loss = (policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss) / self.gradient_accumulation_steps
                loss.backward()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                minibatch_count += 1

                if minibatch_count % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    self.policy.optimizer.zero_grad()

                with torch.no_grad():
                    approx_kl_divs.append(
                        torch.mean(
                            (torch.exp(log_prob - rollout_data.old_log_prob) - 1) - (log_prob - rollout_data.old_log_prob)
                        ).cpu().numpy()
                    )

            if minibatch_count % self.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad()

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                break

        self._n_updates += self.n_epochs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


class ModalityCurriculumCallback(BaseCallback):
    """
    Curriculum that gradually removes coordinate hints and increases state dropout.
    """

    def __init__(
        self,
        *,
        total_timesteps: int,
        hint_start: Tuple[float, float],
        hint_end: Tuple[float, float],
        dropout_start: float,
        dropout_end: float,
        block_start: float,
        block_end: float,
        update_interval: int = 25_000,
    ) -> None:
        super().__init__()
        self.total_timesteps = max(1, int(total_timesteps))
        self.hint_start = (float(hint_start[0]), float(hint_start[1]))
        self.hint_end = (float(hint_end[0]), float(hint_end[1]))
        self.dropout_start = float(dropout_start)
        self.dropout_end = float(dropout_end)
        self.block_start = float(block_start)
        self.block_end = float(block_end)
        self.update_interval = max(1, int(update_interval))
        self._last_update_step = -self.update_interval

    @staticmethod
    def _lerp(start: float, end: float, alpha: float) -> float:
        return start + (end - start) * alpha

    def _init_callback(self) -> None:
        self._apply_curriculum(force=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update_step >= self.update_interval:
            self._apply_curriculum()
        return True

    def _on_training_end(self) -> None:
        self._apply_curriculum(force=True, final=True)

    def _apply_curriculum(self, force: bool = False, final: bool = False) -> None:
        if not force and self.num_timesteps - self._last_update_step < self.update_interval:
            return
        progress = 1.0 if final else min(1.0, self.num_timesteps / self.total_timesteps)

        hint_min = self._lerp(self.hint_start[0], self.hint_end[0], progress)
        hint_max = self._lerp(self.hint_start[1], self.hint_end[1], progress)
        hint_min, hint_max = sorted((float(np.clip(hint_min, 0.0, 1.0)), float(np.clip(hint_max, 0.0, 1.0))))

        dropout = float(np.clip(self._lerp(self.dropout_start, self.dropout_end, progress), 0.0, 1.0))
        block = float(np.clip(self._lerp(self.block_start, self.block_end, progress), 0.0, 1.0))

        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method(
                "set_modality_params",
                env_hint_range=(hint_min, hint_max),
                state_dropout=dropout,
                state_block_dropout=block,
            )
        else:
            self.training_env.set_modality_params(  # type: ignore[attr-defined]
                env_hint_range=(hint_min, hint_max),
                state_dropout=dropout,
                state_block_dropout=block,
            )

        if self.logger is not None:
            self.logger.record("curriculum/progress", progress)
            self.logger.record("curriculum/env_hint_min", hint_min)
            self.logger.record("curriculum/env_hint_max", hint_max)
            self.logger.record("curriculum/state_dropout", dropout)
            self.logger.record("curriculum/state_block_dropout", block)

        self._last_update_step = self.num_timesteps if not final else self.total_timesteps


def make_env(
    rank,
    seed=0,
    *,
    training: bool = True,
    state_dropout: float = 0.25,
    state_block: float = 0.05,
    env_hint: Union[float, Tuple[float, float]] = 0.1,
):
    def _init():
        env = HoverEnvWithScreenshots(
            lines_file="data/text_lines.json",
            training=training,
            state_dropout_prob=state_dropout,
            state_block_dropout_prob=state_block,
            env_hint_weight=env_hint,
        )
        env = Monitor(env)
        env.action_space.seed(seed + rank)
        return env

    return _init


def transfer_weights(new_model: PPOAsyncTrainer, old_model_path: str) -> bool:
    """
    Transfer policy/value weights from Phase 1 model.
    Skip feature extractor (different architecture).
    """
    print()
    print("=" * 70)
    print("[INFO] TRANSFER LEARNING")
    print("=" * 70)

    if not os.path.exists(old_model_path):
        print(f"[WARN] Phase 1 model not found: {old_model_path}")
        print("       Training from scratch instead.")
        return False

    try:
        print(f"[INFO] Loading Phase 1 model from: {old_model_path}")
        old_model = PPO.load(old_model_path, device="cpu")
        print("[INFO] Phase 1 model loaded successfully.")

        transferred = 0
        skipped = 0
        total_layers = 0

        print()
        print("[INFO] Transferring weights...")

        old_params_dict = dict(old_model.policy.named_parameters())

        with torch.no_grad():
            for name, new_param in new_model.policy.named_parameters():
                total_layers += 1

                if "features_extractor" in name:
                    print(f"    [SKIP] {name} (new visual extractor)")
                    skipped += 1
                    continue

                if name not in old_params_dict:
                    print(f"    [SKIP] {name} (missing in Phase 1 model)")
                    skipped += 1
                    continue

                old_param = old_params_dict[name]

                if new_param.shape != old_param.shape:
                    print(
                        f"    [SKIP] {name} (shape mismatch: {tuple(new_param.shape)} vs {tuple(old_param.shape)})"
                    )
                    skipped += 1
                    continue

                new_param.copy_(old_param.to(new_param.device))
                print(f"    [OK]   {name} {list(new_param.shape)}")
                transferred += 1

        del old_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print()
        print("=" * 70)
        print("[INFO] TRANSFER SUMMARY")
        print("=" * 70)
        print(f"    Total layers:   {total_layers}")
        print(f"    Transferred:    {transferred}")
        print(f"    Skipped:        {skipped}")
        success_rate = (transferred / total_layers * 100.0) if total_layers > 0 else 0.0
        print(f"    Success rate:   {success_rate:.1f}%")
        print("=" * 70)
        print()

        return True

    except Exception as exc:
        print(f"[ERROR] Transfer failed: {exc}")
        print("        Training from scratch instead.")
        return False


def _select_training_device() -> Tuple[torch.device, str]:
    """
    Prefer DirectML when available, fall back to CUDA or CPU.
    """
    if torch_directml is not None:
        try:
            dml_device = torch_directml.device()
            return dml_device, "DirectML"
        except Exception:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)

    return torch.device("cpu"), "CPU"


def train():
    print("=" * 70)
    print("PPO PHASE 2 - TRANSFER LEARNING + LIGHT CNN FEATURES")
    print("=" * 70)
    print()

    device, device_name = _select_training_device()
    print(f"[INFO] Using device: {device_name}")

    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {total_vram:.1f} GB")
    elif torch_directml is not None and device_name == "DirectML":
        print("[INFO] DirectML backend active.")
    else:
        print("[WARN] GPU acceleration unavailable; running on CPU will be slower.")

    monitor = GPUMonitor(interval_s=2.0)
    monitor.start()

    # Training settings (lightweight CNN tuning)
    n_envs = 4
    n_steps = 256
    batch_size = 512
    gradient_accumulation_steps = 1
    n_epochs = 6
    total_timesteps = 500_000

    # Transfer learning
    use_transfer_learning = False
    phase1_model_path = (
        r"C:\Users\user\Desktop\Nowy folder\BOT ANK\bot\moje_AI\yolov8\FULL BOT\models\saved\phase1\best_model\best_model.zip"
    )

    print()
    print("[INFO] Training settings:")
    print(f"    n_envs:             {n_envs}")
    print(f"    n_steps:            {n_steps}")
    print(f"    batch_size:         {batch_size}")
    print(f"    gradient_accum:     {gradient_accumulation_steps}x")
    print(f"    effective_batch:    {batch_size * gradient_accumulation_steps}")
    print(f"    n_epochs:           {n_epochs}")
    print(f"    total_timesteps:    {total_timesteps:,}")
    print()
    print("[INFO] Memory expectations:")
    print("    - Tiny CNN extractor: <0.5 GB VRAM per env")
    print("    - PPO batch size:     ~512 samples/update")
    print("    - Expected GPU utilisation: 40-60% (depends on device)")
    print()

    if use_transfer_learning:
        print("[INFO] Transfer learning configuration:")
        print(f"    Phase 1 model: {phase1_model_path}")
        print("    Strategy: transfer policy/value weights, new visual extractor")
        print()

    print("[INFO] Creating environments...")
    env_factories = [
        make_env(
            i,
            42,
            training=True,
            state_dropout=0.0,
            state_block=0.0,
            env_hint=0.0,
        )
        for i in range(n_envs)
    ]
    if n_envs == 1:
        env = DummyVecEnv(env_factories)
    else:
        env = SubprocVecEnv(env_factories, start_method="spawn")

    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=50.0)

    eval_env = DummyVecEnv(
        [
            make_env(
                0,
                42,
                training=False,
                state_dropout=0.0,
                state_block=0.0,
                env_hint=0.0,
            )
        ]
    )
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=True, clip_reward=50.0)
    eval_env.ret_rms = env.ret_rms
    eval_env = VecCheckNan(eval_env, raise_exception=False, warn_once=True)
    print(f"[INFO] {n_envs} training environments + 1 eval environment created.")
    print("[INFO] Modality parameters locked (no curriculum).")
    print("    - Coord hint weight: 0.00 fixed")
    print("    - Per-dimension dropout: 0.00 fixed")
    print("    - Full-state dropout: 0.00 fixed")
    print("[INFO] Reward normalization (VecNormalize) enabled for training and eval.")
    print()

    policy_kwargs = dict(
        features_extractor_class=TinyHoverCNNExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.ReLU,
        normalize_images=False,
    )

    print("[INFO] Creating PPO model with lightweight CNN feature extractor...")

    model = PPOAsyncTrainer(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=2e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=1.0,
        target_kl=0.02,
        clip_range_vf=0.15,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log="models/saved/phase2_transfer/tensorboard",
        seed=42,
        policy_kwargs=policy_kwargs,
    )

    print("[INFO] PPO model created.")

    if use_transfer_learning:
        transfer_weights(model, phase1_model_path)
    else:
        print("[INFO] Transfer learning disabled; training from scratch.")
        print()

    os.makedirs("models/saved/phase2_transfer/best_model", exist_ok=True)
    os.makedirs("models/saved/phase2_transfer/checkpoints", exist_ok=True)
    os.makedirs("models/saved/phase2_transfer/eval_logs", exist_ok=True)
    vecnorm_path = "models/saved/phase2_transfer/vecnormalize.pkl"
    vecnorm_saved = False

    def _save_vecnormalize() -> None:
        nonlocal vecnorm_saved
        if vecnorm_saved:
            return
        if isinstance(env, VecNormalize):
            try:
                env.save(vecnorm_path)
                print(f"[INFO] VecNormalize stats saved to: {vecnorm_path}")
            except Exception as save_exc:
                print(f"[WARN] Failed to save VecNormalize stats: {save_exc}")
        vecnorm_saved = True

    print("[INFO] Setting up callbacks...")

    progress_bar = tqdm(
        total=total_timesteps,
        desc="Training",
        dynamic_ncols=True,
        smoothing=0.01,
        leave=True,
    )

    curriculum_cb = ModalityCurriculumCallback(
        total_timesteps=total_timesteps,
        hint_start=(0.0, 0.0),
        hint_end=(0.0, 0.0),
        dropout_start=0.0,
        dropout_end=0.0,
        block_start=0.0,
        block_end=0.0,
        update_interval=max(total_timesteps // 5, 50_000),
    )
    stats_cb = DetailedStatsCallback(log_freq=500, show_gpu=torch.cuda.is_available())
    stats_cb.attach_gpu_monitor(monitor)
    progress_cb = GPUProgressBarCallback(monitor, progress_bar, update_interval=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/saved/phase2_transfer/best_model",
        log_path="models/saved/phase2_transfer/eval_logs",
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 100),
        save_path="models/saved/phase2_transfer/checkpoints",
        name_prefix="ppo_phase2",
    )

    print("[INFO] Callbacks ready.")
    print()

    print("[INFO] Running warmup rollouts...")
    obs = env.reset()
    for _ in range(10):
        model.predict(obs, deterministic=False)

    if device.type == "cuda":
        torch.cuda.synchronize()
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        usage_pct = (vram_used / vram_total * 100.0) if vram_total > 0 else 0.0
        print(f"[INFO] Warmup complete. VRAM: {vram_used:.2f} GB / {vram_total:.1f} GB ({usage_pct:.0f}%)")
    else:
        print("[INFO] Warmup complete.")

    print()
    print("=" * 70)
    print("READY TO TRAIN")
    print("=" * 70)
    print()
    print("[INFO] Expected performance (reference from CUDA run):")
    print("    - Initial reward: ~3000 (from Phase 1 transfer)")
    print("    - Target reward:  ~5000+")
    print("    - Training time:  ~30-45 min for 500k steps")
    print("    - Throughput:     ~1500-2000 FPS")
    print()
    print("[INFO] Checkpoints:")
    print("    - Best model: models/saved/phase2_transfer/best_model/")
    print("    - Periodic:   models/saved/phase2_transfer/checkpoints/")
    print("    - Logs:       models/saved/phase2_transfer/eval_logs/")
    print()
    print("[INFO] Monitoring tips:")
    print("    - TensorBoard: tensorboard --logdir models/saved/phase2_transfer/tensorboard")
    if device.type == "cuda":
        print("    - GPU:         nvidia-smi -l 1")
    else:
        print("    - GPU:         use vendor-specific tooling if available")
    print()

    input("Press ENTER to start training...")

    print()
    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)
    print()

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[curriculum_cb, stats_cb, eval_cb, ckpt_cb, progress_cb],
            progress_bar=False,
            reset_num_timesteps=False,
        )

        elapsed = time.time() - start_time
        fps = total_timesteps / elapsed if elapsed > 0 else 0.0

        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"    Duration:     {elapsed / 60:.1f} minutes")
        print(f"    Average FPS:  {fps:.0f}")
        print(f"    Total steps:  {total_timesteps:,}")
        print()

        final_path = "models/saved/phase2_transfer/final_model"
        model.save(final_path)
        print(f"[INFO] Final model saved to: {final_path}")
        _save_vecnormalize()

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("TRAINING INTERRUPTED (CTRL+C)")
        print("=" * 70)

        interrupted_path = "models/saved/phase2_transfer/interrupted_model"
        model.save(interrupted_path)
        print(f"[INFO] Progress saved to: {interrupted_path}")
        _save_vecnormalize()

    except Exception as exc:
        print()
        print("=" * 70)
        print(f"[ERROR] Training failed: {exc}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        _save_vecnormalize()

    finally:
        _save_vecnormalize()
        progress_bar.close()
        monitor.stop()
        print()
        print("[INFO] Cleaning up environments...")
        env.close()
        eval_env.close()
        print("[INFO] Cleanup done.")

    print()
    print("=" * 70)
    print("SESSION ENDED")
    print("=" * 70)


if __name__ == "__main__":
    train()
