"""
Custom callbacks for detailed training statistics.
"""

import math
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from tqdm.auto import tqdm
    from utils.gpu_monitor import GPUMonitor

__all__ = [
    "DetailedStatsCallback",
    "GPUProgressBarCallback",
    "LiveTrainingMonitor",
]


class DetailedStatsCallback(BaseCallback):
    """
    Callback that logs detailed statistics every N steps.
    Shows training metrics, environment stats, and optional GPU usage.
    """

    def __init__(
        self,
        log_freq: int = 500,
        show_gpu: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.show_gpu = show_gpu

        self.episode_rewards: Deque[float] = deque(maxlen=100)
        self.episode_lengths: Deque[int] = deque(maxlen=100)
        self.episode_count = 0
        self.last_log_step = 0
        self.start_time: Optional[float] = None

        self.env_stats: Dict[str, Deque[float]] = {
            "dots_collected": deque(maxlen=100),
            "lines_completed": deque(maxlen=100),
            "success_rate": deque(maxlen=100),
        }

        self._nvml_initialized = False
        self._gpu_monitor = None  # type: Optional["GPUMonitor"]

    # ------------------------------------------------------------------ #
    # BaseCallback hooks
    # ------------------------------------------------------------------ #
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print()
        print("=" * 100)
        print(" " * 35 + "TRAINING STATISTICS")
        print("=" * 100)
        self._print_header()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode_data = info.get("episode")
            if episode_data is not None:
                self.episode_rewards.append(episode_data["r"])
                self.episode_lengths.append(episode_data["l"])
                self.episode_count += 1

            if "total_dots" in info:
                self.env_stats["dots_collected"].append(info["total_dots"])
            if "lines_completed" in info:
                self.env_stats["lines_completed"].append(info["lines_completed"])
            if "all_complete" in info:
                self.env_stats["success_rate"].append(1.0 if info["all_complete"] else 0.0)

        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_stats()
            self.last_log_step = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        print("=" * 100)
        print()

        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0.0

        print("=" * 100)
        print(" " * 38 + "FINAL SUMMARY")
        print("=" * 100)
        print(f"   Total steps:        {self.num_timesteps:,}")
        print(f"   Total episodes:     {self.episode_count:,}")
        print(f"   Training time:      {elapsed/60:.1f} min")
        print(f"   Average FPS:        {fps:.0f}")

        if self.episode_rewards:
            print(f"   Mean reward:        {np.mean(self.episode_rewards):.2f}")
            print(f"   Best reward:        {np.max(self.episode_rewards):.2f}")
            print(f"   Mean episode len:   {np.mean(self.episode_lengths):.0f}")

        if self.env_stats["dots_collected"]:
            print()
            print(f"   Mean dots:          {np.mean(self.env_stats['dots_collected']):.1f}")
            print(f"   Mean lines:         {np.mean(self.env_stats['lines_completed']):.2f}")
            print(f"   Success rate:       {np.mean(self.env_stats['success_rate'])*100:.1f}%")

        print()
        print("=" * 100)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def attach_gpu_monitor(self, monitor: "GPUMonitor") -> None:
        """Attach a GPUMonitor instance for richer GPU stats."""
        self._gpu_monitor = monitor

    def _print_header(self) -> None:
        print()
        header = (
            f"{'Step':>8} | {'FPS':>6} | {'Reward':>10} | {'Ep.Len':>7} | "
            f"{'Dots':>6} | {'Lines':>6} | {'Success':>8} | "
            f"{'PG Loss':>9} | {'V Loss':>8} | {'GPU%':>5} | {'VRAM':>7}"
        )
        print(header)
        print("-" * len(header))

    def _log_stats(self) -> None:
        assert self.start_time is not None, "Training start time not set."

        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0.0

        mean_reward = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
        mean_ep_len = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
        mean_dots = float(np.mean(self.env_stats["dots_collected"])) if self.env_stats["dots_collected"] else 0.0
        mean_lines = float(np.mean(self.env_stats["lines_completed"])) if self.env_stats["lines_completed"] else 0.0
        success_rate = (
            float(np.mean(self.env_stats["success_rate"]) * 100.0) if self.env_stats["success_rate"] else 0.0
        )

        pg_loss = 0.0
        v_loss = 0.0
        if getattr(self.model, "logger", None) is not None:
            name_to_value = self.model.logger.name_to_value
            pg_loss = name_to_value.get("train/policy_gradient_loss", 0.0)
            v_loss = name_to_value.get("train/value_loss", 0.0)

        gpu_util, vram_used, vram_total = self._gpu_stats()

        row = (
            f"{self.num_timesteps:8d} | "
            f"{fps:6.0f} | "
            f"{mean_reward:10.2f} | "
            f"{mean_ep_len:7.0f} | "
            f"{mean_dots:6.1f} | "
            f"{mean_lines:6.2f} | "
            f"{success_rate:7.1f}% | "
            f"{pg_loss:9.4f} | "
            f"{v_loss:8.4f} | "
            f"{gpu_util:4.0f}% | "
            f"{vram_used:6.2f}G"
        )
        print(row)
        self._print_gpu_bars(gpu_util, vram_used, vram_total)

        if getattr(self.model, "logger", None) is not None:
            self.logger.record("rollout/mean_reward", mean_reward)
            self.logger.record("rollout/mean_ep_length", mean_ep_len)
            self.logger.record("rollout/fps", fps)
            self.logger.record("env/mean_dots", mean_dots)
            self.logger.record("env/mean_lines", mean_lines)
            self.logger.record("env/success_rate", success_rate)

            if self.show_gpu and torch.cuda.is_available():
                self.logger.record("system/gpu_util", gpu_util)
                self.logger.record("system/vram_gb", vram_used)

    def _print_gpu_bars(self, gpu_util: float, vram_used: float, vram_total: float) -> None:
        if not (self.show_gpu and torch.cuda.is_available()):
            return

        bar_width = 40
        def _make_bar(fraction: float) -> str:
            fraction = np.clip(fraction, 0.0, 1.0)
            filled = int(round(fraction * bar_width))
            return "[" + "#" * filled + "-" * (bar_width - filled) + "]"

        util_bar = _make_bar(gpu_util / 100.0 if gpu_util >= 0 else 0.0)
        mem_bar = _make_bar(vram_used / vram_total if vram_total > 0 else 0.0)

        print(f"        GPU Util  {util_bar}  {gpu_util:5.1f}%")
        print(f"        VRAM      {mem_bar}  {vram_used:5.2f}/{vram_total:5.2f} GB")

    def _gpu_stats(self) -> Tuple[float, float, float]:
        if not (self.show_gpu and torch.cuda.is_available()):
            return 0.0, 0.0, 1.0

        # Prefer attached GPU monitor if available.
        if self._gpu_monitor is not None:
            sample = self._gpu_monitor.latest_summary()
            if sample:
                util = sample.get("util", float("nan"))
                mem = sample.get("mem", 0.0)
                total = sample.get("mem_total", 1.0)
                if math.isnan(util):
                    util = 0.0
                return float(util), float(mem), float(total) if total else 1.0

        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_used = torch.cuda.memory_allocated() / 1e9
        gpu_util = 0.0

        try:
            import pynvml

            if not self._nvml_initialized:
                pynvml.nvmlInit()
                self._nvml_initialized = True

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(util.gpu)
        except Exception:
            gpu_util = 0.0

        return gpu_util, vram_used, vram_total


class GPUProgressBarCallback(BaseCallback):
    """
    Custom progress bar that updates a provided tqdm instance with GPU stats.
    """

    def __init__(
        self,
        monitor: Optional["GPUMonitor"],
        progress: "tqdm",
        update_interval: int = 10,
    ):
        super().__init__()
        self.monitor = monitor
        self.progress = progress
        self.update_interval = max(1, int(update_interval))
        self._last_call = 0
        self._last_timestep = 0
        self._total_timesteps = progress.total

    def _on_training_start(self) -> None:
        total = getattr(self.model, "_total_timesteps", None)
        if not total:
            total = self.locals.get("total_timesteps", None)
        if isinstance(total, int) and total > 0:
            self._total_timesteps = total
            self.progress.reset(total=total)
        else:
            self.progress.reset()
        self.progress.set_description("Training")
        self._last_timestep = 0
        self._last_call = 0

    def _on_step(self) -> bool:
        current = self.num_timesteps
        delta = current - self._last_timestep
        if delta > 0:
            self.progress.update(delta)
            self._last_timestep = current

        if (self.n_calls - self._last_call) >= self.update_interval:
            sample = self.monitor.latest_summary() if self.monitor is not None else None
            if sample:
                util = sample.get("util", 0.0)
                mem = sample.get("mem", 0.0)
                total = sample.get("mem_total", 1.0) or 1.0
                postfix = self._format_postfix(util, mem, total)
                self.progress.set_postfix_str(postfix, refresh=False)
                self.progress.refresh()
            self._last_call = self.n_calls

        return True

    def _on_training_end(self) -> None:
        if self._total_timesteps:
            self.progress.total = self._total_timesteps
        self.progress.n = min(self.progress.total, self.num_timesteps)
        if self.monitor is not None:
            sample = self.monitor.latest_summary()
            if sample:
                util = sample.get("util", 0.0)
                mem = sample.get("mem", 0.0)
                total = sample.get("mem_total", 1.0) or 1.0
                postfix = self._format_postfix(util, mem, total)
                self.progress.set_postfix_str(postfix, refresh=False)
        self.progress.refresh()

    @staticmethod
    def _format_postfix(util: float, mem: float, total: float) -> str:
        width = 14

        def _bar(fraction: float) -> str:
            fraction = max(0.0, min(1.0, fraction))
            filled = int(round(fraction * width))
            return "#" * filled + "-" * (width - filled)

        util_value = 0.0 if math.isnan(util) else util
        gpu_bar = _bar(util_value / 100.0)
        mem_bar = _bar(mem / total if total > 0 else 0.0)

        return (
            f"GPU [{gpu_bar}] {util_value:5.1f}% | "
            f"VRAM [{mem_bar}] {mem:4.2f}/{total:4.2f} GB"
        )


class LiveTrainingMonitor(BaseCallback):
    """
    Compact live monitor - shows current training status in one line.
    Updates every few steps (overwrites the same console line).
    """

    def __init__(self, update_freq: int = 10):
        super().__init__()
        self.update_freq = update_freq
        self.start_time: Optional[float] = None
        self.last_rewards: Deque[float] = deque(maxlen=10)

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print("\nLive training monitor active (updates every step).\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq != 0:
            return True

        infos = self.locals.get("infos", [])
        for info in infos:
            episode_data = info.get("episode")
            if episode_data is not None:
                self.last_rewards.append(episode_data["r"])

        elapsed = time.time() - self.start_time if self.start_time is not None else 0.0
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0.0
        mean_reward = float(np.mean(self.last_rewards)) if self.last_rewards else 0.0
        vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        print(
            f"\rStep: {self.num_timesteps:>8,} | "
            f"FPS: {fps:>5.0f} | "
            f"Reward: {mean_reward:>7.1f} | "
            f"VRAM: {vram_used:>4.1f}G | "
            f"Episodes: {len(self.last_rewards)}",
            end="",
            flush=True,
        )

        return True

    def _on_training_end(self) -> None:
        print()
