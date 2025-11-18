"""Lightweight GPU utilisation monitor for training runs."""

from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

import torch

Sample = Dict[str, float]


class GPUMonitor:
    """
    Background GPU monitor that records utilisation/VRAM without spamming stdout.

    By default, metrics are only stored internally and can be queried via ``latest``.
    Set ``emit_stdout=True`` to stream periodic log lines (debug mode).
    """

    def __init__(self, interval_s: float = 2.0, *, emit_stdout: bool = False):
        self.interval_s = max(0.5, float(interval_s))
        self.emit_stdout = emit_stdout

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sampler: Optional[Callable[[], List[Sample]]] = None
        self._nvml_handles: List[object] = []
        self._nvml_module = None

        self._latest_samples: List[Sample] = []
        self._lock = threading.Lock()
        self.active = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        """
        Start the background monitoring thread.

        Returns ``True`` when monitoring begins, ``False`` if no backend was available.
        """
        if self._thread is not None:
            return True

        self._sampler = self._resolve_sampler()
        if self._sampler is None:
            if self.emit_stdout:
                print("[GPU Monitor] No GPU monitoring backend available.")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        self.active = True
        if self.emit_stdout:
            print(f"[GPU Monitor] Started (interval {self.interval_s:.1f}s).")
        return True

    def stop(self) -> None:
        """Stop the background thread and release NVML resources."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
            self._thread = None
        self._shutdown_nvml()
        self.active = False

    def latest(self, index: int = 0) -> Optional[Sample]:
        """Return the latest sample for GPU ``index`` (or ``None`` if unavailable)."""
        with self._lock:
            if 0 <= index < len(self._latest_samples):
                return dict(self._latest_samples[index])
        return None

    def latest_summary(self) -> Optional[Sample]:
        """Return the first GPU sample (useful for single-GPU setups)."""
        return self.latest(0)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                samples = self._sampler() if self._sampler else []  # type: ignore[operator]
            except Exception as exc:
                samples = []
                if self.emit_stdout:
                    print(f"[GPU Monitor] Sampling error: {exc}")

            with self._lock:
                self._latest_samples = samples

            if self.emit_stdout and samples:
                for sample in samples:
                    idx = int(sample.get("index", 0))
                    util = sample.get("util", float("nan"))
                    mem = sample.get("mem", float("nan"))
                    mem_total = sample.get("mem_total", float("nan"))
                    temp = sample.get("temp")
                    reserved = sample.get("reserved")
                    util_str = f"{util:5.1f}%" if not math.isnan(util) else "  n/a"
                    msg = f"[GPU Monitor] GPU{idx}: util={util_str} mem={mem:6.2f}/{mem_total:6.2f} GB"
                    if reserved is not None:
                        msg += f" (reserved {reserved:6.2f} GB)"
                    if temp is not None:
                        msg += f" temp={temp:4.0f}°C"
                    print(msg)

            time.sleep(self.interval_s)

    def _resolve_sampler(self) -> Optional[Callable[[], List[Sample]]]:
        sampler = self._try_init_nvml()
        if sampler is not None:
            return sampler

        sampler = self._try_init_nvidia_smi()
        if sampler is not None:
            return sampler

        if torch.cuda.is_available():
            return self._torch_allocator_sampler

        return None

    # ----------------------- NVML backend ------------------------------ #
    def _try_init_nvml(self) -> Optional[Callable[[], List[Sample]]]:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
            self._nvml_module = pynvml

            def _nvml_sampler() -> List[Sample]:
                samples: List[Sample] = []
                for idx, handle in enumerate(self._nvml_handles):
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except pynvml.NVMLError:
                        temp = None
                    samples.append(
                        {
                            "index": float(idx),
                            "util": float(util.gpu),
                            "mem": mem.used / 1e9,
                            "mem_total": mem.total / 1e9,
                            "temp": float(temp) if temp is not None else None,
                        }
                    )
                return samples

            return _nvml_sampler
        except Exception:
            self._nvml_handles = []
            self._nvml_module = None
            return None

    def _shutdown_nvml(self) -> None:
        if self._nvml_module is not None:
            try:
                self._nvml_module.nvmlShutdown()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._nvml_module = None
        self._nvml_handles = []

    # -------------------- nvidia-smi backend --------------------------- #
    def _try_init_nvidia_smi(self) -> Optional[Callable[[], List[Sample]]]:
        if shutil.which("nvidia-smi") is None:
            return None

        def _smi_sampler() -> List[Sample]:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "nvidia-smi error")

            samples: List[Sample] = []
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                idx = float(parts[0])
                util = float(parts[1])
                mem = float(parts[2]) / 1024  # MB -> GB
                mem_total = float(parts[3]) / 1024
                temp = float(parts[4]) if len(parts) > 4 else None
                samples.append({"index": idx, "util": util, "mem": mem, "mem_total": mem_total, "temp": temp})
            return samples

        return _smi_sampler

    # -------------------- Torch allocator fallback -------------------- #
    def _torch_allocator_sampler(self) -> List[Sample]:
        samples: List[Sample] = []
        device_count = torch.cuda.device_count()
        for idx in range(device_count):
            torch.cuda.synchronize(idx)
            mem_alloc = torch.cuda.memory_allocated(idx) / 1e9
            mem_reserved = torch.cuda.memory_reserved(idx) / 1e9
            total = torch.cuda.get_device_properties(idx).total_memory / 1e9
            samples.append(
                {
                    "index": float(idx),
                    "util": float("nan"),
                    "mem": mem_alloc,
                    "mem_total": total,
                    "temp": None,
                    "reserved": mem_reserved,
                }
            )
        return samples


__all__ = ["GPUMonitor"]

