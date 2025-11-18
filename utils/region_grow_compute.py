#!/usr/bin/env python3
"""
Prosty monitor zasobów dla pipeline'u region_grow.

Zbiera zużycie CPU/GPU wybranych procesów przez określony czas i zapisuje
podsumowanie do JSON w data/region_grow (domyślnie).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import psutil

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - opcjonalne GPU
    pynvml = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "region_grow"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "usage_snapshot.json"


@dataclass
class ProcStats:
    pid: int
    name: str
    cmdline: List[str]
    cpu_samples: List[float] = field(default_factory=list)
    rss_samples: List[float] = field(default_factory=list)
    gpu_mem_samples: List[float] = field(default_factory=list)  # MB

    def as_summary(self) -> Dict[str, object]:
        def _avg(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        def _max(values: List[float]) -> float:
            return float(max(values)) if values else 0.0

        return {
            "pid": self.pid,
            "name": self.name,
            "cmdline": self.cmdline,
            "samples": len(self.cpu_samples),
            "cpu_percent_avg": round(_avg(self.cpu_samples), 3),
            "cpu_percent_max": round(_max(self.cpu_samples), 3),
            "rss_mb_avg": round(_avg(self.rss_samples), 3),
            "rss_mb_max": round(_max(self.rss_samples), 3),
            "gpu_mem_mb_avg": round(_avg(self.gpu_mem_samples), 3),
            "gpu_mem_mb_max": round(_max(self.gpu_mem_samples), 3),
        }


def _fetch_processes(filter_text: Optional[str], explicit_pids: Optional[List[int]]) -> List[psutil.Process]:
    procs: List[psutil.Process] = []
    if explicit_pids:
        for pid in explicit_pids:
            try:
                procs.append(psutil.Process(pid))
            except psutil.Error:
                pass
        return procs

    if not filter_text:
        filter_text = "region_grow.py"

    filter_lower = filter_text.lower()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if filter_lower in name or filter_lower in cmdline:
                procs.append(proc)
        except psutil.Error:
            continue
    return procs


def _gpu_memory_by_pid() -> Dict[int, float]:
    stats: Dict[int, float] = {}
    if pynvml is None:
        return stats
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return stats

    for idx in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception:
            continue
        processes = []
        for getter in (
            getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses_v3", None),
            getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
            getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
            getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
        ):
            if getter is None:
                continue
            try:
                processes = getter(handle)  # type: ignore
                if processes:
                    break
            except Exception:
                continue
        for proc in processes or []:
            pid = getattr(proc, "pid", None)
            mem = getattr(proc, "usedGpuMemory", 0)
            if pid is None or mem is None or mem <= 0:
                continue
            stats[int(pid)] = stats.get(int(pid), 0.0) + float(mem) / (1024 * 1024)
    return stats


def _ensure_nvml():
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def monitor_processes(
    processes: List[psutil.Process],
    duration: float,
    interval: float,
) -> Dict[int, ProcStats]:
    stats: Dict[int, ProcStats] = {}
    for proc in processes:
        try:
            proc.cpu_percent(None)  # prime
        except psutil.Error:
            continue
        stats[proc.pid] = ProcStats(proc.pid, proc.name(), proc.cmdline())

    if not stats:
        return {}

    gpu_ready = _ensure_nvml()
    start = time.time()
    next_sample = start + interval

    while time.time() - start < duration:
        sleep_for = max(0.0, next_sample - time.time())
        if sleep_for:
            time.sleep(sleep_for)
        next_sample += interval

        gpu_mem = _gpu_memory_by_pid() if gpu_ready else {}
        for pid, stat in list(stats.items()):
            try:
                proc = psutil.Process(pid)
                cpu = proc.cpu_percent(None)
                mem = proc.memory_info().rss / (1024 * 1024)
            except psutil.Error:
                continue
            stat.cpu_samples.append(cpu)
            stat.rss_samples.append(mem)
            stat.gpu_mem_samples.append(gpu_mem.get(pid, 0.0))

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor CPU/GPU usage for region_grow.")
    parser.add_argument("--duration", type=float, default=15.0, help="Czas monitorowania [s].")
    parser.add_argument("--interval", type=float, default=1.0, help="Odstęp między próbkami [s].")
    parser.add_argument(
        "--filter",
        type=str,
        default="region_grow.py",
        help="Tekst wyszukiwany w nazwie/cmdline procesu.",
    )
    parser.add_argument(
        "--pid",
        type=int,
        action="append",
        help="Monitoruj konkretny PID (można podać wielokrotnie).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Ścieżka do wyjściowego JSON.",
    )
    args = parser.parse_args()

    procs = _fetch_processes(args.filter, args.pid)
    if not procs:
        print(f"[WARN] Nie znaleziono procesu zawierającego '{args.filter}' ani PID z listy.")
        sys.exit(1)

    print(f"[INFO] Monitoruję {len(procs)} proces(y) przez {args.duration}s (dt={args.interval}s).")
    stats = monitor_processes(procs, args.duration, args.interval)
    if not stats:
        print("[WARN] Brak próbek (procesy mogły zakończyć się zbyt szybko).")
        sys.exit(1)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_sec": args.duration,
        "interval_sec": args.interval,
        "filter": args.filter,
        "pids": args.pid or [],
        "processes": [stat.as_summary() for stat in stats.values()],
        "gpu_monitoring": bool(pynvml),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Zapisano podsumowanie do: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Przerwano przez użytkownika.")
