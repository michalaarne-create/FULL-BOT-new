#!/usr/bin/env python3
"""
Main orchestrator for the live dropdown pipeline.

It launches the Chrome recorder (ai_recorder_live) once and then, every
PIPELINE_INTERVAL seconds, captures the full screen, runs utils/region_grow.py
on the screenshot (region_grow already performs OCR internally) and finally
feeds the JSON emitted by region_grow into scripts/numpy_rate/rating.py.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_SCREEN_DIR = ROOT / "data" / "screen"
AI_RECORDER_SCRIPT = ROOT / "dom_renderer" / "ai_recorder_live.py"
REGION_GROW_SCRIPT = ROOT / "utils" / "region_grow.py"
RATING_SCRIPT = ROOT / "scripts" / "numpy_rate" / "rating.py"
ARROW_POST_SCRIPT = ROOT / "scripts" / "arrow_post_region.py"
HOVER_SINGLE_SCRIPT = ROOT / "scripts" / "hard_bot" / "hover_single.py"
SCREENSHOT_DIR = DATA_SCREEN_DIR / "raw screen"
SCREEN_BOXES_DIR = DATA_SCREEN_DIR / "numpy_points" / "screen_boxes"
DOM_LIVE_DIR = ROOT / "dom_live"
CURRENT_QUESTION_PATH = DOM_LIVE_DIR / "current_question.json"
HOVER_INPUT_DIR = DATA_SCREEN_DIR / "hover_input"
HOVER_OUTPUT_DIR = DATA_SCREEN_DIR / "hover_output"
HOVER_SIDE_CROP = 300
HOVER_TOP_CROP = int(os.environ.get("HOVER_TOP_CROP", "120"))
CONTROL_AGENT_PORT = int(os.environ.get("CONTROL_AGENT_PORT", "8765"))
HOVER_FALLBACK_SECONDS = float(os.environ.get("HOVER_FALLBACK_SECONDS", "10"))
RATE_SUMMARY_DIR = DATA_SCREEN_DIR / "numpy_points" / "rate_summary"
BRAIN_STATE_FILE = ROOT / "data" / "brain_state.json"

CREATE_NO_WINDOW = 0
if os.name == "nt":
    CREATE_NO_WINDOW = 0x08000000
SUBPROCESS_KW = {"creationflags": CREATE_NO_WINDOW} if os.name == "nt" else {}
_hover_fallback_timer: Optional[threading.Timer] = None

class StatusOverlay:
    def __init__(self):
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="StatusOverlay", daemon=True)
        self._running = threading.Event()
        self._lines: list[str] = []

    def start(self):
        if self._running.is_set():
            return
        self._running.set()
        self._thread.start()

    def stop(self):
        if not self._running.is_set():
            return
        self._running.clear()
        self._queue.put("__quit__")

    def set_status(self, text: str):
        if self._running.is_set():
            self._queue.put(text)

    def _run(self):
        try:
            import tkinter as tk
        except Exception as exc:
            log(f"[WARN] Could not start overlay (tkinter unavailable: {exc})")
            return

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.85)
        except Exception:
            pass
        root.configure(bg="#104010")

        label = tk.Label(
            root,
            text="Pipeline idle",
            fg="#00FF00",
            bg="#104010",
            font=("Consolas", 11),
            justify="left",
            anchor="nw",
        )
        label.pack(padx=10, pady=6)

        root.update_idletasks()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        width, height = 420, 160
        x = sw - width - 25
        y = sh - height - 40
        root.geometry(f"{width}x{height}+{x}+{y}")

        def poll_queue():
            try:
                while True:
                    msg = self._queue.get_nowait()
                    if msg == "__quit__":
                        root.destroy()
                        return
                    self._lines.append(msg)
                    self._lines = self._lines[-5:]
                    label.config(text="\n".join(self._lines))
            except queue.Empty:
                pass
            root.after(200, poll_queue)

        poll_queue()
        root.mainloop()


_status_overlay: Optional[StatusOverlay] = None


def update_overlay_status(message: str):
    if _status_overlay is not None:
        _status_overlay.set_status(message)
def cancel_hover_fallback_timer():
    global _hover_fallback_timer
    if _hover_fallback_timer is not None:
        _hover_fallback_timer.cancel()
        _hover_fallback_timer = None


def start_hover_fallback_timer():
    if HOVER_FALLBACK_SECONDS <= 0:
        return
    cancel_hover_fallback_timer()

    def _timeout():
        msg = f"Hover fallback! no response for {HOVER_FALLBACK_SECONDS:.0f}s. Stopping."
        log(f"[ERROR] {msg}")
        update_overlay_status(msg)
        os._exit(3)

    timer = threading.Timer(HOVER_FALLBACK_SECONDS, _timeout)
    timer.daemon = True
    timer.start()
    globals()["_hover_fallback_timer"] = timer
    update_overlay_status(f"Hover path sent. Awaiting response ({HOVER_FALLBACK_SECONDS:.0f}s timeout).")


def log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[main {ts}] {message}")


def capture_fullscreen(target: Path) -> Path:
    """
    Capture the primary monitor into `target`. Tries mss first and falls back to
    Pillow's ImageGrab if needed.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    errors: List[Exception] = []

    try:
        from mss import mss, tools

        with mss() as sct:
            monitor = sct.monitors[0]
            raw = sct.grab(monitor)
            tools.to_png(raw.rgb, raw.size, output=str(target))
            return target
    except Exception as exc:
        errors.append(exc)
        log(f"[WARN] mss capture failed: {exc}")

    try:
        from PIL import ImageGrab

        img = ImageGrab.grab(all_screens=True)
        img.save(target)
        return target
    except Exception as exc:
        errors.append(exc)
        log(f"[WARN] ImageGrab capture failed: {exc}")

    last = errors[-1] if errors else RuntimeError("Unknown capture error")
    raise RuntimeError("Unable to capture fullscreen screenshot") from last


def prepare_hover_image(full_image: Path) -> Optional[Path]:
    """
    Create a cropped version of the screenshot for hover_bot:
    - removes HOVER_TOP_CROP pixels from the top (to skip tab bars)
    - removes 300 px from both left and right sides.
    """
    try:
        with Image.open(full_image) as im:
            width, height = im.size
            if width <= HOVER_SIDE_CROP * 2 or height <= HOVER_TOP_CROP + 10:
                cropped = im.copy()
            else:
                left = HOVER_SIDE_CROP
                right = width - HOVER_SIDE_CROP
                top = min(max(0, HOVER_TOP_CROP), height - 10)
                box = (left, top, right, height)
                cropped = im.crop(box)
        HOVER_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = HOVER_INPUT_DIR / f"{full_image.stem}_hover.png"
        cropped.save(out_path)
        return out_path
    except Exception as exc:
        log(f"[WARN] Could not prepare hover image: {exc}")
        return None


def run_region_grow(image_path: Path) -> Optional[Path]:
    """
    Run utils/region_grow.py for the provided screenshot and return the JSON
    path it generates (if any).
    """
    log(f"[INFO] Running region_grow on {image_path.name}")
    cmd = [sys.executable, str(REGION_GROW_SCRIPT), str(image_path)]
    result = subprocess.run(cmd, cwd=str(ROOT), **SUBPROCESS_KW)
    if result.returncode != 0:
        log(f"[ERROR] region_grow failed with code {result.returncode}")
        return None

    json_path = SCREEN_BOXES_DIR / f"{image_path.stem}.json"
    if not json_path.exists():
        log(f"[ERROR] Expected JSON missing: {json_path}")
        return None

    return json_path


def run_arrow_post(json_path: Path) -> None:
    """
    Opcjonalny krok po region_grow: wykrywa strzałki na obrazie na podstawie
    JSON-a ze screen_boxes i uzupełnia ten JSON o pole `triangles`.
    """
    if not ARROW_POST_SCRIPT.exists():
        return
    log(f"[INFO] Running arrow_post_region on {json_path.name}")
    cmd = [sys.executable, str(ARROW_POST_SCRIPT), str(json_path)]
    result = subprocess.run(cmd, cwd=str(ROOT), **SUBPROCESS_KW)
    if result.returncode != 0:
        log(f"[WARN] arrow_post_region failed with code {result.returncode}")


def run_rating(json_path: Path) -> bool:
    """Invoke scripts/numpy_rate/rating.py for the produced JSON."""
    log(f"[INFO] Running rating on {json_path.name}")
    cmd = [sys.executable, str(RATING_SCRIPT), str(json_path)]
    result = subprocess.run(cmd, cwd=str(ROOT), **SUBPROCESS_KW)
    if result.returncode != 0:
        log(f"[ERROR] rating failed with code {result.returncode}")
        return False
    return True


def run_hover_bot(hover_image: Path, base_name: str) -> Optional[Tuple[subprocess.Popen, Path]]:
    """Launch hover_single.py for the prepared image."""
    if not HOVER_SINGLE_SCRIPT.exists():
        log("[WARN] hover_single.py not found; skipping hover bot.")
        return None
    points_json = HOVER_OUTPUT_DIR / f"{base_name}_hover.json"
    annot_out = HOVER_OUTPUT_DIR / f"{base_name}_hover.png"
    HOVER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(HOVER_SINGLE_SCRIPT),
        "--image",
        str(hover_image),
        "--json-out",
        str(points_json),
        "--annot-out",
        str(annot_out),
    ]
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), **SUBPROCESS_KW)
        log(f"[INFO] hover_bot started for {hover_image.name}")
        update_overlay_status(f"hover_bot running ({hover_image.name})")
        return proc, points_json
    except Exception as exc:
        log(f"[WARN] Failed to launch hover bot: {exc}")
        return None


def _hover_rect(box: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _inside_any(x: float, y: float, rects: List[Tuple[int, int, int, int]]) -> bool:
    for (x0, y0, x1, y1) in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _group_hover_lines(seqs: List[Dict[str, Any]]) -> List[List[int]]:
    entries = []
    for idx, seq in enumerate(seqs):
        box = seq.get("box") or []
        if not box:
            continue
        ys = [float(p[1]) for p in box]
        min_y, max_y = min(ys), max(ys)
        height = max(1.0, max_y - min_y)
        dots = seq.get("dots") or []
        if dots:
            line_y = float(sum(d[1] for d in dots) / len(dots))
        else:
            line_y = 0.5 * (min_y + max_y)
        entries.append((idx, min_y, max_y, line_y, height))

    groups: List[List[int]] = []
    ranges: List[Tuple[float, float]] = []
    for idx, min_y, max_y, line_y, height in sorted(entries, key=lambda t: t[3]):
        placed = False
        for gi, (gmin, gmax) in enumerate(ranges):
            gc = 0.5 * (gmin + gmax)
            gh = max(1.0, gmax - gmin)
            if abs(line_y - gc) <= 0.45 * max(height, gh):
                ranges[gi] = (min(gmin, min_y), max(gmax, max_y))
                groups[gi].append(idx)
                placed = True
                break
        if not placed:
            ranges.append((min_y, max_y))
            groups.append([idx])
    return groups


def _build_hover_path(
    seqs: List[Dict[str, Any]],
    offset_x: int,
    offset_y: int,
) -> Optional[Dict[str, Any]]:
    if not seqs:
        return None

    real_indices = [i for i, s in enumerate(seqs) if float(s.get("confidence", 0.0)) >= 0.0]
    ordered_groups = _group_hover_lines([seqs[i] for i in real_indices])
    points: List[Dict[str, int]] = []
    line_jump_indices: List[int] = []
    seg_index = -1
    for group in ordered_groups:
        order = sorted(group, key=lambda i: min(p[0] for p in seqs[real_indices[i]].get("box", [[0, 0]])))
        first = True
        for local_idx in order:
            seq = seqs[real_indices[local_idx]]
            dots = seq.get("dots") or []
            if not dots:
                continue
            if first and points:
                line_jump_indices.append(max(0, seg_index))
            first = False
            for d in dots:
                seg_index += 1
                points.append(
                    {
                        "x": int(round(d[0])) + offset_x,
                        "y": int(round(d[1])) + offset_y,
                    }
                )

    if len(points) < 2:
        return None

    rects_all = [
        _hover_rect([[float(x), float(y)] for x, y in seq.get("box", [])])
        for seq in seqs
        if float(seq.get("confidence", 0.0)) >= 0.0 and seq.get("box")
    ]

    def outside_ratio(p0: Tuple[int, int], p1: Tuple[int, int], samples: int = 9) -> float:
        if samples <= 1:
            return 1.0
        out = 0
        for k in range(samples):
            t = k / (samples - 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            if not _inside_any(x, y, rects_all):
                out += 1
        return out / samples

    for i in range(len(points) - 1):
        if outside_ratio((points[i]["x"], points[i]["y"]), (points[i + 1]["x"], points[i + 1]["y"])) >= 0.8:
            line_jump_indices.append(i)

    payload = {
        "cmd": "path",
        "points": points,
        "speed": "normal",
        "min_total_ms": 0.0,
        "speed_factor": 1.0,
        "min_dt": 0.004,
        "gap_rects": [],
        "gap_boost": 3.0,
        "line_jump_indices": line_jump_indices,
        "line_jump_boost": 1.5,
    }
    return payload


def _send_udp_payload(payload: Dict[str, Any], port: int) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(payload).encode("utf-8"), ("127.0.0.1", port))
        return True
    except Exception as exc:
        log(f"[WARN] Failed to send payload to control agent: {exc}")
        return False


def _send_control_agent(payload: Dict[str, Any], port: int) -> bool:
    cmd = payload.get("cmd")
    if cmd == "path":
        points = payload.get("points") or []
        if not points:
            return False
        min_dt = float(payload.get("min_dt", 0.01))
        ok = False
        for pt in points:
            move_payload = {"cmd": "move", "x": int(pt.get("x", 0)), "y": int(pt.get("y", 0))}
            if _send_udp_payload(move_payload, port):
                ok = True
                time.sleep(max(0.0, min_dt))
        return ok
    else:
        return _send_udp_payload(payload, port)


def dispatch_hover_to_control_agent(points_json: Path) -> None:
    try:
        seqs = json.loads(points_json.read_text(encoding="utf-8"))
        if not isinstance(seqs, list):
            raise ValueError("Hover JSON must contain a list")
    except Exception as exc:
        log(f"[WARN] Could not read hover JSON {points_json}: {exc}")
        return

    payload = _build_hover_path(
        seqs,
        offset_x=HOVER_SIDE_CROP,
        offset_y=HOVER_TOP_CROP,
    )
    if not payload:
        log("[WARN] hover_bot produced insufficient points.")
        return

    if _send_control_agent(payload, CONTROL_AGENT_PORT):
        log(f"[INFO] Sent hover path ({len(payload['points'])} pts) to control agent port {CONTROL_AGENT_PORT}")
        start_hover_fallback_timer()


def finalize_hover_bot(task: Tuple[subprocess.Popen, Path]) -> None:
    proc, json_path = task
    try:
        code = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        log("[WARN] hover_bot timed out and was killed.")
        return
    if code == 0:
        log(f"[INFO] hover_bot completed ({json_path.name})")
        dispatch_hover_to_control_agent(json_path)
    else:
        log(f"[WARN] hover_bot exited with code {code}")


def send_random_click(summary_path: Path, image_path: Path) -> None:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[WARN] Could not read summary JSON {summary_path}: {exc}")
        update_overlay_status("Summary JSON missing or invalid.")
        return
    top = data.get("top_labels") or {}
    candidates = [entry for entry in top.values() if isinstance(entry, dict) and entry.get("bbox")]
    if not candidates:
        log("[WARN] Summary has no candidates for random click.")
        update_overlay_status("No candidates for random click.")
        return
    try:
        with Image.open(image_path) as im:
            screen_w, screen_h = im.size
    except Exception:
        screen_w, screen_h = (1920, 1080)
    chosen = random.choice(candidates)
    bbox = chosen.get("bbox") or []
    if not bbox or len(bbox) != 4:
        log("[WARN] Candidate without bbox for random click.")
        update_overlay_status("Candidate without bbox for random click.")
        return
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(screen_w, int(x2))
    y2 = min(screen_h, int(y2))
    if x2 - x1 <= 10 or y2 - y1 <= 10:
        log("[WARN] Bounding box too small for random click.")
        update_overlay_status("Bounding box too small for random click.")
        return
    rx_min = max(5, x1 + 5)
    rx_max = min(screen_w - 5, x2 - 5)
    ry_min = max(5, y1 + 5)
    ry_max = min(screen_h - 5, y2 - 5)
    if rx_max <= rx_min or ry_max <= ry_min:
        log("[WARN] No space to place random click inside bbox.")
        update_overlay_status("No space for random click.")
        return
    rand_x = random.randint(rx_min, rx_max)
    rand_y = random.randint(ry_min, ry_max)
    move_payload = {"cmd": "move", "x": rand_x, "y": rand_y}
    if _send_control_agent(move_payload, CONTROL_AGENT_PORT):
        cancel_hover_fallback_timer()
        log(f"[INFO] Random click sent at ({rand_x}, {rand_y}) from {summary_path.name}")
        update_overlay_status(f"Random click at ({rand_x}, {rand_y})")
    else:
        update_overlay_status("Failed to send random click.")


def start_ai_recorder(extra_args: Optional[Iterable[str]] = None) -> Optional[subprocess.Popen]:
    """Launch ai_recorder_live.py in the background."""
    if not AI_RECORDER_SCRIPT.exists():
        log(f"[WARN] ai_recorder_live not found at {AI_RECORDER_SCRIPT}")
        return None

    args = [sys.executable, str(AI_RECORDER_SCRIPT)]
    if extra_args:
        args.extend(extra_args)

    log(f"[INFO] Launching ai_recorder_live ({' '.join(args[2:]) or 'default args'})")
    return subprocess.Popen(args, cwd=str(ROOT), **SUBPROCESS_KW)


def stop_process(proc: Optional[subprocess.Popen], timeout: float = 5.0) -> None:
    """Terminate a subprocess politely and fall back to kill if needed."""
    if proc is None:
        return
    if proc.poll() is not None:
        return

    log("[INFO] Stopping ai_recorder_live...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log("[WARN] ai_recorder_live did not stop in time; killing.")
        proc.kill()


def pipeline_iteration(loop_idx: int, screenshot_prefix: str = "screen") -> None:
    name = f"{screenshot_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    screenshot_path = SCREENSHOT_DIR / name
    capture_fullscreen(screenshot_path)
    log(f"[INFO] Saved screenshot -> {screenshot_path}")
    update_overlay_status(f"Screenshot captured ({screenshot_path.name})")

    hover_task: Optional[Tuple[subprocess.Popen, Path]] = None
    hover_input = prepare_hover_image(screenshot_path)
    if hover_input:
        hover_task = run_hover_bot(hover_input, screenshot_path.stem)

    update_overlay_status("Running region_grow...")
    json_path = run_region_grow(screenshot_path)
    if not json_path:
        update_overlay_status("region_grow failed.")
        return
    update_overlay_status("region_grow done. Running rating...")
    run_arrow_post(json_path)
    if run_rating(json_path):
        summary_path = RATE_SUMMARY_DIR / f"{screenshot_path.stem}_summary.json"
        send_random_click(summary_path, screenshot_path)
        update_overlay_status("rating completed.")
    else:
        update_overlay_status("rating failed.")
    if hover_task:
        finalize_hover_bot(hover_task)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch ai_recorder_live and run OCR -> region_grow -> rating every few seconds.",
    )
    parser.add_argument("--interval", type=float, default=3.0, help="Delay between pipeline iterations.")
    parser.add_argument(
        "--loop-count",
        type=int,
        default=None,
        help="Number of iterations to run (default: infinite). Useful for testing.",
    )
    parser.add_argument(
        "--disable-recorder",
        action="store_true",
        help="Do not spawn ai_recorder_live.py (keeps only the pipeline).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run pipeline continuously without waiting for hotkey.",
    )
    parser.add_argument(
        "--recorder-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Additional args passed to ai_recorder_live.py. Use as: --recorder-args -- --url https://example.com",
    )
    return parser.parse_args()


def start_hotkey_listener(event: threading.Event) -> Optional["keyboard.Listener"]:
    try:
        from pynput import keyboard  # type: ignore
    except Exception as exc:
        log(f"[WARN] Hotkey listener unavailable (pynput import failed: {exc}). Falling back to auto mode.")
        return None

    def on_press(key):
        try:
            if key.char and key.char.lower() == "p":
                log("[INFO] Hotkey 'P' pressed — starting pipeline.")
                event.set()
                update_overlay_status("Hotkey 'P' pressed — pipeline starting.")
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    log("[INFO] Hotkey listener active — press 'P' to start pipeline iteration.")
    update_overlay_status("Ready. Press 'P' to start.")
    return listener


def main() -> None:
    args = parse_args()
    recorder_proc = None

    trigger_event = threading.Event()
    hotkey_listener = None
    overlay = StatusOverlay()
    overlay.start()
    globals()["_status_overlay"] = overlay
    update_overlay_status("Initializing pipeline...")

    if not args.disable_recorder:
        recorder_proc = start_ai_recorder(args.recorder_args)

    if not args.auto:
        hotkey_listener = start_hotkey_listener(trigger_event)
        if hotkey_listener is None:
            args.auto = True
            update_overlay_status("Auto mode active.")
    else:
        update_overlay_status("Auto mode active.")

    log(
        f"[INFO] Pipeline start (interval={args.interval}s"
        + (f", max_loops={args.loop_count}" if args.loop_count else ", continuous")
        + ")"
    )

    try:
        loop_idx = 0
        while True:
            loop_idx += 1
            if not args.auto:
                log(f"[INFO] Waiting for hotkey 'P' to start iteration #{loop_idx}...")
                update_overlay_status(f"Waiting for 'P' (iteration {loop_idx})")
                trigger_event.wait()
                trigger_event.clear()
            cancel_hover_fallback_timer()
            log(f"[INFO] Iteration {loop_idx} start")
            update_overlay_status(f"Iteration {loop_idx} started")
            iter_start = time.perf_counter()
            try:
                pipeline_iteration(loop_idx)
            except Exception as exc:
                log(f"[ERROR] Pipeline iteration failed: {exc}")
            if args.loop_count and loop_idx >= args.loop_count:
                break
            if args.auto:
                elapsed = time.perf_counter() - iter_start
                delay = max(0.0, args.interval - elapsed)
                if delay:
                    time.sleep(delay)
    except KeyboardInterrupt:
        log("[INFO] Stopped by user.")
    finally:
        stop_process(recorder_proc)
        if hotkey_listener is not None:
            try:
                hotkey_listener.stop()
            except Exception:
                pass
        cancel_hover_fallback_timer()
        if overlay:
            overlay.stop()


if __name__ == "__main__":
    main()
