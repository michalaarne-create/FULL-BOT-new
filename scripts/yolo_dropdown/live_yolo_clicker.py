#!/usr/bin/env python3
from __future__ import annotations
"""
Live YOLO clicker: scrolls to bottom and clicks last option.

Policy:
  - If no panel -> click header center
  - If panel and no bottom_flag -> scroll down (small ticks) and re-detect
  - If bottom_flag -> click last_option center (or bottom row of panel as fallback)

Usage:
  python scripts/yolo_dropdown/live_yolo_clicker.py --weights runs_dropdown/train/weights/best.pt --conf 0.25 --flag-conf 0.15
"""
import argparse
from pathlib import Path
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from pynput.mouse import Controller, Button


CLS_HEADER = 0
CLS_PANEL = 1
CLS_LAST_OPTION = 2
CLS_BOTTOM_FLAG = 3


def center_of_box(xyxy: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) // 2, (y1 + y2) // 2


def pick_boxes(res, conf: float, flag_conf: float):
    has_panel = False
    flag = None
    header = None
    last = None
    panel = None
    for b in res.boxes:
        c = int(b.cls.item())
        score = float(b.conf.item()) if hasattr(b, 'conf') else 1.0
        xyxy = b.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        if c == CLS_PANEL and score >= conf:
            has_panel = True
            panel = (x1, y1, x2, y2)
        elif c == CLS_BOTTOM_FLAG and score >= flag_conf:
            flag = (x1, y1, x2, y2)
        elif c == CLS_HEADER and score >= conf:
            header = (x1, y1, x2, y2)
        elif c == CLS_LAST_OPTION and score >= conf:
            last = (x1, y1, x2, y2)
    return has_panel, panel, flag, header, last


def fallback_last_from_panel(panel: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = panel
    cx = (x1 + x2) // 2
    cy = int(y2 - (y2 - y1) * 0.1)
    return cx, cy


def main() -> None:
    ap = argparse.ArgumentParser(description="Live YOLO clicker for dropdowns")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--conf", type=float, default=0.25, help="General confidence threshold")
    ap.add_argument("--flag-conf", type=float, default=0.15, help="Bottom flag confidence threshold")
    ap.add_argument("--interval", type=float, default=0.2, help="Delay between steps")
    ap.add_argument("--scroll-ticks", type=int, default=4, help="Mouse scroll ticks per step")
    ap.add_argument("--max-steps", type=int, default=30, help="Max scroll/detect loops before giving up")
    ap.add_argument("--source", type=str, default=None, help="Optional image path for offline test mode")
    args = ap.parse_args()

    model = YOLO(args.weights)
    mouse = Controller()

    def capture_frame() -> np.ndarray:
        if args.source:
            im = cv2.imread(args.source)
            if im is None:
                raise RuntimeError(f"Failed to read source image: {args.source}")
            return im
        # fallback to screen grab if mss available
        try:
            import mss
            with mss.mss() as sct:
                mon = sct.monitors[0]
                raw = sct.grab(mon)
                frame = cv2.cvtColor(np.array(raw, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
                return frame
        except Exception as e:
            raise RuntimeError("No screen capture available. Provide --source image.") from e

    steps = 0
    while steps < args.max_steps:
        frame = capture_frame()
        res = model.predict(source=frame, imgsz=640, conf=args.conf, verbose=False)[0]
        has_panel, panel, flag, header, last = pick_boxes(res, args.conf, args.flag_conf)

        if not has_panel:
            if header is None:
                print("[Policy] No panel, no header -> idle")
                time.sleep(args.interval)
                steps += 1
                continue
            x, y = center_of_box(header)
            print(f"[Policy] Click header at ({x},{y})")
            mouse.position = (x, y)
            mouse.click(Button.left, 1)
            time.sleep(args.interval)
            steps += 1
            continue

        # Panel exists
        if flag is None:
            print(f"[Policy] Scroll down ({args.scroll_ticks})")
            mouse.scroll(0, -args.scroll_ticks)
            time.sleep(args.interval)
            steps += 1
            continue

        # At bottom
        if last is not None:
            x, y = center_of_box(last)
        else:
            # Fallback: click bottom row region inside panel
            x, y = fallback_last_from_panel(panel)
        print(f"[Policy] Click last option at ({x},{y})")
        mouse.position = (x, y)
        mouse.click(Button.left, 1)
        print("[Done]")
        return

    print("[Abort] Max steps reached without completion.")


if __name__ == "__main__":
    main()

