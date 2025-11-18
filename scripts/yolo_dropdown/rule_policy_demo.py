from __future__ import annotations

"""
Demo: rule-based action policy on top of trained YOLO detector.

Reads an image (from the synthetic dataset) and prints the action the agent
would take based on detections:
  - If no panel -> click header center (to open)
  - If panel and no bottom_flag -> scroll down
  - If bottom_flag -> click last_option center

Run:
  python scripts/yolo_dropdown/rule_policy_demo.py --weights runs/detect/train/weights/best.pt \
         --image data/yolo_dropdown/val/images/000001.jpg
"""

import argparse
from ultralytics import YOLO
import cv2


CLS_HEADER = 0
CLS_PANEL = 1
CLS_LAST_OPTION = 2
CLS_BOTTOM_FLAG = 3


def _center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--flag-conf", type=float, default=0.15)
    args = ap.parse_args()

    model = YOLO(args.weights)
    im = cv2.imread(args.image)
    if im is None:
        raise RuntimeError(f"Failed to read {args.image}")
    res = model.predict(source=im, imgsz=640, conf=args.conf, verbose=False)[0]

    has_panel = False
    has_bottom = False
    header_box = None
    last_box = None

    for b in res.boxes:
        cls = int(b.cls.item())
        conf = float(b.conf.item()) if hasattr(b, 'conf') else 1.0
        xyxy = b.xyxy[0].tolist()  # x1,y1,x2,y2
        x1, y1, x2, y2 = map(int, xyxy)
        if cls == CLS_PANEL and conf >= args.conf:
            has_panel = True
            panel_box = (x1, y1, x2, y2)
        elif cls == CLS_BOTTOM_FLAG and conf >= args.flag_conf:
            has_bottom = True
        elif cls == CLS_HEADER and conf >= args.conf:
            header_box = (x1, y1, x2, y2)
        elif cls == CLS_LAST_OPTION and conf >= args.conf:
            last_box = (x1, y1, x2, y2)

    if not has_panel:
        if header_box:
            x, y = _center(*header_box)
            print(f"ACTION: click header at ({x},{y})")
        else:
            print("ACTION: click (header not detected, fallback to screen center)")
    else:
        if not has_bottom:
            print("ACTION: scroll down")
        else:
            if last_box:
                x, y = _center(*last_box)
                print(f"ACTION: click last option at ({x},{y})")
            else:
                # Fallback: bottom of panel
                if 'panel_box' in locals():
                    px1, py1, px2, py2 = panel_box
                    x = (px1 + px2) // 2
                    y = int(py2 - 0.1 * (py2 - py1))
                    print(f"ACTION: click bottom panel row (~last) at ({x},{y})")
                else:
                    print("ACTION: click (last option not detected; at bottom)")


if __name__ == "__main__":
    main()
