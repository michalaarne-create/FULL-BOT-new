from __future__ import annotations

"""
Synthetic YOLO dataset generator for a simple dropdown task (uniform background).

Classes (small, action-friendly):
  0 = header         (closed dropdown header area)
  1 = panel          (open list panel area)
  2 = last_option    (click target: last visible option row; only when at bottom)
  3 = bottom_flag    (small marker at panel bottom; present only when scrolled to end)

We generate minimal graphics (rectangles) so the detector can quickly latch onto
the geometry. Later, you can curriculum-tune with richer styles.

Notes for robustness:
 - ensure a solid fraction of bottom states (configurable) so the model learns
   to stop scrolling and click.
 - keep bottom_flag big enough (>= 3-4 px) to survive 640 downsizing.
 - last_option is labeled only when at_bottom and is the last visible row rect.
"""

import argparse
from pathlib import Path
import random
from typing import Tuple, List

import cv2
import numpy as np


IMG_W, IMG_H = 640, 480


def _clip_box(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(IMG_W - 1, x))
    y = max(0, min(IMG_H - 1, y))
    w = max(1, min(IMG_W - x, w))
    h = max(1, min(IMG_H - y, h))
    return x, y, w, h


def _yolo_line(cls: int, x: int, y: int, w: int, h: int) -> str:
    cx = (x + w / 2.0) / IMG_W
    cy = (y + h / 2.0) / IMG_H
    nw = w / IMG_W
    nh = h / IMG_H
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"


def _draw_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color=(64, 64, 64), fill=(230, 230, 230)):
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def generate_one(
    rng: random.Random,
    *,
    open_prob: float = 0.7,
    bottom_prob: float = 0.35,
    flag_min_px: int = 3,
) -> Tuple[np.ndarray, List[str]]:
    img = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 255  # uniform background
    labels: List[str] = []

    # Random dropdown geometry
    hdr_w = rng.randint(160, 280)
    hdr_h = rng.randint(28, 36)
    hdr_x = rng.randint(20, IMG_W - hdr_w - 20)
    hdr_y = rng.randint(20, IMG_H - 200)

    # Draw header and label class 0
    _draw_rect(img, hdr_x, hdr_y, hdr_w, hdr_h)
    labels.append(_yolo_line(0, *(_clip_box(hdr_x, hdr_y, hdr_w, hdr_h))))

    is_open = rng.random() < open_prob
    if not is_open:
        return img, labels

    # Panel just below header
    opt_h = rng.randint(24, 30)
    vis_rows = rng.randint(4, 5)
    total_rows = rng.randint(vis_rows + 2, vis_rows + 10)
    panel_x, panel_y = hdr_x, hdr_y + hdr_h
    panel_w, panel_h = hdr_w, vis_rows * opt_h
    panel_x, panel_y, panel_w, panel_h = _clip_box(panel_x, panel_y, panel_w, panel_h)
    _draw_rect(img, panel_x, panel_y, panel_w, panel_h, color=(40, 40, 40), fill=(245, 245, 245))
    labels.append(_yolo_line(1, panel_x, panel_y, panel_w, panel_h))

    # Scroll state
    max_scroll = max(0, total_rows - vis_rows)
    # choose bottom with configured probability
    if max_scroll > 0 and (rng.random() < bottom_prob):
        scroll_idx = max_scroll
    else:
        scroll_idx = rng.randint(0, max_scroll) if max_scroll > 0 else 0
    at_bottom = (scroll_idx == max_scroll)

    # bottom_flag marker only if at bottom
    if at_bottom:
        # Draw a thin bottom bar for visual context (not labeled)
        bar_h = max(flag_min_px, int(0.06 * opt_h))
        bx = panel_x + 2
        bw = panel_w - 4
        by = panel_y + panel_h - bar_h - 2
        bx, by, bw, bh = _clip_box(bx, by, bw, bar_h)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (128, 64, 0), -1)

        # Label a more robust square flag marker at bottom-right
        sq = max(flag_min_px, int(0.5 * opt_h))  # thicker, easier for detector
        fx = panel_x + panel_w - sq - 4
        fy = panel_y + panel_h - sq - 4
        fx, fy, fw, fh = _clip_box(fx, fy, sq, sq)
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (200, 120, 20), -1)
        labels.append(_yolo_line(3, fx, fy, fw, fh))

        # last option (click target) is the last visible row rect
        lx, ly = panel_x, panel_y + (vis_rows - 1) * opt_h
        lw, lh = panel_w, opt_h
        lx, ly, lw, lh = _clip_box(lx, ly, lw, lh)
        # Draw a light highlight
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (220, 220, 220), -1)
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (160, 160, 160), 1)
        labels.append(_yolo_line(2, lx, ly, lw, lh))

    return img, labels


def generate_split(out_root: Path, n_train: int, n_val: int, seed: int = 123,
                   open_prob: float = 0.7, bottom_prob: float = 0.35, flag_min_px: int = 3) -> None:
    rng = random.Random(seed)
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = out_root / split / "images"
        lbl_dir = out_root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, lines = generate_one(rng, open_prob=open_prob, bottom_prob=bottom_prob, flag_min_px=flag_min_px)
            stem = f"{i:06d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
            with open(lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/yolo_dropdown")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--open-prob", type=float, default=0.7, help="Probability dropdown is open")
    ap.add_argument("--bottom-prob", type=float, default=0.35, help="Probability of bottom among open examples")
    ap.add_argument("--flag-min-px", type=int, default=3, help="Min px height of bottom flag marker")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    generate_split(out_root, n_train=args.train, n_val=args.val, seed=args.seed,
                   open_prob=args.open_prob, bottom_prob=args.bottom_prob, flag_min_px=args.flag_min_px)
    # Write data.yaml with absolute paths for YOLO robustness
    train_images = (out_root / "train" / "images").resolve()
    val_images = (out_root / "val" / "images").resolve()
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        (
            f"train: {train_images.as_posix()}\n"
            f"val: {val_images.as_posix()}\n"
            "names:\n"
            "  0: header\n"
            "  1: panel\n"
            "  2: last_option\n"
            "  3: bottom_flag\n"
        ),
        encoding="utf-8",
    )
    print(f"Dataset ready at {out_root}. Train={args.train}, Val={args.val}")


if __name__ == "__main__":
    main()
