#!/usr/bin/env python3
from __future__ import annotations
"""
Validate YOLO dropdown dataset labels for logical consistency and distribution.

Checks:
 - last_option (2) only when bottom_flag (3) present
 - last_option box lies within panel (1)
 - header (0) above panel (1) (y1_header < y1_panel)
 - ensure minimum fraction of bottom frames
 - warn on very small bottom_flag height (< 3 px)

Usage:
  python scripts/yolo_dropdown/validate_dataset.py --root data/yolo_dropdown --split train --min-bottom-frac 0.2
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2


def parse_yolo_line(line: str, img_w: int, img_h: int) -> Tuple[int, Tuple[int,int,int,int]]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid label line: {line}")
    cls = int(float(parts[0]))
    cx = float(parts[1]) * img_w
    cy = float(parts[2]) * img_h
    w = float(parts[3]) * img_w
    h = float(parts[4]) * img_h
    x1 = int(cx - w / 2.0)
    y1 = int(cy - h / 2.0)
    return cls, (x1, y1, int(w), int(h))


def box_inside(inner: Tuple[int,int,int,int], outer: Tuple[int,int,int,int]) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return ix >= ox and iy >= oy and (ix + iw) <= (ox + ow) and (iy + ih) <= (oy + oh)


def validate_split(root: Path, split: str, min_bottom_frac: float) -> None:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    errors: List[str] = []
    n = 0
    bottom_count = 0
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        stem = lbl_path.stem
        img_path = (img_dir / f"{stem}.jpg")
        if not img_path.exists():
            errors.append(f"[missing-image] {img_path}")
            continue
        im = cv2.imread(str(img_path))
        if im is None:
            errors.append(f"[read-fail] {img_path}")
            continue
        H, W = im.shape[:2]
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        boxes = {0: [], 1: [], 2: [], 3: []}
        for ln in lines:
            try:
                c, box = parse_yolo_line(ln, W, H)
            except Exception as e:
                errors.append(f"[parse] {lbl_path}: {e}")
                continue
            if c in boxes:
                boxes[c].append(box)

        n += 1
        has_panel = len(boxes[1]) > 0
        has_last = len(boxes[2]) > 0
        has_bottom = len(boxes[3]) > 0
        if has_bottom:
            bottom_count += 1

        # Rule: last_option only when bottom_flag present
        if has_last and not has_bottom:
            errors.append(f"[rule] last_option present without bottom_flag: {lbl_path}")

        # Rule: last_option inside panel
        if has_last and has_panel:
            p = boxes[1][0]
            for lo in boxes[2]:
                if not box_inside(lo, p):
                    errors.append(f"[rule] last_option not inside panel: {lbl_path}")

        # Rule: header above panel (if both exist)
        if has_panel and len(boxes[0]) > 0:
            hx, hy, hw, hh = boxes[0][0]
            px, py, pw, ph = boxes[1][0]
            if not (hy + hh <= py):
                errors.append(f"[rule] header not above panel: {lbl_path}")

        # Warn: bottom_flag tiny
        for fx, fy, fw, fh in boxes[3]:
            if fh < 3:
                errors.append(f"[warn] tiny bottom_flag h={fh}px: {lbl_path}")

    frac = (bottom_count / max(1, n))
    if frac < min_bottom_frac:
        errors.append(f"[dist] bottom_frac too low: {frac:.3f} < {min_bottom_frac:.3f}")

    if errors:
        print("Validation FAIL:")
        for e in errors[:200]:
            print(" -", e)
        print(f"Total errors: {len(errors)} over {n} files. bottom_frac={frac:.3f}")
        raise SystemExit(1)
    print(f"Validation OK. Files={n}, bottom_frac={frac:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate YOLO dropdown dataset")
    ap.add_argument("--root", type=str, default="data/yolo_dropdown")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--min-bottom-frac", type=float, default=0.2)
    args = ap.parse_args()
    validate_split(Path(args.root), args.split, args.min_bottom_frac)


if __name__ == "__main__":
    main()

