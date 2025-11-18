#!/usr/bin/env python3
from __future__ import annotations
"""
Synthetic YOLO dataset generator for dropdown headers (heavy augmentation).

Classes:
  0 = header (the full clickable bar)
  1 = caret  (small down arrow on the right side)

The generator renders multiple synthetic headers per image, then applies
heavy appearance and geometric augmentations to break template bias.

Usage:
  python scripts/yolo_dropdown/gen_header_dataset.py --out data/yolo_header \
         --train 8000 --val 1600 --seed 42

This will create YOLO-format folders and a data.yaml in the output dir.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


IMG_W, IMG_H = 640, 480


def _rand_color(lo=180, hi=255) -> Tuple[int, int, int]:
    return (
        int(random.randint(lo, hi)),
        int(random.randint(lo, hi)),
        int(random.randint(lo, hi)),
    )


def _draw_linear_gradient(img: np.ndarray, x: int, y: int, w: int, h: int,
                          c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> None:
    if w <= 0 or h <= 0:
        return
    band = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    c1v = np.array(c1, np.float32)[None, :]
    c2v = np.array(c2, np.float32)[None, :]
    line = (c1v * (1.0 - band) + c2v * band)
    block = np.repeat(line[:, None, :], w, axis=1).astype(np.uint8)
    img[y:y+h, x:x+w] = block


def _render_header(canvas: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """Render a single header with border and caret; return (header_box, caret_box)."""
    # background
    bg1 = _rand_color(220, 245)
    bg2 = _rand_color(200, 235)
    _draw_linear_gradient(canvas, x, y, w, h, bg1, bg2)
    # border
    border_col = tuple(int(c*0.55) for c in bg1)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), border_col, 2)
    # text
    text = random.choice(["Gender", "Select", "Options", "Choose", "Dropdown"]) + f" {random.randint(1,99)}"
    font = random.choice([
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN,
    ])
    scale = random.uniform(0.5, 0.8)
    thick = random.randint(1, 2)
    tsize, _ = cv2.getTextSize(text, font, scale, thick)
    tx = x + random.randint(8, 16)
    ty = y + h//2 + tsize[1]//2 - 2
    cv2.putText(canvas, text, (tx, ty), font, scale, (30, 30, 30), thick, cv2.LINE_AA)
    # caret
    caret_w = max(8, int(w*0.04))
    caret_h = max(8, int(h*0.45))
    cx = x + w - caret_w - random.randint(10, 18)
    cy = y + h//2 - caret_h//2
    pts = np.array([[cx, cy], [cx+caret_w, cy], [cx+caret_w//2, cy+caret_h]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, pts, (40, 40, 40))
    header_box = (x, y, w, h)
    caret_box = (cx, cy, caret_w, caret_h)
    return header_box, caret_box


def _rand_bg(canvas: np.ndarray) -> None:
    # base gradient
    c1 = _rand_color(220, 255)
    c2 = _rand_color(220, 255)
    _draw_linear_gradient(canvas, 0, 0, canvas.shape[1], canvas.shape[0], c1, c2)
    # add light noise
    noise = np.random.normal(0, 6, canvas.shape).astype(np.int16)
    tmp = canvas.astype(np.int16) + noise
    np.clip(tmp, 0, 255, out=tmp)
    canvas[:] = tmp.astype(np.uint8)


def _augment(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    # random brightness/contrast
    alpha = random.uniform(0.7, 1.3)
    beta = random.uniform(-25, 25)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # blur or sharpen
    if random.random() < 0.4:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), random.uniform(0.5, 1.2))
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)
    # mild perspective
    if random.random() < 0.6:
        src = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
        dx = random.uniform(-0.04, 0.04)*W
        dy = random.uniform(-0.04, 0.04)*H
        dst = np.float32([[dx,dy],[W-1-dx,0+dy],[W-1+dx,H-1-dy],[0-dx,H-1-dy]])
        M = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # jpeg artifacts (simulate)
    if random.random() < 0.5:
        enc = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 90)])[1]
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def _yolo_line(cls: int, x: int, y: int, w: int, h: int) -> str:
    cx = (x + w/2) / IMG_W
    cy = (y + h/2) / IMG_H
    nw = w / IMG_W
    nh = h / IMG_H
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"


def generate_one(rng: random.Random) -> Tuple[np.ndarray, List[str]]:
    canvas = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8)
    _rand_bg(canvas)
    labels: List[str] = []
    # how many headers per image
    n = rng.randint(1, 4)
    for _ in range(n):
        w = rng.randint(180, 360)
        h = rng.randint(26, 42)
        x = rng.randint(20, IMG_W - w - 20)
        y = rng.randint(20, IMG_H - h - 20)
        hb, cb = _render_header(canvas, x, y, w, h)
        labels.append(_yolo_line(0, hb[0], hb[1], hb[2], hb[3]))
        labels.append(_yolo_line(1, cb[0], cb[1], cb[2], cb[3]))

    # global augment after composition
    canvas = _augment(canvas)
    return canvas, labels


def write_split(root: Path, count: int, seed: int) -> None:
    rng = random.Random(seed)
    img_dir = root / 'images'
    lbl_dir = root / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img, lines = generate_one(rng)
        stem = f"{i:06d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
        with open(lbl_dir / f"{stem}.txt", 'w', encoding='utf-8') as f:
            for ln in lines:
                f.write(ln)


def main() -> None:
    ap = argparse.ArgumentParser(description='Generate YOLO dataset for dropdown headers')
    ap.add_argument('--out', type=str, default='data/yolo_header')
    ap.add_argument('--train', type=int, default=8000)
    ap.add_argument('--val', type=int, default=1600)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    write_split(out / 'train', args.train, args.seed)
    write_split(out / 'val', args.val, args.seed + 1)
    # Write data.yaml with absolute paths for robustness
    train_images = (out / 'train' / 'images').resolve()
    val_images = (out / 'val' / 'images').resolve()
    yaml_txt = '\n'.join([
        f"train: {train_images.as_posix()}",
        f"val: {val_images.as_posix()}",
        'names:',
        '  0: header',
        '  1: caret',
    ])
    (out / 'data.yaml').write_text(yaml_txt, encoding='utf-8')
    print(f"Dataset ready at {out}. Train={args.train}, Val={args.val}")


if __name__ == '__main__':
    main()
