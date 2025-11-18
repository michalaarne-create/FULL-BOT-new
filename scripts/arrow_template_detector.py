#!/usr/bin/env python3
"""
Lekka detekcja strzałek (trójkątnych wskaźników) z użyciem wielu
syntetycznych tekstur szablonowych + OpenCV `matchTemplate`.

- generuje ~150 wariantów strzałek (wypełnione i obrysowane),
  w kilku skalach i rotacjach;
- wyszukuje najlepsze dopasowania w obrazie w trybie TM_CCOEFF_NORMED;
- filtruje wyniki prostym NMS i zapisuje podgląd z zaznaczonymi strzałkami.

Przykład:
    python scripts/arrow_template_detector.py --image data/screen/raw screen/\"Zrzut ekranu 2025-10-25 163249.png\"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

DATA_SCREEN_DIR = Path(__file__).resolve().parents[1] / "data" / "screen"
RAW_SCREEN_DIR = DATA_SCREEN_DIR / "raw screen"
TRIANGLE_OUTPUT_DIR = DATA_SCREEN_DIR / "numpy_triangles"


@dataclass
class ArrowTemplate:
    kernel: np.ndarray
    angle: float
    scale: float
    kind: str  # "filled" lub "outline"


def preprocess_frame(frame: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = 1.0
    if max_dim and max_dim > 0:
        largest = max(h, w)
        if largest > max_dim:
            scale = max_dim / float(largest)
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame, scale


def render_base_arrow(size: int, angle_deg: float, scale: float, kind: str) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    base_len = size * 0.55 * scale
    half_height = size * 0.20 * scale
    pts = np.array(
        [
            [-base_len * 0.2, -half_height],
            [-base_len * 0.2, half_height],
            [base_len * 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    M = cv2.getRotationMatrix2D((0.0, 0.0), angle_deg, 1.0)
    pts_rot = (pts @ M[:, :2].T) + np.array([size / 2.0, size / 2.0], dtype=np.float32)
    pts_i32 = pts_rot.astype(np.int32)
    if kind == "filled":
        cv2.fillConvexPoly(img, pts_i32, 255)
    else:
        cv2.polylines(img, [pts_i32], True, 255, thickness=2, lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def generate_arrow_templates() -> List[ArrowTemplate]:
    templates: List[ArrowTemplate] = []
    base_sizes = (13, 19)
    angles = np.linspace(0.0, 360.0, 18, endpoint=False)  # 20° krok
    scales = (0.8, 1.0, 1.2)
    kinds = ("filled", "outline")
    for size in base_sizes:
        for angle in angles:
            for scale in scales:
                for kind in kinds:
                    kernel = render_base_arrow(size, angle, scale, kind)
                    templates.append(
                        ArrowTemplate(kernel=kernel, angle=float(angle), scale=float(scale), kind=kind)
                    )
    return templates


def rect_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter) / max(1.0, float(area_a + area_b - inter))


def nms_hits(hits: List[dict], iou_thresh: float = 0.3) -> List[dict]:
    hits_sorted = sorted(hits, key=lambda h: h["score"], reverse=True)
    kept: List[dict] = []
    for h in hits_sorted:
        hb = tuple(h["bbox"])
        if all(rect_iou(hb, tuple(k["bbox"])) < iou_thresh for k in kept):
            kept.append(h)
    return kept


def detect_arrows(
    image: np.ndarray,
    templates: List[ArrowTemplate],
    threshold: float,
    min_mean_intensity: float = 0.0,
) -> List[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    hits: List[dict] = []
    for tpl in templates:
        kernel = tpl.kernel
        if kernel.shape[0] > gray.shape[0] or kernel.shape[1] > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, kernel, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val < threshold:
            continue
        x, y = max_loc
        h, w = kernel.shape
        bbox = [x, y, x + w, y + h]
        patch = gray[y : y + h, x : x + w]
        mean_val = float(cv2.mean(patch)[0])
        if mean_val < min_mean_intensity:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        hits.append(
            {
                "bbox": bbox,
                "center": (cx, cy),
                "score": float(max_val),
                "angle": tpl.angle,
                "scale": tpl.scale,
                "kind": tpl.kind,
                "mean": mean_val,
            }
        )
    return nms_hits(hits)


def draw_hits(image: np.ndarray, hits: List[dict]) -> np.ndarray:
    canvas = image.copy()
    for h in hits:
        x1, y1, x2, y2 = map(int, h["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        cx, cy = h["center"]
        label = f"{h['angle']:.0f}deg | s={h['score']:.2f}"
        cv2.putText(
            canvas,
            label,
            (int(cx) + 4, int(cy) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detekcja strzałek przez dopasowanie do wielu szablonów.")
    p.add_argument("--image", required=True, help="Ścieżka do obrazu.")
    p.add_argument(
        "--max-dim",
        type=int,
        default=0,
        help="Maksymalna długość boku po skalowaniu (0 = pełna rozdzielczość).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Próg dopasowania matchTemplate (TM_CCOEFF_NORMED).",
    )
    p.add_argument(
        "--min-mean",
        type=float,
        default=180.0,
        help="Minimalna średnia jasność w oknie (odfiltruje strzałki na ciemnym tle).",
    )
    p.add_argument("--output", help="Ścieżka do zapisu obrazu z zaznaczonymi strzałkami.")
    p.add_argument("--no-gui", action="store_true", help="Nie pokazuj okna podglądu.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    img_path = Path(args.image)
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise SystemExit(f"Nie mogę otworzyć pliku: {img_path}")
    frame, scale = preprocess_frame(frame, args.max_dim)

    templates = generate_arrow_templates()
    hits = detect_arrows(
        frame,
        templates,
        threshold=args.threshold,
        min_mean_intensity=args.min_mean,
    )

    print(f"Skala obrazu: {scale:.2f}")
    print(f"Znaleziono {len(hits)} strzałek (próg={args.threshold:.2f}).")
    for i, h in enumerate(hits, 1):
        x1, y1, x2, y2 = h["bbox"]
        cx, cy = h["center"]
        print(
            f"#{i}: score={h['score']:.3f}, angle={h['angle']:.1f}deg, kind={h['kind']}, "
            f"bbox=({x1},{y1},{x2},{y2}), center=({cx:.1f},{cy:.1f})"
        )

    overlay = draw_hits(frame, hits)
    if args.output:
        out_path = Path(args.output)
    else:
        try:
            resolved = img_path.resolve()
        except FileNotFoundError:
            resolved = img_path
        if RAW_SCREEN_DIR in resolved.parents:
            target_dir = TRIANGLE_OUTPUT_DIR
        else:
            target_dir = img_path.parent
        out_path = target_dir / f"{img_path.stem}_arrows{img_path.suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"Zapisano obraz wynikowy: {out_path}")

    if not args.no_gui:
        cv2.imshow("arrows", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
