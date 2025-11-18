#!/usr/bin/env python3
"""
Wykrywa trójkąty o zbliżonym położeniu kątowym (równoległe +-20%) na obrazach
lub w strumieniu wideo. Pipeline jest lekki obliczeniowo: ostre skalowanie,
Canny + NumPy, filtrowanie konturów i szybka ocena równoległości.

Przykład:
    python scripts/triangle_finder.py --image data/example.png --output out.png
    python scripts/triangle_finder.py --camera 0 --no-gui
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


DATA_SCREEN_DIR = Path(__file__).resolve().parents[1] / "data" / "screen"
RAW_SCREEN_DIR = DATA_SCREEN_DIR / "raw screen"
TRIANGLE_OUTPUT_DIR = DATA_SCREEN_DIR / "numpy_triangles"
DEFAULT_IMAGE_PATH = str(RAW_SCREEN_DIR / "Zrzut ekranu 2025-10-25 163249.png")


@dataclass
class Triangle:
    points: np.ndarray
    centroid: Tuple[float, float]
    orientation: float  # degrees in [0, 180)
    area: float
    score: float


class OrientationGate:
    """Zapamiętuje kierunek pierwszego dobrego trójkąta i filtruje kolejne."""

    def __init__(self, tolerance_deg: float) -> None:
        self.tolerance = tolerance_deg
        self.reference: Optional[float] = None

    def keep(self, angle_deg: float) -> bool:
        if self.reference is None:
            self.reference = angle_deg
            return True
        diff = abs(angle_deg - self.reference) % 180.0
        diff = min(diff, 180.0 - diff)
        return diff <= self.tolerance


def preprocess_frame(frame: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Szybkie skalowanie aby nie przekraczać limitu wielkości."""
    h, w = frame.shape[:2]
    scale = 1.0
    if max_dim and max_dim > 0:
        largest = max(h, w)
        if largest > max_dim:
            scale = max_dim / float(largest)
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame, scale


def adaptive_canny(gray: np.ndarray, k: float = 1.33) -> np.ndarray:
    """Lekka heurystyka progu wykorzystująca medianę pikseli."""
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - k * 0.3) * median))
    upper = int(min(255, (1.0 + k * 0.3) * median))
    return cv2.Canny(gray, lower, upper, L2gradient=False)


def edge_density_mask(edges: np.ndarray) -> float:
    return float(np.count_nonzero(edges)) / float(edges.size)


def enhance_edges(edges: np.ndarray) -> np.ndarray:
    """Domyka kontury dzięki prostej morfologii (ważne dla trójkątów UI)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated


def polygon_from_contour(contour: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.045 * peri, True)
    if len(approx) != 3 or not cv2.isContourConvex(approx):
        return None
    return approx.reshape(-1, 2).astype(np.float32)


def triangle_orientation(points: np.ndarray) -> float:
    shifted = np.roll(points, -1, axis=0)
    vecs = shifted - points
    lengths = np.linalg.norm(vecs, axis=1)
    longest = vecs[np.argmax(lengths)]
    angle = np.degrees(np.arctan2(longest[1], longest[0])) % 180.0
    return angle


def triangle_score(points: np.ndarray, edge_map: np.ndarray) -> float:
    mask = np.zeros(edge_map.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, points.astype(np.int32), 255)
    overlap = np.count_nonzero(cv2.bitwise_and(mask, edge_map))
    perimeter = cv2.arcLength(points.reshape(-1, 1, 2).astype(np.float32), True)
    return float(overlap) / (perimeter + 1e-3)


def triangle_fit_ratio(contour: np.ndarray) -> float:
    if contour is None or len(contour) < 3:
        return 0.0
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    try:
        enclosing_area, _ = cv2.minEnclosingTriangle(contour)
    except cv2.error:
        enclosing_area = 0.0
    if enclosing_area <= 0:
        return 0.0
    return float(area) / float(enclosing_area)


def output_paths_for_image(image_path: str) -> Tuple[Path, Path]:
    image_file = Path(image_path)
    try:
        resolved = image_file.resolve()
    except FileNotFoundError:
        resolved = image_file
    if RAW_SCREEN_DIR in resolved.parents:
        target_dir = TRIANGLE_OUTPUT_DIR
    else:
        target_dir = image_file.parent
    annotated = target_dir / f"{image_file.stem}_triangle{image_file.suffix}"
    json_file = target_dir / f"{image_file.stem}_triangle.json"
    return annotated, json_file


def save_triangles_json(
    triangles: Iterable[Triangle], json_path: Path, image_path: str, scale: float
) -> None:
    inv_scale = 1.0 if scale == 0 else 1.0 / scale
    payload = {
        "image": str(Path(image_path)),
        "generated_at": time.time(),
        "scale_used": scale,
        "triangles": [],
    }
    for tri in triangles:
        scaled_points = [[float(x), float(y)] for x, y in tri.points]
        original_points = [[float(x * inv_scale), float(y * inv_scale)] for x, y in tri.points]
        payload["triangles"].append(
            {
                "points_scaled": scaled_points,
                "points_original": original_points,
                "centroid_scaled": {"x": float(tri.centroid[0]), "y": float(tri.centroid[1])},
                "centroid_original": {
                    "x": float(tri.centroid[0] * inv_scale),
                    "y": float(tri.centroid[1] * inv_scale),
                },
                "orientation_deg": float(tri.orientation),
                "area_px2_scaled": float(tri.area),
                "score": float(tri.score),
            }
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_triangles(
    frame: np.ndarray,
    min_area: float,
    min_side_px: float,
    density_range: Tuple[float, float],
    orientation_gate: OrientationGate,
    max_intensity_std: float,
    min_fit_ratio: float,
) -> Tuple[List[Triangle], np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = adaptive_canny(gray)

    density = edge_density_mask(edges)
    if not (density_range[0] <= density <= density_range[1]):
        return [], edges

    processed_edges = enhance_edges(edges)

    triangles: List[Triangle] = []
    contours, _ = cv2.findContours(processed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        fit_ratio = triangle_fit_ratio(contour)
        if fit_ratio < min_fit_ratio:
            continue
        polygon = polygon_from_contour(contour)
        if polygon is None:
            continue
        side_lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        if np.min(side_lengths) < min_side_px:
            continue
        angle = triangle_orientation(polygon)
        if not orientation_gate.keep(angle):
            continue
        centroid = tuple(np.mean(polygon, axis=0))
        fill_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillConvexPoly(fill_mask, polygon.astype(np.int32), 255)
        std_dev = cv2.meanStdDev(gray, mask=fill_mask)[1].mean()
        if std_dev > max_intensity_std:
            continue
        score = triangle_score(polygon, processed_edges)
        triangles.append(
            Triangle(points=polygon, centroid=centroid, orientation=angle, area=area, score=score)
        )
    return triangles, processed_edges


def draw_triangles(frame: np.ndarray, triangles: Iterable[Triangle]) -> np.ndarray:
    canvas = frame.copy()
    for tri in triangles:
        pts = tri.points.astype(np.int32)
        cv2.fillConvexPoly(canvas, pts, (0, 200, 0))
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        cx, cy = map(int, tri.centroid)
        cv2.circle(canvas, (cx, cy), 3, (0, 200, 0), -1)
        label = f"{tri.orientation:5.1f}deg | S={tri.score:.2f}"
        cv2.putText(canvas, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 255, 40), 1)
    return canvas


def process_single_image(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Nie mogę otworzyć pliku: {args.image}")
    frame, scale = preprocess_frame(frame, args.max_dim)
    gate = OrientationGate(args.parallel_tolerance * 180.0)
    triangles, edges = detect_triangles(
        frame,
        min_area=args.min_area * (scale**2),
        min_side_px=args.min_side * scale,
        density_range=(args.min_density, args.max_density),
        orientation_gate=gate,
        max_intensity_std=args.max_variance,
        min_fit_ratio=args.min_fit,
    )

    print(f"Znaleziono {len(triangles)} trójkąty (skala {scale:.2f}).")
    for idx, tri in enumerate(triangles, 1):
        cx, cy = tri.centroid
        print(
            f"#{idx}: orientacja={tri.orientation:.1f}deg, pole={tri.area:.1f}, "
            f"centroid=({cx:.1f}, {cy:.1f}), score={tri.score:.2f}"
        )

    overlay = draw_triangles(frame, triangles)
    output_image_path, json_output_path = output_paths_for_image(str(image_path))
    if args.output:
        output_image_path = Path(args.output)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), overlay)
    save_triangles_json(triangles, json_output_path, str(image_path), scale)
    print(f"Zapisano obraz: {output_image_path}")
    print(f"Zapisano opis JSON: {json_output_path}")

    if args.no_gui:
        return
    cv2.imshow("frame", overlay)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_stream(cap: cv2.VideoCapture, args: argparse.Namespace) -> None:
    prev_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, scale = preprocess_frame(frame, args.max_dim)
        gate = OrientationGate(args.parallel_tolerance * 180.0)
        triangles, edges = detect_triangles(
            frame,
            min_area=args.min_area * (scale**2),
            min_side_px=args.min_side * scale,
            density_range=(args.min_density, args.max_density),
            orientation_gate=gate,
            max_intensity_std=args.max_variance,
            min_fit_ratio=args.min_fit,
        )
        overlay = draw_triangles(frame, triangles)
        if not args.no_gui:
            stacked = np.hstack([overlay, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])
            cv2.imshow("Triangles | Edges", stacked)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        now = time.time()
        if now - prev_print > 1.0:
            prev_print = now
            print(f"[stream] {len(triangles)} trójkąty, fps ~{cap.get(cv2.CAP_PROP_FPS):.1f}")
    cap.release()
    cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lightweight triangle finder powered by Canny + NumPy."
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--image",
        help="Path to an image file (default: data/screen/raw screen sample).",
    )
    source.add_argument("--video", help="Path to a video file (also supports RTSP/HTTP).")
    source.add_argument("--camera", type=int, help="Camera index (e.g. 0) for live capture.")
    parser.add_argument("--output", help="Path where annotated image will be saved.")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=0,
        help="Maximum side length after scaling (0 = full resolution).",
    )
    parser.add_argument("--min-area", type=float, default=450.0, help="Minimum contour area (px^2).")
    parser.add_argument("--min-side", type=float, default=8.0, help="Minimum triangle side length (px).")
    parser.add_argument(
        "--parallel-tolerance",
        type=float,
        default=0.20,
        help="Relative tolerance for parallelism (0.2 ~ 36 degrees).",
    )
    parser.add_argument("--min-density", type=float, default=0.01, help="Minimum edge density.")
    parser.add_argument("--max-density", type=float, default=0.28, help="Maximum edge density.")
    parser.add_argument(
        "--max-variance",
        type=float,
        default=55.0,
        help="Max std deviation of intensities inside triangle.",
    )
    parser.add_argument(
        "--min-fit",
        type=float,
        default=0.45,
        help="Minimum ratio of contour area to minimum enclosing triangle.",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI windows.")
    parser.add_argument(
        "--batch-folder",
        action="store_true",
        help="Process every image from data/screen/raw screen without GUI.",
    )
    return parser


def _iter_images_in_dir(dir_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not dir_path.is_dir():
        return []
    return sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.batch_folder:
        folder = RAW_SCREEN_DIR
        images = _iter_images_in_dir(folder)
        if not images:
            print(f"No images found in batch folder: {folder}")
            return
        print(f"Processing {len(images)} images from folder: {folder}")
        args.no_gui = True
        args.output = None
        for img in images:
            args.image = str(img)
            process_single_image(args)
        return

    if not (args.image or args.video or args.camera is not None):
        args.image = str(DEFAULT_IMAGE_PATH)
        print(f"Using default image: {args.image}")
    if args.image:
        process_single_image(args)
        return
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Unable to open video source.")
    process_stream(cap, args)


if __name__ == "__main__":
    main()
