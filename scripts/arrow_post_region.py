#!/usr/bin/env python3
"""
Post-processing step uruchamiany PO utils/region_grow.py.

Bierze JSON ze screen_boxes (wynik region_grow), wczytuje oryginalny
screenshot i używa `arrow_template_detector` do wykrycia strzałek
na całym ekranie. Następnie:

- zapisuje listę strzałek w polu `triangles` (każda ma bbox + centroid),
- ustawia `has_triangle = True` dla tych elementów, których bbox
  nachodzi na dowolną wykrytą strzałkę.

`rating.py` wykorzystuje pole `triangles` (oraz opcjonalnie
`has_triangle`) przy score_dropdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2

import arrow_template_detector as atd


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def arrow_hits_to_triangles(hits: List[dict]) -> List[dict]:
    triangles: List[dict] = []
    for h in hits:
        x1, y1, x2, y2 = h["bbox"]
        cx, cy = h["center"]
        triangles.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "centroid": {"x": float(cx), "y": float(cy)},
                "angle": float(h.get("angle", 0.0)),
                "score": float(h.get("score", 0.0)),
                "kind": h.get("kind", ""),
            }
        )
    return triangles


def mark_elements_with_triangles(elements: List[dict], triangles: List[dict], iou_thresh: float = 0.18) -> None:
    for elem in elements:
        bbox = (
            elem.get("dropdown_box")
            or elem.get("text_box")
            or elem.get("bbox")
            or elem.get("box")
            or elem.get("rect")
        )
        if not bbox or len(bbox) != 4:
            elem["has_triangle"] = False
            continue
        ex1, ey1, ex2, ey2 = [int(b) for b in bbox]
        elem_box = (ex1, ey1, ex2, ey2)
        has_tri = False
        for tri in triangles:
            tb = tri.get("bbox")
            if not tb or len(tb) != 4:
                continue
            tx1, ty1, tx2, ty2 = tb
            tri_box = (int(tx1), int(ty1), int(tx2), int(ty2))
            if atd.rect_iou(elem_box, tri_box) >= iou_thresh:
                has_tri = True
                break
        elem["has_triangle"] = has_tri


def process_screen_boxes(json_path: Path, threshold: float, min_mean: float, max_dim: int) -> None:
    data = load_json(json_path)
    image_path = data.get("image")
    if not image_path:
        print(f"[arrow_post] Brak pola 'image' w JSON: {json_path}")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[arrow_post] Nie mogę otworzyć obrazu: {image_path}")
        return

    frame, scale = atd.preprocess_frame(img, max_dim)
    templates = atd.generate_arrow_templates()
    hits = atd.detect_arrows(frame, templates, threshold=threshold, min_mean_intensity=min_mean)

    print(
        f"[arrow_post] image={Path(image_path).name}, scale={scale:.2f}, "
        f"threshold={threshold:.2f}, min_mean={min_mean:.1f}, hits={len(hits)}"
    )

    if scale != 1.0:
        for h in hits:
            x1, y1, x2, y2 = h["bbox"]
            h["bbox"] = [x1 / scale, y1 / scale, x2 / scale, y2 / scale]
            cx, cy = h["center"]
            h["center"] = (cx / scale, cy / scale)

    triangles = arrow_hits_to_triangles(hits)
    data["triangles"] = triangles

    elements = data.get("results") or []
    mark_elements_with_triangles(elements, triangles)

    save_json(json_path, data)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Uzupełnia screen_boxes JSON o strzałki po region_grow.")
    p.add_argument("json_path", help="Ścieżka do pliku JSON wygenerowanego przez utils/region_grow.py.")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Próg dopasowania matchTemplate (TM_CCOEFF_NORMED).",
    )
    p.add_argument(
        "--min-mean",
        type=float,
        default=180.0,
        help="Minimalna średnia jasność w oknie (odfiltruje strzałki na ciemnym tle).",
    )
    p.add_argument(
        "--max-dim",
        type=int,
        default=0,
        help="Maksymalna długość boku po skalowaniu (0 = pełna rozdzielczość).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    process_screen_boxes(Path(args.json_path), threshold=args.threshold, min_mean=args.min_mean, max_dim=args.max_dim)


if __name__ == "__main__":
    main()

