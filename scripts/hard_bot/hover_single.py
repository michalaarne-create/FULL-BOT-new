#!/usr/bin/env python3
"""
Lightweight wrapper for hover_bot.process_image to process a single screenshot.

This allows the main orchestrator to feed a cropped screenshot into the hover-bot
pipeline without batching through folders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2

try:
    from scripts.hard_bot.hover_bot import (
        OCRNotAvailableError,
        create_easyocr_reader,
        process_image,
        DotSequence,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Cannot import from scripts.hard_bot.hover_bot. Ensure PYTHONPATH includes project root."
    ) from exc


def save_sequences_json(sequences: list[DotSequence], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [seq.__dict__ for seq in sequences]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hover_bot on a single image.")
    parser.add_argument("--image", required=True, help="Ścieżka do obrazu wejściowego.")
    parser.add_argument(
        "--json-out",
        required=True,
        help="Ścieżka wyjściowa dla JSON (punkty i boxy).",
    )
    parser.add_argument(
        "--annot-out",
        default=None,
        help="Opcjonalna ścieżka do zapisu obrazu z adnotacjami.",
    )
    parser.add_argument("--lang", default="en", help="Język EasyOCR (domyślnie: en).")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Brak pliku: {image_path}")

    json_out = Path(args.json_out)
    annot_out: Optional[Path] = Path(args.annot_out) if args.annot_out else None

    try:
        reader = create_easyocr_reader(lang=args.lang)
    except OCRNotAvailableError as exc:
        raise SystemExit(f"[ERROR] EasyOCR reader nie jest dostępny: {exc}")

    sequences, annotated = process_image(image_path, reader=reader)

    save_sequences_json(sequences, json_out)
    if annot_out:
        annot_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annot_out), annotated)


if __name__ == "__main__":
    main()
