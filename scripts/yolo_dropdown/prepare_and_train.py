#!/usr/bin/env python3
from __future__ import annotations
"""
One-shot: generate dataset and train YOLO dropdown detector.

Usage:
  python scripts/yolo_dropdown/prepare_and_train.py \
      --out data/yolo_dropdown --train 5000 --val 1000 --seed 42 \
      --model yolo11/yolo11n.pt --epochs 40 --imgsz 640
"""
import argparse
from pathlib import Path
import sys
from ultralytics import YOLO
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Ensure project root on sys.path to import generator
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "scripts").exists()), Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.yolo_dropdown.gen_simple_dataset import generate_split  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare dataset + train YOLO")
    ap.add_argument("--out", type=str, default="data/yolo_dropdown")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default="yolo11/yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--overrides", type=str, default="scripts/yolo_dropdown/train_overrides.yaml")
    ap.add_argument("--open-prob", type=float, default=0.7)
    ap.add_argument("--bottom-prob", type=float, default=0.5)
    ap.add_argument("--flag-min-px", type=int, default=6)
    ap.add_argument("--min-bottom-frac", type=float, default=0.2, help="Fail if bottom_frac below this")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    # 1) Generate dataset
    generate_split(out_root, n_train=args.train, n_val=args.val, seed=args.seed,
                   open_prob=args.open_prob, bottom_prob=args.bottom_prob, flag_min_px=args.flag_min_px)
    data_yaml = out_root / "data.yaml"
    # Always (re)write data.yaml with absolute paths for robustness
    train_images = (out_root / "train" / "images").resolve()
    val_images = (out_root / "val" / "images").resolve()
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

    # 2) Validate dataset logical rules
    try:
        from scripts.yolo_dropdown.validate_dataset import validate_split  # type: ignore
        validate_split(out_root, 'train', args.min_bottom_frac)
        validate_split(out_root, 'val', 0.1)
    except SystemExit as e:
        print("[Validate] Dataset validation failed. Aborting training.")
        raise
    except Exception as e:
        print(f"[Validate] Warning: validation skipped due to error: {e}")

    # 3) Train
    print(f"[Prep] Dataset ready at {out_root}. Training starts...")
    model = YOLO(args.model)
    kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        optimizer="sgd",
        batch=32,
        device=0,
        project="runs_dropdown",
        name="train",
    )
    if args.overrides and Path(args.overrides).exists() and yaml is not None:
        over = yaml.safe_load(Path(args.overrides).read_text(encoding="utf-8")) or {}
        if isinstance(over, dict):
            kwargs.update(over)
    model.train(**kwargs)
    print("[Done] Training complete.")


if __name__ == "__main__":
    main()
