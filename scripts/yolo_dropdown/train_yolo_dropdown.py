from __future__ import annotations

"""
Quick training script for the simple dropdown detector using Ultralytics.

Edit the CONFIG block below to change settings without CLI flags.
CLI flags are still supported and take precedence if provided.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional


# --- Default config (edit here) ----------------------------------------------
CONFIG = {  
    "data": "data/yolo_header/data.yaml",
    "model": "runs_dropdown/train/weights/best.pt",
    "epochs": 20,
    "imgsz": 640,
    # Runtime/throughput
    "batch": 16,          # speed preset for 4GB (avoid swap)
    "workers": 4,         # Windows: fewer workers often faster
    "cache": "",         # no caching (avoid RAM/disk overhead)
    # Project/output
    "project": "runs_dropdown",
    "name": "train",
    # Optional overrides YAML (ultralytics args)
    "overrides": "scripts/yolo_dropdown/train_overrides.yaml",
    # Quick profile switch: 'speed' or 'gpu'
    "profile": "speed",
}

# Optional profiles to quickly switch between presets
PROFILES = {
    "speed": {
        "imgsz": 512,
        "batch": 12,
        "workers": 2,
        "cache": "",
    },
    "gpu": {
        # Heavier per-iteration compute to raise GPU util
        "imgsz": 704,
        "batch": 10,   # adjust to fit ~3.8–3.9 GB
        "workers": 4,
        "cache": "disk",
    },
}


def main():
    ap = argparse.ArgumentParser(description="Train YOLO dropdown detector (uses in-file defaults)")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, help="Batch size (-1 = auto-tune to VRAM)")
    ap.add_argument("--workers", type=int, default=None, help="Dataloader workers")
    ap.add_argument("--cache", type=str, default=None, help="Cache images to RAM/Disk (ram|disk)")
    ap.add_argument("--overrides", type=str, default=None, help="Optional YAML with Ultralytics train overrides")
    args = ap.parse_args()

    # Merge CONFIG with selected PROFILE, then CLI (CLI highest priority)
    cfg = CONFIG.copy()
    prof = cfg.get("profile", "speed")
    if prof in PROFILES:
        for k, v in PROFILES[prof].items():
            cfg[k] = v
    for k in ("data", "model", "epochs", "imgsz", "batch", "workers", "cache", "overrides"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    model = YOLO(cfg["model"])

    # Resolve data.yaml to absolute train/val paths if it contains relatives
    data_arg = cfg["data"]
    data_path = Path(str(data_arg))
    if data_path.suffix.lower() == ".yaml" and data_path.exists():
        try:
            txt = data_path.read_text(encoding="utf-8")
            if yaml is not None:
                data_obj = yaml.safe_load(txt) or {}
                if isinstance(data_obj, dict):
                    changed = False
                    for key in ("train", "val"):
                        p = data_obj.get(key)
                        if isinstance(p, str):
                            pp = Path(p)
                            if not pp.is_absolute():
                                # Make relative to yaml location
                                abs_p = (data_path.parent / pp).resolve()
                                data_obj[key] = abs_p.as_posix()
                                changed = True
                    if changed:
                        tmp_yaml = data_path.parent / (data_path.stem + "_abs.yaml")
                        tmp_yaml.write_text(yaml.safe_dump(data_obj), encoding="utf-8")
                        cfg["data"] = str(tmp_yaml)
        except Exception:
            # fallback: leave as-is; Ultralytics may still handle it
            pass

    kwargs = dict(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        optimizer="sgd",
        batch=cfg["batch"],
        workers=cfg["workers"],
        cache=(cfg["cache"] if cfg["cache"] in ("ram", "disk") else False),
        device=0 if True else "cpu",
        project=CONFIG["project"],
        name=CONFIG["name"],
    )

    # Load overrides if available
    overrides_path = cfg.get("overrides")
    if overrides_path and Path(overrides_path).exists():
        if yaml is None:
            print(f"[Warn] YAML not available; ignoring overrides file: {overrides_path}")
        else:
            with open(overrides_path, "r", encoding="utf-8") as f:
                over = yaml.safe_load(f) or {}
            if not isinstance(over, dict):
                print(f"[Warn] Overrides must be a mapping, got {type(over)}; ignoring")
            else:
                # CLI args take precedence; only add keys not explicitly set via CLI
                for k, v in over.items():
                    if k not in kwargs:
                        kwargs[k] = v

    # Print effective config for traceability
    print("[Train] Effective config:")
    for k in ("data", "model", "epochs", "imgsz", "batch", "workers", "cache", "project", "name"):
        print(f"  - {k}: {kwargs.get(k, cfg.get(k, None))}")

    model.train(**kwargs)


if __name__ == "__main__":
    main()
