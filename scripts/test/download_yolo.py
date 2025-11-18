# Utwórz plik: download_yolo.py
from ultralytics import YOLO

print("Downloading YOLOv11n...")
model = YOLO('yolo11n.pt')  # ✅ Automatycznie pobierze z internetu
print(f"✅ Downloaded to: {model.ckpt_path}")