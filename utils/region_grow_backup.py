# -*- coding: utf-8 -*-
"""
Dropdown / box detector (PaddleOCR 2.9.x) + lasery + trójkąty
WSZYSTKIE OCR wykrycia + informacja o obramowaniu (4 lasery)
"""

# ====== KONFIG =================================================================
import os
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
ROOT = str(ROOT_PATH)
DATA_SCREEN_DIR = ROOT_PATH / "data" / "screen"
RAW_SCREEN_DIR = DATA_SCREEN_DIR / "raw screen"
DEFAULT_IMAGE_PATH = str(RAW_SCREEN_DIR / "Zrzut ekranu 2025-10-25 163249.png")
PADDLE_GPU_ID = int(os.environ.get("PADDLEOCR_GPU_ID", "0"))
PADDLE_GPU_MEM = int(os.environ.get("PADDLEOCR_GPU_MEM", "6000"))

# === ŚCIEŻKI INTEGRACJI Z CNN ===

# gdzie zapisujemy boxy dla CNN (UWAGA: nowa lokalizacja!)
JSON_OUT_DIR = str(DATA_SCREEN_DIR / "region_grow")

# runner (plik 1) który wytnie cropy +10% i puści inferencję
CNN_RUNNER = str(ROOT_PATH / "utils" / "CNN" / "cnn_dropdown_runner.py")

# katalog na cropy z jednej rundy (czyścimy przed uruchomieniem)
CNN_CROPS_DIR = str(DATA_SCREEN_DIR / "temp" / "OCR_boxes+10%")

# Flood-fill / box
RADIUS = 1200
TOL_RGB = 3
NEIGHBOR_8 = True
MIN_BOX = 5  # Obniżony próg
SEED_PAD = 10

# OCR
FAST_OCR = False              # używaj rozbudowanego pipeline'u
OCR_CONF_MIN = 0.01
OCR_MAX_SIDE = 2800
MAX_OCR_ITEMS = 400
OCR_NMS_IOU = 0.60
DB_THRESH = 0.12
DB_BOX_THRESH = 0.32
DB_UNCLIP = 1.45
OCR_SCALES = [0.75, 1.0, 1.3, 1.6]            # kilka skal, by łapać mały tekst
FORCE_DET_ONLY_IF_EMPTY_TEXT = True
MIN_OCR_RESULTS = 5  # jeśli mniej wyników, dołóż fallback det-only

# Tło / histogram
HIST_BITS_PER_CH = 4
HIST_TOP_K = 8
GLOBAL_BG_OVER_PCT = 0.50

# Rysowanie
OCR_BOX_COLOR = (0, 128, 255, 255)
REGION_FILL_RGBA = (255, 0, 0, 150)
REGION_EDGE_RGBA = (255, 0, 0, 255)
LASER_ACCEPTED_RGBA = (255, 255, 0, 220)
LASER_REJECTED_RGBA = (255, 105, 180, 180)
LASER_WIDTH_ACCEPT = 3
LASER_WIDTH_REJECT = 2

# Lasery
EDGE_DELTA_E = 8.0
EDGE_CONSEC_N = 1
EDGE_MAX_LEN_PX = 2000
FRAME_SEARCH_MAX_PX = 500
TEXT_MASK_DILATE = 1

# Trójkąty
ENABLE_TRIANGLE_DETECT = False  # wyłącz trójkąty (dużo przyspiesza)
TRI_BITS = 4
TRI_NEIGHBOR_QDIST = 1
TRI_MIN_AREA = 40
TRI_MIN_W = 6
TRI_MIN_H = 6
TRI_MIN_GEOM_AREA = 28.0
TRI_FILL_RATIO = 0.72
TRI_ISO_TOL = 0.08
TRI_BG_DELTA_E = 8.0
TRI_FILL_RGBA = (0, 255, 0, 70)
TRI_EDGE_RGBA = (0, 160, 0, 230)
TRI_EDGE_WIDTH = 3

DEBUG_OCR = False
ENABLE_TIMINGS = True
TIMING_PREFIX = "[TIMER] "

# ==============================================================================

import contextlib
import json
import os
import subprocess, shutil
import threading
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw
try:
    import pytesseract
except Exception:
    pytesseract = None
import cv2
from paddleocr import PaddleOCR
try:
    import psutil  # type: ignore
except Exception:
    psutil = None
try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None

# OpenCV optymalizacja
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4)//2)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass

_K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# ===================== Timer =====================
class StageTimer:
    def __init__(self, enabled=False, prefix=TIMING_PREFIX):
        self.enabled = enabled
        self.prefix = prefix
        self._last = time.perf_counter()
        self._start = self._last
    
    def mark(self, label: str):
        if not self.enabled: return 0.0
        now = time.perf_counter()
        dt = now - self._last
        print(f"{self.prefix}{label}: {dt*1000:.1f} ms")
        self._last = now
        return dt
    
    def total(self, label: str = "TOTAL"):
        if not self.enabled: return 0.0
        now = time.perf_counter()
        dt = now - self._start
        print(f"{self.prefix}{label}: {dt*1000:.1f} ms")
        return dt

# ===================== UTIL =====================
def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)

def deltaE76(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.int16) - b.astype(np.int16)
    return float(np.sqrt((d * d).sum()))

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return inter / float(ua + ub - inter + 1e-9)

def suppress_ocr_overlaps(ocr_raw, iou_thr=OCR_NMS_IOU):
    if not ocr_raw: return ocr_raw
    
    def box_from_quad(q):
        xs = [int(p[0]) for p in q]
        ys = [int(p[1]) for p in q]
        return [min(xs), min(ys), max(xs), max(ys)]
    
    items = [(box_from_quad(q), q, t, c) for (q, t, c) in ocr_raw]
    items.sort(key=lambda x: x[3], reverse=True)
    
    keep = []
    used = [False] * len(items)
    
    for i, (b, q, t, c) in enumerate(items):
        if used[i]: continue
        keep.append((q, t, c))
        for j in range(i+1, len(items)):
            if not used[j] and iou_xyxy(b, items[j][0]) >= iou_thr:
                used[j] = True
    
    return keep

def color_close_rgb(a: np.ndarray, b: np.ndarray, tol: int = TOL_RGB) -> bool:
    return bool(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))) <= tol)

# Zamień funkcję to_py() na bezpieczniejszą wersję:
def to_py(obj):
    """Konwersja numpy/Python do czystego Python (rekurencyjna, bezpieczna)"""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        # Sprawdź NaN/Inf
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (str, bytes)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    # Fallback
    try:
        return obj.item()
    except (AttributeError, ValueError):
        return str(obj)

# Zamień funkcję main() na tę z lepszym debugowaniem:
def main():
    print("="*60)
    print("[DEBUG] SCRIPT START")
    print("="*60)
    
    import sys
    
    try:
        path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
        print(f"[DEBUG] Image path: {path}")
        
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}")
            return
        
        print("[DEBUG] Running detection...")
        out = run_dropdown_detection(path)
        print(f"[DEBUG] Detection returned {len(out.get('results', []))} results")
        
        print("[DEBUG] Creating annotation...")
        out_path = annotate_and_save(path, out.get("results", []), out.get("triangles"), output_dir=JSON_OUT_DIR)
        print(f"[DEBUG] Annotation created: {out_path}")
        
        print("\n[DEBUG] Converting to Python types...")
        try:
            out_py = to_py(out)
            print(f"[DEBUG] Conversion OK, keys: {list(out_py.keys())}")
        except Exception as e:
            print(f"[ERROR] to_py() failed: {e}")
            import traceback
            traceback.print_exc()
            out_py = {"error": str(e)}
        
        print("\n[DEBUG] Generating JSON...")
        try:
            json_output = json.dumps(out_py, ensure_ascii=False, indent=2)
            print(f"[DEBUG] JSON length: {len(json_output)} chars")
        except Exception as e:
            print(f"[ERROR] json.dumps() failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback - prosta wersja
            json_output = json.dumps({
                "image": out_py.get("image"),
                "results_count": len(out_py.get("results", [])),
                "error": str(e)
            }, indent=2)
        
        print("\n" + "="*60)
        print("JSON OUTPUT:")
        print("="*60)
        print(json_output)
        print("\n" + "="*60)
        print(f"Annotation: {out_path}")
        print("="*60)
    
    except Exception as e:
        print(f"\n[ERROR] Script failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[DEBUG] SCRIPT END")

# ===================== OCR =====================
_paddle = None
_last_ocr_debug = {}

def get_ocr():
    global _paddle
    if _paddle is None:
        print("[DEBUG] Initializing PaddleOCR on GPU...")
        try:
            _paddle = PaddleOCR(
                lang="en",
                use_gpu=True,
                use_angle_cls=False,
                show_log=False,
                rec_batch_num=16,
                gpu_mem=PADDLE_GPU_MEM,
                gpu_id=PADDLE_GPU_ID,
            )
            print(f"[DEBUG] PaddleOCR initialized on GPU (id={PADDLE_GPU_ID}, mem={PADDLE_GPU_MEM})")
        except Exception as exc:
            print(f"[WARN] PaddleOCR GPU init failed ({exc}), falling back to CPU")
            try:
                _paddle = PaddleOCR(lang="en")
                print("[DEBUG] PaddleOCR initialized on CPU")
            except Exception as exc2:
                print(f"[ERROR] PaddleOCR initialization failed: {exc2}")
                _paddle = None
    return _paddle

def _preprocess_for_ocr(img_pil: Image.Image):
    w, h = img_pil.size
    base_scale = 1.0
    target_side = 1700
    if max(w, h) < target_side:
        base_scale = target_side / float(max(w, h))
    
    tgt = img_pil.resize((int(w * base_scale), int(h * base_scale)), Image.BILINEAR)
    arr0 = np.array(tgt)
    
    lab = cv2.cvtColor(arr0, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(2.0, (8, 8)).apply(L)
    arr0 = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2RGB)
    arr0 = cv2.addWeighted(arr0, 1.5, cv2.GaussianBlur(arr0, (0, 0), 1.0), -0.5, 0)
    
    return arr0, base_scale

def _parse_rec_result(res) -> Tuple[str, float]:
    try:
        if not isinstance(res, list) or not res:
            return "", 0.0
        entry = res[0]

        def _try_unpack(node):
            if isinstance(node, (list, tuple)) and len(node) >= 2 and isinstance(node[1], (int, float)):
                return (node[0] or "").strip(), float(node[1] or 0.0)
            if (
                isinstance(node, (list, tuple))
                and len(node) >= 2
                and isinstance(node[1], (list, tuple))
                and len(node[1]) >= 2
            ):
                return (node[1][0] or "").strip(), float(node[1][1] or 0.0)
            return None

        # Case 1: entry is already a (text, conf) tuple/list
        unpacked = _try_unpack(entry)
        if unpacked:
            return unpacked

        # Case 2: entry is a list whose first element is (text, conf)
        if isinstance(entry, (list, tuple)) and entry:
            unpacked = _try_unpack(entry[0])
            if unpacked:
                return unpacked
    except Exception:
        pass
    return "", 0.0

def _tesseract_fallback(img_pil: Image.Image) -> List[Tuple[List[Tuple[int, int]], str, float]]:
    """Generuje (quad, text, conf) wykorzystując pytesseract, gdy Paddle nie widzi pól."""
    if pytesseract is None:
        return []
    try:
        data = pytesseract.image_to_data(
            img_pil,
            output_type=getattr(pytesseract, "Output", None).DICT if hasattr(pytesseract, "Output") else None,
        )
    except Exception:
        return []

    # Jeśli pytesseract.Output.DICT nie jest dostępny (stare wersje), spróbuj parsować ręcznie
    if not isinstance(data, dict) or "text" not in data:
        return []

    outs = []
    n = len(data.get("text", []))
    for i in range(n):
        try:
            txt = (data["text"][i] or "").strip()
            if len(txt) < 2:
                continue
            conf = float(data.get("conf", ["0"]*n)[i])
            if conf <= 0:
                continue
            x = int(data.get("left", [0]*n)[i])
            y = int(data.get("top", [0]*n)[i])
            w = int(data.get("width", [0]*n)[i])
            h = int(data.get("height", [0]*n)[i])
            if w <= 1 or h <= 1:
                continue
            quad = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
            ]
            outs.append((quad, txt, conf / 100.0))
        except Exception:
            continue

    return outs

def _cv_text_regions(img_rgb: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Wykrywa prostokąty tekstu klasycznymi metodami CV (fallback gdy OCR nic nie widzi)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 23, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dil = cv2.dilate(bw, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 30 or h < 12:
            continue
        if w * h < 400:
            continue
        aspect = w / float(h)
        if aspect < 1.5:
            continue
        x2, y2 = x + w, y + h
        quads.append([(x, y), (x2, y), (x2, y2), (x, y2)])
    return quads

def read_ocr_faster(img_pil: Image.Image, timer: Optional[StageTimer] = None):
    print("[DEBUG] OCR: Preprocessing...")
    arr0, base_scale = _preprocess_for_ocr(img_pil)
    if timer: timer.mark("OCR preprocess")
    
    ocr = get_ocr()
    det_quads = []
    
    # DET multi-scale
    print("[DEBUG] OCR: Detection phase...")
    for s in OCR_SCALES:
        try:
            arr = cv2.resize(arr0, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            det = ocr.ocr(arr, det=True, rec=False, cls=False)
            qlist = det[0] if isinstance(det, list) and len(det) else []
            inv = 1.0 / (base_scale * s)
            for q in qlist:
                det_quads.append([(int(x*inv), int(y*inv)) for (x, y) in q])
            print(f"[DEBUG] OCR: Scale {s} -> {len(qlist)} boxes")
        except Exception as e:
            print(f"[ERROR] OCR detection scale {s}: {e}")

    if not det_quads:
        print("[DEBUG] OCR: No detections found")
        if timer: timer.mark("OCR det (no boxes)")
        return []

    # NMS + limit
    det_boxes = suppress_ocr_overlaps([(q, "", 0.0) for q in det_quads], OCR_NMS_IOU)
    if len(det_boxes) > MAX_OCR_ITEMS:
        det_boxes = det_boxes[:MAX_OCR_ITEMS]
    
    print(f"[DEBUG] OCR: After NMS -> {len(det_boxes)} boxes")

    # REC batchem
    print("[DEBUG] OCR: Recognition phase...")
    arr_full = np.array(img_pil)
    crops, boxes = [], []
    
    for q, _, _ in det_boxes:
        xs = [int(p[0]) for p in q]
        ys = [int(p[1]) for p in q]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(arr_full.shape[1], max(xs)), min(arr_full.shape[0], max(ys))
        if x2 - x1 <= 1 or y2 - y1 <= 1: 
            continue
        crops.append(arr_full[y1:y2, x1:x2])
        boxes.append(q)

    rec_outs = []
    if crops:
        try:
            res_list = ocr.ocr(crops, det=False, rec=True, cls=True)
        except Exception:
            print("[DEBUG] OCR: Batch rec failed, using loop fallback")
            res_list = []
            for c in crops:
                res_list.append(ocr.ocr(c, det=False, rec=True, cls=True))

        for q, res in zip(boxes, res_list):
            txt, conf = _parse_rec_result(res)
            rec_outs.append((q, txt, conf))

    outs = [r for r in rec_outs if float(r[2]) >= OCR_CONF_MIN]
    
    if (not outs or not any((t.strip() != "") for _, t, _ in outs)) and FORCE_DET_ONLY_IF_EMPTY_TEXT:
        print("[DEBUG] OCR: No text found, using det-only mode")
        outs = [(q, "", 0.5) for (q, _, _) in det_boxes]

    outs.sort(key=lambda r: float(r[2]), reverse=True)
    outs = suppress_ocr_overlaps(outs, OCR_NMS_IOU)
    
    if len(outs) > MAX_OCR_ITEMS:
        outs = outs[:MAX_OCR_ITEMS]
    
    print(f"[DEBUG] OCR: Final -> {len(outs)} results")
    if timer: timer.mark("OCR finalize")
    return outs

def read_ocr_full(img_pil: Image.Image, timer: Optional[StageTimer] = None):
    global _last_ocr_debug
    print("[DEBUG] OCR: Using FULL mode...")
    arr0, base_scale = _preprocess_for_ocr(img_pil)
    if timer: timer.mark("OCR preprocess")
    
    ocr = get_ocr()
    outs = []
    dbg = {
        "ppocr": "v4", 
        "base_scale": base_scale, 
        "scales": [],
        "boxes_rec_raw": 0, 
        "boxes_kept_after_thr_nms": 0,
        "det_only_used": False, 
        "det_only_boxes": 0, 
        "tesseract_boxes": 0,
        "cv_boxes": 0,
        "errors": []
    }

    for s in OCR_SCALES:
        try:
            arr = cv2.resize(arr0, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            res = ocr.ocr(arr, cls=True)
            lines = res[0] if isinstance(res, list) and len(res) > 0 else []
            dbg["scales"].append({"scale": s, "raw": len(lines)})
            inv = 1.0 / (base_scale * s)
            for it in lines:
                quad = it[0]
                txt = (it[1][0] or "").strip()
                conf = float(it[1][1] or 0.0)
                quad = [(int(x * inv), int(y * inv)) for (x, y) in quad]
                outs.append((quad, txt, conf))
        except Exception as e:
            dbg["errors"].append(f"rec_scale{s}: {e}")

    if timer: timer.mark("OCR full total")

    dbg["boxes_rec_raw"] = len(outs)
    has_text = any((t.strip() != "") for (_, t, _) in outs)

    # Fallback: jeśli wyników jest mało lub brak tekstu, spróbuj
    # (a) wykryć boxy det-only na kilku skalach, a następnie
    # (b) uruchomić rozpoznawanie na każdym wykrytym boxie (bez kolejnej detekcji).
    need_det_only = (
        ((len(outs) == 0) or not has_text) and FORCE_DET_ONLY_IF_EMPTY_TEXT
    ) or (len(outs) < MIN_OCR_RESULTS)
    if need_det_only:
        det_quads = []
        try:
            arr_full = np.array(img_pil)
        except Exception:
            arr_full = None
        for s in OCR_SCALES:
            try:
                arr = cv2.resize(arr0, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                res = ocr.ocr(arr, det=True, rec=False, cls=False)
                boxes = res[0] if isinstance(res, list) and len(res) > 0 else []
                inv = 1.0 / (base_scale * s)
                for quad in boxes:
                    quad = [(int(x * inv), int(y * inv)) for (x, y) in quad]
                    det_quads.append(quad)
            except Exception as e:
                dbg["errors"].append(f"det_scale{s}: {e}")

        if not det_quads and arr_full is not None:
            cv_quads = _cv_text_regions(arr_full)
            det_quads.extend(cv_quads)
            dbg["cv_boxes"] = len(cv_quads)

        # Rozpoznanie per-box na oryginalnym obrazie (bez dodatkowej detekcji)
        rec_outs = []
        for quad in det_quads:
            try:
                xs = [int(p[0]) for p in quad]
                ys = [int(p[1]) for p in quad]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                if arr_full is None:
                    crop = None
                else:
                    x1c = max(0, x1); y1c = max(0, y1)
                    x2c = min(arr_full.shape[1], x2); y2c = min(arr_full.shape[0], y2)
                    if x2c - x1c > 1 and y2c - y1c > 1:
                        crop = arr_full[y1c:y2c, x1c:x2c]
                    else:
                        crop = None
                if crop is not None:
                    res = ocr.ocr(crop, det=False, rec=True, cls=True)
                    txt, c = _parse_rec_result(res)
                    rec_outs.append((quad, txt, float(c)))
                else:
                    rec_outs.append((quad, "", 0.5))
            except Exception as e:
                dbg["errors"].append(f"rec_box: {e}")
                rec_outs.append((quad, "", 0.5))

        outs.extend(rec_outs)
        dbg["det_only_used"] = True
        dbg["det_only_boxes"] = len(det_quads)

        if len(outs) < MIN_OCR_RESULTS:
            tess_outs = _tesseract_fallback(img_pil)
            if tess_outs:
                outs.extend(tess_outs)
                dbg["tesseract_boxes"] = len(tess_outs)

    outs = [r for r in outs if float(r[2]) >= OCR_CONF_MIN]
    outs.sort(key=lambda r: float(r[2]), reverse=True)
    outs = suppress_ocr_overlaps(outs, OCR_NMS_IOU)
    
    if len(outs) > MAX_OCR_ITEMS:
        outs = outs[:MAX_OCR_ITEMS]
    
    dbg["boxes_kept_after_thr_nms"] = len(outs)
    _last_ocr_debug = dbg
    return outs

def read_ocr_wrapper(img_pil: Image.Image, timer: Optional[StageTimer] = None):
    return read_ocr_faster(img_pil, timer=timer) if FAST_OCR else read_ocr_full(img_pil, timer=timer)

def _ocr_text_for_bbox(img_rgb: np.ndarray, bbox: Optional[List[int]], pad: int = 4,
                       min_conf: float = 0.2) -> str:
    """
    Dodatkowe rozpoznanie tekstu wewnątrz zadanego bboxa (np. po flood fillu).
    Przydaje się, gdy początkowy OCR zwrócił pusty string, a chcemy zapisać
    tekst opisujący cały box odpowiedzi/dropdown.
    """
    if bbox is None:
        return ""
    try:
        ocr = get_ocr()
    except Exception:
        ocr = None
    if ocr is None or img_rgb is None or img_rgb.size == 0:
        return ""

    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(W, int(x2) + pad)
    y2 = min(H, int(y2) + pad)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return ""

    crop_rgb = img_rgb[y1:y2, x1:x2]
    try:
        crop_pil = Image.fromarray(crop_rgb)
        arr_proc, _ = _preprocess_for_ocr(crop_pil)
        res = ocr.ocr(arr_proc, cls=True)
        lines = res[0] if isinstance(res, list) and len(res) > 0 else []
        texts = []
        for it in lines:
            t = (it[1][0] or "").strip()
            c = float(it[1][1] or 0.0)
            if t and c >= min_conf:
                texts.append(t)
        if texts:
            return " ".join(texts)
    except Exception:
        pass
    return ""

# ===================== MASKA TEKSTU =====================
def build_text_mask_cv(ocr_items, W, H):
    m = np.zeros((H, W), np.uint8)
    for q, _, _ in ocr_items:
        xs = [int(p[0]) for p in q]
        ys = [int(p[1]) for p in q]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        cv2.rectangle(m, (x1, y1), (min(W, x2+1), min(H, y2+1)), 1, thickness=-1)
    
    if TEXT_MASK_DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1+2*TEXT_MASK_DILATE, 1+2*TEXT_MASK_DILATE))
        m = cv2.dilate(m, k, iterations=1)
    
    return m.astype(bool)

# ===================== HISTOGRAM =====================
def quantize_rgb(img: np.ndarray, bits: int = HIST_BITS_PER_CH) -> np.ndarray:
    return (img >> (8 - bits)).astype(np.uint8)

def hist_color_percent(img: np.ndarray, bits: int = HIST_BITS_PER_CH, top_k: int = HIST_TOP_K):
    q = quantize_rgb(img, bits)
    key = (q[...,0].astype(np.uint32)<<(2*bits)) | (q[...,1].astype(np.uint32)<<bits) | q[...,2].astype(np.uint32)
    uniq, cnt = np.unique(key.reshape(-1), return_counts=True)
    total = float(img.shape[0]*img.shape[1])
    order = np.argsort(cnt)[::-1]
    uniq = uniq[order]
    cnt = cnt[order]
    
    items = []
    step = 256 // (1 << bits)
    for k_, c in zip(uniq[:top_k], cnt[:top_k]):
        r = int(((k_>>(2*bits)) & ((1<<bits)-1)) * step + step//2)
        g = int(((k_>>bits) & ((1<<bits)-1)) * step + step//2)
        b = int((k_ & ((1<<bits)-1)) * step + step//2)
        items.append({"rgb": [r, g, b], "pct": float(round(c/total, 6))})
    
    dom_pct = float(cnt[0]/total) if len(cnt) else 0.0
    if len(uniq):
        r0 = int(((uniq[0]>>(2*bits)) & ((1<<bits)-1)) * step + step//2)
        g0 = int(((uniq[0]>>bits) & ((1<<bits)-1)) * step + step//2)
        b0 = int((uniq[0] & ((1<<bits)-1)) * step + step//2)
        dom = [r0, g0, b0]
    else:
        dom = [255, 255, 255]
    
    return {"top": items, "dominant_pct": dom_pct, "dominant_rgb": dom}

# ===================== FLOOD i LASERY =====================
def pick_seed(img_rgb: np.ndarray, text_box_xyxy: Tuple[int, int, int, int], pad: int = SEED_PAD):
    H, W, _ = img_rgb.shape
    x1, y1, x2, y2 = text_box_xyxy
    cy = (y1 + y2)//2
    
    for dx in range(2, pad*3):
        x = x2 + dx
        if x >= W: break
        y0, y1b = clamp(cy-1, 0, H-1), clamp(cy+1, 0, H-1)
        x0, x1b = clamp(x-1, 0, W-1), clamp(x+1, 0, W-1)
        neigh = img_rgb[y0:y1b+1, x0:x1b+1].astype(np.int16)
        m = neigh.mean(axis=(0,1))
        if np.max(np.abs(neigh - m)) <= 6.0:
            return cy, x, img_rgb[cy, x]
    
    cx = (x1 + x2)//2
    return cy, cx, img_rgb[cy, cx]

def flood_same_color_bbox_cv(img_rgb: np.ndarray, seed_y: int, seed_x: int,
                             tol_rgb: int = TOL_RGB, radius: int = RADIUS, neighbor8: bool = True):
    H, W = img_rgb.shape[:2]
    ry1, ry2 = max(0, seed_y-radius), min(H, seed_y+radius+1)
    rx1, rx2 = max(0, seed_x-radius), min(W, seed_x+radius+1)
    roi = img_rgb[ry1:ry2, rx1:rx2]
    
    if roi.size == 0: return None
    
    mask = np.zeros((roi.shape[0]+2, roi.shape[1]+2), np.uint8)
    flags = ((8 if neighbor8 else 4) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)
    sy_r, sx_r = seed_y - ry1, seed_x - rx1
    
    try:
        _, _, _, rect = cv2.floodFill(roi, mask, (int(sx_r), int(sy_r)), newVal=(0,0,0),
                                      loDiff=(tol_rgb,)*3, upDiff=(tol_rgb,)*3, flags=flags)
    except Exception:
        return None
    
    m_local = mask[1:-1, 1:-1]
    if rect[2] < MIN_BOX or rect[3] < MIN_BOX: return None
    
    region_big = np.zeros((H, W), np.uint8)
    region_big[ry1:ry2, rx1:rx2] = m_local
    region_big = cv2.morphologyEx(region_big, cv2.MORPH_CLOSE, _K3, iterations=1)
    
    ys, xs = np.where(region_big > 0)
    if ys.size == 0: return None
    
    y1b, y2b = int(ys.min()), int(ys.max())
    x1b, x2b = int(xs.min()), int(xs.max())
    bbox_xyxy = (x1b, y1b, x2b+1, y2b+1)
    
    return region_big.astype(bool), bbox_xyxy

def boundary_colors_from_region_fast(img_rgb: np.ndarray, region_mask: np.ndarray,
                                     text_mask: np.ndarray, tol_rgb: int = TOL_RGB) -> Dict[str, dict]:
    H, W = region_mask.shape
    ys, xs = np.where(region_mask)
    if ys.size == 0: return {}

    y1, y2 = max(0, ys.min()-1), min(H-1, ys.max()+1)
    x1, x2 = max(0, xs.min()-1), min(W-1, xs.max()+1)
    reg = region_mask[y1:y2+1, x1:x2+1]
    txt = text_mask[y1:y2+1, x1:x2+1]
    sub = img_rgb[y1:y2+1, x1:x2+1]

    rim = cv2.dilate(reg.astype(np.uint8), _K3, iterations=1).astype(bool)
    rim = rim & (~reg) & (~txt)
    rys, rxs = np.where(rim)
    if rys.size == 0: return {}

    cols = sub[rys, rxs]
    posy = rys + y1
    posx = rxs + x1

    s = tol_rgb + 1
    keys = (cols // s).astype(np.int16)

    clusters = []
    bucket = {}

    for i, c in enumerate(cols):
        k = (int(keys[i,0]), int(keys[i,1]), int(keys[i,2]))
        matched = False
        
        for dr in (-1, 0, 1):
            for dg in (-1, 0, 1):
                for db in (-1, 0, 1):
                    lst = bucket.get((k[0]+dr, k[1]+dg, k[2]+db))
                    if not lst:
                        continue
                    for ci in lst:
                        rc = clusters[ci]["rgb"]
                        if np.max(np.abs(c.astype(np.int16) - rc.astype(np.int16))) <= tol_rgb:
                            clusters[ci]["count"] += 1
                            matched = True
                            break
                    if matched: break
                if matched: break
            if matched: break
        
        if not matched:
            ci = len(clusters)
            clusters.append({"rgb": c.copy(), "count": 1, "pos": (int(posy[i]), int(posx[i]))})
            bucket.setdefault(k, []).append(ci)

    out = {}
    for cl in clusters:
        r, g, b = int(cl["rgb"][0]), int(cl["rgb"][1]), int(cl["rgb"][2])
        out[f"{r},{g},{b}"] = {
            "rgb": [r, g, b], 
            "count": int(cl["count"]),
            "sample_pos": [cl["pos"][0], cl["pos"][1]]
        }
    
    return out

def region_ref_lab(lab_img: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(region_mask)
    if len(ys) == 0: return np.array([0, 0, 0], dtype=np.uint8)
    return np.median(lab_img[ys, xs], axis=0).astype(np.uint8)

def _deltaE_map(lab_win: np.ndarray, ref_lab: np.ndarray) -> np.ndarray:
    d = lab_win.astype(np.int16) - ref_lab.reshape(1,1,3).astype(np.int16)
    return np.sqrt((d*d).sum(axis=2)).astype(np.float32)

def laser_box_check_fast(img_rgb: np.ndarray, lab_img: np.ndarray, bbox_text, 
                        region_mask, frame_rgb, text_mask, tol_rgb: int):
    x1, y1, x2, y2 = bbox_text
    cy, cx = (y1+y2)//2, (x1+x2)//2
    ys, xs = np.where(region_mask)
    
    if xs.size == 0: return 0, [(cx, cy)]*4
    
    ry1, ry2 = int(ys.min()), int(ys.max())
    rx1, rx2 = int(xs.min()), int(xs.max())
    pad = min(30, FRAME_SEARCH_MAX_PX)
    wy1, wy2 = max(0, ry1-pad), min(lab_img.shape[0], ry2+pad+1)
    wx1, wx2 = max(0, rx1-pad), min(lab_img.shape[1], rx2+pad+1)
    sub_lab = lab_img[wy1:wy2, wx1:wx2]
    ref_lab = region_ref_lab(lab_img, region_mask)
    dEmap = _deltaE_map(sub_lab, ref_lab)

    def cast_dir(dy, dx):
        y, x = cy - wy1, cx - wx1
        H, W = dEmap.shape
        steps = 0
        consec = 0
        searched = 0
        last = (x, y)
        
        while steps < EDGE_MAX_LEN_PX and 0 <= y < H and 0 <= x < W and text_mask[wy1+y, wx1+x]:
            y += dy
            x += dx
            steps += 1
        
        while steps < EDGE_MAX_LEN_PX and 0 <= y < H and 0 <= x < W and region_mask[wy1+y, wx1+x]:
            y += dy
            x += dx
            steps += 1
        
        while steps < EDGE_MAX_LEN_PX and 0 <= y < H and 0 <= x < W and searched < FRAME_SEARCH_MAX_PX:
            if not text_mask[wy1+y, wx1+x]:
                ok = False
                if frame_rgb is not None and color_close_rgb(img_rgb[wy1+y, wx1+x], frame_rgb, tol_rgb):
                    ok = True
                else:
                    if dEmap[y, x] > EDGE_DELTA_E:
                        consec += 1
                        if consec >= EDGE_CONSEC_N: 
                            ok = True
                    else:
                        consec = 0
                
                if ok: return True, (wx1+x, wy1+y)
            
            last = (x, y)
            y += dy
            x += dx
            steps += 1
            searched += 1
        
        return False, (wx1+last[0], wy1+last[1])

    hits = 0
    pts = []
    for dy, dx in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
        ok, pt = cast_dir(dy, dx)
        hits += int(ok)
        pts.append(pt)
    
    return hits, pts

# ===================== TRIANGLES ===============================================
def _tri_quantize(img: np.ndarray, bits: int):
    q = (img >> (8 - bits)).astype(np.uint8)
    return q, q[...,0], q[...,1], q[...,2]

def _tri_bin_center_rgb(bin_r: int, bin_g: int, bin_b: int, bits: int):
    step = 256 // (1 << bits)
    return (bin_r*step + step//2, bin_g*step + step//2, bin_b*step + step//2)

def _is_isosceles(pts: np.ndarray, rel_tol: float) -> bool:
    if pts.shape[0] != 3: return False
    
    def sq(a, b): 
        return (a[0]-b[0])**2 + (a[1]-b[1])**2
    
    d = [sq(pts[0], pts[1]), sq(pts[1], pts[2]), sq(pts[2], pts[0])]
    if min(d) == 0: return False
    
    def close(u, v): 
        return abs(u-v) <= rel_tol*max(u, v)
    
    return close(d[0], d[1]) or close(d[1], d[2]) or close(d[0], d[2])

def find_filled_isosceles_triangles_fast(
    img_rgb: np.ndarray,
    lab_img: np.ndarray,
    bg_rgb: Tuple[int,int,int],
    bits: int = TRI_BITS,
    neighbor_qdist: int = TRI_NEIGHBOR_QDIST,
    min_area: int = TRI_MIN_AREA,
    min_w: int = TRI_MIN_W,
    min_h: int = TRI_MIN_H,
    min_geom_area: float = TRI_MIN_GEOM_AREA,
    fill_ratio: float = TRI_FILL_RATIO,
    iso_tol: float = TRI_ISO_TOL,
    bg_delta_e: float = TRI_BG_DELTA_E
) -> List[Dict]:
    bg_lab = cv2.cvtColor(np.uint8([[[bg_rgb[0], bg_rgb[1], bg_rgb[2]]]]), cv2.COLOR_RGB2LAB)[0, 0]

    q, qR, qG, qB = _tri_quantize(img_rgb, bits)
    key = (qR.astype(np.uint32)<<(2*bits)) | (qG.astype(np.uint32)<<bits) | qB.astype(np.uint32)
    uniq, cnt = np.unique(key.reshape(-1), return_counts=True)

    cand_bins = []
    for k_, c in zip(uniq, cnt):
        if c < max(8, min_area//2): continue
        br = int((k_>>(2*bits)) & ((1<<bits)-1))
        bg_ = int((k_>>bits) & ((1<<bits)-1))
        bb = int(k_ & ((1<<bits)-1))
        cr, cg, cb = _tri_bin_center_rgb(br, bg_, bb, bits)
        col_lab = cv2.cvtColor(np.uint8([[[cr, cg, cb]]]), cv2.COLOR_RGB2LAB)[0, 0]
        if deltaE76(col_lab, bg_lab) <= bg_delta_e:
            continue
        cand_bins.append((br, bg_, bb))

    diff = lab_img.astype(np.int16) - bg_lab.reshape(1,1,3).astype(np.int16)
    d2 = (diff * diff).sum(axis=2).astype(np.int32)
    thr2 = float(bg_delta_e) * float(bg_delta_e)
    not_bg = (d2 > thr2).astype(np.uint8)

    results = []
    for br, bg__, bb in cand_bins:
        mr = (np.abs(qR.astype(np.int16) - br) <= neighbor_qdist)
        mg = (np.abs(qG.astype(np.int16) - bg__) <= neighbor_qdist)
        mb = (np.abs(qB.astype(np.int16) - bb) <= neighbor_qdist)
        mask = (mr & mg & mb).astype(np.uint8)
        mask = (mask & not_bg)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K3, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue

        for cnt_ in contours:
            area_pix = cv2.contourArea(cnt_)
            if area_pix < min_area: continue
            x, y, w, h = cv2.boundingRect(cnt_)
            if w < min_w or h < min_h: continue
            hull = cv2.convexHull(cnt_)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.04*peri, True)
            if approx.shape[0] != 3: continue
            tri = approx.reshape(-1, 2)
            area_geom = cv2.contourArea(approx)
            if area_geom <= min_geom_area or area_geom <= 0: continue
            if (area_pix / (area_geom + 1e-9)) < fill_ratio: continue
            if not _is_isosceles(tri, rel_tol=iso_tol): continue
            cr, cg, cb = _tri_bin_center_rgb(br, bg__, bb, bits)
            results.append({
                "color": [int(cr), int(cg), int(cb)],
                "hull": [(int(tri[0][0]), int(tri[0][1])),
                        (int(tri[1][0]), int(tri[1][1])),
                        (int(tri[2][0]), int(tri[2][1]))],
                "bbox": (int(x), int(y), int(x+w), int(y+h)),
                "area_pixels": float(area_pix),
                "area_geom": float(area_geom),
                "fill_ratio": float(area_pix/(area_geom+1e-9)),
            })
    
    return results

# ===================== PIPELINE =================================================
def run_dropdown_detection(image_path: str) -> dict:
    print(f"[DEBUG] Starting detection: {image_path}")
    timer = StageTimer(ENABLE_TIMINGS)

    print("[DEBUG] Loading image...")
    img_pil = Image.open(image_path).convert("RGB")
    img = np.ascontiguousarray(np.array(img_pil))
    H, W = img.shape[:2]
    print(f"[DEBUG] Image size: {W}x{H}")
    
    lab = np.ascontiguousarray(rgb_to_lab(img))
    timer.mark("Load + LAB")

    print("[DEBUG] Computing histogram...")
    hist = hist_color_percent(img, bits=HIST_BITS_PER_CH, top_k=HIST_TOP_K)
    is_plain_bg_global = bool(hist["dominant_pct"] >= float(GLOBAL_BG_OVER_PCT))
    dominant_bg_rgb = np.array(hist["dominant_rgb"], dtype=np.uint8)
    timer.mark("Histogram")

    print("[DEBUG] Running OCR...")
    try:
        ocr_raw = read_ocr_wrapper(img_pil, timer=timer)
        print(f"[DEBUG] OCR found {len(ocr_raw)} items")
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        import traceback
        traceback.print_exc()
        ocr_raw = []
    
    timer.mark("OCR wrapper")
    
    print("[DEBUG] Building text mask...")
    text_mask = build_text_mask_cv(ocr_raw, W, H)
    timer.mark("Text mask")

    results = []

    print(f"[DEBUG] Processing {len(ocr_raw)} detections...")
    for idx, (quad, txt, conf) in enumerate(ocr_raw):
        xs = [int(p[0]) for p in quad]
        ys = [int(p[1]) for p in quad]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Podstawowy wynik
        result = {
            "text": txt,
            "conf": float(conf),
            "text_box": [x1, y1, x2, y2],
            "has_frame": False,
            "frame_hits": 0,
            "dropdown_box": None,
            "frame_rgb": None,
            "laser_endpoints": None
        }

        try:
            sy, sx, _ = pick_seed(img, (x1, y1, x2, y2), pad=SEED_PAD)
            ff = flood_same_color_bbox_cv(img, sy, sx, tol_rgb=TOL_RGB, radius=RADIUS, neighbor8=NEIGHBOR_8)
            
            if ff is None:
                cy, cx = (y1+y2)//2, (x1+x2)//2
                ff = flood_same_color_bbox_cv(img, cy, cx, tol_rgb=TOL_RGB, radius=RADIUS, neighbor8=NEIGHBOR_8)

            if ff is not None:
                region_mask, flood_bbox = ff
                bcolors = boundary_colors_from_region_fast(img, region_mask, text_mask, tol_rgb=TOL_RGB)
                frame_rgb = None if not bcolors else np.array(max(bcolors.values(), key=lambda v: v["count"])["rgb"], dtype=np.uint8)
                hits, endpoints = laser_box_check_fast(img, lab, (x1, y1, x2, y2), region_mask, frame_rgb, text_mask, TOL_RGB)
                
                result["dropdown_box"] = [int(flood_bbox[0]), int(flood_bbox[1]), int(flood_bbox[2]), int(flood_bbox[3])]
                result["frame_rgb"] = [int(frame_rgb[0]), int(frame_rgb[1]), int(frame_rgb[2])] if frame_rgb is not None else None
                result["frame_hits"] = int(hits)
                result["has_frame"] = (hits == 4)
                result["laser_endpoints"] = [[int(px), int(py)] for (px, py) in endpoints]
        
        except Exception as e:
            print(f"[ERROR] Processing item {idx+1} failed: {e}")

        results.append(result)

    print(f"[DEBUG] Collected {len(results)} results")

    # Uzupełnij tekst opisujący cały box (np. całą odpowiedź) – jeżeli
    # OCR pierwotnie zwrócił pusty string, spróbujemy ponownie na flood-bboxie.
    if results:
        for result in results:
            current_text = (result.get("text") or "").strip()
            bbox_for_text = result.get("dropdown_box") or result.get("text_box")
            extra_text = _ocr_text_for_bbox(img, bbox_for_text, pad=6, min_conf=0.15)
            if extra_text:
                result["box_text"] = extra_text
                if not current_text:
                    result["text"] = extra_text

    triangles = []
    if ENABLE_TRIANGLE_DETECT:
        print("[DEBUG] Detecting triangles...")
        try:
            triangles = find_filled_isosceles_triangles_fast(
                img_rgb=img,
                lab_img=lab,
                bg_rgb=tuple(int(x) for x in hist["dominant_rgb"]),
                bits=TRI_BITS,
                neighbor_qdist=TRI_NEIGHBOR_QDIST,
                min_area=TRI_MIN_AREA,
                min_w=TRI_MIN_W,
                min_h=TRI_MIN_H,
                min_geom_area=TRI_MIN_GEOM_AREA,
                fill_ratio=TRI_FILL_RATIO,
                iso_tol=TRI_ISO_TOL,
                bg_delta_e=TRI_BG_DELTA_E
            )
            print(f"[DEBUG] Found {len(triangles)} triangles")
        except Exception as e:
            print(f"[ERROR] Triangle detection failed: {e}")

    timer.total("Pipeline TOTAL")

    return {
        "image": image_path,
        "color_histogram": hist,
        "dominant_bg_over_50": bool(is_plain_bg_global),
        "results": results,
        "triangles": triangles,
        "ocr_debug": _last_ocr_debug if DEBUG_OCR else None
    }

# ===================== ANOTACJA =================================================
def _build_text_mask_from_results(W: int, H: int, results: Optional[List[dict]]) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    if not results: return m
    
    for r in results:
        x1, y1, x2, y2 = r["text_box"]
        x1 = clamp(x1, 0, W-1)
        x2 = clamp(x2, 1, W)
        y1 = clamp(y1, 0, H-1)
        y2 = clamp(y2, 1, H)
        m[y1:y2, x1:x2] = True
    
    if TEXT_MASK_DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1+2*TEXT_MASK_DILATE, 1+2*TEXT_MASK_DILATE))
        m = cv2.dilate(m.astype(np.uint8), k, iterations=1).astype(bool)
    
    return m

def _ray_stop_on_edge(lab_img: np.ndarray, text_mask: np.ndarray, cx: int, cy: int, dx: int, dy: int,
                     max_len: int = EDGE_MAX_LEN_PX, thr_delta_e: float = EDGE_DELTA_E, consec_req: int = EDGE_CONSEC_N):
    H, W = lab_img.shape[:2]
    x, y = cx, cy
    steps = 0
    
    while steps < max_len and 0 <= x < W and 0 <= y < H and text_mask[y, x]:
        x += dx
        y += dy
        steps += 1
    
    if not (0 <= x < W and 0 <= y < H):
        return (clamp(x, 0, W-1), clamp(y, 0, H-1))
    
    base = lab_img[y, x].copy()
    consec = 0
    last = (x, y)
    
    while steps < max_len and 0 <= x < W and 0 <= y < H:
        if not text_mask[y, x]:
            dE = deltaE76(lab_img[y, x], base)
            if dE > thr_delta_e:
                consec += 1
                if consec >= consec_req: 
                    return (x, y)
            else:
                consec = 0
                last = (x, y)
        x += dx
        y += dy
        steps += 1
    
    return (clamp(last[0], 0, W-1), clamp(last[1], 0, H-1))

def annotate_and_save(image_path: str, results: List[dict], triangles: Optional[List[dict]] = None,
                      output_dir: Optional[str] = None) -> str:
    print(f"[DEBUG ANNOT] Starting annotation for {len(results)} results")
    im = Image.open(image_path).convert("RGBA")
    W, H = im.size
    im_rgb = np.array(im.convert("RGB"))
    im_lab = rgb_to_lab(im_rgb)
    text_mask = _build_text_mask_from_results(W, H, results)

    base = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    # Regiony z obramowaniem
    frame_count = 0
    for r in results:
        if r.get("has_frame") and r.get("dropdown_box"):
            bx1, by1, bx2, by2 = r["dropdown_box"]
            bx1 = clamp(bx1, 0, W-1)
            by1 = clamp(by1, 0, H-1)
            bx2 = clamp(bx2, 1, W)
            by2 = clamp(by2, 1, H)
            od.rectangle((bx1, by1, bx2, by2), fill=REGION_FILL_RGBA, outline=REGION_EDGE_RGBA, width=3)
            frame_count += 1
    print(f"[DEBUG ANNOT] Drew {frame_count} frame regions")

    # Wszystkie OCR boxy
    box_count = 0
    for r in results:
        x1, y1, x2, y2 = r["text_box"]
        draw.rectangle((x1, y1, x2, y2), outline=OCR_BOX_COLOR, width=2)
        label = (r.get("text") or "")[:40]
        if label: 
            draw.text((x1+2, max(0, y1-14)), label, fill=OCR_BOX_COLOR)
        box_count += 1
    print(f"[DEBUG ANNOT] Drew {box_count} OCR boxes")

    # Lasery
    laser_count = 0
    for r in results:
        if r.get("laser_endpoints") is None:
            continue
            
        x1, y1, x2, y2 = r["text_box"]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        ends = [
            _ray_stop_on_edge(im_lab, text_mask, cx, cy, +1, 0),
            _ray_stop_on_edge(im_lab, text_mask, cx, cy, -1, 0),
            _ray_stop_on_edge(im_lab, text_mask, cx, cy, 0, -1),
            _ray_stop_on_edge(im_lab, text_mask, cx, cy, 0, +1),
        ]
        color = LASER_ACCEPTED_RGBA if r.get("has_frame") else LASER_REJECTED_RGBA
        width = LASER_WIDTH_ACCEPT if r.get("has_frame") else LASER_WIDTH_REJECT
        for ex, ey in ends:
            od.line([(cx, cy), (ex, ey)], fill=color, width=width)
        laser_count += 1
    print(f"[DEBUG ANNOT] Drew {laser_count} laser sets")

    # Trójkąty
    tri_count = 0
    if triangles:
        for tri in triangles:
            hull = tri["hull"]
            od.polygon(hull, fill=TRI_FILL_RGBA, outline=TRI_EDGE_RGBA)
            od.line(hull + [hull[0]], fill=TRI_EDGE_RGBA, width=TRI_EDGE_WIDTH)
            tri_count += 1
    print(f"[DEBUG ANNOT] Drew {tri_count} triangles")

    print("[DEBUG ANNOT] Compositing layers...")
    merged = Image.alpha_composite(im, base)
    out = Image.alpha_composite(merged, overlay).convert("RGB")
    
    target_dir = output_dir or os.path.dirname(image_path)
    os.makedirs(target_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(target_dir, f"{base_name}_annot.png")
    out.save(out_path)
    print(f"[DEBUG ANNOT] Saved: {out_path}")
    return out_path

# ===================== CLI ======================================================
# ===================== CLI ======================================================
def main():
    print("="*60)
    print("[DEBUG] SCRIPT START")
    print("="*60)
    
    import sys
    
    try:
        path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
        print(f"[DEBUG] Image path: {path}")
        
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}")
            return
        
        out = run_dropdown_detection(path)
        out_path = annotate_and_save(path, out.get("results", []), out.get("triangles"), output_dir=JSON_OUT_DIR)
        
        # ========== ZAPIS JSON DO PLIKU (NOWA ŚCIEŻKA) ==========
        os.makedirs(JSON_OUT_DIR, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(JSON_OUT_DIR, f"{base_name}.json")

        print(f"[DEBUG] Saving JSON to: {json_path}")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(to_py(out), f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] JSON saved successfully!")
        # ========================================================

        # ========== CZYŚCIMY CROPY Z POPRZEDNIEJ RUNDY ==========
        try:
            if os.path.isdir(CNN_CROPS_DIR):
                shutil.rmtree(CNN_CROPS_DIR, ignore_errors=True)
            os.makedirs(CNN_CROPS_DIR, exist_ok=True)
            print(f"[DEBUG] Cleaned: {CNN_CROPS_DIR}")
        except Exception as e:
            print(f"[WARN] Could not clean {CNN_CROPS_DIR}: {e}")

        # ========== AUTO-START RUNNERA CNN ==========
        try:
            if os.path.isfile(CNN_RUNNER):
                print("[DEBUG] Launching CNN runner...")
                cmd = [
                    sys.executable, CNN_RUNNER,
                    "--json-dir", JSON_OUT_DIR,        # skąd czytać boxy (screen_boxes)
                    "--out-dir",  CNN_CROPS_DIR,       # dokąd zapisać cropy +10%
                    "--model",    fr"{ROOT}\tri_cnn.pt",
                    "--img-size", "128",
                    "--padding",  "0.10",              # +10% wokół textboxa
                    "--batch",    "256",
                    "--thresh",   "0.50"
                ]
                t0 = time.perf_counter()
                subprocess.run(cmd, check=False)
                print(f"[DEBUG] CNN runner done in {(time.perf_counter()-t0)*1000:.1f} ms")
            else:
                print(f"[WARN] CNN runner not found: {CNN_RUNNER}")
        except Exception as e:
            print(f"[WARN] CNN runner failed: {e}")
        
        print("\n" + "="*60)
        print("JSON OUTPUT (console):")
        print("="*60)
        print(json.dumps(to_py(out), ensure_ascii=False, indent=2))
        print("\n" + "="*60)
        print(f"Annotation: {out_path}")
        print(f"JSON file:  {json_path}")
        print("="*60)
    
    except Exception as e:
        print(f"\n[ERROR] Script failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[DEBUG] SCRIPT END")


# ===================== MONITORING ==============================================
class ResourceMonitor:
    def __init__(self, pid: int, interval: float, output_path: Path):
        self.pid = int(pid)
        self.interval = max(0.2, float(interval))
        self.output_path = Path(output_path)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional["psutil.Process"] = None
        self._samples: List[Dict[str, float]] = []
        self._started_at = 0.0
        self._gpu_handles = []
        self._gpu_ready = False

    def start(self) -> bool:
        if psutil is None:
            print("[WARN] Debug monitor wymaga biblioteki psutil (pomijam).")
            return False
        try:
            self._proc = psutil.Process(self.pid)
            self._proc.cpu_percent(None)
        except Exception as exc:
            print(f"[WARN] Debug monitor nie może połączyć się z procesem: {exc}")
            return False
        self._gpu_ready = self._init_gpu()
        self._started_at = time.time()
        self._thread = threading.Thread(
            target=self._run, name="RegionGrowMonitor", daemon=True
        )
        self._thread.start()
        print(
            f"[DEBUG] Resource monitor aktywny (dt={self.interval:.2f}s, output={self.output_path})"
        )
        return True

    def stop(self):
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
        if self._gpu_ready and pynvml is not None:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
        if self._samples:
            self._write_summary()

    def _init_gpu(self) -> bool:
        if pynvml is None:
            return False
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)
            ]
            return bool(self._gpu_handles)
        except Exception:
            return False

    def _gpu_usage_now(self) -> float:
        if not (self._gpu_ready and pynvml is not None):
            return 0.0
        total = 0.0
        for handle in self._gpu_handles:
            processes = []
            for getter_name in (
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
                "nvmlDeviceGetGraphicsRunningProcesses",
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses",
            ):
                getter = getattr(pynvml, getter_name, None)
                if getter is None:
                    continue
                try:
                    processes = getter(handle)  # type: ignore
                    if processes:
                        break
                except Exception:
                    continue
            for proc in processes or []:
                pid = getattr(proc, "pid", None)
                mem = getattr(proc, "usedGpuMemory", 0)
                if pid == self.pid and mem:
                    total = max(total, float(mem) / (1024 * 1024))
        return total

    def _run(self):
        while not self._stop_event.is_set():
            try:
                cpu = self._proc.cpu_percent(None) if self._proc else 0.0
                rss = (
                    self._proc.memory_info().rss / (1024 * 1024)
                    if self._proc
                    else 0.0
                )
            except Exception:
                break
            gpu = self._gpu_usage_now()
            self._samples.append(
                {
                    "ts": time.time(),
                    "cpu_percent": float(cpu),
                    "rss_mb": float(rss),
                    "gpu_mem_mb": float(gpu),
                }
            )
            self._stop_event.wait(self.interval)

    def _write_summary(self):
        def _avg(key: str) -> float:
            if not self._samples:
                return 0.0
            return float(
                sum(sample.get(key, 0.0) for sample in self._samples) / len(self._samples)
            )

        def _max(key: str) -> float:
            if not self._samples:
                return 0.0
            return float(max(sample.get(key, 0.0) for sample in self._samples))

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "pid": self.pid,
            "process_name": self._proc.name() if self._proc else "",
            "duration_sec": round(time.time() - self._started_at, 3),
            "interval_sec": self.interval,
            "sample_count": len(self._samples),
            "cpu_percent_avg": round(_avg("cpu_percent"), 3),
            "cpu_percent_max": round(_max("cpu_percent"), 3),
            "rss_mb_avg": round(_avg("rss_mb"), 3),
            "rss_mb_max": round(_max("rss_mb"), 3),
            "gpu_mem_mb_avg": round(_avg("gpu_mem_mb"), 3),
            "gpu_mem_mb_max": round(_max("gpu_mem_mb"), 3),
            "gpu_monitoring": bool(self._gpu_ready),
            "samples": [
                {
                    "ts": datetime.fromtimestamp(sample["ts"]).isoformat(timespec="seconds"),
                    "cpu_percent": round(sample["cpu_percent"], 3),
                    "rss_mb": round(sample["rss_mb"], 3),
                    "gpu_mem_mb": round(sample["gpu_mem_mb"], 3),
                }
                for sample in self._samples
            ],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] Resource monitor zapisany do: {self.output_path}")


def main_cli():
    """
    Wariant CLI, który potrafi przetwarzać cały folder z RAW screenami.
    - bez argumentów: wszystkie obrazy z RAW_SCREEN_DIR
    - argument = plik: pojedynczy obraz
    - argument = folder: wszystkie obsługiwane obrazy z tego folderu
    """
    print("=" * 60)
    print("[DEBUG] CLI (batch) START")
    print("=" * 60)

    import sys
    from pathlib import Path as _Path

    def _iter_images_in_dir(dir_path: _Path):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        if not dir_path.is_dir():
            return []
        return sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts)

    raw_args = sys.argv[1:]
    debug_enabled = False
    debug_interval = 1.0
    debug_output: Optional[_Path] = None
    cleaned_args: List[str] = []
    i = 0
    while i < len(raw_args):
        tok = raw_args[i]
        if tok == "--debug":
            debug_enabled = True
            i += 1
            continue
        if tok.startswith("--debug-interval="):
            debug_enabled = True
            try:
                debug_interval = float(tok.split("=", 1)[1])
            except ValueError:
                print("[ERROR] Invalid value for --debug-interval")
                return
            i += 1
            continue
        if tok == "--debug-interval":
            if i + 1 >= len(raw_args):
                print("[ERROR] Missing value for --debug-interval")
                return
            try:
                debug_interval = float(raw_args[i + 1])
            except ValueError:
                print("[ERROR] Invalid value for --debug-interval")
                return
            debug_enabled = True
            i += 2
            continue
        if tok.startswith("--debug-output="):
            debug_enabled = True
            debug_output = _Path(tok.split("=", 1)[1]).expanduser()
            i += 1
            continue
        if tok == "--debug-output":
            if i + 1 >= len(raw_args):
                print("[ERROR] Missing value for --debug-output")
                return
            debug_enabled = True
            debug_output = _Path(raw_args[i + 1]).expanduser()
            i += 2
            continue
        cleaned_args.append(tok)
        i += 1

    arg = cleaned_args[0] if cleaned_args else None

    monitor: Optional[ResourceMonitor] = None
    if debug_enabled:
        default_name = f"usage_debug_{int(time.time())}.json"
        out_path = debug_output or _Path(JSON_OUT_DIR) / default_name
        monitor = ResourceMonitor(os.getpid(), debug_interval, out_path)
        if not monitor.start():
            monitor = None

    try:

        if arg is None:
            images = _iter_images_in_dir(_Path(RAW_SCREEN_DIR))
            if images:
                print(
                    f"[DEBUG] No CLI path -> processing folder: {RAW_SCREEN_DIR} "
                    f"({len(images)} images)"
                )
            else:
                print(f"[WARN] No images found in folder: {RAW_SCREEN_DIR}")
                if os.path.isfile(DEFAULT_IMAGE_PATH):
                    images = [_Path(DEFAULT_IMAGE_PATH)]
                    print(
                        f"[DEBUG] Falling back to single default image: {DEFAULT_IMAGE_PATH}"
                    )
                else:
                    print(f"[ERROR] Default image not found: {DEFAULT_IMAGE_PATH}")
                    print("\n[DEBUG] CLI END")
                    return
        else:
            p = _Path(arg)
            if p.is_dir():
                images = _iter_images_in_dir(p)
                if not images:
                    print(f"[ERROR] Folder is empty or has no supported images: {p}")
                    print("\n[DEBUG] CLI END")
                    return
                print(f"[DEBUG] Processing folder: {p} ({len(images)} images)")
            else:
                images = [p]
                print(f"[DEBUG] Processing single image: {p}")

        os.makedirs(JSON_OUT_DIR, exist_ok=True)
        processed = 0

        for img_path in images:
            path_str = str(img_path)
            if not os.path.isfile(path_str):
                print(f"[WARN] File not found, skipping: {path_str}")
                continue

            print("\n" + "=" * 60)
            print(f"[DEBUG] Image path: {path_str}")

            out = run_dropdown_detection(path_str)
            out_path = annotate_and_save(path_str, out.get("results", []), out.get("triangles"), output_dir=JSON_OUT_DIR)

            base_name = os.path.splitext(os.path.basename(path_str))[0]
            json_path = os.path.join(JSON_OUT_DIR, f"{base_name}.json")

            print(f"[DEBUG] Saving JSON to: {json_path}")
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(to_py(out), f, ensure_ascii=False, indent=2)
                print(f"[DEBUG] JSON saved successfully!")
                print(f"[DEBUG] Annotation: {out_path}")
                processed += 1
            except Exception as e:
                print(f"[ERROR] Could not save JSON for {path_str}: {e}")

        if processed == 0:
            print("[WARN] No images processed, skipping CNN runner.")
            print("\n[DEBUG] CLI END")
            return

        # ========== CZYŚCIMY CROPY Z POPRZEDNIEJ RUNDY ==========
        try:
            if os.path.isdir(CNN_CROPS_DIR):
                shutil.rmtree(CNN_CROPS_DIR, ignore_errors=True)
            os.makedirs(CNN_CROPS_DIR, exist_ok=True)
            print(f"[DEBUG] Cleaned: {CNN_CROPS_DIR}")
        except Exception as e:
            print(f"[WARN] Could not clean {CNN_CROPS_DIR}: {e}")

        # ========== AUTO-START RUNNERA CNN ==========
        try:
            if os.path.isfile(CNN_RUNNER):
                print("[DEBUG] Launching CNN runner...")
                cmd = [
                    sys.executable,
                    CNN_RUNNER,
                    "--json-dir",
                    JSON_OUT_DIR,
                    "--out-dir",
                    CNN_CROPS_DIR,
                    "--model",
                    fr"{ROOT}\tri_cnn.pt",
                    "--img-size",
                    "128",
                    "--padding",
                    "0.10",
                    "--batch",
                    "256",
                    "--thresh",
                    "0.50",
                ]
                t0 = time.perf_counter()
                subprocess.run(cmd, check=False)
                print(f"[DEBUG] CNN runner done in {(time.perf_counter()-t0)*1000:.1f} ms")
            else:
                print(f"[WARN] CNN runner not found: {CNN_RUNNER}")
        except Exception as e:
            print(f"[WARN] CNN runner failed: {e}")

        print("\n[DEBUG] CLI END")

    except Exception as e:
        print(f"\n[ERROR] CLI failed: {e}")
        import traceback

        traceback.print_exc()
        print("\n[DEBUG] CLI END")
    finally:
        if monitor:
            monitor.stop()


if __name__ == "__main__":
    main_cli()
