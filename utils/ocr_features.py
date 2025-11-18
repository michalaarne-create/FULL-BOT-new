"""Shared OCR helpers for feature extraction."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import logging
import queue
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

STATE_DIM = 12
ZERO_STATE = np.zeros(STATE_DIM, dtype=np.float32)

try:
    import easyocr
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class OCRNotAvailableError(RuntimeError):
    """Raised when OCR backend is unavailable but required."""


def _close_reader(reader: "easyocr.Reader") -> None:  # type: ignore[name-defined]
    try:
        reader.close()  # type: ignore[attr-defined]
    except Exception:
        pass


def _resolve_model_dir(model_dir: Optional[str | os.PathLike[str]]) -> str:
    if model_dir:
        return str(Path(model_dir))
    env_dir = os.environ.get("EASYOCR_MODEL_PATH")
    if env_dir:
        return str(Path(env_dir))
    default_dir = Path(__file__).parent.parent / "models" / "easyocr_cache"
    default_dir.mkdir(parents=True, exist_ok=True)
    return str(default_dir)


@dataclass
class OCRResult:
    state: np.ndarray
    boxes: List[Dict[str, Any]]
    frame_id: Optional[int]
    timestamp: float


@lru_cache(maxsize=None)
def create_easyocr_reader(
    lang: str = "en",
    *,
    model_dir: Optional[str | os.PathLike[str]] = None,
    **kwargs: Any,
) -> "easyocr.Reader":  # type: ignore[name-defined]
    """
    Lazily construct an EasyOCR reader with shared caching.

    Parameters
    ----------
    lang:
        OCR language code (default: "en").
    model_dir:
        Optional directory containing the EasyOCR weights. If not provided,
        the helper looks at ``EASYOCR_MODEL_PATH`` env var, otherwise uses
        ``models/easyocr_cache`` inside the project.
    kwargs:
        Additional keyword arguments forwarded to ``easyocr.Reader``.
    """
    if easyocr is None:
        raise OCRNotAvailableError(
            "EasyOCR is required but not installed. Install with `pip install easyocr`."
        )

    gpu = kwargs.pop("gpu", None)
    if gpu is None:
        gpu = torch.cuda.is_available()

    model_storage_directory = _resolve_model_dir(model_dir)

    try:
        reader = easyocr.Reader(
            [lang],
            gpu=gpu,
            download_enabled=False,
            model_storage_directory=model_storage_directory,
            **kwargs,
        )
        reader._download_warning = False  # type: ignore[attr-defined]
    except Exception:
        try:
            reader = easyocr.Reader(
                [lang],
                gpu=gpu,
                download_enabled=True,
                model_storage_directory=model_storage_directory,
                **kwargs,
            )
            reader._download_warning = True  # type: ignore[attr-defined]
        except Exception as exc:
            raise OCRNotAvailableError(
                f"Failed to initialise EasyOCR. Manually place required weights in '{model_storage_directory}' "
                "or set the EASYOCR_MODEL_PATH environment variable."
            ) from exc

    atexit.register(_close_reader, reader)
    return reader


def _aggregate_polygons(polygons: List[np.ndarray], frame_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = frame_shape[:2]

    boxes: List[Tuple[float, float, float, float]] = []
    heights: List[float] = []
    bottoms: List[float] = []
    area_sum = 0.0

    for poly in polygons:
        if poly.shape != (4, 2):
            poly = poly.reshape(-1, 2)
            if poly.shape[0] < 4:
                continue
        xs = np.clip(poly[:, 0], 0, w)
        ys = np.clip(poly[:, 1], 0, h)
        x_min = float(xs.min())
        y_min = float(ys.min())
        x_max = float(xs.max())
        y_max = float(ys.max())
        if x_max <= x_min or y_max <= y_min:
            continue
        boxes.append((x_min, y_min, x_max, y_max))
        height = y_max - y_min
        heights.append(height)
        bottoms.append(y_max)
        area_sum += (x_max - x_min) * height

    if not boxes:
        return np.zeros(STATE_DIM, dtype=np.float32)

    x_union_min = min(b[0] for b in boxes)
    y_union_min = min(b[1] for b in boxes)
    x_union_max = max(b[2] for b in boxes)
    y_union_max = max(b[3] for b in boxes)

    center_x = (x_union_min + x_union_max) / 2.0
    center_y = (y_union_min + y_union_max) / 2.0
    width = x_union_max - x_union_min
    height = y_union_max - y_union_min

    coverage = np.clip(area_sum / float(w * h), 0.0, 1.0)
    count_norm = min(len(boxes) / 10.0, 1.0)
    avg_height = np.clip(np.mean(heights) / h, 0.0, 1.0)
    height_std = np.clip(np.std(heights) / h, 0.0, 1.0)
    bottom_norm = np.clip(max(bottoms) / h, 0.0, 1.0)

    def _norm(value: float, max_value: float) -> float:
        if max_value <= 0:
            return -1.0
        return float(np.clip(value / max_value, 0.0, 1.0) * 2.0 - 1.0)

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:8] = [
        _norm(center_x, w),
        _norm(center_y, h),
        _norm(width, w),
        _norm(height, h),
        coverage * 2.0 - 1.0,
        count_norm * 2.0 - 1.0,
        avg_height * 2.0 - 1.0,
        bottom_norm * 2.0 - 1.0,
    ]
    state[8] = _norm(x_union_min, w)
    state[9] = _norm(y_union_min, h)
    state[10] = height_std * 2.0 - 1.0
    state[11] = (1.0 if len(boxes) > 1 else 0.0) * 2.0 - 1.0
    return state


def compute_ocr_state(
    reader: "easyocr.Reader",  # type: ignore[name-defined]
    frame: np.ndarray,
    *,
    return_boxes: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Run EasyOCR on a frame and return the extracted state vector.

    Parameters
    ----------
    reader:
        EasyOCR reader instance.
    frame:
        Image array (H x W x 3).
    return_boxes:
        If True, also return detailed bounding boxes with text and confidence.
    """
    if reader is None:
        state = np.zeros(STATE_DIM, dtype=np.float32)
        return (state, []) if return_boxes else state

    results = reader.readtext(frame, detail=1, paragraph=False)
    polygons: List[np.ndarray] = []
    boxes_payload: List[Dict[str, Any]] = []

    for item in results:
        if len(item) != 3:
            continue
        box, text, score = item
        poly = np.array(box, dtype=np.float32)
        polygons.append(poly)
        if return_boxes:
            boxes_payload.append(
                {
                    "box": poly.tolist(),
                    "text": text,
                    "confidence": float(score),
                }
            )

    state = _aggregate_polygons(polygons, frame.shape)
    if return_boxes:
        return state, boxes_payload
    return state


class AsyncOCRWorker:
    """
    Background worker that runs OCR asynchronously and stores the latest result.
    """

    def __init__(
        self,
        reader: Optional["easyocr.Reader"],  # type: ignore[name-defined]
        *,
        max_queue: int = 4,
    ) -> None:
        self.reader = reader
        self.enabled = reader is not None
        self._queue: "queue.Queue[Tuple[np.ndarray, Optional[int]]]" = queue.Queue(maxsize=max_queue)
        self._latest: Optional[OCRResult] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="EasyOCRWorker", daemon=True)
            self._thread.start()

    def submit(self, frame: np.ndarray, frame_id: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait((frame.copy(), frame_id))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((frame.copy(), frame_id))
            except queue.Full:
                logger.debug("OCR queue still full; dropping frame.")

    def get_latest(self) -> Optional[OCRResult]:
        with self._lock:
            if self._latest is None:
                return None
            # return copies to avoid accidental mutation
            return OCRResult(
                state=self._latest.state.copy(),
                boxes=list(self._latest.boxes),
                frame_id=self._latest.frame_id,
                timestamp=self._latest.timestamp,
            )

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        assert self.reader is not None
        while not self._stop_event.is_set():
            try:
                frame, frame_id = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                state, boxes = compute_ocr_state(self.reader, frame, return_boxes=True)
            except Exception as exc:
                logger.debug("Async OCR error: %s", exc)
                state = ZERO_STATE.copy()
                boxes = []
            result = OCRResult(
                state=state.astype(np.float32, copy=False),
                boxes=boxes,
                frame_id=frame_id,
                timestamp=time.time(),
            )
            with self._lock:
                self._latest = result

    def __del__(self) -> None:
        self.stop()


__all__ = [
    "STATE_DIM",
    "ZERO_STATE",
    "create_easyocr_reader",
    "compute_ocr_state",
    "OCRNotAvailableError",
    "OCRResult",
    "AsyncOCRWorker",
]
