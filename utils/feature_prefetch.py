"""
Shared singleton that coordinates asynchronous YOLO feature prefetching
between the hover environment and the ONNX feature extractor.
"""

from __future__ import annotations

import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from models.feature_extractor_onnx_optimized import YOLOv11ONNXExtractorOptimized


class FeaturePrefetchManager:
    _instance: Optional["FeaturePrefetchManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._extractor: Optional["YOLOv11ONNXExtractorOptimized"] = None
        self._extractor_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "FeaturePrefetchManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_extractor(self, extractor: Optional["YOLOv11ONNXExtractorOptimized"]) -> None:
        with self._extractor_lock:
            self._extractor = extractor

    def submit(self, frame) -> None:
        """
        Submit frame copy for background prefetching. The extractor will
        silently drop requests when the prefetch queue is full.
        """
        with self._extractor_lock:
            extractor = self._extractor
        if extractor is None:
            return
        extractor.enqueue_prefetch(frame)


__all__ = ["FeaturePrefetchManager"]
