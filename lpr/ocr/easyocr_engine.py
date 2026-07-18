"""Motor de OCR por defecto: EasyOCR (offline tras la primera descarga de pesos)."""
from __future__ import annotations

import re

import numpy as np

from lpr.ocr.engine_base import OCREngine, OCRResult

_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class EasyOCREngine(OCREngine):
    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        import easyocr  # import perezoso: es pesado y no siempre se necesita

        self._reader = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        detections = self._reader.readtext(plate_image, allowlist=_ALLOWLIST)
        if not detections:
            return OCRResult(text="", confidence=0.0)

        detections.sort(key=lambda d: d[0][0][0])  # ordenar de izquierda a derecha por bbox
        text = "".join(det[1] for det in detections)
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        confidence = float(np.mean([det[2] for det in detections]))
        return OCRResult(text=text, confidence=confidence)
