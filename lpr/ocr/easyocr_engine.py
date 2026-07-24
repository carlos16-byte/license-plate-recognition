"""Motor de OCR por defecto: EasyOCR (offline tras la primera descarga de pesos)."""
from __future__ import annotations

import re

import numpy as np

from lpr.ocr.engine_base import OCREngine, OCRResult

_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class EasyOCREngine(OCREngine):
    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, gpu: bool = False, min_height_ratio: float = 0.6):
        import easyocr  # import perezoso: es pesado y no siempre se necesita

        self._reader = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)
        # las cajas de placa suelen incluir texto chico decorativo (pais, ciudad,
        # año/mes) ademas del numero de placa. En vez de depender de que el
        # recorte venga ya ajustado (algo que un detector real nunca garantiza),
        # nos quedamos solo con el texto mas alto/prominente: el numero de placa
        # es casi siempre el elemento de texto mas grande del recorte.
        self.min_height_ratio = min_height_ratio

    @staticmethod
    def _bbox_height(bbox) -> float:
        ys = [pt[1] for pt in bbox]
        return max(ys) - min(ys)

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        detections = self._reader.readtext(plate_image, allowlist=_ALLOWLIST)
        if not detections:
            return OCRResult(text="", confidence=0.0)

        max_height = max(self._bbox_height(bbox) for bbox, _, _ in detections)
        main_detections = [
            det for det in detections
            if self._bbox_height(det[0]) >= self.min_height_ratio * max_height
        ]
        if not main_detections:
            main_detections = detections

        main_detections.sort(key=lambda d: d[0][0][0])  # ordenar de izquierda a derecha por bbox
        text = "".join(det[1] for det in main_detections)
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        confidence = float(np.mean([det[2] for det in main_detections]))
        return OCRResult(text=text, confidence=confidence)
