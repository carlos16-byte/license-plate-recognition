"""Fase 2 - Motor de OCR basado en un clasificador ML (SVM/RandomForest) entrenado
por scripts/train_char_classifier.py sobre caracteres sinteticos.

A diferencia de EasyOCR/Tesseract (que reciben la placa entera), este motor
opera sobre los recortes YA segmentados por lpr.plates.segmentation, porque
un clasificador de un solo caracter no puede ubicar donde empieza y termina
cada letra por si mismo.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import joblib
import numpy as np

from lpr.config import CHAR_CLASSIFIER_PATH
from lpr.ocr.engine_base import OCREngine, OCRResult
from lpr.plates.segmentation import segmentar_caracteres


class MLCharClassifierEngine(OCREngine):
    name = "ml_char_classifier"

    def __init__(self, model_path: Path | str = CHAR_CLASSIFIER_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontro el clasificador de caracteres en {model_path}. "
                "Corre primero: python scripts/train_char_classifier.py"
            )
        bundle = joblib.load(model_path)
        self._model = bundle["model"]
        self._glyph_size = bundle["glyph_size"]
        self._hog = cv2.HOGDescriptor(
            _winSize=(self._glyph_size, self._glyph_size),
            _blockSize=(14, 14),
            _blockStride=(7, 7),
            _cellSize=(7, 7),
            _nbins=9,
        )

    def _extract_features(self, glyph: np.ndarray) -> np.ndarray:
        glyph = cv2.resize(glyph, (self._glyph_size, self._glyph_size), interpolation=cv2.INTER_AREA)
        hog_features = self._hog.compute(glyph).flatten()
        pixel_features = cv2.resize(glyph, (14, 14)).flatten() / 255.0
        return np.concatenate([hog_features, pixel_features])

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        caracteres: List[np.ndarray] = segmentar_caracteres(plate_image)
        if not caracteres:
            return OCRResult(text="", confidence=0.0)

        features = np.array([self._extract_features(c) for c in caracteres], dtype=np.float32)
        predictions = self._model.predict(features)

        confidence = 0.0
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(features)
            confidence = float(np.mean(probs.max(axis=1)))

        text = "".join(predictions)
        return OCRResult(text=text, confidence=confidence)
