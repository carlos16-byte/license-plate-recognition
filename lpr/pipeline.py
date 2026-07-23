"""Orquestacion de alto nivel: detectar->(segmentar)->reconocer, con timing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from lpr.faces.detector import FaceBox, FaceDetector
from lpr.faces.recognizer import FaceRecognizer
from lpr.ocr.engine_base import OCREngine
from lpr.plates.detector_base import PlateCandidate, PlateDetector


@dataclass
class PlateResult:
    candidate: Optional[PlateCandidate]
    text: str
    ocr_confidence: float
    detector_name: str
    ocr_name: str
    detect_time_s: float
    ocr_time_s: float


class PlatePipeline:
    def __init__(self, detector: PlateDetector, ocr_engine: OCREngine):
        self.detector = detector
        self.ocr_engine = ocr_engine

    def run(self, image: np.ndarray) -> PlateResult:
        t0 = time.perf_counter()
        candidate = self.detector.detect_best(image)
        detect_time = time.perf_counter() - t0

        if candidate is None:
            return PlateResult(
                candidate=None,
                text="",
                ocr_confidence=0.0,
                detector_name=self.detector.name,
                ocr_name=self.ocr_engine.name,
                detect_time_s=detect_time,
                ocr_time_s=0.0,
            )

        t1 = time.perf_counter()
        ocr_result = self.ocr_engine.recognize(candidate.image)
        ocr_time = time.perf_counter() - t1

        return PlateResult(
            candidate=candidate,
            text=ocr_result.text,
            ocr_confidence=ocr_result.confidence,
            detector_name=self.detector.name,
            ocr_name=self.ocr_engine.name,
            detect_time_s=detect_time,
            ocr_time_s=ocr_time,
        )

    @staticmethod
    def draw(image: np.ndarray, result: PlateResult) -> np.ndarray:
        annotated = image.copy()
        if result.candidate is None:
            return annotated
        x, y, w, h = result.candidate.box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{result.text} ({result.ocr_confidence:.2f})" if result.text else "?"
        cv2.putText(annotated, label, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated


@dataclass
class FaceResult:
    box: FaceBox
    name: Optional[str]
    distance: float
    confidence_pct: float = 0.0


@dataclass
class FramesFaceResult:
    faces: List[FaceResult] = field(default_factory=list)
    detect_time_s: float = 0.0


class FacePipeline:
    def __init__(self, detector: FaceDetector, recognizer: Optional[FaceRecognizer] = None):
        self.detector = detector
        self.recognizer = recognizer

    def run(self, image: np.ndarray) -> FramesFaceResult:
        t0 = time.perf_counter()
        boxes = self.detector.detect(image)
        detect_time = time.perf_counter() - t0

        results = []
        for box in boxes:
            name, distance, confidence_pct = None, 0.0, 0.0
            if self.recognizer is not None and self.recognizer.is_trained:
                face_gray = self.detector.crop_gray(image, box)
                name, distance = self.recognizer.predict(face_gray)
                if name:
                    # LBPH: distancia mas baja = mas parecido. La convertimos a un
                    # porcentaje legible relativo al umbral de aceptacion.
                    threshold = self.recognizer.confidence_threshold
                    confidence_pct = max(0.0, 1.0 - distance / threshold) * 100
            results.append(
                FaceResult(box=box, name=name, distance=distance, confidence_pct=confidence_pct)
            )

        return FramesFaceResult(faces=results, detect_time_s=detect_time)

    @staticmethod
    def draw(image: np.ndarray, result: FramesFaceResult) -> np.ndarray:
        annotated = image.copy()
        for face in result.faces:
            x, y, w, h = face.box.box
            color = (0, 255, 0) if face.name else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            # distancia LBPH cruda (mas baja = mas parecido); mas facil de calibrar
            # a ojo que un porcentaje, que cerca del umbral queda enganoso (ej: 1%
            # aunque el reconocimiento sea correcto)
            label = f"{face.name} (dist={face.distance:.0f})" if face.name else "New face"
            cv2.putText(annotated, label, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return annotated
