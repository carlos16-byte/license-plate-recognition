"""Detector de placas basado en YOLO (Deep Learning, Fase 3).

Carga los pesos entrenados por scripts/train_yolo_plate.py. Si no existen
(porque todavia no se corrio el entrenamiento), delega automaticamente en
ClassicalPlateDetector para que el pipeline siga siendo funcional de punta a
punta sin exigir GPU ni un modelo previo.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from lpr.config import YOLO_PLATE_WEIGHTS
from lpr.logging_utils import get_logger
from lpr.plates.classical_detector import ClassicalPlateDetector
from lpr.plates.detector_base import PlateCandidate, PlateDetector

logger = get_logger(__name__)


class YoloPlateDetector(PlateDetector):
    name = "yolo"

    def __init__(self, weights_path: Path | str = YOLO_PLATE_WEIGHTS, conf_threshold: float = 0.25):
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self._model = None
        self._fallback = ClassicalPlateDetector()

        if self.weights_path.exists():
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(self.weights_path))
            except Exception:
                logger.exception("No se pudo cargar el modelo YOLO, se usara el detector clasico como fallback")
                self._model = None
        else:
            logger.warning(
                "No se encontraron pesos YOLO en %s. Corre scripts/train_yolo_plate.py primero; "
                "mientras tanto se usa el detector clasico como fallback.",
                self.weights_path,
            )

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def detect(self, image: np.ndarray) -> List[PlateCandidate]:
        if self._model is None:
            return self._fallback.detect(image)

        results = self._model.predict(image, conf=self.conf_threshold, verbose=False)
        candidates: List[PlateCandidate] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                candidates.append(
                    PlateCandidate(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=conf, image=crop)
                )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        if not candidates:
            return self._fallback.detect(image)
        return candidates
