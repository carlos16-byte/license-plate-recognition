"""Reconocimiento facial offline con LBPH (cv2.face), sin descargas de modelos
grandes. Requiere haber enrolado al menos una persona con `face-enroll`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from lpr.config import FACE_LABELS_PATH, FACE_MODEL_PATH
from lpr.faces.detector import FaceDetector
from lpr.faces.gallery import FaceGallery, load_label_map, save_label_map


class FaceRecognizer:
    def __init__(self, confidence_threshold: float = 80.0):
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._label_map: dict[int, str] = {}
        self.confidence_threshold = confidence_threshold  # LBPH: menor = mas parecido

    @property
    def is_trained(self) -> bool:
        return bool(self._label_map)

    def train(self, gallery: FaceGallery, detector: FaceDetector) -> int:
        faces, names = gallery.load_training_faces(detector)
        if not faces:
            raise RuntimeError(
                "La galeria de rostros esta vacia. Usa `face-enroll` para registrar al menos una persona."
            )

        unique_names = sorted(set(names))
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        labels = np.array([name_to_id[n] for n in names])

        self._recognizer.train(faces, labels)
        self._label_map = {idx: name for name, idx in name_to_id.items()}
        return len(faces)

    def predict(self, face_gray: np.ndarray) -> tuple[Optional[str], float]:
        if not self.is_trained:
            return None, 0.0
        label_id, distance = self._recognizer.predict(face_gray)
        if distance > self.confidence_threshold:
            return None, distance
        return self._label_map.get(label_id), distance

    def save(self, model_path: Path = FACE_MODEL_PATH, labels_path: Path = FACE_LABELS_PATH) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self._recognizer.write(str(model_path))
        save_label_map(self._label_map, labels_path)

    def load(self, model_path: Path = FACE_MODEL_PATH, labels_path: Path = FACE_LABELS_PATH) -> None:
        if not model_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"No hay modelo facial entrenado en {model_path}. Corre `face-enroll` y luego entrena la galeria."
            )
        self._recognizer.read(str(model_path))
        self._label_map = load_label_map(labels_path)
