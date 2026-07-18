"""Deteccion de rostros con el clasificador Haar Cascade incluido en OpenCV
(cero descargas adicionales). Suficiente para deteccion frontal en tiempo real;
la identificacion de la persona la hace lpr.faces.recognizer por separado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class FaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 8, min_size: int = 60):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"No se pudo cargar el cascade de rostros desde {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        return [FaceBox(x=int(x), y=int(y), w=int(w), h=int(h)) for (x, y, w, h) in faces]

    @staticmethod
    def crop_gray(image: np.ndarray, face: FaceBox, size: int = 200) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        crop = gray[face.y : face.y + face.h, face.x : face.x + face.w]
        return cv2.resize(crop, (size, size))
