"""Interfaz comun para detectores de placas (clasico, YOLO, ...)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PlateCandidate:
    """Un candidato a placa detectado en una imagen."""

    x: int
    y: int
    w: int
    h: int
    confidence: float
    image: np.ndarray  # recorte BGR de la placa

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def iou(self, other: "PlateCandidate") -> float:
        ax1, ay1, ax2, ay2 = self.x, self.y, self.x + self.w, self.y + self.h
        bx1, by1, bx2, by2 = other.x, other.y, other.x + other.w, other.y + other.h

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = self.w * self.h
        area_b = other.w * other.h
        union = area_a + area_b - inter_area
        return inter_area / union if union else 0.0


class PlateDetector(ABC):
    """Interfaz que deben implementar todos los detectores de placas."""

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[PlateCandidate]:
        """Devuelve candidatos a placa ordenados por confianza descendente."""
        raise NotImplementedError

    def detect_best(self, image: np.ndarray) -> PlateCandidate | None:
        candidates = self.detect(image)
        return candidates[0] if candidates else None
