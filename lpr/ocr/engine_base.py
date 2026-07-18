"""Interfaz comun para motores de OCR (EasyOCR, Tesseract, clasificador ML)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class OCRResult:
    text: str
    confidence: float


class OCREngine(ABC):
    name: str = "base"

    @abstractmethod
    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        """Recibe el recorte BGR de una placa y devuelve el texto reconocido."""
        raise NotImplementedError
