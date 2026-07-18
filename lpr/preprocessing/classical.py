"""Preprocesamiento clasico de imagenes previo a la deteccion de placas.

Migrado y mejorado desde classical/image_processing.py: los umbrales de Canny
ahora se calculan de forma adaptativa segun la mediana de intensidad de la
imagen (metodo estandar de "auto Canny"), en vez de usar constantes fijas que
solo funcionaban bien con las fotos originales del dataset de prueba.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def procesar_imagen(imagen: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte a gris, reduce ruido preservando bordes y detecta bordes.

    Devuelve (gris, bordes).
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.bilateralFilter(gris, 11, 17, 17)
    bordes = auto_canny(suavizada)
    return gris, bordes
