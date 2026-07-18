"""Segmentacion de caracteres dentro de una placa ya recortada.

Migrado desde classical/character_segmentation.py. Mejoras:
  * threshold adaptativo (Otsu) en vez de un umbral fijo de 150 que solo
    funcionaba con buena iluminacion
  * filtro de tamano relativo al alto de la placa en vez de pixeles fijos
    (una placa recortada de 40px de alto y una de 400px necesitan distintos
    umbrales absolutos)
  * descarta contornos que ocupan casi todo el ancho/alto (el borde de la
    placa misma, ruido de recorte)
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from lpr.config import SegmentationConfig


def segmentar_caracteres(placa: np.ndarray, config: SegmentationConfig | None = None) -> List[np.ndarray]:
    config = config or SegmentationConfig()

    gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY) if placa.ndim == 3 else placa
    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    _, thresh = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # RETR_LIST (no RETR_EXTERNAL): si la placa tiene un borde/marco dibujado,
    # ese marco cerrado seria el UNICO contorno externo y los caracteres
    # quedarian "adentro" como hijos que RETR_EXTERNAL nunca devuelve.
    contornos, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    plate_h, plate_w = gris.shape[:2]
    candidatos = []

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)

        height_ratio = h / plate_h
        if height_ratio < config.min_char_height_ratio or height_ratio > config.max_char_height_ratio:
            continue
        if w < config.min_char_width_px:
            continue
        if w > 0.5 * plate_w:  # descarta bloques que abarcan medio ancho de placa (ruido/marco)
            continue

        char = thresh[y : y + h, x : x + w]
        candidatos.append((x, char))

    candidatos.sort(key=lambda item: item[0])
    return [c[1] for c in candidatos]
