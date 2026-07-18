"""Detector de placas por contornos (vision por computadora clasica).

Migrado desde classical/plate_detection.py. La version original tomaba el
*primer* contorno de la lista (ya ordenada solo por area) que aproximara a un
poligono de 4 lados, sin verificar que tuviera forma de placa. Eso producia
falsos positivos frecuentes (cualquier objeto rectangular grande, p.ej. un
parabrisas o una ventana). Esta version:

  * filtra candidatos por area relativa a la imagen (ni ruido ni el fondo entero)
  * filtra por aspect ratio tipico de una placa (ancho/alto entre ~1.5 y 6.0)
  * en vez de devolver el primero que matchea, rankea todos los candidatos
    validos y devuelve varios ordenados por un score de "cuanto parece una
    placa" (combinacion de area, aspect ratio y densidad de bordes internos)
  * pondera densidad de bordes internos: una placa tiene muchos bordes chicos
    (los caracteres); un parabrisas o una ventana son regiones mucho mas lisas
    y esto evita el falso positivo mas comun de la version original
"""
from __future__ import annotations

from typing import List

import cv2
import imutils
import numpy as np

from lpr.config import PlateDetectionConfig
from lpr.plates.detector_base import PlateCandidate, PlateDetector

IDEAL_ASPECT_RATIO = 3.14  # placas rectangulares tipicas (~ISO/ancho-alto)
EXPECTED_CHAR_COUNT = 7  # placas tipicas: 6-7 caracteres


def _character_blob_score(crop_gray: np.ndarray) -> float:
    """Cuenta blobs con forma/tamano de caracter dentro del candidato.

    Es la senal mas especifica de "esto es una placa": una region con 5-8
    blobs verticales de tamano parejo separados horizontalmente. Un faro o
    un parabrisas con reflejos genera bordes densos pero no esta forma.
    """
    if crop_gray.size == 0:
        return 0.0
    h, w = crop_gray.shape[:2]
    if h < 8 or w < 8:
        return 0.0

    _, thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_like = 0
    for c in contours:
        _, _, cw, ch = cv2.boundingRect(c)
        height_ratio = ch / h
        if 0.35 <= height_ratio <= 0.95 and cw >= 3 and cw <= 0.5 * w:
            char_like += 1

    return min(char_like / EXPECTED_CHAR_COUNT, 1.0)


class ClassicalPlateDetector(PlateDetector):
    name = "classical"

    def __init__(self, config: PlateDetectionConfig | None = None):
        self.config = config or PlateDetectionConfig()

    def detect(self, image: np.ndarray) -> List[PlateCandidate]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        median = float(np.median(blurred))
        edges = cv2.Canny(blurred, int(max(0, 0.66 * median)), int(min(255, 1.33 * median)))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[: self.config.max_candidates]

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        # bordes "crudos" (sin dilatar) para medir densidad de textura interna:
        # una placa con caracteres tiene muchos bordes chicos, una superficie
        # lisa (parabrisas, capot) casi ninguno.
        texture_edges = cv2.Canny(blurred, int(max(0, 0.66 * median)), int(min(255, 1.33 * median)))

        candidates: List[PlateCandidate] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue

            area_ratio = (w * h) / img_area
            aspect_ratio = w / h

            if not (self.config.min_area_ratio <= area_ratio <= self.config.max_area_ratio):
                continue
            if not (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio):
                continue

            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            rectangularity = 1.0 if len(approx) == 4 else 0.6

            region_edges = texture_edges[y : y + h, x : x + w]
            edge_density = float(np.count_nonzero(region_edges)) / (w * h)
            # una placa suele tener 5-25% de pixeles de borde por los caracteres;
            # normalizamos contra un techo razonable en vez de premiar sin limite
            texture_score = min(edge_density / 0.15, 1.0)

            char_score = _character_blob_score(gray[y : y + h, x : x + w])

            aspect_score = 1.0 - min(abs(aspect_ratio - IDEAL_ASPECT_RATIO) / IDEAL_ASPECT_RATIO, 1.0)
            score = (
                0.2 * aspect_score
                + 0.1 * rectangularity
                + 0.1 * min(area_ratio / self.config.max_area_ratio, 1.0)
                + 0.15 * texture_score
                + 0.45 * char_score
            )

            crop = image[y : y + h, x : x + w]
            candidates.append(PlateCandidate(x=x, y=y, w=w, h=h, confidence=float(score), image=crop))

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates
