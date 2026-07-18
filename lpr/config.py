"""Rutas y parametros compartidos por todo el paquete."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

PLATES_SYNTHETIC_DIR = DATA_DIR / "plates" / "synthetic"
FACES_DIR = DATA_DIR / "faces"

YOLO_PLATE_WEIGHTS = MODELS_DIR / "yolo_plate.pt"
CHAR_CLASSIFIER_PATH = MODELS_DIR / "char_classifier.joblib"
FACE_MODEL_DIR = MODELS_DIR / "face_model"
FACE_MODEL_PATH = FACE_MODEL_DIR / "lbph.yml"
FACE_LABELS_PATH = FACE_MODEL_DIR / "labels.json"


@dataclass
class PlateDetectionConfig:
    """Umbrales del detector clasico de placas por contornos."""

    min_area_ratio: float = 0.001   # area minima del candidato relativa a la imagen
    max_area_ratio: float = 0.25    # area maxima del candidato relativa a la imagen
    min_aspect_ratio: float = 1.5   # ancho/alto tipico de una placa
    max_aspect_ratio: float = 6.0
    max_candidates: int = 15


@dataclass
class SegmentationConfig:
    """Umbrales de la segmentacion clasica de caracteres."""

    min_char_height_ratio: float = 0.35  # alto minimo del caracter relativo al alto de la placa
    max_char_height_ratio: float = 0.98
    min_char_width_px: int = 4


for _dir in (DATA_DIR, MODELS_DIR, RESULTS_DIR, PLATES_SYNTHETIC_DIR, FACES_DIR, FACE_MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
