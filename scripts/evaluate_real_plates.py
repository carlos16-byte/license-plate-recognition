"""Mide accuracy de OCR sobre las fotos reales de data/plates/real, usando
las cajas ya etiquetadas (formato YOLO) para recortar la placa y el texto de
verdad en ground_truth.json. A diferencia de benchmark_compare.py (que solo
usa placas 100% sinteticas), esto da una cifra honesta de que tan bien anda
cada motor de OCR contra fotos reales.

Uso:
    python scripts/evaluate_real_plates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.logging_utils import get_logger  # noqa: E402
from scripts.benchmark_compare import char_accuracy  # noqa: E402

logger = get_logger(__name__)

REAL_DIR = ROOT_DIR / "data" / "plates" / "real"


def _crop_from_label(image, label_path: Path):
    h, w = image.shape[:2]
    _, cx, cy, bw, bh = map(float, label_path.read_text().split())
    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    return image[max(y1, 0):y2, max(x1, 0):x2]


def _build_engines():
    engines = {}
    try:
        from lpr.ocr.easyocr_engine import EasyOCREngine
        engines["easyocr"] = EasyOCREngine()
    except Exception:
        logger.exception("EasyOCR no disponible")

    try:
        from lpr.ocr.ml_char_classifier import MLCharClassifierEngine
        engines["ml_char_classifier"] = MLCharClassifierEngine()
    except FileNotFoundError as exc:
        logger.warning("Clasificador ML no disponible: %s", exc)

    return engines


def main():
    gt_path = REAL_DIR / "ground_truth.json"
    ground_truth = json.loads(gt_path.read_text())

    engines = _build_engines()
    if not engines:
        logger.warning("Ningun motor de OCR disponible.")
        return

    results = {name: [] for name in engines}

    for stem, expected in ground_truth.items():
        images = list((REAL_DIR / "images").glob(f"{stem}.*"))
        label_path = REAL_DIR / "labels" / f"{stem}.txt"
        if not images or not label_path.exists():
            logger.warning("Sin imagen/label para %s, se omite", stem)
            continue

        image = cv2.imread(str(images[0]))
        crop = _crop_from_label(image, label_path)

        for name, engine in engines.items():
            result = engine.recognize(crop)
            acc = char_accuracy(expected, result.text)
            exact = result.text == expected
            results[name].append((stem, expected, result.text, acc, exact))

    for name, rows in results.items():
        accs = [r[3] for r in rows]
        exact_count = sum(1 for r in rows if r[4])
        logger.info("=== %s === accuracy_promedio=%.3f  exact=%d/%d",
                     name, sum(accs) / len(accs) if accs else 0.0, exact_count, len(rows))
        for stem, expected, got, acc, exact in rows:
            marker = "OK" if exact else ("~~" if acc >= 0.5 else "XX")
            logger.info("  %s  %-16s esperado=%-10s leido=%-10s acc=%.2f", marker, stem, expected, got, acc)


if __name__ == "__main__":
    main()
