"""Compara los enfoques clasico vs IA en cada etapa del pipeline y guarda los
resultados en results/comparison.csv (+ un grafico de barras en
results/comparison.png).

  * Deteccion de placas: detector clasico (contornos) vs YOLO, sobre el split
    de validacion sintetico (que tiene ground-truth de bounding box) -> IoU
    promedio, tasa de deteccion y tiempo.
  * OCR: EasyOCR vs Tesseract (si esta instalado) vs clasificador ML, sobre
    placas sinteticas generadas al vuelo con texto conocido -> accuracy a
    nivel de caracter (distancia de edicion normalizada) y tiempo.

Uso:
    python scripts/benchmark_compare.py
    python main.py compare
"""
from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.config import PLATES_SYNTHETIC_DIR, RESULTS_DIR, YOLO_PLATE_WEIGHTS  # noqa: E402
from lpr.logging_utils import get_logger  # noqa: E402
from lpr.plates.classical_detector import ClassicalPlateDetector  # noqa: E402
from lpr.plates.detector_base import PlateCandidate  # noqa: E402
from scripts.generate_synthetic_plates import _random_plate_text, _render_plate  # noqa: E402

logger = get_logger(__name__)


def _edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def char_accuracy(expected: str, got: str) -> float:
    if not expected:
        return 1.0 if not got else 0.0
    distance = _edit_distance(expected, got)
    return max(0.0, 1.0 - distance / len(expected))


def _yolo_box(candidate: PlateCandidate) -> tuple[int, int, int, int]:
    return candidate.x, candidate.y, candidate.x + candidate.w, candidate.y + candidate.h


def _iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def benchmark_detectors(rows: list[dict]) -> None:
    val_images = sorted((PLATES_SYNTHETIC_DIR / "images" / "val").glob("*.jpg"))
    if not val_images:
        logger.warning("No hay dataset sintetico de validacion. Corre generate_synthetic_plates.py primero.")
        return

    detectors = {"classical": ClassicalPlateDetector()}
    if YOLO_PLATE_WEIGHTS.exists():
        from lpr.plates.yolo_detector import YoloPlateDetector

        yolo = YoloPlateDetector()
        if yolo.is_trained:
            detectors["yolo"] = yolo
    else:
        logger.warning("No hay pesos YOLO entrenados (%s); se omite del benchmark de deteccion.", YOLO_PLATE_WEIGHTS)

    for name, detector in detectors.items():
        ious, hits, times = [], 0, []
        for img_path in val_images:
            label_path = PLATES_SYNTHETIC_DIR / "labels" / "val" / f"{img_path.stem}.txt"
            image = cv2.imread(str(img_path))
            h, w = image.shape[:2]
            cls, cx, cy, bw, bh = map(float, label_path.read_text().split())
            gt_box = (
                int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                int((cx + bw / 2) * w), int((cy + bh / 2) * h),
            )

            t0 = time.perf_counter()
            candidate = detector.detect_best(image)
            times.append(time.perf_counter() - t0)

            if candidate is None:
                ious.append(0.0)
                continue
            iou = _iou_xyxy(_yolo_box(candidate), gt_box)
            ious.append(iou)
            if iou >= 0.5:
                hits += 1

        rows.append({
            "stage": "deteccion_placa",
            "method": name,
            "metric": "mean_iou",
            "value": round(float(np.mean(ious)), 4),
            "extra": f"deteccion@IoU0.5={hits}/{len(val_images)}",
            "mean_time_ms": round(float(np.mean(times)) * 1000, 2),
        })
        logger.info("[deteccion] %s -> IoU medio=%.3f, det@0.5=%d/%d, t=%.1fms",
                    name, float(np.mean(ious)), hits, len(val_images), float(np.mean(times)) * 1000)


def _build_ocr_engines() -> dict:
    engines = {}
    try:
        from lpr.ocr.easyocr_engine import EasyOCREngine

        engines["easyocr"] = EasyOCREngine()
    except Exception:
        logger.exception("No se pudo inicializar EasyOCR")

    try:
        from lpr.ocr.tesseract_engine import TesseractOCREngine

        engines["tesseract"] = TesseractOCREngine()
    except Exception as exc:
        logger.warning("Tesseract no disponible, se omite del benchmark: %s", exc)

    try:
        from lpr.ocr.ml_char_classifier import MLCharClassifierEngine

        engines["ml_char_classifier"] = MLCharClassifierEngine()
    except FileNotFoundError as exc:
        logger.warning("Clasificador ML no disponible, se omite del benchmark: %s", exc)

    return engines


def benchmark_ocr(rows: list[dict], num_samples: int = 15, seed: int = 123) -> None:
    engines = _build_ocr_engines()
    if not engines:
        logger.warning("Ningun motor de OCR disponible para el benchmark.")
        return

    rng = random.Random(seed)
    samples = []
    for _ in range(num_samples):
        text = _random_plate_text(rng)
        plate_img = _render_plate(text, rng)
        samples.append((text, cv2.cvtColor(np.array(plate_img), cv2.COLOR_RGB2BGR)))

    for name, engine in engines.items():
        accuracies, times, exact = [], [], 0
        for expected_text, plate_bgr in samples:
            t0 = time.perf_counter()
            result = engine.recognize(plate_bgr)
            times.append(time.perf_counter() - t0)
            acc = char_accuracy(expected_text, result.text)
            accuracies.append(acc)
            if result.text == expected_text:
                exact += 1

        rows.append({
            "stage": "ocr",
            "method": name,
            "metric": "char_accuracy",
            "value": round(float(np.mean(accuracies)), 4),
            "extra": f"exact_match={exact}/{len(samples)}",
            "mean_time_ms": round(float(np.mean(times)) * 1000, 2),
        })
        logger.info("[ocr] %s -> char_accuracy=%.3f, exact=%d/%d, t=%.1fms",
                    name, float(np.mean(accuracies)), exact, len(samples), float(np.mean(times)) * 1000)


def _save_plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    det_rows = [r for r in rows if r["stage"] == "deteccion_placa"]
    if det_rows:
        axes[0].bar([r["method"] for r in det_rows], [r["value"] for r in det_rows], color="steelblue")
        axes[0].set_title("Deteccion de placas (IoU medio)")
        axes[0].set_ylim(0, 1)

    ocr_rows = [r for r in rows if r["stage"] == "ocr"]
    if ocr_rows:
        axes[1].bar([r["method"] for r in ocr_rows], [r["value"] for r in ocr_rows], color="darkorange")
        axes[1].set_title("OCR (accuracy por caracter)")
        axes[1].set_ylim(0, 1)

    fig.tight_layout()
    out_path = RESULTS_DIR / "comparison.png"
    fig.savefig(out_path)
    logger.info("Grafico guardado en %s", out_path)


def run_benchmark() -> None:
    rows: list[dict] = []
    benchmark_detectors(rows)
    benchmark_ocr(rows)

    if not rows:
        logger.warning("No se genero ningun resultado de benchmark.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "method", "metric", "value", "extra", "mean_time_ms"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV guardado en %s", csv_path)

    _save_plot(rows)


if __name__ == "__main__":
    run_benchmark()
