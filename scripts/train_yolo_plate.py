"""Fase 3 - Entrena un detector YOLOv8n de placas sobre el dataset sintetico
generado por scripts/generate_synthetic_plates.py y guarda los pesos en
models/yolo_plate.pt.

Sin GPU disponible, los defaults se mantienen deliberadamente livianos
(imagenes chicas, pocas epocas) para que el entrenamiento termine en un
tiempo razonable en CPU. Este modelo es una prueba de concepto funcional del
pipeline de Deep Learning, no un detector de nivel produccion: para mejorar
la precision real, reemplaza/agrega imagenes reales de placas en
data/plates/synthetic/ (mismo formato YOLO) y vuelve a correr este script.

Uso:
    python scripts/generate_synthetic_plates.py   # si todavia no existe el dataset
    python scripts/train_yolo_plate.py [--epochs 25] [--imgsz 320]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.config import MODELS_DIR, PLATES_SYNTHETIC_DIR, YOLO_PLATE_WEIGHTS  # noqa: E402
from lpr.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--base-model", default="yolov8n.pt")
    args = parser.parse_args()

    dataset_yaml = PLATES_SYNTHETIC_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        raise SystemExit(
            f"No existe {dataset_yaml}. Corre primero: python scripts/generate_synthetic_plates.py"
        )

    from ultralytics import YOLO

    logger.info("Entrenando YOLOv8n sobre %s (epochs=%d, imgsz=%d)", dataset_yaml, args.epochs, args.imgsz)
    model = YOLO(args.base_model)
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=str(MODELS_DIR / "yolo_runs"),
        name="plate_detector",
        exist_ok=True,
        verbose=False,
        patience=10,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise SystemExit(f"El entrenamiento no genero pesos en {best_weights}")

    YOLO_PLATE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, YOLO_PLATE_WEIGHTS)
    logger.info("Pesos copiados a %s", YOLO_PLATE_WEIGHTS)


if __name__ == "__main__":
    main()
