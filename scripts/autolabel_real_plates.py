"""Auto-etiqueta fotos reales de placas usando el detector clasico como punto
de partida, para poder mezclarlas con el dataset sintetico y reentrenar YOLO.

No reemplaza una revision humana: guarda ademas una vista previa con el
cuadro dibujado (y un contact sheet con todas juntas) para que se pueda
confirmar o corregir cada caja antes de usarla para entrenar.

Uso:
    python scripts/autolabel_real_plates.py --source data/plates/synthetic/new_plates
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.config import DATA_DIR  # noqa: E402
from lpr.logging_utils import get_logger  # noqa: E402
from lpr.plates.classical_detector import ClassicalPlateDetector  # noqa: E402

logger = get_logger(__name__)

REAL_PLATES_DIR = DATA_DIR / "plates" / "real"
PLATE_CLASS_ID = 0

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Carpeta con fotos reales de autos con placa")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                         help="Descarta candidatos por debajo de esta confianza del detector clasico")
    args = parser.parse_args()

    source_dir = Path(args.source)
    images_out = REAL_PLATES_DIR / "images"
    labels_out = REAL_PLATES_DIR / "labels"
    review_out = REAL_PLATES_DIR / "review"
    for d in (images_out, labels_out, review_out):
        d.mkdir(parents=True, exist_ok=True)

    detector = ClassicalPlateDetector()
    accepted, rejected = [], []

    files = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in VALID_EXTS)
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("No se pudo leer %s, se omite", path)
            continue

        candidate = detector.detect_best(image)
        preview = image.copy()

        if candidate is None or candidate.confidence < args.min_confidence:
            rejected.append(path.name)
            cv2.putText(preview, "SIN CAJA CONFIABLE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            h, w = image.shape[:2]
            x, y, cw, ch = candidate.box
            cx, cy = (x + cw / 2) / w, (y + ch / 2) / h
            nw, nh = cw / w, ch / h

            stem = path.stem.replace(" ", "_")
            shutil.copy(path, images_out / f"{stem}{path.suffix.lower()}")
            (labels_out / f"{stem}.txt").write_text(f"{PLATE_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            accepted.append((path.name, candidate.confidence))

            cv2.rectangle(preview, (x, y), (x + cw, y + ch), (0, 255, 0), 3)
            cv2.putText(preview, f"conf={candidate.confidence:.2f}", (x, max(y - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        preview_small = cv2.resize(preview, (400, int(preview.shape[0] * 400 / preview.shape[1])))
        cv2.imwrite(str(review_out / f"{path.stem}_preview.jpg"), preview_small)

    logger.info("Aceptadas (con caja): %d", len(accepted))
    for name, conf in accepted:
        logger.info("  OK  %.2f  %s", conf, name)
    logger.info("Rechazadas (sin caja confiable, revisar a mano): %d", len(rejected))
    for name in rejected:
        logger.info("  ??  %s", name)

    _build_contact_sheet(review_out)


def _build_contact_sheet(review_dir: Path, cols: int = 5) -> None:
    previews = sorted(review_dir.glob("*_preview.jpg"))
    if not previews:
        return

    thumbs = [cv2.imread(str(p)) for p in previews]
    cell_w, cell_h = 260, 220
    thumbs = [cv2.resize(t, (cell_w, cell_h)) for t in thumbs]

    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 40, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = thumb

    out_path = review_dir.parent / "contact_sheet.jpg"
    cv2.imwrite(str(out_path), sheet)
    logger.info("Contact sheet guardado en %s", out_path)


if __name__ == "__main__":
    main()
