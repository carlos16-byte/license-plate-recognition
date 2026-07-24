"""Fase 2 - Entrena un clasificador de caracteres (SVM y RandomForest) para
leer placas ya segmentadas.

v2: en vez de entrenar con letras aisladas y perfectas (una fuente de Windows,
centradas, sin ruido de fondo), generamos PLACAS SINTETICAS COMPLETAS con
scripts.generate_synthetic_plates._render_plate, les aplicamos degradaciones
tipicas de una foto real (blur, ruido, compresion JPEG, iluminacion pareja),
y corremos el MISMO segmentador (lpr.plates.segmentation.segmentar_caracteres)
que se usa en produccion para extraer los caracteres.

Por que el cambio: la v1 media 69% de accuracy en el benchmark sintetico pero
solo 31% contra fotos reales -- el clasificador nunca habia visto la fuente
real de una placa, ni los artefactos que deja la segmentacion (recortes
irregulares, bordes de threshold, etc). Entrenar con el mismo pipeline de
segmentacion que despues se usa en inferencia cierra ese hueco.

Si segmentar_caracteres no separa exactamente tantos caracteres como el
texto conocido, esa muestra se descarta (no hay forma de saber que blob
corresponde a que letra), en vez de arriesgar una etiqueta incorrecta.

Uso:
    python scripts/train_char_classifier.py [--num-plates 4000]

Guarda el mejor modelo (por accuracy en un split de validacion) en
models/char_classifier.joblib junto con el mapeo de clases.
"""
from __future__ import annotations

import argparse
import random
import string
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.config import CHAR_CLASSIFIER_PATH  # noqa: E402
from lpr.logging_utils import get_logger  # noqa: E402
from lpr.plates.segmentation import segmentar_caracteres  # noqa: E402
from scripts.generate_synthetic_plates import _render_plate  # noqa: E402

logger = get_logger(__name__)

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
GLYPH_SIZE = 28  # tamano final de cada caracter, en pixeles (cuadrado)


def _random_plate_text(rng: random.Random) -> str:
    letters = "".join(rng.choices(string.ascii_uppercase, k=rng.choice([2, 3])))
    digits = "".join(rng.choices(string.digits, k=rng.choice([3, 4])))
    return (letters + digits) if rng.random() < 0.5 else (digits + letters)


def _degrade(plate_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simula lo que le pasa a una placa real entre la camara y el recorte:
    desenfoque de movimiento/foco, ruido, compresion JPEG y iluminacion
    pareja/dispareja. Sin esto el clasificador solo aprende glifos perfectos."""
    img = plate_bgr.copy()

    if rng.random() < 0.6:
        k = rng.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if rng.random() < 0.5:
        factor = rng.uniform(0.6, 1.5)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        noise = np.random.normal(0, rng.uniform(3, 15), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.6:
        quality = rng.randint(25, 70)
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if rng.random() < 0.3:
        # sombra/gradiente de luz pareja como en fotos al aire libre
        h, w = img.shape[:2]
        gradient = np.tile(np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0), w), (h, 1))
        img = np.clip(img.astype(np.float32) * gradient[:, :, None], 0, 255).astype(np.uint8)

    return img


def build_dataset(num_plates: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)

    X, y = [], []
    matched, discarded = 0, 0

    for _ in range(num_plates):
        text = _random_plate_text(rng)
        plate_pil = _render_plate(text, rng)
        plate_bgr = cv2.cvtColor(np.array(plate_pil), cv2.COLOR_RGB2BGR)
        plate_bgr = _degrade(plate_bgr, rng)

        chars = segmentar_caracteres(plate_bgr)
        if len(chars) != len(text):
            discarded += 1
            continue

        matched += 1
        for glyph, label in zip(chars, text):
            glyph = cv2.resize(glyph, (GLYPH_SIZE, GLYPH_SIZE), interpolation=cv2.INTER_AREA)
            X.append(_extract_features(glyph))
            y.append(label)

    logger.info("Placas generadas: %d | segmentadas correctamente: %d | descartadas: %d",
                num_plates, matched, discarded)
    return np.array(X, dtype=np.float32), np.array(y)


def _extract_features(glyph: np.ndarray) -> np.ndarray:
    """Vector de features: HOG + intensidades crudas reducidas (simple pero efectivo
    para glifos binarizados pequenos, sin depender de una CNN)."""
    hog = cv2.HOGDescriptor(
        _winSize=(GLYPH_SIZE, GLYPH_SIZE),
        _blockSize=(14, 14),
        _blockStride=(7, 7),
        _cellSize=(7, 7),
        _nbins=9,
    )
    hog_features = hog.compute(glyph).flatten()
    pixel_features = cv2.resize(glyph, (14, 14)).flatten() / 255.0
    return np.concatenate([hog_features, pixel_features])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-plates", type=int, default=4000,
                         help="Cuantas placas sinteticas generar (cada una aporta varios caracteres)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Generando dataset a partir de %d placas sinteticas degradadas...", args.num_plates)
    X, y = build_dataset(args.num_plates, args.seed)
    logger.info("Dataset: %d caracteres, %d clases", len(X), len(set(y)))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    candidates = {
        "svm": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=args.seed),
    }

    best_name, best_model, best_acc = None, None, -1.0
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        logger.info("%s -> accuracy validacion: %.4f", name, acc)
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    # reentrenar el ganador con todos los datos disponibles
    best_model.fit(X, y)

    CHAR_CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best_model, "model_name": best_name, "classes": sorted(set(y)), "glyph_size": GLYPH_SIZE},
        CHAR_CLASSIFIER_PATH,
    )
    logger.info("Mejor modelo: %s (acc=%.4f). Guardado en %s", best_name, best_acc, CHAR_CLASSIFIER_PATH)


if __name__ == "__main__":
    main()
