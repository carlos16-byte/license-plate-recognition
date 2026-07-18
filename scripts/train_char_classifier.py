"""Fase 2 - Entrena un clasificador de caracteres (SVM y RandomForest) usando
un dataset SINTETICO de glifos A-Z/0-9, ya que no existe un dataset real de
caracteres de placa etiquetado. Es la tecnica estandar para bootstrapear un
clasificador de OCR de placas cuando no hay datos anotados: se renderizan los
mismos caracteres que aparecen en una placa con varias fuentes, rotaciones y
ruido, y se entrena sobre eso.

Uso:
    python scripts/train_char_classifier.py [--samples-per-char 300]

Guarda el mejor modelo (por accuracy en un split de validacion) en
models/char_classifier.joblib junto con el mapeo de clases.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.config import CHAR_CLASSIFIER_PATH  # noqa: E402
from lpr.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
GLYPH_SIZE = 28  # tamano final de cada caracter, en pixeles (cuadrado)

FONT_CANDIDATES = [
    "arial.ttf",
    "arialbd.ttf",
    "cour.ttf",
    "courbd.ttf",
    "consola.ttf",
    "consolab.ttf",
    "tahoma.ttf",
]
FONTS_DIR = Path("C:/Windows/Fonts")


def _load_fonts(font_size: int = 34) -> list[ImageFont.FreeTypeFont]:
    fonts = []
    for name in FONT_CANDIDATES:
        path = FONTS_DIR / name
        if path.exists():
            fonts.append(ImageFont.truetype(str(path), font_size))
    if not fonts:
        logger.warning("No se encontraron fuentes TTF de Windows, se usa la fuente por defecto de PIL")
        fonts = [ImageFont.load_default()]
    return fonts


def _render_glyph(char: str, font: ImageFont.FreeTypeFont, rng: random.Random) -> np.ndarray:
    canvas_size = 48
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), char, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_size - text_w) / 2 - bbox[0] + rng.uniform(-2, 2)
    y = (canvas_size - text_h) / 2 - bbox[1] + rng.uniform(-2, 2)
    draw.text((x, y), char, fill=255, font=font)

    array = np.array(img, dtype=np.uint8)

    angle = rng.uniform(-8, 8)
    matrix = cv2.getRotationMatrix2D((canvas_size / 2, canvas_size / 2), angle, 1.0)
    array = cv2.warpAffine(array, matrix, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)

    if rng.random() < 0.5:
        noise = rng.uniform(2, 10)
        array = array.astype(np.float32) + np.random.normal(0, noise, array.shape)
        array = np.clip(array, 0, 255).astype(np.uint8)

    ys, xs = np.where(array > 40)
    if len(xs) and len(ys):
        pad = 2
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, canvas_size)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, canvas_size)
        array = array[y0:y1, x0:x1]

    array = cv2.resize(array, (GLYPH_SIZE, GLYPH_SIZE), interpolation=cv2.INTER_AREA)
    return array


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


def build_dataset(samples_per_char: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    fonts = _load_fonts()

    X, y = [], []
    for char in CHARSET:
        for _ in range(samples_per_char):
            font = rng.choice(fonts)
            glyph = _render_glyph(char, font, rng)
            X.append(_extract_features(glyph))
            y.append(char)

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-char", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Generando dataset sintetico de caracteres (%d muestras/caracter)...", args.samples_per_char)
    X, y = build_dataset(args.samples_per_char, args.seed)
    logger.info("Dataset: %d muestras, %d clases", len(X), len(set(y)))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    candidates = {
        "svm": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=args.seed),
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
