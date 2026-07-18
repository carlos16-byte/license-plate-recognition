"""Fase 3 - Genera un dataset sintetico (formato YOLO) de "placas" compuestas
sobre fondos variados, ya que no existe un dataset real de placas etiquetado.

Es un dataset de *bootstrap*: alcanza para que scripts/train_yolo_plate.py
tenga con que entrenar y todo el pipeline de deteccion por Deep Learning
funcione de punta a punta ahora mismo. Cuando existan fotos reales de placas,
basta con reemplazar/agregar imagenes+labels en el mismo formato y
reentrenar - no hace falta tocar el resto del sistema.

Uso:
    python scripts/generate_synthetic_plates.py [--num-images 360] [--val-split 0.15]
"""
from __future__ import annotations

import argparse
import random
import shutil
import string
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.config import DATA_DIR, PLATES_SYNTHETIC_DIR  # noqa: E402
from lpr.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

IMG_SIZE = 416
FONT_PATH = Path("C:/Windows/Fonts/arialbd.ttf")
PLATE_CLASS_ID = 0

REAL_PHOTOS = [DATA_DIR / "car1.jpg", DATA_DIR / "car2.jpg"]


def _random_plate_text(rng: random.Random) -> str:
    letters = "".join(rng.choices(string.ascii_uppercase, k=3))
    digits = "".join(rng.choices(string.digits, k=3))
    return f"{letters}{digits}"


def _make_background(rng: random.Random) -> Image.Image:
    if REAL_PHOTOS and rng.random() < 0.4:
        photo_path = rng.choice([p for p in REAL_PHOTOS if p.exists()] or [None])
        if photo_path is not None:
            img = Image.open(photo_path).convert("RGB")
            # recorte aleatorio + resize para variar el encuadre
            w, h = img.size
            side = min(w, h)
            left = rng.randint(0, max(w - side, 0))
            top = rng.randint(0, max(h - side, 0))
            img = img.crop((left, top, left + side, top + side)).resize((IMG_SIZE, IMG_SIZE))
            return img

    base_color = tuple(rng.randint(60, 200) for _ in range(3))
    array = np.full((IMG_SIZE, IMG_SIZE, 3), base_color, dtype=np.uint8)
    noise = np.random.normal(0, rng.uniform(5, 25), array.shape)
    array = np.clip(array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # algunas formas simples simulando carrocerias / carretera para variar el fondo
    for _ in range(rng.randint(1, 3)):
        color = tuple(rng.randint(0, 255) for _ in range(3))
        x1, y1 = rng.randint(0, IMG_SIZE), rng.randint(0, IMG_SIZE)
        x2, y2 = rng.randint(0, IMG_SIZE), rng.randint(0, IMG_SIZE)
        cv2.rectangle(array, (x1, y1), (x2, y2), color, thickness=-1)

    return Image.fromarray(array)


def _render_plate(text: str, rng: random.Random) -> Image.Image:
    plate_w, plate_h = rng.randint(160, 220), 0
    plate_h = int(plate_w / rng.uniform(2.8, 3.6))

    bg_color = rng.choice([(245, 245, 245), (255, 255, 255), (240, 220, 90)])
    plate = Image.new("RGB", (plate_w, plate_h), color=bg_color)
    draw = ImageDraw.Draw(plate)
    draw.rectangle([1, 1, plate_w - 2, plate_h - 2], outline=(20, 20, 20), width=max(2, plate_h // 20))

    font_size = int(plate_h * 0.62)
    font = ImageFont.truetype(str(FONT_PATH), font_size) if FONT_PATH.exists() else ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((plate_w - text_w) / 2 - bbox[0], (plate_h - text_h) / 2 - bbox[1]),
        text,
        fill=(15, 15, 15),
        font=font,
    )
    return plate


def _composite(background: Image.Image, plate: Image.Image, rng: random.Random):
    bg_w, bg_h = background.size
    scale = rng.uniform(0.18, 0.35)
    new_w = int(bg_w * scale)
    new_h = int(plate.height * (new_w / plate.width))
    plate_resized = plate.resize((new_w, new_h))

    angle = rng.uniform(-12, 12)
    plate_rotated = plate_resized.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

    max_x = bg_w - plate_rotated.width
    max_y = bg_h - plate_rotated.height
    if max_x <= 0 or max_y <= 0:
        px, py = 0, 0
    else:
        px, py = rng.randint(0, max_x), rng.randint(0, max_y)

    composed = background.copy()
    composed.paste(plate_rotated, (px, py))

    cx = (px + plate_rotated.width / 2) / bg_w
    cy = (py + plate_rotated.height / 2) / bg_h
    w = plate_rotated.width / bg_w
    h = plate_rotated.height / bg_h
    return composed, (cx, cy, w, h)


def generate_split(split: str, num_images: int, seed: int) -> None:
    images_dir = PLATES_SYNTHETIC_DIR / "images" / split
    labels_dir = PLATES_SYNTHETIC_DIR / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    for i in range(num_images):
        text = _random_plate_text(rng)
        background = _make_background(rng)
        plate = _render_plate(text, rng)
        composed, (cx, cy, w, h) = _composite(background, plate, rng)

        stem = f"{split}_{i:05d}"
        composed.convert("RGB").save(images_dir / f"{stem}.jpg", quality=90)
        (labels_dir / f"{stem}.txt").write_text(f"{PLATE_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    logger.info("Split '%s': %d imagenes generadas en %s", split, num_images, images_dir)


def write_dataset_yaml() -> Path:
    yaml_path = PLATES_SYNTHETIC_DIR / "dataset.yaml"
    yaml_path.write_text(
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: plate\n".format(PLATES_SYNTHETIC_DIR.as_posix())
    )
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=360)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if PLATES_SYNTHETIC_DIR.exists():
        shutil.rmtree(PLATES_SYNTHETIC_DIR)

    num_val = max(int(args.num_images * args.val_split), 1)
    num_train = args.num_images - num_val

    generate_split("train", num_train, args.seed)
    generate_split("val", num_val, args.seed + 1)
    yaml_path = write_dataset_yaml()
    logger.info("dataset.yaml escrito en %s", yaml_path)


if __name__ == "__main__":
    main()
