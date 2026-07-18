"""Carga y guardado de imagenes/video, y captura de webcam."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

import cv2
import numpy as np

from lpr.logging_utils import get_logger

logger = get_logger(__name__)


def load_image(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return image


def save_image(image: np.ndarray, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def resize_width(image: np.ndarray, width: int = 600) -> np.ndarray:
    h, w = image.shape[:2]
    if w == 0:
        return image
    new_h = int(h * (width / w))
    return cv2.resize(image, (width, max(new_h, 1)))


def open_video_source(source: Union[str, int]) -> cv2.VideoCapture:
    """source puede ser un indice de webcam (int/'0') o una ruta de archivo de video."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
    return cap


def iter_video_frames(source: Union[str, int]) -> Iterator[np.ndarray]:
    cap = open_video_source(source)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


class VideoWriter:
    """Wrapper fino sobre cv2.VideoWriter que infiere tamano en el primer frame."""

    def __init__(self, path: Union[str, Path], fps: float = 20.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
