"""Utilidades compartidas por las paginas de Streamlit: carga cacheada de
modelos (pesados, no se quieren recargar en cada interaccion) y procesamiento
de video subido (frame a frame, con muestreo para no correr el OCR en cada
cuadro).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import cv2
import streamlit as st

from lpr.config import FACE_LABELS_PATH, FACE_MODEL_PATH, YOLO_PLATE_WEIGHTS
from lpr.faces.detector import FaceDetector
from lpr.faces.recognizer import FaceRecognizer
from lpr.io.media import VideoWriter
from lpr.ocr.engine_base import OCREngine
from lpr.pipeline import FacePipeline, PlatePipeline
from lpr.plates.detector_base import PlateDetector


@st.cache_resource(show_spinner="Cargando detector de placas...")
def get_plate_detector(name: str) -> PlateDetector:
    if name == "classical":
        from lpr.plates.classical_detector import ClassicalPlateDetector
        return ClassicalPlateDetector()
    from lpr.plates.yolo_detector import YoloPlateDetector
    return YoloPlateDetector()


@st.cache_resource(show_spinner="Cargando motor de OCR (la primera vez puede tardar)...")
def get_ocr_engine(name: str) -> OCREngine:
    if name == "easyocr":
        from lpr.ocr.easyocr_engine import EasyOCREngine
        return EasyOCREngine()
    if name == "tesseract":
        from lpr.ocr.tesseract_engine import TesseractOCREngine
        return TesseractOCREngine()
    from lpr.ocr.ml_char_classifier import MLCharClassifierEngine
    return MLCharClassifierEngine()


@st.cache_resource(show_spinner="Cargando detector de rostros...")
def get_face_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_resource(show_spinner="Cargando modelo de reconocimiento facial...")
def get_face_recognizer() -> FaceRecognizer | None:
    if not (FACE_MODEL_PATH.exists() and FACE_LABELS_PATH.exists()):
        return None
    recognizer = FaceRecognizer()
    recognizer.load()
    return recognizer


def yolo_weights_available() -> bool:
    return YOLO_PLATE_WEIGHTS.exists()


def save_upload_to_tempfile(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return Path(tmp.name)


def process_video_plates(
    video_path: Path,
    pipeline: PlatePipeline,
    process_every: int,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[Path, dict[str, float]]:
    """Corre PlatePipeline sobre un video, muestreando cada N frames.
    Devuelve (ruta del video anotado, {texto_placa: confianza_maxima_vista})."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    out_path = Path(tempfile.mktemp(suffix=".mp4"))
    writer = VideoWriter(out_path, fps=fps)

    seen_plates: dict[str, float] = {}
    last_result = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % max(process_every, 1) == 0:
            last_result = pipeline.run(frame)
            if last_result.text:
                seen_plates[last_result.text] = max(
                    seen_plates.get(last_result.text, 0.0), last_result.ocr_confidence
                )

        annotated = pipeline.draw(frame, last_result) if last_result else frame
        writer.write(annotated)

        frame_idx += 1
        if progress_cb:
            progress_cb(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    return out_path, seen_plates


def process_video_faces(
    video_path: Path,
    pipeline: FacePipeline,
    process_every: int,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[Path, set[str]]:
    """Corre FacePipeline sobre un video, muestreando cada N frames.
    Devuelve (ruta del video anotado, nombres reconocidos)."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    out_path = Path(tempfile.mktemp(suffix=".mp4"))
    writer = VideoWriter(out_path, fps=fps)

    seen_names: set[str] = set()
    last_result = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % max(process_every, 1) == 0:
            last_result = pipeline.run(frame)
            for face in last_result.faces:
                if face.name:
                    seen_names.add(face.name)

        annotated = pipeline.draw(frame, last_result) if last_result else frame
        writer.write(annotated)

        frame_idx += 1
        if progress_cb:
            progress_cb(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    return out_path, seen_names
