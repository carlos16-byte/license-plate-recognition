"""CLI unificada del sistema. Punto de entrada: `python main.py <subcomando>`."""
from __future__ import annotations

import argparse
import sys

import cv2

from lpr.config import RESULTS_DIR
from lpr.io.media import VideoWriter, iter_video_frames, load_image, resize_width, save_image
from lpr.logging_utils import get_logger
from lpr.ocr.engine_base import OCREngine
from lpr.plates.detector_base import PlateDetector
from lpr.pipeline import FacePipeline, PlatePipeline

logger = get_logger(__name__)


def _build_plate_detector(name: str) -> PlateDetector:
    if name == "classical":
        from lpr.plates.classical_detector import ClassicalPlateDetector

        return ClassicalPlateDetector()
    if name == "yolo":
        from lpr.plates.yolo_detector import YoloPlateDetector

        return YoloPlateDetector()
    raise ValueError(f"Detector desconocido: {name}")


def _build_ocr_engine(name: str) -> OCREngine:
    if name == "easyocr":
        from lpr.ocr.easyocr_engine import EasyOCREngine

        return EasyOCREngine()
    if name == "tesseract":
        from lpr.ocr.tesseract_engine import TesseractOCREngine

        return TesseractOCREngine()
    if name == "ml":
        from lpr.ocr.ml_char_classifier import MLCharClassifierEngine

        return MLCharClassifierEngine()
    raise ValueError(f"Motor de OCR desconocido: {name}")


def cmd_plate_image(args: argparse.Namespace) -> None:
    detector = _build_plate_detector(args.detector)
    ocr = _build_ocr_engine(args.ocr)
    pipeline = PlatePipeline(detector, ocr)

    image = load_image(args.source)
    result = pipeline.run(image)

    if result.candidate is None:
        print("No se detecto ninguna placa.")
    else:
        print(f"Placa detectada -> texto='{result.text}' confianza={result.ocr_confidence:.2f}")
        print(f"detector={result.detector_name} ocr={result.ocr_name} "
              f"t_deteccion={result.detect_time_s * 1000:.1f}ms t_ocr={result.ocr_time_s * 1000:.1f}ms")

    annotated = pipeline.draw(image, result)
    out_path = RESULTS_DIR / f"plate_{args.detector}_{args.ocr}.jpg"
    save_image(annotated, out_path)
    print(f"Resultado guardado en {out_path}")

    if args.show:
        cv2.imshow("Placa detectada", resize_width(annotated))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cmd_plate_video(args: argparse.Namespace) -> None:
    detector = _build_plate_detector(args.detector)
    ocr = _build_ocr_engine(args.ocr)
    pipeline = PlatePipeline(detector, ocr)

    writer = VideoWriter(RESULTS_DIR / "plate_video_output.mp4") if args.save else None

    for i, frame in enumerate(iter_video_frames(args.source)):
        if i % max(args.ocr_every, 1) == 0:
            result = pipeline.run(frame)
            last_result = result
        annotated = pipeline.draw(frame, last_result)

        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("Placas (video)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()


def _draw_enroll_hud(frame, name: str, count: int, total: int, face_visible: bool):
    """Panel superior con nombre/progreso e instrucciones, en vez de texto
    crudo pegado sobre el video (dificil de leer con cualquier fondo)."""
    h, w = frame.shape[:2]
    bar_h = 78
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(frame, name, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    progress_text = f"{count}/{total} fotos"
    (pt_w, _), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, progress_text, (w - pt_w - 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

    bar_x, bar_y, bar_w, bar_h_px = 16, 42, w - 32, 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (80, 80, 80), -1)
    filled_w = int(bar_w * min(count / max(total, 1), 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h_px), (0, 255, 120), -1)

    hint_color = (0, 255, 0) if face_visible else (120, 120, 255)
    hint = "ESPACIO: capturar   |   Q: salir" if face_visible else "Buscando rostro...   |   Q: salir"
    cv2.putText(frame, hint, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hint_color, 1)
    return frame


def cmd_face_enroll(args: argparse.Namespace) -> None:
    from lpr.faces.detector import FaceDetector
    from lpr.faces.gallery import FaceGallery

    detector = FaceDetector()
    gallery = FaceGallery()

    if args.source == "webcam":
        cap_source = 0
        count = 0
        for frame in iter_video_frames(cap_source):
            boxes = detector.detect(frame)
            display = frame.copy()
            for box in boxes:
                x, y, w, h = box.box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            display = _draw_enroll_hud(display, args.name, count, args.num_samples, face_visible=bool(boxes))
            cv2.imshow("Enrolamiento facial", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and boxes:
                best = max(boxes, key=lambda b: b.w * b.h)
                x, y, w, h = best.box
                gallery.add_image(args.name, frame[y : y + h, x : x + w])
                count += 1
            if key == ord("q") or count >= args.num_samples:
                break
        cv2.destroyAllWindows()
        print(f"Capturadas {count} fotos de '{args.name}'.")
    else:
        from pathlib import Path

        src_dir = Path(args.source)
        photos = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        for photo in photos:
            image = load_image(photo)
            gallery.add_image(args.name, image)
        print(f"Copiadas {len(photos)} fotos de '{args.name}' desde {src_dir}.")

    print("Entrenando modelo de reconocimiento con la galeria actualizada...")
    _train_face_model()


def _train_face_model() -> None:
    from lpr.faces.detector import FaceDetector
    from lpr.faces.gallery import FaceGallery
    from lpr.faces.recognizer import FaceRecognizer

    detector = FaceDetector()
    gallery = FaceGallery()
    recognizer = FaceRecognizer()
    n = recognizer.train(gallery, detector)
    recognizer.save()
    print(f"Modelo facial entrenado con {n} rostros de {len(gallery.list_people())} persona(s).")


def cmd_face_recognize(args: argparse.Namespace) -> None:
    from lpr.faces.detector import FaceDetector
    from lpr.faces.recognizer import FaceRecognizer

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    try:
        recognizer.load()
    except FileNotFoundError as exc:
        print(exc)
        recognizer = None

    pipeline = FacePipeline(detector, recognizer)
    image = load_image(args.source)
    result = pipeline.run(image)

    for face in result.faces:
        print(f"Rostro en {face.box.box} -> {face.name or 'desconocido'} (distancia={face.distance:.1f})")

    annotated = pipeline.draw(image, result)
    out_path = RESULTS_DIR / "face_recognition_output.jpg"
    save_image(annotated, out_path)
    print(f"Resultado guardado en {out_path}")

    if args.show:
        cv2.imshow("Rostros", resize_width(annotated))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cmd_face_video(args: argparse.Namespace) -> None:
    from lpr.faces.detector import FaceDetector
    from lpr.faces.recognizer import FaceRecognizer

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    try:
        recognizer.load()
    except FileNotFoundError:
        recognizer = None

    pipeline = FacePipeline(detector, recognizer)
    writer = VideoWriter(RESULTS_DIR / "face_video_output.mp4") if args.save else None

    for frame in iter_video_frames(args.source):
        result = pipeline.run(frame)
        annotated = pipeline.draw(frame, result)
        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("Rostros (video)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()


def cmd_compare(args: argparse.Namespace) -> None:
    from scripts.benchmark_compare import run_benchmark

    run_benchmark()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lpr", description="Sistema de reconocimiento de placas y rostros")
    sub = parser.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser("plate-image", help="Detecta y reconoce la placa en una imagen")
    p_img.add_argument("source", help="Ruta a la imagen")
    p_img.add_argument("--detector", choices=["classical", "yolo"], default="classical")
    p_img.add_argument("--ocr", choices=["easyocr", "tesseract", "ml"], default="easyocr")
    p_img.add_argument("--show", action="store_true")
    p_img.set_defaults(func=cmd_plate_image)

    p_vid = sub.add_parser("plate-video", help="Detecta placas en video/webcam en tiempo real")
    p_vid.add_argument("source", help="Indice de webcam (0) o ruta a un video")
    p_vid.add_argument("--detector", choices=["classical", "yolo"], default="classical")
    p_vid.add_argument("--ocr", choices=["easyocr", "tesseract", "ml"], default="easyocr")
    p_vid.add_argument("--ocr-every", type=int, default=5, help="Correr OCR cada N frames (costoso)")
    p_vid.add_argument("--show", action="store_true", default=True)
    p_vid.add_argument("--save", action="store_true")
    p_vid.set_defaults(func=cmd_plate_video)

    p_enroll = sub.add_parser("face-enroll", help="Registra fotos de una persona y reentrena el modelo")
    p_enroll.add_argument("--name", required=True)
    p_enroll.add_argument("--source", default="webcam", help="'webcam' o ruta a una carpeta de fotos")
    p_enroll.add_argument("--num-samples", type=int, default=20)
    p_enroll.set_defaults(func=cmd_face_enroll)

    p_frec = sub.add_parser("face-recognize", help="Reconoce rostros en una imagen")
    p_frec.add_argument("source", help="Ruta a la imagen")
    p_frec.add_argument("--show", action="store_true")
    p_frec.set_defaults(func=cmd_face_recognize)

    p_fvid = sub.add_parser("face-video", help="Reconoce rostros en video/webcam en tiempo real")
    p_fvid.add_argument("source", help="Indice de webcam (0) o ruta a un video")
    p_fvid.add_argument("--show", action="store_true", default=True)
    p_fvid.add_argument("--save", action="store_true")
    p_fvid.set_defaults(func=cmd_face_video)

    p_cmp = sub.add_parser("compare", help="Compara deteccion clasica vs YOLO y OCR EasyOCR vs Tesseract vs ML")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
