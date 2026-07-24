"""Pagina de deteccion y lectura de placas: imagen o video subido."""
import sys
import time
from pathlib import Path

import cv2
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.io.media import load_image  # noqa: E402
from lpr.pipeline import PlatePipeline  # noqa: E402
from lpr.webapp import (  # noqa: E402
    get_ocr_engine,
    get_plate_detector,
    process_video_plates,
    save_upload_to_tempfile,
    yolo_weights_available,
)

st.set_page_config(page_title="Placas", page_icon="🚙", layout="wide")
st.title("🚙 Deteccion y lectura de placas")

with st.sidebar:
    st.header("Opciones")
    detector_options = ["classical"] + (["yolo"] if yolo_weights_available() else [])
    if "yolo" not in detector_options:
        st.caption("⚠️ No hay pesos YOLO entrenados (`models/yolo_plate.pt`); solo esta disponible el detector clasico.")
    detector_name = st.selectbox("Detector de placas", detector_options,
                                  format_func=lambda x: {"classical": "Clasico (contornos)", "yolo": "YOLO (Deep Learning)"}[x])

    ocr_name = st.selectbox(
        "Motor de OCR", ["easyocr", "tesseract", "ml"],
        format_func=lambda x: {"easyocr": "EasyOCR (recomendado)", "tesseract": "Tesseract", "ml": "Clasificador ML (SVM/RF)"}[x],
    )

    st.divider()
    st.caption("Solo para video:")
    process_every = st.slider("Procesar 1 de cada N frames", 1, 30, 8,
                               help="Correr deteccion+OCR en cada frame es lento; se puede muestrear.")

tab_img, tab_vid = st.tabs(["📷 Imagen", "🎞️ Video"])

with tab_img:
    uploaded = st.file_uploader("Subi una foto de un auto", type=["jpg", "jpeg", "png", "webp"], key="plate_image")
    if uploaded is not None:
        tmp_path = save_upload_to_tempfile(uploaded)
        image = load_image(tmp_path)

        try:
            detector = get_plate_detector(detector_name)
            ocr = get_ocr_engine(ocr_name)
        except Exception as exc:
            st.error(f"No se pudo cargar el motor seleccionado: {exc}")
            st.stop()

        pipeline = PlatePipeline(detector, ocr)
        with st.spinner("Detectando y leyendo..."):
            t0 = time.perf_counter()
            result = pipeline.run(image)
            elapsed = time.perf_counter() - t0

        annotated = pipeline.draw(image, result)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("Resultado")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if result.candidate is None:
            st.warning("No se detecto ninguna placa.")
        else:
            st.success(f"Texto leido: **{result.text or '(vacio)'}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Confianza OCR", f"{result.ocr_confidence:.0%}")
            c2.metric("Tiempo deteccion", f"{result.detect_time_s * 1000:.0f} ms")
            c3.metric("Tiempo OCR", f"{result.ocr_time_s * 1000:.0f} ms")
            st.caption(f"Total: {elapsed * 1000:.0f} ms · detector={result.detector_name} · ocr={result.ocr_name}")

with tab_vid:
    uploaded_video = st.file_uploader("Subi un video con autos", type=["mp4", "avi", "mov", "mkv"], key="plate_video")
    if uploaded_video is not None:
        tmp_path = save_upload_to_tempfile(uploaded_video)

        try:
            detector = get_plate_detector(detector_name)
            ocr = get_ocr_engine(ocr_name)
        except Exception as exc:
            st.error(f"No se pudo cargar el motor seleccionado: {exc}")
            st.stop()

        pipeline = PlatePipeline(detector, ocr)

        if st.button("Procesar video", type="primary"):
            progress_bar = st.progress(0.0, text="Procesando...")
            out_path, seen_plates = process_video_plates(
                tmp_path, pipeline, process_every,
                progress_cb=lambda p: progress_bar.progress(p, text=f"Procesando... {p:.0%}"),
            )
            progress_bar.empty()

            st.video(str(out_path))

            if seen_plates:
                st.subheader("Placas detectadas")
                rows = sorted(seen_plates.items(), key=lambda kv: kv[1], reverse=True)
                st.table({"Placa": [r[0] for r in rows], "Confianza maxima": [f"{r[1]:.0%}" for r in rows]})
            else:
                st.warning("No se detecto ninguna placa en el video.")
