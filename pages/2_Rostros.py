"""Pagina de deteccion y reconocimiento facial: imagen o video subido.

El enrolamiento de personas se hace por CLI (`face-enroll`) porque requiere
capturar fotos interactivamente con la webcam; esta pagina solo detecta/
reconoce con el modelo ya entrenado.
"""
import sys
import time
from pathlib import Path

import cv2
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.faces.gallery import FaceGallery  # noqa: E402
from lpr.io.media import load_image  # noqa: E402
from lpr.pipeline import FacePipeline  # noqa: E402
from lpr.webapp import get_face_detector, get_face_recognizer, process_video_faces, save_upload_to_tempfile  # noqa: E402

st.set_page_config(page_title="Rostros", page_icon="🙂", layout="wide")
st.title("🙂 Deteccion y reconocimiento facial")

detector = get_face_detector()
recognizer = get_face_recognizer()
gallery = FaceGallery()
people = gallery.list_people()

with st.sidebar:
    st.header("Galeria")
    if people:
        st.success(f"{len(people)} persona(s) enrolada(s):")
        for name in people:
            st.write(f"- {name}")
    else:
        st.warning("Nadie enrolado todavia.")
    st.caption(
        "Para enrolar a alguien nuevo, corre desde la terminal:\n\n"
        "`python main.py face-enroll --name \"Nombre\" --source webcam`"
    )

    st.divider()
    st.caption("Solo para video:")
    process_every = st.slider("Procesar 1 de cada N frames", 1, 30, 5, key="face_every")

if recognizer is None:
    st.info("No hay modelo de reconocimiento entrenado todavia — se van a detectar rostros pero no identificar a nadie.")

pipeline = FacePipeline(detector, recognizer)

tab_img, tab_vid = st.tabs(["📷 Imagen", "🎞️ Video"])

with tab_img:
    uploaded = st.file_uploader("Subi una foto con personas", type=["jpg", "jpeg", "png", "webp"], key="face_image")
    if uploaded is not None:
        tmp_path = save_upload_to_tempfile(uploaded)
        image = load_image(tmp_path)

        with st.spinner("Detectando rostros..."):
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

        if not result.faces:
            st.warning("No se detecto ningun rostro.")
        else:
            st.success(f"{len(result.faces)} rostro(s) detectado(s) en {elapsed * 1000:.0f} ms")
            for face in result.faces:
                label = face.name if face.name else "desconocido"
                st.write(f"- **{label}** (distancia LBPH={face.distance:.1f})" if face.name else f"- {label}")

with tab_vid:
    uploaded_video = st.file_uploader("Subi un video con personas", type=["mp4", "avi", "mov", "mkv"], key="face_video")
    if uploaded_video is not None:
        tmp_path = save_upload_to_tempfile(uploaded_video)

        if st.button("Procesar video", type="primary", key="process_face_video"):
            progress_bar = st.progress(0.0, text="Procesando...")
            out_path, seen_names = process_video_faces(
                tmp_path, pipeline, process_every,
                progress_cb=lambda p: progress_bar.progress(p, text=f"Procesando... {p:.0%}"),
            )
            progress_bar.empty()

            st.video(str(out_path))

            if seen_names:
                st.subheader("Personas reconocidas")
                for name in sorted(seen_names):
                    st.write(f"- {name}")
            else:
                st.warning("No se reconocio a nadie en el video (puede que no haya nadie enrolado, o nadie coincida).")
