"""Punto de entrada de la app: `streamlit run streamlit_app.py`."""
import streamlit as st

st.set_page_config(page_title="Reconocimiento de Placas y Rostros", page_icon="🚗", layout="wide")

st.title("🚗 Sistema de Reconocimiento de Placas y Rostros")

st.markdown(
    """
Bienvenido. Este sistema compara enfoques de **Vision por Computadora clasica**,
**Machine Learning** y **Deep Learning** para detectar placas vehiculares y
reconocer rostros, en imagenes o video.

Usa el menu de la izquierda para navegar:

- **🚙 Placas** — sube una imagen o video y elegi que detector (clasico o YOLO)
  y que motor de OCR (EasyOCR, Tesseract o el clasificador ML) usar.
- **🙂 Rostros** — deteccion y reconocimiento facial. Para que reconozca a
  alguien primero hay que enrolarlo desde la terminal (`python main.py
  face-enroll --name "Nombre"`).
- **📊 Comparativa** — resultados de comparar clasico vs YOLO y los tres
  motores de OCR entre si, con metricas reales.
    """
)

st.info(
    "El motor de OCR por defecto es **EasyOCR** (mejor accuracy medida en fotos reales: ~81%). "
    "El clasificador ML (Fase 2) esta pensado como comparacion educativa, no como el motor "
    "principal — con SVM/RandomForest anda ~57% en fotos reales."
)
