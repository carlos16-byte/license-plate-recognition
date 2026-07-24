"""Pagina de comparacion clasico vs IA (deteccion y OCR), con metricas reales."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.config import RESULTS_DIR  # noqa: E402

st.set_page_config(page_title="Comparativa", page_icon="📊", layout="wide")
st.title("📊 Comparativa: clasico vs Machine Learning vs Deep Learning")

csv_path = RESULTS_DIR / "comparison.csv"
png_path = RESULTS_DIR / "comparison.png"

st.markdown(
    "Corre `python main.py compare` (o el boton de abajo) para comparar el detector "
    "clasico contra YOLO, y los tres motores de OCR entre si, sobre datos con "
    "resultado conocido (ground truth)."
)

if st.button("🔄 Correr comparacion ahora (puede tardar unos minutos)"):
    with st.spinner("Corriendo benchmark..."):
        from scripts.benchmark_compare import run_benchmark
        run_benchmark()
    st.success("Listo.")
    st.rerun()

if csv_path.exists():
    df = pd.read_csv(csv_path)
    st.subheader("Resultados")
    st.dataframe(df, use_container_width=True)

    if png_path.exists():
        st.image(str(png_path), use_container_width=True)
else:
    st.info("Todavia no hay resultados guardados. Apreta el boton de arriba para generarlos.")

st.divider()
st.subheader("Accuracy medida contra fotos reales (no solo sinteticas)")
st.markdown(
    """
| Motor | Accuracy en fotos reales | Exactas |
|---|---|---|
| EasyOCR | ~80.8% | 10/24 |
| Clasificador ML (SVM/RF) | ~57.2% | 2/24 |

Estas cifras salen de `scripts/evaluate_real_plates.py`, corrido contra las
fotos reales en `data/plates/real/` con su texto verdadero anotado a mano.
El benchmark de arriba usa datos sinteticos (mas facil), asi que los numeros
de esta tabla son mas representativos de uso real.
    """
)
