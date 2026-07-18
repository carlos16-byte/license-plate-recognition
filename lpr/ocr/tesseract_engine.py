"""Motor de OCR alternativo: Tesseract (requiere el binario instalado aparte)."""
from __future__ import annotations

import re
import shutil

import cv2
import numpy as np

from lpr.ocr.engine_base import OCREngine, OCRResult

INSTALL_HINT = (
    "Tesseract no esta instalado o no esta en el PATH. En Windows: descarga el instalador desde "
    "https://github.com/UB-Mannheim/tesseract/wiki, instalalo y agrega la carpeta de instalacion "
    "al PATH (o configura pytesseract.pytesseract.tesseract_cmd)."
)


class TesseractOCREngine(OCREngine):
    name = "tesseract"

    def __init__(self):
        import pytesseract

        self._pytesseract = pytesseract
        if shutil.which("tesseract") is None:
            raise RuntimeError(INSTALL_HINT)

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if plate_image.ndim == 3 else plate_image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        try:
            data = self._pytesseract.image_to_data(
                thresh, config=config, output_type=self._pytesseract.Output.DICT
            )
        except Exception as exc:  # binario roto/ausente en tiempo de ejecucion
            raise RuntimeError(INSTALL_HINT) from exc

        words = [w for w in data["text"] if w.strip()]
        confidences = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]

        text = re.sub(r"[^A-Z0-9]", "", "".join(words).upper())
        confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
        return OCRResult(text=text, confidence=confidence)
