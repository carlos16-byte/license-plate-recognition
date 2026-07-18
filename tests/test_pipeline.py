import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lpr.config import CHAR_CLASSIFIER_PATH, DATA_DIR, YOLO_PLATE_WEIGHTS  # noqa: E402
from lpr.faces.detector import FaceDetector  # noqa: E402
from lpr.plates.classical_detector import ClassicalPlateDetector  # noqa: E402
from lpr.plates.segmentation import segmentar_caracteres  # noqa: E402
from lpr.preprocessing.classical import procesar_imagen  # noqa: E402

CAR_IMAGES = [DATA_DIR / "car1.jpg", DATA_DIR / "car2.jpg"]


@pytest.fixture(params=CAR_IMAGES, ids=[p.name for p in CAR_IMAGES])
def car_image(request):
    image = cv2.imread(str(request.param))
    assert image is not None, f"no se pudo cargar {request.param}"
    return image


def test_preprocesamiento_no_falla(car_image):
    gris, bordes = procesar_imagen(car_image)
    assert gris.shape[:2] == car_image.shape[:2]
    assert bordes.shape[:2] == car_image.shape[:2]


def test_detector_clasico_encuentra_candidato(car_image):
    detector = ClassicalPlateDetector()
    candidates = detector.detect(car_image)
    assert len(candidates) > 0
    best = candidates[0]
    assert best.w > 0 and best.h > 0
    assert 1.0 <= best.w / best.h <= 7.0


def test_segmentacion_sobre_placa_sintetica():
    from scripts.generate_synthetic_plates import _render_plate

    rng = random.Random(7)
    plate = _render_plate("ABC1234", rng)
    plate_bgr = cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2BGR)
    caracteres = segmentar_caracteres(plate_bgr)
    assert len(caracteres) >= 5  # deberia separar la mayoria de los 7 caracteres


def test_generador_dataset_sintetico(tmp_path, monkeypatch):
    import scripts.generate_synthetic_plates as gen

    monkeypatch.setattr(gen, "PLATES_SYNTHETIC_DIR", tmp_path)
    gen.generate_split("train", num_images=3, seed=1)

    images = list((tmp_path / "images" / "train").glob("*.jpg"))
    labels = list((tmp_path / "labels" / "train").glob("*.txt"))
    assert len(images) == 3
    assert len(labels) == 3

    for label_path in labels:
        values = label_path.read_text().split()
        assert len(values) == 5
        cls, cx, cy, w, h = int(values[0]), *map(float, values[1:])
        assert cls == 0
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_face_detector_no_falla(car_image):
    detector = FaceDetector()
    faces = detector.detect(car_image)
    assert isinstance(faces, list)


@pytest.mark.skipif(not CHAR_CLASSIFIER_PATH.exists(), reason="corre scripts/train_char_classifier.py primero")
def test_clasificador_ml_caracteres_roundtrip():
    from lpr.ocr.ml_char_classifier import MLCharClassifierEngine
    from scripts.generate_synthetic_plates import _render_plate

    engine = MLCharClassifierEngine()
    rng = random.Random(3)
    plate = _render_plate("ZX9182", rng)
    plate_bgr = cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2BGR)
    result = engine.recognize(plate_bgr)
    assert isinstance(result.text, str)
    assert len(result.text) > 0


@pytest.mark.skipif(not YOLO_PLATE_WEIGHTS.exists(), reason="corre scripts/train_yolo_plate.py primero")
def test_yolo_detector_carga_pesos_entrenados():
    from lpr.plates.yolo_detector import YoloPlateDetector

    detector = YoloPlateDetector()
    assert detector.is_trained
