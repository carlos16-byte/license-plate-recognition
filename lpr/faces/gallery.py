"""Gestion de la galeria de personas conocidas: data/faces/<nombre>/*.jpg."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from lpr.config import FACES_DIR
from lpr.faces.detector import FaceDetector


class FaceGallery:
    def __init__(self, root: Path = FACES_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def person_dir(self, name: str) -> Path:
        safe_name = name.strip().replace(" ", "_")
        path = self.root / safe_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def add_image(self, name: str, image: np.ndarray) -> Path:
        person_dir = self.person_dir(name)
        existing = list(person_dir.glob("*.jpg"))
        out_path = person_dir / f"{len(existing):04d}.jpg"
        cv2.imwrite(str(out_path), image)
        return out_path

    def list_people(self) -> List[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.glob("*.jpg")))

    def load_training_faces(self, detector: FaceDetector, face_size: int = 200) -> tuple[List[np.ndarray], List[str]]:
        """Detecta y recorta la cara en cada foto enrolada. Devuelve (recortes, nombres)."""
        faces: List[np.ndarray] = []
        labels: List[str] = []

        for name in self.list_people():
            for img_path in self.person_dir(name).glob("*.jpg"):
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                detections = detector.detect(image)
                if not detections:
                    continue
                best = max(detections, key=lambda f: f.w * f.h)
                faces.append(detector.crop_gray(image, best, size=face_size))
                labels.append(name)

        return faces, labels


def save_label_map(label_map: Dict[int, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label_map, indent=2))


def load_label_map(path: Path) -> Dict[int, str]:
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data.items()}
