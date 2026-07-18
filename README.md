# Sistema de Reconocimiento de Placas y Rostros

Sistema modular de Vision por Computadora e Inteligencia Artificial que detecta
placas vehiculares y rostros humanos, comparando un enfoque **clasico**
(OpenCV) contra enfoques de **IA** (Machine Learning y Deep Learning) en cada
etapa del pipeline. Funciona con imagenes y con video/webcam en tiempo real.

## Arquitectura

```
lpr/                       paquete instalable (pip install -e .)
├── config.py               rutas y umbrales compartidos
├── io/media.py             carga/guardado de imagenes, video y webcam
├── preprocessing/          gris + reduccion de ruido + Canny adaptativo
├── plates/
│   ├── classical_detector.py   Fase 1: contornos + heuristicas (aspect ratio,
│   │                            densidad de bordes, blobs tipo-caracter)
│   ├── yolo_detector.py        Fase 3: YOLOv8, cae al detector clasico si
│   │                            no hay pesos entrenados
│   └── segmentation.py         segmentacion de caracteres (Otsu adaptativo)
├── ocr/
│   ├── easyocr_engine.py       motor de OCR por defecto
│   ├── tesseract_engine.py     motor alternativo (requiere binario aparte)
│   └── ml_char_classifier.py   Fase 2: SVM/RandomForest sobre caracteres
├── faces/
│   ├── detector.py             Haar Cascade (incluido en OpenCV)
│   ├── gallery.py               galeria de personas enroladas
│   └── recognizer.py            LBPH (cv2.face), offline
├── pipeline.py              orquesta deteccion -> (segmentacion) -> OCR/reconocimiento
└── cli.py                   subcomandos de la CLI

scripts/
├── generate_synthetic_plates.py  dataset sintetico de placas (formato YOLO)
├── train_yolo_plate.py           entrena YOLOv8n -> models/yolo_plate.pt
├── train_char_classifier.py      entrena SVM/RF sobre glifos sinteticos
└── benchmark_compare.py          compara clasico vs IA y guarda resultados/
```

## Instalacion

```bash
pip install -e .
pip install -r requirements.txt
```

Notas del entorno:
- **EasyOCR** es el motor de OCR por defecto: no requiere instalar nada mas
  aparte de `pip install`, descarga sus pesos la primera vez que se usa.
- **Tesseract** es opcional: si queres usarlo (`--ocr tesseract`), instala el
  binario desde https://github.com/UB-Mannheim/tesseract/wiki y agregalo al
  PATH.
- Se usa `opencv-contrib-python` (no `opencv-python`) porque el reconocimiento
  facial (Fase 4) necesita `cv2.face`.
- `numpy` esta fijado en `<2` porque `torch`/`ultralytics` (Fase 3) todavia no
  son compatibles con NumPy 2.x en esta version.

## Uso rapido

```bash
# Fase 1: deteccion clasica + OCR sobre una imagen
python main.py plate-image data/car2.jpg --detector classical --ocr easyocr --show

# Fase 3: deteccion con YOLO (requiere haber entrenado antes, ver abajo)
python main.py plate-image data/car2.jpg --detector yolo --ocr easyocr

# Video / webcam en tiempo real (0 = camara por defecto)
python main.py plate-video 0 --detector classical --ocr easyocr

# Fase 4: enrolar una persona (webcam o carpeta de fotos) y reconocerla
python main.py face-enroll --name "Juan" --source webcam
python main.py face-recognize data/alguna_foto.jpg --show
python main.py face-video 0

# Comparar clasico vs IA (deteccion y OCR) y guardar resultados/comparison.csv
python main.py compare
```

## Entrenar los modelos de IA

Ningun dataset real de placas/caracteres estaba disponible al iniciar este
proyecto, asi que Fase 2 y Fase 3 se **bootstrapean con datos sinteticos**
generados por PIL/OpenCV (tecnica estandar en ANPR cuando no hay datos
etiquetados). El pipeline queda funcional de punta a punta desde ya, y
reentrenar con datos reales el dia de mañana es tan simple como reemplazar
las imagenes en el mismo formato y volver a correr el script.

```bash
# Fase 2: clasificador de caracteres (SVM/RandomForest) sobre glifos sinteticos
python scripts/train_char_classifier.py --samples-per-char 300

# Fase 3: dataset sintetico de placas + entrenamiento YOLOv8n (CPU)
python scripts/generate_synthetic_plates.py --num-images 320
python scripts/train_yolo_plate.py --epochs 25 --imgsz 320
```

## Limites honestos

- El detector YOLO entrenado con datos sinteticos es una **prueba de concepto
  funcional** del pipeline de Deep Learning, no un detector de nivel
  produccion. Para mejorar la precision real: agrega fotos reales de placas
  (mismo formato YOLO) en `data/plates/synthetic/` y vuelve a entrenar.
- El detector clasico por contornos puede confundir objetos rectangulares con
  textura densa (parabrisas, faros) con una placa; se mitigo con heuristicas
  de aspect ratio + densidad de bordes + conteo de blobs tipo-caracter, pero
  no es infalible.
- El reconocimiento facial no reconoce a nadie hasta correr `face-enroll` al
  menos una vez. La deteccion de rostros (Haar Cascade) funciona de
  inmediato, aunque como cualquier Haar Cascade puede dar algun falso
  positivo en texturas complejas.
- Sin GPU disponible, el entrenamiento YOLO usa imagenes chicas y pocas
  epocas para terminar en tiempos razonables en CPU.

## Tests

```bash
pytest
```
