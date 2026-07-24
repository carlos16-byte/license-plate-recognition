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
├── autolabel_real_plates.py      propone cajas YOLO sobre fotos reales (revision humana)
├── train_yolo_plate.py           entrena YOLOv8n -> models/yolo_plate.pt
├── train_char_classifier.py      entrena SVM/RF sobre placas sinteticas segmentadas
├── benchmark_compare.py          compara clasico vs IA (datos sinteticos) y guarda resultados/
└── evaluate_real_plates.py       mide accuracy de OCR contra fotos reales (mas representativo)

streamlit_app.py + pages/         app web (imagen y video subido, ver "App web" abajo)
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

## App web

```bash
streamlit run streamlit_app.py
```

Abre una app con 3 paginas (navegacion en la barra lateral):

- **Placas** — subi una imagen o video, elegi detector (clasico/YOLO) y motor
  de OCR (EasyOCR/Tesseract/ML), y ve el resultado anotado con texto y
  confianza.
- **Rostros** — deteccion y reconocimiento facial sobre imagen o video. El
  enrolamiento de personas se sigue haciendo por CLI (`face-enroll`) porque
  requiere capturar fotos interactivamente con la webcam.
- **Comparativa** — resultados de `benchmark_compare.py` y de
  `evaluate_real_plates.py` (accuracy contra fotos reales).

Para video, se puede elegir "procesar 1 de cada N frames" ya que correr
deteccion+OCR en cada frame es lento; los frames salteados repiten la ultima
deteccion para que el video anotado no parpadee.

## Entrenar los modelos de IA

Ningun dataset real de placas/caracteres estaba disponible al iniciar este
proyecto, asi que Fase 2 y Fase 3 se **bootstrapean con datos sinteticos**
generados por PIL/OpenCV (tecnica estandar en ANPR cuando no hay datos
etiquetados). El pipeline queda funcional de punta a punta desde ya, y
reentrenar con datos reales el dia de mañana es tan simple como reemplazar
las imagenes en el mismo formato y volver a correr el script.

```bash
# Fase 2: clasificador de caracteres (SVM/RandomForest). Genera placas
# sinteticas completas, las degrada (blur/ruido/jpeg) y las segmenta con el
# MISMO pipeline que produccion, en vez de entrenar con letras aisladas
# perfectas (eso fue lo que hacia que rindiera bien en synthetic pero mal en
# fotos reales -- ver "Limites honestos" abajo).
python scripts/train_char_classifier.py --num-plates 4000

# Fase 3: dataset sintetico de placas + entrenamiento YOLOv8n (CPU)
python scripts/generate_synthetic_plates.py --num-images 320
python scripts/train_yolo_plate.py --epochs 25 --imgsz 320

# Opcional: sumar fotos reales de placas al entrenamiento de YOLO.
# scripts/autolabel_real_plates.py propone la caja con el detector clasico
# (guarda un contact_sheet.jpg para revisar visualmente antes de usarla),
# lo que no matchee bien se etiqueta a mano y se agrega igual.
python scripts/autolabel_real_plates.py --source ruta/a/tus/fotos
# luego copiar imagenes+labels de data/plates/real/ dentro de
# data/plates/synthetic/images|labels/{train,val}/ y volver a correr train_yolo_plate.py
```

## Limites honestos

Accuracy medida contra 24 fotos reales (no solo sinteticas), con
`scripts/evaluate_real_plates.py`:

| Motor | Accuracy por caracter | Exactas |
|---|---|---|
| EasyOCR | ~80.8% | 10/24 |
| Clasificador ML (SVM/RandomForest) | ~57.2% | 2/24 |

- YOLO ya fue reentrenado con 24 fotos reales ademas de las sinteticas
  (mAP50=0.984 en el set de validacion combinado) y generaliza a fotos que
  nunca vio en entrenamiento. Sigue siendo un dataset chico -- mas fotos
  reales variadas (angulos, paises, condiciones de luz) mejoran la
  robustez.
- El detector clasico por contornos puede confundir objetos rectangulares con
  textura densa (parabrisas, faros, marcas de agua) con una placa; se mitigo
  con heuristicas de aspect ratio + densidad de bordes + conteo de blobs
  tipo-caracter, pero no es infalible.
- El clasificador ML (Fase 2) queda bastante atras de EasyOCR incluso
  entrenando con el mismo pipeline de segmentacion de produccion (ver arriba)
  -- tiene sentido, EasyOCR es una red neuronal entrenada con millones de
  imagenes de texto real, mientras que SVM/RandomForest sobre HOG es un
  enfoque mucho mas simple. Esta pensado como comparacion educativa
  (clasico vs ML vs Deep Learning), no como motor principal de la app.
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
