# Procesamiento de Imagenes para el Reconocimiento de Placas de Vehículos 1.0
import cv2
from src.image_processing import procesar_imagen
from src.plate_detection import detectar_placa
from src.character_segmentation import segmentar_caracteres 

# Cargar imagen
imagen = cv2.imread('license-plate-recognition\data\car2.jpg')

if imagen is None:
    print("Error al cargar la imagen.")
    exit()

# Procesar imagen
gris, bordes = procesar_imagen(imagen)

# Redimensionar imágenes para mejor visualización
def redimensionar(img, ancho=600):
    alto = int(img.shape[0] * (ancho / img.shape[1]))
    return cv2.resize(img, (ancho, alto))

imagen_small = redimensionar(imagen)
gris_small = redimensionar(gris)
bordes_small = redimensionar(bordes)

# Detectar placa
placa = detectar_placa(bordes, imagen)

if placa is not None:
    placa_small = redimensionar(placa, 300)
    cv2.imshow('Placa Detectada', placa_small)
else:
    print("No se detecto ninguna placa")

# Segmentar caracteres
if placa is not None:
    caracteres = segmentar_caracteres(placa)
    
    for i, char in enumerate(caracteres):
        char_small = redimensionar(char, 100)
        cv2.imshow(f"char {i}", char_small)

# Mostrar resultados
cv2.imshow('Imagen Original', imagen_small)
cv2.imshow('Imagen en Escala de Grises', gris_small)
cv2.imshow('Imagen con Bordes', bordes_small)
cv2.waitKey(0)
cv2.destroyAllWindows()