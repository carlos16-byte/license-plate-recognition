# Procesamiento de Imagenes para el Reconocimiento de Placas de Vehículos 1.0
import cv2

def procesar_imagen(imagen):
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Reducir ruido manteniendo los bordes
    bordes = cv2.bilateralFilter(gris, 11, 17, 17)

    # Detectar bordes
    bordes = cv2.Canny(bordes, 30, 200)
    return gris, bordes
