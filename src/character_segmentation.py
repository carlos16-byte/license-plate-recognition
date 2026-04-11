import cv2
import numpy as np

def segmentar_caracteres(placa):
    # Convertir a gris
    gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

    # Binarizar imagen
    _, thresh = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentar_caracteres = []

    for contornos in contornos:
        x, y, w, h = cv2.boundingRect(c)

        # Filtrar por tamaño (evitar ruido)
        if h > 20 and w > 5:
            char = thresh[y:y+h, x:x+w]
            segmentar_caracteres.append((x, char))

   # Odenar de izquiera a derecha
    segmentar_caracteres = sorted(segmentar_caracteres, key=lambda x: x[0])

    # Solo devolver las imágenes de los caracteres
    segmentar_caracteres = [char for _, char in segmentar_caracteres]

    return segmentar_caracteres