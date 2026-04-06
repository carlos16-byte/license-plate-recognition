import cv2
import imutils

def detectar_placa(bordes, imagen_original):
    contornos = cv2.findContours(bordes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos = imutils.grab_contours(contornos)

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

    placa = None

    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimetro, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            placa = imagen_original[y:y+h, x:x+w]
            break

    return placa