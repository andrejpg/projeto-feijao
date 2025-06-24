import cv2
import numpy as np

def extrair_caracteristicas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return [0, 0, 0, 0]

    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    circularidade = 4 * np.pi * area / (perimetro ** 2 + 1e-5)
    cor_media = np.mean(gray)
    return [area, perimetro, circularidade, cor_media]
