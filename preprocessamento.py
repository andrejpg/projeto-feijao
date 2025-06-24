import cv2
import os

def segmentar_imagem(path_imagem, salvar_em=None):
    img = cv2.imread(path_imagem)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imagens_segmentadas = []

    for i, contorno in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(contorno)
        recorte = img[y:y+h, x:x+w]
        imagens_segmentadas.append(recorte)
        if salvar_em:
            nome = os.path.join(salvar_em, f"{os.path.basename(path_imagem).split('.')[0]}_{i}.png")
            cv2.imwrite(nome, recorte)

    return imagens_segmentadas