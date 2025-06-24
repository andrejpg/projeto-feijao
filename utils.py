import os
import pandas as pd
import cv2
from extrator import extrair_caracteristicas

def gerar_csv(pasta_boas, pasta_ruins, arquivo_saida):
    dados = []
    classes = []

    for img_file in os.listdir(pasta_boas):
        img = cv2.imread(os.path.join(pasta_boas, img_file))
        features = extrair_caracteristicas(img)
        dados.append(features)
        classes.append(1)

    for img_file in os.listdir(pasta_ruins):
        img = cv2.imread(os.path.join(pasta_ruins, img_file))
        features = extrair_caracteristicas(img)
        dados.append(features)
        classes.append(0)

    df = pd.DataFrame(dados, columns=["area", "perimetro", "circularidade", "cor_media"])
    df["classe"] = classes
    df.to_csv(arquivo_saida, index=False)
