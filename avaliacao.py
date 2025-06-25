from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("dados/padroes_feijao.csv")

X = df.drop("classe", axis=1).values
y = df["classe"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = load_model("modelos/modelo_feijao.h5")
y_pred = model.predict(X_test) > 0.5


matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nMatriz de Confusão:")
print("                     Previsto")
print("                 Ruim     Bom")
print(f"Real  Ruim   {matriz[0][0]:>6}   {matriz[0][1]:>6}")
print(f"      Bom    {matriz[1][0]:>6}   {matriz[1][1]:>6}")

# classificação em português
print("\nRelatório de Classificação:")
relatorio = classification_report(y_test, y_pred, target_names=["Ruim", "Bom"])
print(relatorio)