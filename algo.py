import pandas as pd

# Cargar el archivo de predicciones
df = pd.read_csv("preds_healthy.csv")

# Contar cuántas predicciones fueron "maize"
correctas = (df["pred"] == "tizon_foliar").sum()
total = len(df)

print(f"Correctas: {correctas}/{total}")
print(f"Accuracy: {correctas/total:.4f}")
