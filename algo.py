import pandas as pd

df = pd.read_csv("preds_healthy.csv")

correctas = (df["pred"] == "tizon_foliar").sum()
total = len(df)

print(f"Correctas: {correctas}/{total}")
print(f"Accuracy: {correctas/total:.4f}")
