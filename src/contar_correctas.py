import pandas as pd

df = pd.read_csv("preds.csv")

correctas = (df["pred"] == "maize").sum()
total = len(df)

print(f"Correctas: {correctas}/{total}")
print(f"Accuracy: {correctas/total:.4f}")
