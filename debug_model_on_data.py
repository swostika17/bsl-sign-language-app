import pandas as pd
import joblib

df = pd.read_csv("bsl_data_combined.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = joblib.load("bsl_model.pkl")

pred = model.predict(X)

print("Accuracy on full dataset:", (pred == y).mean())
print(pd.crosstab(y, pred))
