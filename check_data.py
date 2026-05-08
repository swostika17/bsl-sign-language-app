import pandas as pd

df = pd.read_csv("bsl_data_combined.csv")
print(df["label"].value_counts())
