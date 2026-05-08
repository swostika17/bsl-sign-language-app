import pandas as pd
import os

CSV_FILE = "hand_landmarks.csv"   # ✅ correct file name

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found in current directory")

df = pd.read_csv("hand_landmarks.csv")


print("Total samples:", len(df))
print("Labels found:")
print(df["label"].value_counts())

# Optional: save a cleaned copy
df.to_csv("bsl_data_combined.csv", index=False)
print("Saved as bsl_data_combined.csv")
