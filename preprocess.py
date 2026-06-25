import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

df = pd.read_csv("crop_yield_2.csv")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

str_cols = df.select_dtypes(include=["object"]).columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

df["Yield_log"] = np.log1p(df["Yield"])

categorical_cols = ["Crop", "Season", "State"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"Encoded '{col}': {len(le.classes_)} classes")

feature_cols = ["Crop_enc","Season_enc","State_enc","Area","Annual_Rainfall","Fertilizer","Pesticide"]
X = df[feature_cols]
y = df["Yield_log"]

os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(encoders, "model_artifacts/label_encoders.pkl")
df.to_csv("model_artifacts/crop_yield_cleaned.csv", index=False)
print("Done! Saved encoders and cleaned CSV.")