import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("model_artifacts/crop_yield_cleaned.csv")
encoders = joblib.load("model_artifacts/label_encoders.pkl")
print(f"Loaded cleaned data: {df.shape[0]} rows")

feature_cols = ["Crop_enc","Season_enc","State_enc","Area","Annual_Rainfall","Fertilizer","Pesticide"]
X = df[feature_cols]
y = df["Yield_log"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
print("\nTraining Random Forest... (takes ~30 seconds)")

model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Training complete!")

y_pred = np.expm1(model.predict(X_test))
y_true = np.expm1(y_test)

print(f"\n── Model Evaluation ──────────────────")
print(f"  R² Score : {r2_score(y_true, y_pred):.4f}")
print(f"  RMSE     : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"  MAE      : {mean_absolute_error(y_true, y_pred):.4f}")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n── Feature Importance ────────────────")
for feat, score in importances.items():
    print(f"  {feat:<20} {score:.4f}")

os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(model, "model_artifacts/model.pkl")
print(f"\nSaved model → model_artifacts/model.pkl")
print("All done!")