import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. Load data ───────────────────────────────────────────────────────────────
df       = pd.read_csv("model_artifacts/crop_yield_cleaned.csv")
encoders = joblib.load("model_artifacts/label_encoders.pkl")

feature_cols = ["Crop_enc","Season_enc","State_enc","Area","Annual_Rainfall","Fertilizer","Pesticide"]
X = df[feature_cols]
y = df["Yield_log"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("evaluation", exist_ok=True)
print("=" * 55)
print("  CROP YIELD MODEL EVALUATION REPORT")
print("=" * 55)


# ── 2. Model Comparison ────────────────────────────────────────────────────────
print("\n── STEP 1: Model Comparison (5-fold Cross Validation) ──")

models = {
    "Linear Regression"  : LinearRegression(),
    "Decision Tree"      : DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

comparison_results = {}
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=5, scoring='r2', n_jobs=-1)
    comparison_results[name] = scores.mean()
    print(f"  {name:<25} R² = {scores.mean():.4f} ± {scores.std():.4f}")

# Plot model comparison
fig, ax = plt.subplots(figsize=(8, 4))
names  = list(comparison_results.keys())
values = list(comparison_results.values())
colors = ['#C0392B' if v < 0.7 else '#E67E22' if v < 0.85 else '#27AE60' for v in values]
bars   = ax.barh(names, values, color=colors, edgecolor='white', height=0.5)
ax.set_xlabel('R² Score (Cross Validation)')
ax.set_title('Model Comparison — 5-Fold Cross Validation')
ax.set_xlim(0, 1)
for bar, val in zip(bars, values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('evaluation/1_model_comparison.png', dpi=150)
plt.close()
print("  Saved → evaluation/1_model_comparison.png")


# ── 3. Best model: Random Forest with Cross Validation ─────────────────────────
print("\n── STEP 2: Random Forest — Cross Validation Scores ────")

rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2,
                           random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"  Fold scores : {[round(s,4) for s in cv_scores]}")
print(f"  Mean R²     : {cv_scores.mean():.4f}")
print(f"  Std Dev     : {cv_scores.std():.4f}")
print("  (Low std dev = model is consistent across different data splits)")

# Train final model on full training set
rf.fit(X_train, y_train)
y_pred_log = rf.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_true     = np.expm1(y_test)

print(f"\n  Test Set Results:")
print(f"  R² Score : {r2_score(y_true, y_pred):.4f}")
print(f"  RMSE     : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"  MAE      : {mean_absolute_error(y_true, y_pred):.4f}")


# ── 4. Feature Importance ──────────────────────────────────────────────────────
print("\n── STEP 3: Feature Importance ──────────────────────────")

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
for feat, score in importances.items():
    bar = "█" * int(score * 50)
    print(f"  {feat:<20} {bar} {score:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
colors  = ['#27AE60' if v > 0.1 else '#2980B9' for v in importances.values]
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance — What Affects Crop Yield Most?')
for i, (val, name) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.003, i, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('evaluation/2_feature_importance.png', dpi=150)
plt.close()
print("\n  Saved → evaluation/2_feature_importance.png")


# ── 5. Learning Curve ──────────────────────────────────────────────────────────
print("\n── STEP 4: Learning Curve ──────────────────────────────")

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    X, y, cv=5, scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_mean, 'o-', color='#27AE60', label='Training Score')
ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color='#27AE60')
ax.plot(train_sizes, val_mean, 'o-', color='#2980B9', label='Validation Score')
ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color='#2980B9')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('R² Score')
ax.set_title('Learning Curve — Does More Data Help?')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evaluation/3_learning_curve.png', dpi=150)
plt.close()
print("  Saved → evaluation/3_learning_curve.png")
if val_mean[-1] > val_mean[-2]:
    print("  Insight: Model still improving with more data — more records would help!")
else:
    print("  Insight: Model has plateaued — adding more data won't help much.")


# ── 6. Residual Analysis ───────────────────────────────────────────────────────
print("\n── STEP 5: Residual Analysis ───────────────────────────")

residuals = y_true.values - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residual scatter plot
axes[0].scatter(y_pred, residuals, alpha=0.3, color='#2980B9', s=10)
axes[0].axhline(0, color='red', linewidth=1.5)
axes[0].set_xlabel('Predicted Yield')
axes[0].set_ylabel('Residual (Actual - Predicted)')
axes[0].set_title('Residual Plot')
axes[0].grid(True, alpha=0.3)

# Residual distribution
axes[1].hist(residuals, bins=60, color='#27AE60', alpha=0.7, edgecolor='white')
axes[1].axvline(0, color='red', linewidth=1.5)
axes[1].set_xlabel('Residual Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution (should be centered at 0)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation/4_residual_analysis.png', dpi=150)
plt.close()
print(f"  Mean residual : {residuals.mean():.4f} (closer to 0 = less bias)")
print(f"  Std residual  : {residuals.std():.4f}")
print("  Saved → evaluation/4_residual_analysis.png")


# ── 7. Actual vs Predicted ────────────────────────────────────────────────────
print("\n── STEP 6: Actual vs Predicted Plot ────────────────────")

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_true, y_pred, alpha=0.2, color='#2980B9', s=10)
max_val = max(y_true.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Actual Yield')
ax.set_ylabel('Predicted Yield')
ax.set_title(f'Actual vs Predicted Yield (R² = {r2_score(y_true, y_pred):.4f})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evaluation/5_actual_vs_predicted.png', dpi=150)
plt.close()
print("  Saved → evaluation/5_actual_vs_predicted.png")


# ── 8. Save updated model ─────────────────────────────────────────────────────
joblib.dump(rf, "model_artifacts/model.pkl")
print("\n  Updated model saved → model_artifacts/model.pkl")

print("\n" + "=" * 55)
print("  ALL DONE! Check the evaluation/ folder for charts.")
print("=" * 55)