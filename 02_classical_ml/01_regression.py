"""
01 - Regression: Zahlen vorhersagen
====================================
Regression = ein Modell lernt, eine Zahl vorherzusagen.
Beispiel: Hauspreis basierend auf Größe, Zimmer, Lage.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. Daten generieren
# ============================================================

X, y = make_regression(n_samples=500, n_features=5, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training: {X_train.shape[0]} Samples, {X_train.shape[1]} Features")
print(f"Test:     {X_test.shape[0]} Samples")

# ============================================================
# 2. Verschiedene Modelle vergleichen
# ============================================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"mse": mse, "rmse": np.sqrt(mse), "r2": r2}

    print(f"\n{name}:")
    print(f"  RMSE: {np.sqrt(mse):.2f} (niedriger = besser)")
    print(f"  R²:   {r2:.4f} (1.0 = perfekt)")

# ============================================================
# 3. Visualisierung
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_title(f"{name}\nR² = {results[name]['r2']:.3f}")
    ax.set_xlabel("Tatsächlicher Wert")
    ax.set_ylabel("Vorhersage")

plt.tight_layout()
plt.savefig("regression_vergleich.png", dpi=150)
plt.show()

# ============================================================
# 4. Feature Importance - was ist dem Modell wichtig?
# ============================================================

rf = models["Random Forest"]
importances = rf.feature_importances_
for i, imp in enumerate(importances):
    bar = "█" * int(imp * 50)
    print(f"  Feature {i}: {imp:.3f} {bar}")

print("\n💡 Feature Importance zeigt dir, welche Eingaben das Modell am meisten nutzt.")
