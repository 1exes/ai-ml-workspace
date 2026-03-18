"""
02 - Daten laden, verstehen und visualisieren
==============================================
Bevor du ML machst, musst du deine Daten verstehen.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Daten laden mit Pandas
# ============================================================

# Wir erstellen einen Beispiel-Datensatz (normalerweise: pd.read_csv("datei.csv"))
np.random.seed(42)
n = 200

data = pd.DataFrame({
    "alter": np.random.randint(18, 65, n),
    "einkommen": np.random.normal(45000, 15000, n).astype(int),
    "stunden_online": np.random.exponential(3, n).round(1),
    "kauft_produkt": np.random.choice([0, 1], n, p=[0.6, 0.4]),
})

print("Erste 5 Zeilen:")
print(data.head())

print("\nStatistiken:")
print(data.describe())

print(f"\nShape: {data.shape} ({data.shape[0]} Samples, {data.shape[1]} Features)")
print(f"Datentypen:\n{data.dtypes}")

# ============================================================
# 2. Visualisierung - dein wichtigstes Werkzeug
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Daten-Exploration", fontsize=16)

# Histogramm - Verteilung eines Features
axes[0, 0].hist(data["einkommen"], bins=30, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Einkommensverteilung")
axes[0, 0].set_xlabel("Einkommen (€)")

# Scatter - Beziehung zwischen Features
colors = ["red" if k == 1 else "blue" for k in data["kauft_produkt"]]
axes[0, 1].scatter(data["alter"], data["einkommen"], c=colors, alpha=0.5, s=20)
axes[0, 1].set_title("Alter vs Einkommen (rot=kauft)")
axes[0, 1].set_xlabel("Alter")
axes[0, 1].set_ylabel("Einkommen")

# Box Plot - Verteilung pro Kategorie
data.boxplot(column="stunden_online", by="kauft_produkt", ax=axes[1, 0])
axes[1, 0].set_title("Online-Stunden nach Kaufverhalten")
axes[1, 0].set_xlabel("Kauft Produkt (0=Nein, 1=Ja)")

# Korrelationsmatrix - welche Features hängen zusammen?
corr = data.corr()
im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr.columns)))
axes[1, 1].set_yticks(range(len(corr.columns)))
axes[1, 1].set_xticklabels(corr.columns, rotation=45, ha="right")
axes[1, 1].set_yticklabels(corr.columns)
axes[1, 1].set_title("Korrelationsmatrix")
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("exploration.png", dpi=150)
plt.show()
print("\nPlot gespeichert als 'exploration.png'")

# ============================================================
# 3. Daten vorbereiten für ML
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features (X) und Target (y) trennen
X = data[["alter", "einkommen", "stunden_online"]].values
y = data["kauft_produkt"].values

# Train/Test Split - IMMER machen!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} Samples")
print(f"Test:  {X_test.shape[0]} Samples")

# Normalisierung - Features auf gleiche Skala bringen
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NUR transform, NICHT fit!

print(f"\nVor Normalisierung - Mean: {X_train.mean(axis=0).round(1)}")
print(f"Nach Normalisierung - Mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"Nach Normalisierung - Std:  {X_train_scaled.std(axis=0).round(4)}")

print("\n✅ Daten sind bereit für ML!")
