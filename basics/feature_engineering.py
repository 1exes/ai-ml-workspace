import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Feature Engineering & Selection
================================
Gute Features sind wichtiger als komplexe Modelle.
Hier lernst du, wie man Rohdaten in nuetzliche Features verwandelt.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder,
    OneHotEncoder, PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    mutual_info_classif, RFE, SelectKBest, f_classif,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# ============================================================
# 1. Realistischen Datensatz erstellen (Hauspreise)
# ============================================================
print("=" * 60)
print("1. DATENSATZ ERSTELLEN")
print("=" * 60)

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "wohnflaeche": np.random.normal(100, 30, n).clip(30, 250).astype(int),
    "zimmer": np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.05, 0.15, 0.3, 0.25, 0.15, 0.1]),
    "baujahr": np.random.randint(1950, 2024, n),
    "stockwerk": np.random.choice(["EG", "1.OG", "2.OG", "3.OG", "DG"], n),
    "balkon": np.random.choice([0, 1], n, p=[0.3, 0.7]),
    "stadt": np.random.choice(["Berlin", "Muenchen", "Hamburg", "Koeln", "Frankfurt"], n),
    "entfernung_zentrum_km": np.random.exponential(5, n).round(1).clip(0.5, 30),
    "laerm_index": np.random.uniform(20, 80, n).round(1),
})

# Preis berechnen (realistisch)
data["preis"] = (
    data["wohnflaeche"] * 3500
    + data["zimmer"] * 15000
    - (2024 - data["baujahr"]) * 800
    + data["balkon"] * 20000
    - data["entfernung_zentrum_km"] * 5000
    - data["laerm_index"] * 200
    + np.random.normal(0, 30000, n)
).astype(int).clip(50000, 1500000)

# Preiskategorie (fuer Classification)
data["preisklasse"] = pd.cut(data["preis"], bins=3, labels=["guenstig", "mittel", "teuer"])

print(data.head(10).to_string())
print(f"\nShape: {data.shape}")
print(f"\nDatentypen:\n{data.dtypes}")

# ============================================================
# 2. Encoding: Kategorien in Zahlen
# ============================================================
print("\n" + "=" * 60)
print("2. ENCODING - Kategorien zu Zahlen")
print("=" * 60)

# Label Encoding (fuer ordinale Features - haben eine Reihenfolge)
print("\n--- Label Encoding (Stockwerk - hat Reihenfolge) ---")
stockwerk_mapping = {"EG": 0, "1.OG": 1, "2.OG": 2, "3.OG": 3, "DG": 4}
data["stockwerk_encoded"] = data["stockwerk"].map(stockwerk_mapping)
print(f"  EG -> 0, 1.OG -> 1, ..., DG -> 4")

# One-Hot Encoding (fuer nominale Features - keine Reihenfolge)
print("\n--- One-Hot Encoding (Stadt - keine Reihenfolge) ---")
stadt_dummies = pd.get_dummies(data["stadt"], prefix="stadt")
print(stadt_dummies.head(3).to_string())
print(f"  1 Feature -> {stadt_dummies.shape[1]} Features (eine Spalte pro Stadt)")

# ============================================================
# 3. Feature Transformationen
# ============================================================
print("\n" + "=" * 60)
print("3. FEATURE TRANSFORMATIONEN")
print("=" * 60)

# Log-Transformation (fuer schiefe Verteilungen)
print("\n--- Log-Transformation ---")
data["preis_log"] = np.log1p(data["preis"])
print(f"  Preis:     Min={data['preis'].min():>10,}, Max={data['preis'].max():>10,}, Std={data['preis'].std():>10,.0f}")
print(f"  Log(Preis): Min={data['preis_log'].min():>10.2f}, Max={data['preis_log'].max():>10.2f}, Std={data['preis_log'].std():>10.2f}")

# Binning (kontinuierlich -> kategorisch)
print("\n--- Binning ---")
data["baujahr_epoche"] = pd.cut(data["baujahr"], bins=[1949, 1970, 1990, 2010, 2025],
                                 labels=["Nachkrieg", "70er-80er", "90er-2000er", "Modern"])
print(data["baujahr_epoche"].value_counts().to_string())

# Polynomial Features (Interaktionen)
print("\n--- Polynomial Features ---")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
sample = data[["wohnflaeche", "zimmer"]].head(3)
poly_features = poly.fit_transform(sample)
print(f"  Input:  {sample.columns.tolist()} ({sample.shape[1]} Features)")
print(f"  Output: {poly.get_feature_names_out().tolist()} ({poly_features.shape[1]} Features)")
print(f"  -> 'wohnflaeche zimmer' ist das Interaktions-Feature!")

# Eigene Features
print("\n--- Eigene Features ---")
data["qm_pro_zimmer"] = (data["wohnflaeche"] / data["zimmer"]).round(1)
data["alter"] = 2024 - data["baujahr"]
data["zentral"] = (data["entfernung_zentrum_km"] < 5).astype(int)
print(f"  qm_pro_zimmer: Durchschnitt = {data['qm_pro_zimmer'].mean():.1f}")
print(f"  alter: Durchschnitt = {data['alter'].mean():.0f} Jahre")
print(f"  zentral (< 5km): {data['zentral'].sum()} von {len(data)} Wohnungen")

# ============================================================
# 4. Feature Selection
# ============================================================
print("\n" + "=" * 60)
print("4. FEATURE SELECTION - Welche Features sind wichtig?")
print("=" * 60)

# Numerische Features vorbereiten
feature_cols = ["wohnflaeche", "zimmer", "alter", "stockwerk_encoded",
                "balkon", "entfernung_zentrum_km", "laerm_index",
                "qm_pro_zimmer", "zentral"]
X = data[feature_cols].values
y = LabelEncoder().fit_transform(data["preisklasse"])

# Methode 1: Korrelation
print("\n--- Korrelation mit Preis ---")
correlations = data[feature_cols + ["preis"]].corr()["preis"].drop("preis").abs().sort_values(ascending=False)
for feat, corr in correlations.items():
    bar = "#" * int(corr * 40)
    print(f"  {feat:.<30s} {corr:.3f} {bar}")

# Methode 2: Mutual Information
print("\n--- Mutual Information (nicht-lineare Abhaengigkeiten) ---")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_ranking = sorted(zip(feature_cols, mi_scores), key=lambda x: -x[1])
for feat, score in mi_ranking:
    bar = "#" * int(score * 30)
    print(f"  {feat:.<30s} {score:.3f} {bar}")

# Methode 3: Recursive Feature Elimination
print("\n--- RFE (Recursive Feature Elimination) ---")
rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=5)
rfe.fit(X, y)
for feat, rank, selected in zip(feature_cols, rfe.ranking_, rfe.support_):
    marker = "[X]" if selected else "[ ]"
    print(f"  {marker} {feat} (Rang: {rank})")

# ============================================================
# 5. PCA - Dimensionsreduktion
# ============================================================
print("\n" + "=" * 60)
print("5. PCA - Dimensionsreduktion")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("\nVarianz erklaert pro Komponente:")
cumulative = 0
for i, var in enumerate(pca.explained_variance_ratio_):
    cumulative += var
    bar = "#" * int(var * 100)
    print(f"  PC{i+1}: {var:.1%} (kumulativ: {cumulative:.1%}) {bar}")
    if cumulative > 0.95:
        print(f"  -> {i+1} von {len(feature_cols)} Komponenten erklaeren 95% der Varianz!")
        break

# ============================================================
# 6. Vorher/Nachher Vergleich
# ============================================================
print("\n" + "=" * 60)
print("6. VORHER vs NACHHER - Machen Features einen Unterschied?")
print("=" * 60)

# Einfache Features
X_simple = data[["wohnflaeche", "zimmer", "balkon"]].values
X_train_s, X_test_s, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
clf_simple = RandomForestClassifier(n_estimators=100, random_state=42)
score_simple = cross_val_score(clf_simple, X_simple, y, cv=5).mean()

# Engineered Features
X_engineered = data[feature_cols].values
X_eng_with_dummies = np.hstack([X_engineered, stadt_dummies.values])
clf_eng = RandomForestClassifier(n_estimators=100, random_state=42)
score_eng = cross_val_score(clf_eng, X_eng_with_dummies, y, cv=5).mean()

print(f"\n  Einfache Features (3):     Accuracy = {score_simple:.1%}")
print(f"  Engineered Features ({X_eng_with_dummies.shape[1]}): Accuracy = {score_eng:.1%}")
print(f"  Verbesserung:              +{(score_eng - score_simple):.1%}")
print(f"\n  * Gute Features machen oft mehr aus als ein besserer Algorithmus!")

# ============================================================
# 7. Visualisierung
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Feature Engineering Uebersicht", fontsize=16)

# Korrelationsheatmap
corr_matrix = data[feature_cols].corr()
im = axes[0, 0].imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
axes[0, 0].set_xticks(range(len(feature_cols)))
axes[0, 0].set_yticks(range(len(feature_cols)))
axes[0, 0].set_xticklabels([f[:8] for f in feature_cols], rotation=45, ha="right", fontsize=7)
axes[0, 0].set_yticklabels([f[:8] for f in feature_cols], fontsize=7)
axes[0, 0].set_title("Korrelationsmatrix")
plt.colorbar(im, ax=axes[0, 0])

# PCA Varianz
axes[0, 1].bar(range(1, len(pca.explained_variance_ratio_) + 1),
               pca.explained_variance_ratio_, color="steelblue")
axes[0, 1].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                np.cumsum(pca.explained_variance_ratio_), "ro-")
axes[0, 1].set_xlabel("Komponente")
axes[0, 1].set_ylabel("Erklaerte Varianz")
axes[0, 1].set_title("PCA - Varianz pro Komponente")
axes[0, 1].axhline(y=0.95, color="gray", linestyle="--", label="95%")
axes[0, 1].legend()

# Feature Importance
importances = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y).feature_importances_
sorted_idx = np.argsort(importances)
axes[1, 0].barh([feature_cols[i] for i in sorted_idx], importances[sorted_idx], color="steelblue")
axes[1, 0].set_title("Feature Importance (Random Forest)")

# Vorher/Nachher
axes[1, 1].bar(["3 einfache\nFeatures", f"{X_eng_with_dummies.shape[1]} engineered\nFeatures"],
               [score_simple, score_eng], color=["salmon", "steelblue"])
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_title("Vorher vs Nachher")
axes[1, 1].set_ylim(0, 1)
for i, v in enumerate([score_simple, score_eng]):
    axes[1, 1].text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("feature_engineering.png", dpi=150)
print("\n[OK] Plot gespeichert als 'feature_engineering.png'")

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. INTERAKTIONS-FEATURES
   Erstelle 3 neue Interaktions-Features (z.B. wohnflaeche * zentral,
   zimmer * balkon). Verbessert sich die Accuracy?

2. ANDERE SELECTION-METHODE
   Nutze SelectKBest mit f_classif statt RFE.
   Code: selector = SelectKBest(f_classif, k=5).fit(X, y)
   Welche 5 Features waehlt diese Methode?

3. TARGET ENCODING
   Statt One-Hot fuer 'stadt': berechne den Durchschnittspreis
   pro Stadt und nutze diesen als Feature.
   Code: data['stadt_avg_preis'] = data.groupby('stadt')['preis'].transform('mean')

4. PCA-EXPERIMENT
   Trainiere ein Modell nur mit den ersten 3 PCA-Komponenten.
   Wie gut ist es im Vergleich zu allen Features?
""")
