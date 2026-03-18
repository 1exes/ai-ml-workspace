import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
02 - Classification: Kategorien vorhersagen
=============================================
Classification = ein Modell lernt, Dinge in Klassen einzuteilen.
Beispiel: Spam/Kein Spam, Katze/Hund, Sentiment positiv/negativ.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 1. Daten generieren
# ============================================================

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_classes=3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Klassen: {np.unique(y)} (3 Kategorien)")
print(f"Verteilung: {dict(zip(*np.unique(y_train, return_counts=True)))}")

# ============================================================
# 2. Modelle trainieren & vergleichen
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}: {acc:.1%} Accuracy")

# ============================================================
# 3. Detaillierte Auswertung des besten Modells
# ============================================================

best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)

print("\n" + "=" * 50)
print("Random Forest - Detaillierte Auswertung")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["Klasse A", "Klasse B", "Klasse C"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(["Klasse A", "Klasse B", "Klasse C"])
ax.set_yticklabels(["Klasse A", "Klasse B", "Klasse C"])
ax.set_xlabel("Vorhergesagt")
ax.set_ylabel("Tatsaechlich")
ax.set_title("Confusion Matrix")
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
plt.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

print("\n* Precision = 'Wie oft war eine positive Vorhersage korrekt?'")
print("  Recall = 'Wie viele tatsaechlich positive wurden gefunden?'")
print("  F1 = Kombination aus beiden (harmonisches Mittel)")

# ============================================================
# UEBUNGEN
# ============================================================
#
# Aufgabe 1: Aendere den Datensatz auf 2 Klassen (n_classes=2) und
#   nur 5 Features. Trainiere alle 3 Modelle erneut. Welches gewinnt
#   jetzt? Erstelle eine Confusion Matrix fuer das beste Modell.
#
# Aufgabe 2: Fuege ein weiteres Modell hinzu: KNeighborsClassifier
#   (aus sklearn.neighbors). Teste verschiedene Werte fuer n_neighbors
#   (3, 5, 10, 20) und plotte die Accuracy als Balkendiagramm.
#   Speichere als "knn_vergleich.png".
#
# Aufgabe 3: Berechne die ROC-AUC Kurve (sklearn.metrics.roc_curve,
#   roc_auc_score) fuer ein 2-Klassen-Problem. Verwende LogisticRegression
#   mit predict_proba(). Plotte die ROC-Kurve und speichere als "roc.png".
#   Was sagt die Flaeche unter der Kurve (AUC) aus?
