import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Model Evaluation & Vergleich
==============================
Nicht nur Accuracy zaehlt - hier lernst du ALLE wichtigen Metriken.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)
import time

# ============================================================
# 1. Datensatz (absichtlich unbalanciert - realistisch!)
# ============================================================
print("=" * 60)
print("1. DATENSATZ (unbalanciert - wie in der Realitaet)")
print("=" * 60)

X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10,
    n_classes=2, weights=[0.7, 0.3], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Train: {X_train.shape[0]} Samples")
print(f"Test:  {X_test.shape[0]} Samples")
print(f"Klassenverteilung: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  -> Klasse 0: {(y_train == 0).mean():.0%}, Klasse 1: {(y_train == 1).mean():.0%}")

# ============================================================
# 2. Alle Modelle trainieren
# ============================================================
print("\n" + "=" * 60)
print("2. 7 MODELLE IM VERGLEICH")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME"),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
        "train_time": train_time,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

# ============================================================
# 3. Metriken-Tabelle
# ============================================================
print("\n" + "=" * 60)
print("3. METRIKEN-TABELLE")
print("=" * 60)

header = f"{'Modell':<25s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s} {'LogL':>6s} {'Zeit':>6s}"
print(f"\n{header}")
print("-" * len(header))
for name, m in sorted(results.items(), key=lambda x: -x[1]["f1"]):
    print(f"{name:<25s} {m['accuracy']:>5.1%} {m['precision']:>5.1%} {m['recall']:>5.1%} "
          f"{m['f1']:>5.1%} {m['auc_roc']:>5.3f} {m['log_loss']:>5.3f} {m['train_time']:>5.2f}s")

best = max(results.items(), key=lambda x: x[1]["f1"])
print(f"\n  -> Bestes Modell (nach F1): {best[0]} ({best[1]['f1']:.1%})")

# ============================================================
# 4. Metriken erklaert
# ============================================================
print("\n" + "=" * 60)
print("4. WAS BEDEUTEN DIE METRIKEN?")
print("=" * 60)
print("""
  Accuracy:  Wie oft war das Modell richtig? (ACHTUNG: bei unbalancierten
             Daten truegerisch! 70% Accuracy geht auch mit "immer Klasse 0")

  Precision: Von allen als positiv vorhergesagten - wie viele waren wirklich positiv?
             Wichtig wenn: False Positives teuer sind (Spam-Filter, Betrugs-Erkennung)

  Recall:    Von allen tatsaechlich positiven - wie viele wurden gefunden?
             Wichtig wenn: False Negatives gefaehrlich sind (Krebs-Diagnose, Sicherheit)

  F1-Score:  Harmonisches Mittel aus Precision und Recall.
             Guter Gesamtindikator, besonders bei unbalancierten Daten.

  AUC-ROC:   Wie gut trennt das Modell die Klassen? 0.5 = Zufall, 1.0 = perfekt.
             Unabhaengig vom Schwellenwert - robust!

  Log Loss:  Wie sicher ist das Modell bei seinen Vorhersagen?
             Bestraft falsche UND unsichere Vorhersagen. Niedriger = besser.
""")

# ============================================================
# 5. Detaillierte Analyse des besten Modells
# ============================================================
print("=" * 60)
print("5. DETAILLIERTE ANALYSE: " + best[0])
print("=" * 60)

cm = confusion_matrix(y_test, best[1]["y_pred"])
print(f"\nConfusion Matrix:")
print(f"                  Vorhergesagt")
print(f"                  Neg    Pos")
print(f"  Tatsaechlich Neg  {cm[0, 0]:>4}   {cm[0, 1]:>4}  (FP: {cm[0, 1]} False Positives)")
print(f"  Tatsaechlich Pos  {cm[1, 0]:>4}   {cm[1, 1]:>4}  (FN: {cm[1, 0]} False Negatives)")
print(f"\n{classification_report(y_test, best[1]['y_pred'], target_names=['Negativ', 'Positiv'])}")

# ============================================================
# 6. Visualisierungen
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Evaluation Dashboard", fontsize=16)

# ROC Curves
for name, m in results.items():
    fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
    axes[0, 0].plot(fpr, tpr, label=f"{name} ({m['auc_roc']:.3f})")
axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].set_title("ROC Curves")
axes[0, 0].legend(fontsize=7)

# Precision-Recall Curves
for name, m in results.items():
    prec, rec, _ = precision_recall_curve(y_test, m["y_prob"])
    axes[0, 1].plot(rec, prec, label=name)
axes[0, 1].set_xlabel("Recall")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].set_title("Precision-Recall Curves")
axes[0, 1].legend(fontsize=7)

# Metriken-Vergleich (Balkendiagramm)
model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.2
metrics = ["accuracy", "precision", "recall", "f1"]
colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
for i, (metric, color) in enumerate(zip(metrics, colors)):
    values = [results[n][metric] for n in model_names]
    axes[1, 0].bar(x + i * width, values, width, label=metric.capitalize(), color=color)
axes[1, 0].set_xticks(x + width * 1.5)
axes[1, 0].set_xticklabels([n[:12] for n in model_names], rotation=45, ha="right", fontsize=7)
axes[1, 0].set_ylabel("Score")
axes[1, 0].set_title("Metriken pro Modell")
axes[1, 0].legend()
axes[1, 0].set_ylim(0.5, 1.0)

# Lernkurve bestes Modell
best_model_class = models[best[0]].__class__
if best[0] == "Gradient Boosting":
    lc_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
elif best[0] == "Random Forest":
    lc_model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    lc_model = LogisticRegression(max_iter=1000)
train_sizes, train_scores, val_scores = learning_curve(
    lc_model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
)
axes[1, 1].plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
axes[1, 1].plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
axes[1, 1].fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
axes[1, 1].set_xlabel("Training Samples")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_title(f"Lernkurve: {best[0]}")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
print("\n[OK] Plot gespeichert als 'model_evaluation.png'")

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. SCHWELLENWERT ANPASSEN
   Statt 0.5 als Schwelle: Teste 0.3 und 0.7.
   Code: y_pred_custom = (y_prob > 0.3).astype(int)
   Wie aendern sich Precision und Recall?

2. BALANCED ACCURACY
   Nutze balanced_accuracy_score fuer unbalancierte Daten.
   from sklearn.metrics import balanced_accuracy_score
   Welches Modell gewinnt jetzt?

3. NEUES MODELL: XGBoost
   pip install xgboost
   from xgboost import XGBClassifier
   Fuege es zum Vergleich hinzu. Ist es besser?

4. MULTI-CLASS
   Aendere n_classes=5 in make_classification.
   Welche Metriken aendern sich? (Tipp: average='macro' fuer F1)
""")
