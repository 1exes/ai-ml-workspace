import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Hyperparameter Tuning
======================
Modelle haben Stellschrauben (Hyperparameter).
Die richtigen Einstellungen machen den Unterschied.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    learning_curve, validation_curve,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import time

# ============================================================
# 1. Datensatz
# ============================================================
print("=" * 60)
print("1. DATENSATZ")
print("=" * 60)

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=12,
    n_redundant=3, n_classes=2, random_state=42
)
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

# ============================================================
# 2. Warum Hyperparameter wichtig sind
# ============================================================
print("\n" + "=" * 60)
print("2. WARUM HYPERPARAMETER WICHTIG SIND")
print("=" * 60)

print("\n--- Random Forest: Anzahl Baeume ---")
for n_trees in [1, 5, 10, 50, 100, 500]:
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    score = cross_val_score(clf, X, y, cv=5).mean()
    bar = "#" * int(score * 50)
    print(f"  {n_trees:>4} Baeume: {score:.4f} {bar}")

print("\n--- KNN: Anzahl Nachbarn ---")
for k in [1, 3, 5, 10, 20, 50]:
    clf = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(clf, X, y, cv=5).mean()
    bar = "#" * int(score * 50)
    print(f"  k={k:>3}: {score:.4f} {bar}")

# ============================================================
# 3. GridSearch - Alle Kombinationen testen
# ============================================================
print("\n" + "=" * 60)
print("3. GRID SEARCH - Systematisch alle Kombinationen")
print("=" * 60)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
}
total_combos = 3 * 4 * 3
print(f"Parameter-Kombinationen: {total_combos}")
print(f"Mit 5-Fold CV: {total_combos * 5} Trainings-Laeufe")

start = time.time()
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0,
)
grid.fit(X, y)
grid_time = time.time() - start

print(f"\nBeste Parameter: {grid.best_params_}")
print(f"Beste Accuracy:  {grid.best_score_:.4f}")
print(f"Dauer:           {grid_time:.1f}s")

# ============================================================
# 4. RandomSearch - Schneller, oft gleich gut
# ============================================================
print("\n" + "=" * 60)
print("4. RANDOM SEARCH - Zufaellig, aber effizient")
print("=" * 60)

param_distributions = {
    "n_estimators": [10, 50, 100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10, 15, 20, None],
    "min_samples_split": [2, 3, 5, 7, 10, 15, 20],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
}

start = time.time()
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions, n_iter=30, cv=5, scoring="accuracy",
    n_jobs=-1, random_state=42, verbose=0,
)
random_search.fit(X, y)
random_time = time.time() - start

print(f"Getestete Kombinationen: 30 (von tausenden moeglichen)")
print(f"Beste Parameter: {random_search.best_params_}")
print(f"Beste Accuracy:  {random_search.best_score_:.4f}")
print(f"Dauer:           {random_time:.1f}s")

print(f"\n--- Vergleich ---")
print(f"  GridSearch:   {grid.best_score_:.4f} in {grid_time:.1f}s ({total_combos} Kombinationen)")
print(f"  RandomSearch: {random_search.best_score_:.4f} in {random_time:.1f}s (30 Kombinationen)")
print(f"  * RandomSearch ist oft schneller und findet aehnlich gute Werte!")

# ============================================================
# 5. Overfitting vs Underfitting (Lernkurven)
# ============================================================
print("\n" + "=" * 60)
print("5. OVERFITTING vs UNDERFITTING")
print("=" * 60)

models = {
    "Underfitting (max_depth=1)": DecisionTreeClassifier(max_depth=1, random_state=42),
    "Gut (max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Overfitting (max_depth=None)": DecisionTreeClassifier(max_depth=None, random_state=42),
}

for name, model in models.items():
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy", n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    gap = train_mean[-1] - val_mean[-1]
    print(f"\n  {name}:")
    print(f"    Train Accuracy: {train_mean[-1]:.4f}")
    print(f"    Val Accuracy:   {val_mean[-1]:.4f}")
    print(f"    Gap:            {gap:.4f} {'<-- Overfitting!' if gap > 0.05 else '[OK]'}")

# ============================================================
# 6. Cross-Validation erklaert
# ============================================================
print("\n" + "=" * 60)
print("6. CROSS-VALIDATION")
print("=" * 60)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=10, scoring="accuracy")

print(f"\n10-Fold Cross-Validation:")
for i, score in enumerate(scores):
    bar = "#" * int(score * 40)
    print(f"  Fold {i+1:2d}: {score:.4f} {bar}")
print(f"  {'─' * 40}")
print(f"  Mean:   {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"\n  * Standardabweichung zeigt wie stabil das Modell ist.")
print(f"    Kleine Std -> Modell generalisiert gut.")

# ============================================================
# 7. Visualisierung
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Hyperparameter Tuning Uebersicht", fontsize=16)

# Lernkurven
for ax_idx, (name, model) in enumerate(models.items()):
    if ax_idx >= 2:
        break
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
    )
    ax = axes[0, ax_idx]
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
    ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    ax.set_title(name)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_ylim(0.5, 1.05)

# Validation Curve (n_estimators)
param_range = [5, 10, 20, 50, 100, 200]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42), X, y,
    param_name="n_estimators", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1,
)
axes[1, 0].plot(param_range, train_scores.mean(axis=1), "o-", label="Train")
axes[1, 0].plot(param_range, val_scores.mean(axis=1), "o-", label="Validation")
axes[1, 0].set_xlabel("n_estimators")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].set_title("Validation Curve: Random Forest")
axes[1, 0].legend()

# CV Scores
axes[1, 1].bar(range(1, 11), scores, color="steelblue")
axes[1, 1].axhline(y=scores.mean(), color="red", linestyle="--", label=f"Mean: {scores.mean():.3f}")
axes[1, 1].set_xlabel("Fold")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_title("10-Fold Cross-Validation")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("hyperparameter_results.png", dpi=150)
print("\n[OK] Plot gespeichert als 'hyperparameter_results.png'")

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. GRADIENT BOOSTING TUNEN
   Fuehre RandomizedSearchCV mit GradientBoostingClassifier durch.
   Wichtige Params: n_estimators, learning_rate, max_depth, subsample.

2. VALIDATION CURVE FUER max_depth
   Erstelle eine Validation Curve fuer max_depth von 1 bis 30.
   Ab welcher Tiefe beginnt Overfitting?

3. VERSCHACHTELTE CV (Nested Cross-Validation)
   Problem: GridSearchCV + score auf gleichen Daten = optimistisch!
   Loesung: Aeussere CV fuer Evaluation, innere CV fuer Tuning.
   Code-Ansatz:
     from sklearn.model_selection import cross_val_score
     inner_cv = GridSearchCV(model, params, cv=3)
     nested_scores = cross_val_score(inner_cv, X, y, cv=5)
""")
