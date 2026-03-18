import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Automatische Daten-Pipeline
=============================
Ein Agent der automatisch: Daten laden -> EDA -> Cleaning -> Training -> Report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. Synthetischen Datensatz generieren
# ============================================================
print("=" * 60)
print("AUTOMATISCHE DATEN-PIPELINE")
print("=" * 60)

np.random.seed(42)
n = 1000

print("\n[1/6] Daten generieren...")

data = pd.DataFrame({
    "alter": np.random.normal(40, 15, n).clip(18, 80).astype(int),
    "einkommen": np.random.lognormal(10.5, 0.5, n).astype(int),
    "bildung": np.random.choice(["Hauptschule", "Realschule", "Abitur", "Studium", "Promotion"], n,
                                 p=[0.1, 0.2, 0.3, 0.3, 0.1]),
    "berufserfahrung_jahre": np.random.exponential(8, n).clip(0, 40).astype(int),
    "branche": np.random.choice(["IT", "Gesundheit", "Finanzen", "Bildung", "Handwerk", "Handel"], n),
    "wochenstunden": np.random.normal(40, 8, n).clip(10, 60).round(1),
    "pendelzeit_min": np.random.exponential(25, n).clip(5, 120).astype(int),
    "zufriedenheit": np.random.uniform(1, 10, n).round(1),
})

# Missing Values einfuegen (realistisch!)
mask = np.random.random(n) < 0.05
data.loc[mask, "einkommen"] = np.nan
mask = np.random.random(n) < 0.03
data.loc[mask, "wochenstunden"] = np.nan
mask = np.random.random(n) < 0.02
data.loc[mask, "pendelzeit_min"] = np.nan

# Outliers einfuegen
data.loc[0, "einkommen"] = 5000000
data.loc[1, "alter"] = 150
data.loc[2, "wochenstunden"] = 200

# Target: Kuendigung (abhaengig von Zufriedenheit + Pendelzeit + Einkommen)
kuendigungsrisiko = (
    (10 - data["zufriedenheit"]) * 0.3
    + data["pendelzeit_min"].fillna(30) * 0.01
    - np.log1p(data["einkommen"].fillna(40000)) * 0.5
    + np.random.normal(0, 1, n)
)
data["kuendigt"] = (kuendigungsrisiko > np.percentile(kuendigungsrisiko, 70)).astype(int)

print(f"  Shape: {data.shape}")
print(f"  Spalten: {list(data.columns)}")
print(f"  Target: kuendigt (0=bleibt, 1=kuendigt)")

# ============================================================
# 2. Automatische EDA (Exploratory Data Analysis)
# ============================================================
print(f"\n[2/6] Explorative Datenanalyse...")
print(f"\n{'='*60}")
print(f"  DATENTYPEN")
print(f"{'='*60}")
for col in data.columns:
    dtype = data[col].dtype
    nunique = data[col].nunique()
    null_pct = data[col].isnull().mean()
    print(f"  {col:<25s} {str(dtype):<10s} {nunique:>6d} unique  {null_pct:>5.1%} missing")

print(f"\n{'='*60}")
print(f"  STATISTIKEN (numerisch)")
print(f"{'='*60}")
num_cols = data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    s = data[col]
    print(f"  {col:<25s} Mean={s.mean():>10.1f}  Std={s.std():>10.1f}  "
          f"Min={s.min():>10.1f}  Max={s.max():>10.1f}")

print(f"\n{'='*60}")
print(f"  MISSING VALUES")
print(f"{'='*60}")
missing = data.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    for col, count in missing.items():
        pct = count / len(data)
        bar = "#" * int(pct * 100)
        print(f"  {col:<25s} {count:>4d} ({pct:.1%}) {bar}")
else:
    print("  Keine fehlenden Werte!")

print(f"\n{'='*60}")
print(f"  OUTLIERS (IQR Methode)")
print(f"{'='*60}")
outlier_report = {}
for col in num_cols:
    if col == "kuendigt":
        continue
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((data[col] < lower) | (data[col] > upper)).sum()
    if outliers > 0:
        outlier_report[col] = outliers
        print(f"  {col:<25s} {outliers:>4d} Outliers (Range: {lower:.0f} - {upper:.0f})")

print(f"\n{'='*60}")
print(f"  TARGET-VERTEILUNG")
print(f"{'='*60}")
for val, count in data["kuendigt"].value_counts().items():
    pct = count / len(data)
    bar = "#" * int(pct * 50)
    label = "Bleibt" if val == 0 else "Kuendigt"
    print(f"  {label} ({val}): {count:>5d} ({pct:.1%}) {bar}")

# ============================================================
# 3. Automatisches Cleaning
# ============================================================
print(f"\n[3/6] Daten bereinigen...")

df = data.copy()
actions = []

# Missing Values
for col in df.columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            actions.append(f"  [OK] {col}: {null_count} NaN -> Median ({median_val:.0f})")
        else:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            actions.append(f"  [OK] {col}: {null_count} NaN -> Modus ({mode_val})")

# Outliers clippen
for col, count in outlier_report.items():
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df[col] = df[col].clip(lower, upper)
    actions.append(f"  [OK] {col}: {count} Outliers geclippt auf [{lower:.0f}, {upper:.0f}]")

# Encoding
cat_cols = df.select_dtypes(include=["object"]).columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    actions.append(f"  [OK] {col}: Label Encoded ({len(le.classes_)} Klassen)")

for a in actions:
    print(a)

# ============================================================
# 4. Feature-Vorbereitung
# ============================================================
print(f"\n[4/6] Features vorbereiten...")

feature_cols = [c for c in df.columns if c != "kuendigt"]
X = df[feature_cols].values
y = df["kuendigt"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Features: {len(feature_cols)} ({', '.join(feature_cols[:5])}...)")
print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================
# 5. Automatisches Model Selection & Training
# ============================================================
print(f"\n[5/6] Modelle trainieren und vergleichen...")

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

model_results = {}
for name, model in candidates.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    model.fit(X_train_scaled, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    model_results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_acc": test_acc,
        "model": model,
    }
    bar = "#" * int(cv_scores.mean() * 40)
    print(f"  {name:<25s} CV: {cv_scores.mean():.3f}+/-{cv_scores.std():.3f}  Test: {test_acc:.3f}  {bar}")

best_name = max(model_results, key=lambda x: model_results[x]["cv_mean"])
best = model_results[best_name]
print(f"\n  -> Bestes Modell: {best_name} (CV: {best['cv_mean']:.3f})")

# Feature Importance
if hasattr(best["model"], "feature_importances_"):
    importances = best["model"].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  Top-5 wichtigste Features:")
    for i in sorted_idx[:5]:
        bar = "#" * int(importances[i] * 50)
        print(f"    {feature_cols[i]:<25s} {importances[i]:.3f} {bar}")

# ============================================================
# 6. Report generieren
# ============================================================
print(f"\n[6/6] Report generieren...")

print(f"""
{'='*60}
  PIPELINE REPORT
{'='*60}

  DATENSATZ
  - Samples:        {len(data):,}
  - Features:       {len(feature_cols)}
  - Target:         kuendigt (binaer)
  - Klassenbalance: {(y==0).mean():.0%} / {(y==1).mean():.0%}

  CLEANING
  - Missing Values: {data.isnull().sum().sum()} gefixt (Median/Modus)
  - Outliers:       {sum(outlier_report.values())} geclippt (IQR)
  - Encoding:       {len(cat_cols)} kategorische Spalten -> Label Encoded

  MODELL: {best_name}
  - Cross-Val Accuracy: {best['cv_mean']:.1%} (+/- {best['cv_std']:.1%})
  - Test Accuracy:      {best['test_acc']:.1%}

  EMPFEHLUNG
  - Modell ist {"gut" if best['cv_mean'] > 0.75 else "akzeptabel" if best['cv_mean'] > 0.6 else "verbesserungswuerdig"}
  - {"Daten sind ausreichend" if len(data) > 500 else "Mehr Daten wuerden helfen"}
  - {"Klassen sind unbalanciert - SMOTE oder class_weight erwaegen" if abs((y==0).mean() - 0.5) > 0.15 else "Klassen sind ausgewogen"}
{'='*60}
""")

# ============================================================
# 7. Visualisierung
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Daten-Pipeline Report", fontsize=16)

# Feature Importance
if hasattr(best["model"], "feature_importances_"):
    imp = best["model"].feature_importances_
    idx = np.argsort(imp)
    axes[0, 0].barh([feature_cols[i] for i in idx], imp[idx], color="steelblue")
    axes[0, 0].set_title(f"Feature Importance ({best_name})")

# Model Comparison
names = list(model_results.keys())
cv_means = [model_results[n]["cv_mean"] for n in names]
test_accs = [model_results[n]["test_acc"] for n in names]
x = np.arange(len(names))
axes[0, 1].bar(x - 0.15, cv_means, 0.3, label="CV Score", color="steelblue")
axes[0, 1].bar(x + 0.15, test_accs, 0.3, label="Test Score", color="coral")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([n[:12] for n in names], rotation=45, ha="right", fontsize=8)
axes[0, 1].set_title("Modell-Vergleich")
axes[0, 1].legend()
axes[0, 1].set_ylim(0.5, 1.0)

# Target Distribution
axes[1, 0].bar(["Bleibt (0)", "Kuendigt (1)"], [(y==0).sum(), (y==1).sum()], color=["steelblue", "coral"])
axes[1, 0].set_title("Target-Verteilung")

# Correlation
num_data = df[feature_cols].select_dtypes(include=[np.number])
corr = num_data.corr()
im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr.columns)))
axes[1, 1].set_yticks(range(len(corr.columns)))
axes[1, 1].set_xticklabels([c[:6] for c in corr.columns], rotation=45, ha="right", fontsize=7)
axes[1, 1].set_yticklabels([c[:6] for c in corr.columns], fontsize=7)
axes[1, 1].set_title("Feature-Korrelation")
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("pipeline_report.png", dpi=150)
print("[OK] Plot gespeichert als 'pipeline_report.png'")

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. CSV EINLESEN
   Ersetze den synthetischen Datensatz durch eine echte CSV:
     data = pd.read_csv("dein_datensatz.csv")
   Die Pipeline sollte trotzdem funktionieren!

2. SMOTE FUER UNBALANCIERTE DATEN
   pip install imbalanced-learn
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

3. AUTOMATISCHES FEATURE ENGINEERING
   Finde automatisch Interaktions-Features:
   - Einkommen / Wochenstunden = Stundenlohn
   - Berufserfahrung / Alter = Karriere-Ratio

4. PIPELINE ALS KLASSE
   Refactore den Code in eine Klasse AutoMLPipeline mit Methoden:
   load(), eda(), clean(), train(), report()
""")
