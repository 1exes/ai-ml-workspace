#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
  DATEN-AUGMENTATION & PREPROCESSING
=============================================================================
  Techniken zur kuenstlichen Datenanreicherung fuer bessere ML-Modelle.

  Themen:
    1. Bild-Augmentation mit torchvision.transforms
    2. Text-Augmentation (Synonym-Ersetzung, Random Insertion, Back-Translation)
    3. Tabellarische Daten-Augmentation (SMOTE-Konzept, Rauschen)
    4. Visualisierung: Vorher/Nachher-Vergleich

  Ausgabe: augmentation_demo.png
=============================================================================
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("  DATEN-AUGMENTATION & PREPROCESSING")
print("=" * 70)

# ============================================================================
# TEIL 1: Synthetische Beispielbilder erzeugen
# ============================================================================
print("\n[1] Synthetische Beispielbilder erzeugen...")

def erstelle_beispielbild(typ="kreis"):
    """Erzeugt ein 128x128 RGB-Bild als numpy-Array."""
    img = np.ones((128, 128, 3), dtype=np.uint8) * 240  # Heller Hintergrund

    if typ == "kreis":
        # Roter Kreis mit blauem Rand
        y, x = np.ogrid[-64:64, -64:64]
        maske = x**2 + y**2 <= 40**2
        rand = (x**2 + y**2 <= 45**2) & ~maske
        img[maske] = [220, 50, 50]
        img[rand] = [30, 30, 180]

    elif typ == "dreieck":
        # Gruenes Dreieck
        for y in range(30, 100):
            breite = int((y - 30) * 0.7)
            x_start = max(0, 64 - breite)
            x_end = min(128, 64 + breite)
            img[y, x_start:x_end] = [30, 180, 50]

    elif typ == "rechteck":
        # Blaues Rechteck mit Farbverlauf
        for y in range(25, 105):
            for x in range(20, 110):
                faktor = (x - 20) / 90.0
                img[y, x] = [int(30 + 100 * faktor), int(50 + 50 * faktor), 200]

    elif typ == "stern":
        # Gelber Stern (vereinfacht als Raute + Kreuz)
        for y in range(20, 108):
            for x in range(20, 108):
                cy, cx = y - 64, x - 64
                # Raute
                if abs(cy) + abs(cx) <= 35:
                    img[y, x] = [230, 200, 30]
                # Kreuz-Erweiterung
                if (abs(cy) <= 10 and abs(cx) <= 40) or (abs(cy) <= 40 and abs(cx) <= 10):
                    img[y, x] = [230, 200, 30]

    return img

# Vier verschiedene Bilder erzeugen
typen = ["kreis", "dreieck", "rechteck", "stern"]
bilder = {t: erstelle_beispielbild(t) for t in typen}
print(f"  -> {len(bilder)} Beispielbilder erzeugt (128x128 RGB)")

# ============================================================================
# TEIL 2: Bild-Augmentation mit torchvision.transforms
# ============================================================================
print("\n[2] Bild-Augmentation mit torchvision.transforms...")

# Verschiedene Transformations-Pipelines definieren
transformationen = {
    "Original": T.Compose([T.ToTensor()]),

    "Rotation 45grad": T.Compose([
        T.RandomRotation(degrees=(45, 45)),
        T.ToTensor()
    ]),

    "Horizontal Flip": T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor()
    ]),

    "Color Jitter": T.Compose([
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
        T.ToTensor()
    ]),

    "Random Crop": T.Compose([
        T.RandomResizedCrop(size=128, scale=(0.5, 0.8)),
        T.ToTensor()
    ]),

    "Perspektive": T.Compose([
        T.RandomPerspective(distortion_scale=0.4, p=1.0),
        T.ToTensor()
    ]),

    "Gauss Blur": T.Compose([
        T.GaussianBlur(kernel_size=7, sigma=(2.0, 2.0)),
        T.ToTensor()
    ]),

    "Kombination": T.Compose([
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.RandomHorizontalFlip(p=0.5),
        T.GaussianBlur(kernel_size=3),
        T.ToTensor()
    ]),
}

print(f"  -> {len(transformationen)} Transformationen definiert:")
for name in transformationen:
    print(f"     * {name}")

# Augmentation auf Kreis-Bild anwenden
print("\n  Augmentation auf Beispielbild anwenden...")
pil_bild = Image.fromarray(bilder["kreis"])
augmentierte_bilder = {}
for name, transform in transformationen.items():
    augmentierte_bilder[name] = transform(pil_bild)
print(f"  -> {len(augmentierte_bilder)} augmentierte Versionen erzeugt")

# ============================================================================
# TEIL 3: Text-Augmentation
# ============================================================================
print("\n[3] Text-Augmentation Techniken...")

def synonym_ersetzung(text, n=2):
    """Ersetzt zufaellige Woerter durch einfache Synonyme (Demo-Version)."""
    # Einfaches Synonym-Woerterbuch fuer Demonstration
    synonyme = {
        "gut": ["prima", "toll", "super", "fein", "klasse"],
        "schlecht": ["mies", "uebel", "miserabel", "mangelhaft"],
        "gross": ["riesig", "gewaltig", "enorm", "massiv"],
        "klein": ["winzig", "gering", "minimal", "zierlich"],
        "schnell": ["rasch", "flink", "zuegig", "hurtig"],
        "langsam": ["traege", "gemuetlich", "bedaechtig"],
        "Haus": ["Gebaeude", "Wohnung", "Anwesen", "Domizil"],
        "Auto": ["Wagen", "Fahrzeug", "PKW", "Automobil"],
        "lernen": ["studieren", "pauken", "bueffeln"],
        "Modell": ["System", "Algorithmus", "Verfahren"],
    }
    woerter = text.split()
    ersetzt = 0
    for i in range(len(woerter)):
        if woerter[i] in synonyme and ersetzt < n:
            woerter[i] = np.random.choice(synonyme[woerter[i]])
            ersetzt += 1
    return " ".join(woerter)

def zufalls_einfuegung(text, n=1):
    """Fuegt zufaellige Woerter an zufaelligen Positionen ein."""
    fuell_woerter = ["wirklich", "tatsaechlich", "offensichtlich", "natuerlich", "quasi"]
    woerter = text.split()
    for _ in range(n):
        pos = np.random.randint(0, len(woerter) + 1)
        wort = np.random.choice(fuell_woerter)
        woerter.insert(pos, wort)
    return " ".join(woerter)

def zufalls_tausch(text, n=1):
    """Tauscht die Position von n zufaelligen Wortpaaren."""
    woerter = text.split()
    for _ in range(n):
        if len(woerter) >= 2:
            i, j = np.random.choice(len(woerter), 2, replace=False)
            woerter[i], woerter[j] = woerter[j], woerter[i]
    return " ".join(woerter)

def zufalls_loeschung(text, p=0.1):
    """Loescht jedes Wort mit Wahrscheinlichkeit p."""
    woerter = text.split()
    if len(woerter) <= 1:
        return text
    ergebnis = [w for w in woerter if np.random.random() > p]
    return " ".join(ergebnis) if ergebnis else woerter[0]

# Demonstration
beispiel_texte = [
    "Das Modell hat eine gut Leistung auf dem Testdatensatz gezeigt",
    "Das Auto ist schnell und das Haus ist gross",
    "Neuronale Netze lernen komplexe Muster aus Daten",
]

print("\n  Text-Augmentation Beispiele:")
print("  " + "-" * 60)
for text in beispiel_texte:
    print(f"\n  Original:     {text}")
    print(f"  Synonym:      {synonym_ersetzung(text, n=2)}")
    print(f"  Einfuegung:   {zufalls_einfuegung(text, n=1)}")
    print(f"  Wort-Tausch:  {zufalls_tausch(text, n=1)}")
    print(f"  Loeschung:    {zufalls_loeschung(text, p=0.2)}")

# Back-Translation Konzept erklaeren
print("\n  [INFO] Back-Translation Konzept:")
print("  Deutsch -> Englisch -> Deutsch (via Uebersetzungsmodell)")
print("  Beispiel: 'Der Hund rennt schnell'")
print("        ->  'The dog runs fast'")
print("        ->  'Der Hund laeuft rasch'")
print("  -> Semantisch gleich, aber anders formuliert!")

# ============================================================================
# TEIL 4: Tabellarische Daten-Augmentation
# ============================================================================
print("\n[4] Tabellarische Daten-Augmentation...")

# Synthetischen Datensatz erstellen (Kreditkarten-Betrug Szenario)
n_normal = 200
n_betrug = 15  # Stark unbalanciert!

# Normale Transaktionen
X_normal = np.column_stack([
    np.random.normal(50, 20, n_normal),      # Betrag
    np.random.normal(5, 2, n_normal),         # Haeufigkeit
    np.random.uniform(0, 1, n_normal),        # Auslands-Anteil
    np.random.normal(100, 30, n_normal),       # Konto-Alter (Tage)
])
y_normal = np.zeros(n_normal)

# Betrugs-Transaktionen (andere Verteilung)
X_betrug = np.column_stack([
    np.random.normal(500, 200, n_betrug),     # Hohe Betraege
    np.random.normal(15, 5, n_betrug),        # Hohe Frequenz
    np.random.uniform(0.5, 1, n_betrug),      # Viel Ausland
    np.random.normal(10, 5, n_betrug),         # Junge Konten
])
y_betrug = np.ones(n_betrug)

X = np.vstack([X_normal, X_betrug])
y = np.concatenate([y_normal, y_betrug])

print(f"  Originaldaten: {n_normal} normal, {n_betrug} Betrug")
print(f"  -> Verhaeltnis: {n_betrug/(n_normal+n_betrug)*100:.1f}% Betrug (stark unbalanciert!)")

# SMOTE-Konzept (vereinfachte Implementation)
def einfaches_smote(X_minderheit, n_synthetisch, k=5):
    """
    Vereinfachte SMOTE-Implementation.
    Erzeugt synthetische Samples durch Interpolation zwischen
    naechsten Nachbarn der Minderheitsklasse.
    """
    n_samples = len(X_minderheit)
    synthetisch = []

    for _ in range(n_synthetisch):
        # Zufaelliges Sample waehlen
        idx = np.random.randint(0, n_samples)
        sample = X_minderheit[idx]

        # Naechsten Nachbarn finden (vereinfacht: zufaellig aus Minderheit)
        nachbar_idx = idx
        while nachbar_idx == idx:
            nachbar_idx = np.random.randint(0, n_samples)
        nachbar = X_minderheit[nachbar_idx]

        # Interpolation: Punkt auf der Verbindungslinie
        alpha = np.random.random()
        neues_sample = sample + alpha * (nachbar - sample)
        synthetisch.append(neues_sample)

    return np.array(synthetisch)

# SMOTE anwenden
n_synthetisch = n_normal - n_betrug  # Ausgleichen
X_synthetisch = einfaches_smote(X_betrug, n_synthetisch)
y_synthetisch = np.ones(n_synthetisch)

X_balanced = np.vstack([X, X_synthetisch])
y_balanced = np.concatenate([y, y_synthetisch])

n_betrug_neu = int(y_balanced.sum())
n_normal_neu = len(y_balanced) - n_betrug_neu
print(f"  Nach SMOTE: {n_normal_neu} normal, {n_betrug_neu} Betrug")
print(f"  -> Verhaeltnis: {n_betrug_neu/len(y_balanced)*100:.1f}% Betrug (ausbalanciert!)")

# Rauschen-Injektion
def rauschen_hinzufuegen(X, sigma=0.05):
    """Fuegt Gauss-Rauschen hinzu (relativ zur Standardabweichung)."""
    rauschen = np.random.normal(0, sigma, X.shape) * X.std(axis=0)
    return X + rauschen

X_verrauscht = rauschen_hinzufuegen(X, sigma=0.1)
print(f"\n  Rauschen-Injektion: sigma=0.1 * std pro Feature")
print(f"  -> Originale Standardabweichungen: {X.std(axis=0).round(2)}")
print(f"  -> Mittlere Abweichung:            {np.abs(X_verrauscht - X).mean(axis=0).round(3)}")

# ============================================================================
# TEIL 5: Auswirkung auf Modell-Performance
# ============================================================================
print("\n[5] Auswirkung auf Modell-Performance...")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Ohne Augmentation
modell_original = LogisticRegression(max_iter=1000, random_state=42)
scores_original = cross_val_score(modell_original, X, y, cv=5, scoring='f1')

# Mit SMOTE-Augmentation
modell_smote = LogisticRegression(max_iter=1000, random_state=42)
scores_smote = cross_val_score(modell_smote, X_balanced, y_balanced, cv=5, scoring='f1')

print(f"  F1-Score OHNE Augmentation: {scores_original.mean():.3f} (+/- {scores_original.std():.3f})")
print(f"  F1-Score MIT  SMOTE:        {scores_smote.mean():.3f} (+/- {scores_smote.std():.3f})")
print(f"  -> Verbesserung: {(scores_smote.mean() - scores_original.mean())*100:.1f} Prozentpunkte")

# ============================================================================
# TEIL 6: Visualisierung
# ============================================================================
print("\n[6] Visualisierung erstellen...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Daten-Augmentation: Techniken und Ergebnisse', fontsize=18, fontweight='bold', y=0.98)

# --- Zeile 1: Bild-Augmentation (8 Spalten) ---
for i, (name, tensor) in enumerate(augmentierte_bilder.items()):
    ax = fig.add_subplot(4, 8, i + 1)
    img_np = tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    ax.imshow(img_np)
    ax.set_title(name, fontsize=8, fontweight='bold')
    ax.axis('off')

# --- Zeile 1 (rechts): Augmentation auf andere Formen ---
for i, typ in enumerate(typen):
    ax = fig.add_subplot(4, 8, 5 + i)
    pil_img = Image.fromarray(bilder[typ])
    kombi_transform = T.Compose([
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        T.RandomPerspective(distortion_scale=0.3, p=1.0),
        T.ToTensor()
    ])
    aug_tensor = kombi_transform(pil_img)
    img_np = aug_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    ax.imshow(img_np)
    ax.set_title(f'{typ.capitalize()} (Aug)', fontsize=8)
    ax.axis('off')

# --- Zeile 2: SMOTE-Visualisierung ---
ax1 = fig.add_subplot(4, 3, 4)
ax1.scatter(X_normal[:, 0], X_normal[:, 1], c='steelblue', alpha=0.5, s=20, label=f'Normal ({n_normal})')
ax1.scatter(X_betrug[:, 0], X_betrug[:, 1], c='crimson', alpha=0.8, s=40, marker='x', label=f'Betrug ({n_betrug})')
ax1.set_xlabel('Betrag', fontsize=9)
ax1.set_ylabel('Haeufigkeit', fontsize=9)
ax1.set_title('VOR Augmentation\n(stark unbalanciert)', fontweight='bold', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(4, 3, 5)
ax2.scatter(X_normal[:, 0], X_normal[:, 1], c='steelblue', alpha=0.5, s=20, label=f'Normal ({n_normal})')
ax2.scatter(X_betrug[:, 0], X_betrug[:, 1], c='crimson', alpha=0.8, s=40, marker='x', label=f'Betrug orig. ({n_betrug})')
ax2.scatter(X_synthetisch[:, 0], X_synthetisch[:, 1], c='orange', alpha=0.5, s=20, marker='s', label=f'SMOTE ({n_synthetisch})')
ax2.set_xlabel('Betrag', fontsize=9)
ax2.set_ylabel('Haeufigkeit', fontsize=9)
ax2.set_title('NACH SMOTE\n(ausbalanciert)', fontweight='bold', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(4, 3, 6)
kategorien = ['Ohne Aug.', 'Mit SMOTE']
werte = [scores_original.mean(), scores_smote.mean()]
fehler = [scores_original.std(), scores_smote.std()]
farben = ['#e74c3c', '#2ecc71']
bars = ax3.bar(kategorien, werte, yerr=fehler, color=farben, capsize=8, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, werte):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f'{val:.3f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)
ax3.set_ylabel('F1-Score', fontsize=10)
ax3.set_title('Modell-Performance\n(Logistische Regression)', fontweight='bold', fontsize=10)
ax3.set_ylim(0, 1.15)
ax3.grid(True, alpha=0.3, axis='y')

# --- Zeile 3: Rauschen-Visualisierung ---
ax4 = fig.add_subplot(4, 3, 7)
feature_namen = ['Betrag', 'Haeufigkeit', 'Ausland', 'Konto-Alter']
for i, name in enumerate(feature_namen):
    ax4.hist(X[:, i], bins=30, alpha=0.5, label=f'{name} (orig)', density=True)
ax4.set_title('Feature-Verteilungen\n(Original)', fontweight='bold', fontsize=10)
ax4.set_xlabel('Wert', fontsize=9)
ax4.set_ylabel('Dichte', fontsize=9)
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(4, 3, 8)
sigmas = [0.01, 0.05, 0.1, 0.2, 0.5]
feature_idx = 0  # Betrag
for sigma in sigmas:
    verrauscht = rauschen_hinzufuegen(X[:, feature_idx:feature_idx+1], sigma=sigma)
    ax5.hist(verrauscht.flatten(), bins=30, alpha=0.4, label=f'sigma={sigma}', density=True)
ax5.set_title('Rauschen-Staerke\n(Feature: Betrag)', fontweight='bold', fontsize=10)
ax5.set_xlabel('Wert', fontsize=9)
ax5.set_ylabel('Dichte', fontsize=9)
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)

# --- Zeile 3 (rechts): Augmentation-Strategie Uebersicht ---
ax6 = fig.add_subplot(4, 3, 9)
ax6.axis('off')
strategien = [
    ("BILDER", "Rotation, Flip, Crop,\nFarbaenderung, Blur,\nPerspektive, Mixup"),
    ("TEXT", "Synonym-Ersetzung,\nBack-Translation,\nWort-Einfuegung/Tausch"),
    ("TABELLEN", "SMOTE, Rauschen,\nFeature-Kreuzung,\nBootstrap-Sampling"),
]
for i, (titel, beschr) in enumerate(strategien):
    y_pos = 0.85 - i * 0.33
    ax6.text(0.05, y_pos, titel, fontsize=11, fontweight='bold',
             transform=ax6.transAxes, color='navy',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='navy'))
    ax6.text(0.35, y_pos - 0.02, beschr, fontsize=9, transform=ax6.transAxes,
             verticalalignment='top')
ax6.set_title('Augmentation nach Datentyp', fontweight='bold', fontsize=10)

# --- Zeile 4: Augmentierte Bilder-Grid (alle 4 Formen, je Original + 2 Augmentationen) ---
augmentation_pipeline = T.Compose([
    T.RandomRotation(degrees=30),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    T.RandomPerspective(distortion_scale=0.3, p=1.0),
    T.GaussianBlur(kernel_size=3),
    T.ToTensor()
])

col_idx = 0
for typ in typen:
    pil_img = Image.fromarray(bilder[typ])
    # Original
    ax = fig.add_subplot(4, 12, 37 + col_idx)
    ax.imshow(bilder[typ])
    ax.set_title(f'{typ.capitalize()}', fontsize=7)
    ax.axis('off')
    col_idx += 1
    # 2 Augmentationen
    for aug_i in range(2):
        ax = fig.add_subplot(4, 12, 37 + col_idx)
        aug = augmentation_pipeline(pil_img)
        img_np = aug.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(f'Aug {aug_i+1}', fontsize=7)
        ax.axis('off')
        col_idx += 1

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
ausgabe_pfad = "basics/augmentation_demo.png"
plt.savefig(ausgabe_pfad, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  -> Visualisierung gespeichert: {ausgabe_pfad}")

# ============================================================================
# ZUSAMMENFASSUNG
# ============================================================================
print("\n" + "=" * 70)
print("  ZUSAMMENFASSUNG")
print("=" * 70)
print("""
  Daten-Augmentation ist entscheidend fuer gute ML-Modelle:

  [*] BILDER:   torchvision.transforms bietet >20 Transformationen
  [*] TEXT:     Synonym-Ersetzung, Einfuegung, Tausch, Back-Translation
  [*] TABELLEN: SMOTE gleicht Klassen aus, Rauschen erhoehlt Robustheit

  Vorteile:
    -> Mehr Trainingsdaten ohne neues Labeling
    -> Bessere Generalisierung (weniger Overfitting)
    -> Ausgleich von Klassen-Ungleichgewichten
    -> Robustere Modelle

  Faustregel:
    -> Augmentation sollte realistische Variationen erzeugen
    -> Zu starke Augmentation kann die Datenqualitaet verschlechtern
    -> Immer auf Validierungs-Set evaluieren!
""")

# ============================================================================
# UEBUNGEN
# ============================================================================
print("=" * 70)
print("  UEBUNGEN")
print("=" * 70)
print("""
  1. TRANSFORM-KOMBINATOR
     Erstelle 5 verschiedene Augmentation-Pipelines mit unterschiedlicher
     Staerke (leicht, mittel, stark, extrem, destruktiv). Vergleiche die
     Ergebnisse visuell. Ab wann sind die Bilder nicht mehr erkennbar?

     Tipp: T.Compose([T.RandomRotation(90), T.RandomErasing(p=0.5), ...])

  2. TEXT-AUGMENTER KLASSE
     Baue eine TextAugmenter-Klasse, die alle 4 Techniken kombiniert
     und pro Eingabetext N augmentierte Varianten erzeugt. Teste mit
     deutschen Saetzen und messe die semantische Aehnlichkeit der
     Ergebnisse (z.B. mit Sentence-Transformers).

     Bonus: Implementiere echte Synonym-Ersetzung mit WordNet/GermaNet.

  3. SMOTE vs. RANDOM OVERSAMPLING
     Vergleiche drei Strategien auf dem Betrugs-Datensatz:
     a) Kein Oversampling
     b) Einfaches Duplizieren der Minderheitsklasse
     c) SMOTE (synthetische Interpolation)
     Messe jeweils F1, Precision, Recall mit 5-Fold Cross-Validation.

  4. AUGMENTATION-PIPELINE FUER PRODUKTION
     Erstelle eine vollstaendige Augmentation-Pipeline die:
     - Bilder augmentiert UND als PyTorch Dataset bereitstellt
     - On-the-fly augmentiert (nicht vorab gespeichert)
     - Verschiedene Augmentationen pro Epoche anwendet
     Nutze torch.utils.data.Dataset und DataLoader.

  5. RAUSCHEN-STUDIE
     Untersuche systematisch den Einfluss von Rauschen-Staerke (sigma)
     auf die Modell-Performance. Erstelle einen Plot: x=sigma, y=F1-Score.
     Wo liegt das Optimum? Warum verschlechtert zu viel Rauschen?
""")

print("[OK] Script erfolgreich abgeschlossen!")
