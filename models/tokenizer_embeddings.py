"""
01 - Tokenizer & Embeddings mit europaeischen KI-Modellen
==========================================================
Wie versteht ein KI-Modell Text?
1. Text -> Tokens (Tokenizer)
2. Tokens -> Vektoren (Embeddings)
3. Vektoren -> Verarbeitung (Transformer)

Modelle:
- Mistral 7B Tokenizer (Frankreich, Mistral AI)
- GBERT (Deutschland, deepset)
- Multilingual MiniLM (mehrsprachig, sentence-transformers)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import AutoTokenizer
import torch

# ============================================================
# 1. Tokenizer-Vergleich: Mistral vs. GBERT
# ============================================================

print("=" * 60)
print("TOKENIZER-VERGLEICH: Mistral (FR) vs. GBERT (DE)")
print("=" * 60)

# Mistral Tokenizer (franzoesisches Modell von Mistral AI, Paris)
print("\nLade Mistral-7B Tokenizer (Mistral AI, Frankreich)...")
tok_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# GBERT Tokenizer (deutsches BERT-Modell von deepset, Berlin)
print("Lade GBERT Tokenizer (deepset, Deutschland)...")
tok_gbert = AutoTokenizer.from_pretrained("deepset/gbert-base")

# Deutsche Beispielsaetze
deutsche_saetze = [
    "Kuenstliche Intelligenz veraendert die Welt!",
    "Der Datenschutz ist in Europa besonders wichtig.",
    "Maschinelles Lernen hilft bei der Wettervorhersage.",
    "Berlin ist die Hauptstadt von Deutschland.",
    "Transformermodelle revolutionieren die Sprachverarbeitung.",
]

print("\n--- Tokenisierung deutscher Saetze ---")
print(f"{'Satz':<50} | {'Mistral':>8} | {'GBERT':>8}")
print("-" * 72)

for satz in deutsche_saetze:
    n_mistral = len(tok_mistral.encode(satz))
    n_gbert = len(tok_gbert.encode(satz))
    # Satz kuerzen fuer die Anzeige
    display = satz[:47] + "..." if len(satz) > 50 else satz
    print(f"{display:<50} | {n_mistral:>8} | {n_gbert:>8}")

# Detaillierte Token-Ansicht fuer einen Satz
beispiel = "Der Datenschutz ist in Europa besonders wichtig."
print(f"\n--- Detailansicht: '{beispiel}' ---")

tokens_mistral = tok_mistral.encode(beispiel)
decoded_mistral = [tok_mistral.decode([t]) for t in tokens_mistral]
print(f"\nMistral Tokens ({len(tokens_mistral)}):")
print(f"  IDs:    {tokens_mistral}")
print(f"  Text:   {decoded_mistral}")

tokens_gbert = tok_gbert.encode(beispiel)
decoded_gbert = [tok_gbert.decode([t]) for t in tokens_gbert]
print(f"\nGBERT Tokens ({len(tokens_gbert)}):")
print(f"  IDs:    {tokens_gbert}")
print(f"  Text:   {decoded_gbert}")

print("\n* GBERT wurde auf deutschen Texten trainiert und tokenisiert")
print("  deutsche Woerter oft effizienter als ein englisches Modell!")

# Vergleich: Englisch vs Deutsch
print("\n--- Englisch vs. Deutsch ---")
vergleiche = [
    ("Hello World", "Hallo Welt"),
    ("Machine Learning", "Maschinelles Lernen"),
    ("Data Protection", "Datenschutz"),
    ("Weather Forecast", "Wettervorhersage"),
]

print(f"{'Englisch':<25} {'Mistral':>4} | {'Deutsch':<25} {'Mistral':>4} {'GBERT':>4}")
print("-" * 72)
for en, de in vergleiche:
    n_en = len(tok_mistral.encode(en))
    n_de_m = len(tok_mistral.encode(de))
    n_de_g = len(tok_gbert.encode(de))
    print(f"{en:<25} {n_en:>4} | {de:<25} {n_de_m:>4} {n_de_g:>4}")


# ============================================================
# 2. Embeddings - Saetze als Vektoren (Multilingual!)
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDINGS: Multilingual MiniLM (sentence-transformers)")
print("=" * 60)

from sentence_transformers import SentenceTransformer

# Multilinguales Modell - versteht Deutsch, Franzoesisch, Englisch, etc.
print("\nLade paraphrase-multilingual-MiniLM-L12-v2...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Deutsche Saetze fuer Aehnlichkeitsvergleich
saetze = [
    "Der Hund spielt im Park",              # Tier/draussen
    "Die Katze jagt eine Maus im Garten",   # Tier/draussen
    "Python ist eine Programmiersprache",    # IT
    "JavaScript wird fuer Webseiten genutzt",# IT
    "Das Haustier rennt draussen herum",     # Tier/draussen
    "Europa investiert stark in KI-Forschung", # KI/Europa
    "Frankreich und Deutschland foerdern kuenstliche Intelligenz", # KI/Europa
]

embeddings = model.encode(saetze)
print(f"\nEmbedding Shape: {embeddings.shape}")
print(f"Jeder Satz = ein Vektor mit {embeddings.shape[1]} Dimensionen")


# ============================================================
# 3. Semantische Aehnlichkeit
# ============================================================

from sentence_transformers.util import cos_sim

print("\n--- Semantische Aehnlichkeit (Top 10 Paare) ---")

# Alle Paare berechnen und sortieren
paare = []
similarities = cos_sim(embeddings, embeddings)
for i in range(len(saetze)):
    for j in range(i + 1, len(saetze)):
        sim = similarities[i][j].item()
        paare.append((sim, i, j))

paare.sort(reverse=True)

for sim, i, j in paare[:10]:
    bar = "#" * int(sim * 30) if sim > 0 else ""
    print(f"  {sim:.3f} [{bar:<30}]")
    print(f"    '{saetze[i]}'")
    print(f"    '{saetze[j]}'")
    print()

print("* Aehnliche Bedeutung = aehnliche Vektoren (hoher Cosine Score)")
print("  Das ist die Basis fuer RAG, Semantic Search, Chatbots, etc.")


# ============================================================
# 4. Mehrsprachige Aehnlichkeit (gleiche Bedeutung, andere Sprache)
# ============================================================

print("\n" + "=" * 60)
print("MEHRSPRACHIG: Gleiche Bedeutung in verschiedenen Sprachen")
print("=" * 60)

# Gleiche Saetze auf Deutsch, Englisch, Franzoesisch
mehrsprachig = [
    "Kuenstliche Intelligenz veraendert die Welt",     # DE
    "Artificial Intelligence is changing the world",     # EN
    "L'intelligence artificielle change le monde",       # FR
    "Das Wetter ist heute schoen",                       # DE
    "The weather is nice today",                         # EN
    "Le temps est beau aujourd'hui",                     # FR
]

emb_multi = model.encode(mehrsprachig)
sim_multi = cos_sim(emb_multi, emb_multi)

print("\nAehnlichkeitsmatrix (gleiche Bedeutung in DE/EN/FR):")
print(f"{'':>45}", end="")
for idx in range(len(mehrsprachig)):
    print(f" [{idx}]  ", end="")
print()

for i in range(len(mehrsprachig)):
    label = mehrsprachig[i][:42] + "..." if len(mehrsprachig[i]) > 45 else mehrsprachig[i]
    print(f"[{i}] {label:<42}", end=" ")
    for j in range(len(mehrsprachig)):
        val = sim_multi[i][j].item()
        print(f"{val:.2f} ", end=" ")
    print()

print("\n* Das multilingual Modell erkennt, dass 'KI veraendert die Welt'")
print("  und 'AI is changing the world' dasselbe bedeuten!")


# ============================================================
# 5. Embeddings visualisieren
# ============================================================

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive Backend fuer Windows
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Alle Saetze zusammen visualisieren
    alle_saetze = saetze + mehrsprachig
    alle_embeddings = model.encode(alle_saetze)

    # 384D -> 2D fuer Visualisierung
    pca = PCA(n_components=2)
    coords = pca.fit_transform(alle_embeddings)

    plt.figure(figsize=(14, 10))

    # Farben: Blau fuer deutsche Saetze, Rot fuer mehrsprachige
    farben = ['#1f77b4'] * len(saetze) + ['#d62728'] * len(mehrsprachig)

    for i, (x, y) in enumerate(coords):
        plt.scatter(x, y, s=100, c=farben[i], alpha=0.7)
        label = alle_saetze[i][:40] + "..." if len(alle_saetze[i]) > 40 else alle_saetze[i]
        plt.annotate(label, (x, y), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")

    plt.title("Saetze im Embedding-Raum (PCA 2D) - Blau=DE, Rot=Multilingual")
    plt.tight_layout()
    plt.savefig("embeddings_visualisierung.png", dpi=150)
    print("\n[OK] Plot gespeichert als 'embeddings_visualisierung.png'")

except ImportError:
    print("\n(matplotlib nicht installiert - Plot uebersprungen)")


# ============================================================
# UEBUNGEN
# ============================================================

print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)

print("""
1. FINDE DIE 2 AEHNLICHSTEN SAETZE
   Fuege eigene Saetze zur Liste 'saetze' hinzu und finde
   das Paar mit der hoechsten Cosine-Aehnlichkeit.
   Tipp: Die Top-10-Liste oben zeigt dir das Ergebnis!

2. TESTE MIT EIGENEN SAETZEN
   Erstelle 3 Saetze zum gleichen Thema und 3 zu einem
   anderen Thema. Erkennt das Modell die Gruppen?

3. SPRACH-EXPERIMENT
   Schreibe den gleichen Satz auf Deutsch und Englisch.
   Wie hoch ist die Aehnlichkeit? Teste auch mit
   Franzoesisch, Spanisch, Italienisch...

4. TOKENIZER-VERGLEICH
   Finde ein langes deutsches Wort (z.B. "Donaudampfschifffahrt")
   und vergleiche wie Mistral vs. GBERT es tokenisieren.
   Welcher ist effizienter?

5. EMBEDDING-DIMENSIONEN
   Aendere den Satz leicht ("Der Hund spielt im Park" ->
   "Der Hund schlaeft im Park"). Wie stark aendert sich
   die Aehnlichkeit?
""")

print("[OK] Script abgeschlossen!")
