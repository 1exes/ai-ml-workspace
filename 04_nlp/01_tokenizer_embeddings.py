"""
01 - Tokenizer & Embeddings
=============================
Wie versteht ein KI-Modell Text?
1. Text → Tokens (Tokenizer)
2. Tokens → Vektoren (Embeddings)
3. Vektoren → Verarbeitung (Transformer)
"""

from transformers import AutoTokenizer, AutoModel
import torch

# ============================================================
# 1. Tokenizer - Text in Zahlen umwandeln
# ============================================================

print("=" * 50)
print("TOKENIZER")
print("=" * 50)

# Lade einen Tokenizer (GPT-2 als Beispiel)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Künstliche Intelligenz verändert die Welt!"
tokens = tokenizer.encode(text)
decoded = [tokenizer.decode([t]) for t in tokens]

print(f"\nText: '{text}'")
print(f"Token IDs: {tokens}")
print(f"Tokens:    {decoded}")
print(f"Anzahl:    {len(tokens)} Tokens")

# Verschiedene Texte tokenisieren
examples = [
    "Hello World",
    "Maschinelles Lernen",
    "🤖 AI is amazing!",
    "pneumonoultramicroscopicsilicovolcanoconiosis",
]
for ex in examples:
    toks = tokenizer.encode(ex)
    print(f"  '{ex}' → {len(toks)} Tokens")

print("\n💡 Seltene Wörter brauchen mehr Tokens!")
print("   Ein Token ≈ 4 Zeichen in Englisch, weniger in Deutsch")

# ============================================================
# 2. Embeddings - Wörter als Vektoren
# ============================================================

print("\n" + "=" * 50)
print("EMBEDDINGS")
print("=" * 50)

# Sentence-Transformers für semantische Embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # Klein und schnell

sentences = [
    "Der Hund spielt im Park",
    "Die Katze jagt eine Maus",
    "Python ist eine Programmiersprache",
    "JavaScript wird für Webseiten genutzt",
    "Das Haustier rennt draußen herum",
]

embeddings = model.encode(sentences)
print(f"\nEmbedding Shape: {embeddings.shape}")
print(f"Jeder Satz = ein Vektor mit {embeddings.shape[1]} Dimensionen")

# ============================================================
# 3. Semantische Ähnlichkeit - DAS macht Embeddings so mächtig
# ============================================================

from sentence_transformers.util import cos_sim

print("\n--- Semantische Ähnlichkeit ---")
similarities = cos_sim(embeddings, embeddings)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = similarities[i][j].item()
        bar = "█" * int(sim * 30) if sim > 0 else ""
        print(f"  {sim:.3f} {bar}")
        print(f"    '{sentences[i]}'")
        print(f"    '{sentences[j]}'")
        print()

print("💡 Ähnliche Bedeutung = ähnliche Vektoren (hoher Cosine Score)")
print("   Das ist die Basis für RAG, Semantic Search, etc.")

# ============================================================
# 4. Embeddings visualisieren
# ============================================================

try:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # 384D → 2D für Visualisierung
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(coords):
        plt.scatter(x, y, s=100)
        plt.annotate(sentences[i], (x, y), fontsize=9,
                     xytext=(5, 5), textcoords="offset points")

    plt.title("Sätze im Embedding-Raum (PCA 2D)")
    plt.tight_layout()
    plt.savefig("embeddings_visualisierung.png", dpi=150)
    plt.show()
    print("\nPlot gespeichert als 'embeddings_visualisierung.png'")
except ImportError:
    print("\n(matplotlib nicht installiert - Plot übersprungen)")
