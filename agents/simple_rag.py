import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
01 - RAG (Retrieval Augmented Generation) - Standalone Version
===============================================================
RAG = Dein Modell bekommt relevante Dokumente als Kontext.
So kann ein LLM ueber DEINE Daten antworten, ohne Fine-Tuning.

Pipeline: Frage -> Embedding -> Aehnliche Docs finden -> LLM mit Kontext fragen

Funktioniert komplett OFFLINE ohne externen Server!
Benoetigt: pip install sentence-transformers chromadb transformers torch
"""

import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. Deutsche Wissensbasis - AI/ML Konzepte
# ============================================================

print("=" * 60)
print("  RAG Pipeline - Standalone (kein Server noetig)")
print("=" * 60)

# Umfangreiche deutsche Wissensbasis zu AI/ML
WISSENSBASIS = [
    # Grundlagen
    "Kuenstliche Intelligenz (KI) ist ein Teilgebiet der Informatik, das sich mit der "
    "Automatisierung von intelligentem Verhalten befasst. Maschinelles Lernen ist eine "
    "Unterkategorie von KI, bei der Algorithmen aus Daten lernen.",

    "Supervised Learning (ueberwachtes Lernen) trainiert ein Modell mit gelabelten Daten. "
    "Das Modell lernt die Zuordnung von Eingabe zu Ausgabe. Beispiele: Klassifikation, "
    "Regression. Typische Algorithmen: Lineare Regression, Random Forest, Neuronale Netze.",

    "Unsupervised Learning (unueberwachtes Lernen) findet Muster in ungelabelten Daten. "
    "Clustering (z.B. K-Means) gruppiert aehnliche Datenpunkte. Dimensionsreduktion "
    "(z.B. PCA) reduziert die Anzahl der Features.",

    "Reinforcement Learning (Verstaerkungslernen) trainiert einen Agenten durch Belohnung "
    "und Bestrafung. Der Agent lernt eine Policy, die den kumulativen Reward maximiert. "
    "Beispiele: AlphaGo, Roboter-Steuerung, Spielstrategien.",

    # Neuronale Netze
    "Neuronale Netze bestehen aus Schichten von Neuronen (Nodes). Jedes Neuron berechnet "
    "eine gewichtete Summe seiner Eingaben plus Bias und wendet eine Aktivierungsfunktion an. "
    "Typische Aktivierungen: ReLU, Sigmoid, Tanh, Softmax.",

    "Backpropagation ist der Algorithmus zum Training neuronaler Netze. Er berechnet den "
    "Gradienten der Verlustfunktion bezueglich jedes Gewichts durch die Kettenregel. "
    "Der Optimizer (z.B. Adam, SGD) aktualisiert dann die Gewichte.",

    "Overfitting bedeutet, dass das Modell die Trainingsdaten auswendig lernt statt zu "
    "generalisieren. Gegenmassnahmen: Dropout, Regularisierung (L1/L2), Data Augmentation, "
    "Early Stopping, mehr Trainingsdaten.",

    # Transformer und LLMs
    "Transformer wurden 2017 im Paper 'Attention is All You Need' vorgestellt. Sie nutzen "
    "Self-Attention um Beziehungen zwischen allen Positionen einer Sequenz zu berechnen. "
    "Die Formel: Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V.",

    "Large Language Models (LLMs) wie GPT, Claude und Llama basieren auf der Transformer-"
    "Architektur. Sie werden auf riesigen Textmengen vortrainiert (Pre-Training) und dann "
    "mit RLHF oder DPO auf menschliche Praeferenzen feinabgestimmt.",

    "LoRA (Low-Rank Adaptation) ermoeglicht effizientes Fine-Tuning grosser Modelle. Statt "
    "alle Gewichte anzupassen, werden kleine Low-Rank-Matrizen hinzugefuegt. Das spart "
    "90%+ Speicher und Training dauert nur Stunden statt Tage.",

    # RAG und Embeddings
    "Embeddings sind dichte Vektoren, die die Bedeutung von Text kodieren. Aehnliche Texte "
    "haben aehnliche Vektoren. Sentence-Transformers erzeugen Embeddings fuer ganze Saetze. "
    "Cosine Similarity misst die Aehnlichkeit zwischen zwei Vektoren.",

    "RAG (Retrieval Augmented Generation) kombiniert Suche mit Textgenerierung. Statt alles "
    "im Modell zu speichern, werden relevante Dokumente zur Laufzeit gesucht und als Kontext "
    "uebergeben. Vorteile: Aktuelles Wissen, nachvollziehbar, kein Fine-Tuning noetig.",

    "Vector-Datenbanken wie ChromaDB, Pinecone, Weaviate oder FAISS speichern Embeddings "
    "und ermoeglichen schnelle Aehnlichkeitssuche (Approximate Nearest Neighbor). "
    "ChromaDB ist Open Source und laeuft lokal ohne Server.",

    # Quantisierung und Inference
    "Quantisierung reduziert die Praezision von Modellgewichten (z.B. float32 -> int4). "
    "GGUF und GPTQ sind gaengige Quantisierungsformate. Ein 7B-Modell braucht mit float16 "
    "ca. 14GB RAM, mit 4-bit Quantisierung nur ca. 4GB.",

    "Inference-Optimierung umfasst: KV-Cache (vermeidet Neuberechnung), Speculative Decoding "
    "(schnelleres Sampling), Flash Attention (speichereffiziente Attention), "
    "Continuous Batching (mehrere Anfragen gleichzeitig).",
]

# ============================================================
# 2. Embedding-Modell laden und Dokumente indexieren
# ============================================================

print("\n[1/4] Embedding-Modell laden...")
print("      Modell: paraphrase-multilingual-MiniLM-L12-v2")
print("      (Wird beim ersten Mal heruntergeladen, ~120MB)")
t0 = time.time()

# Multilinguales Modell - versteht Deutsch besonders gut
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(f"      Geladen in {time.time() - t0:.1f}s")

print("\n[2/4] Dokumente in ChromaDB indexieren...")
t0 = time.time()

# ChromaDB - lokale Vector-Datenbank (kein Server noetig)
chroma_client = chromadb.Client()

# Collection erstellen (oder vorhandene loeschen und neu erstellen)
try:
    chroma_client.delete_collection("ai_wissensbasis")
except Exception:
    pass

collection = chroma_client.create_collection(
    name="ai_wissensbasis",
    metadata={"hnsw:space": "cosine"},  # Cosine Similarity
)

# Embeddings berechnen
embeddings = embedder.encode(WISSENSBASIS).tolist()

# In ChromaDB speichern
collection.add(
    documents=WISSENSBASIS,
    embeddings=embeddings,
    ids=[f"doc_{i:02d}" for i in range(len(WISSENSBASIS))],
    metadatas=[{"thema": doc.split(".")[0][:50]} for doc in WISSENSBASIS],
)

print(f"      {len(WISSENSBASIS)} Dokumente indexiert in {time.time() - t0:.1f}s")
print(f"      Embedding-Dimension: {len(embeddings[0])}")

# ============================================================
# 3. Semantische Suche
# ============================================================

print("\n" + "=" * 60)
print("[3/4] Semantische Suche testen")
print("=" * 60)


def suche(query: str, n_results: int = 3):
    """Finde die relevantesten Dokumente fuer eine Frage."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    return results["documents"][0], results["distances"][0]


# Verschiedene Testfragen
testfragen = [
    "Wie kann ich ein grosses Modell kleiner machen?",
    "Was ist der Transformer und wie funktioniert Attention?",
    "Wie verhindere ich, dass mein Modell uebertrainiert?",
    "Erklaere mir RAG einfach",
    "Was ist der Unterschied zwischen Supervised und Unsupervised Learning?",
]

for frage in testfragen:
    docs, distances = suche(frage, n_results=2)
    print(f"\n  Frage: '{frage}'")
    for i, (doc, dist) in enumerate(zip(docs, distances)):
        # Cosine Distance: 0 = identisch, 2 = komplett verschieden
        relevanz = (1 - dist) * 100  # Umrechnung in Prozent
        print(f"    [{relevanz:.0f}% relevant] {doc[:90]}...")


# ============================================================
# 4. Vollstaendige RAG-Pipeline mit lokalem LLM
# ============================================================

print("\n" + "=" * 60)
print("[4/4] RAG-Pipeline: Suche + LLM-Generierung")
print("=" * 60)


def rag_nur_suche(frage: str, n_results: int = 3) -> str:
    """RAG-Fallback: Gibt nur die gefundenen Dokumente zurueck (ohne LLM)."""
    docs, distances = suche(frage, n_results=n_results)

    antwort_teile = [f"Basierend auf der Wissensbasis zu '{frage}':\n"]
    for i, (doc, dist) in enumerate(zip(docs, distances)):
        relevanz = (1 - dist) * 100
        antwort_teile.append(f"  ({relevanz:.0f}%) {doc}")
    return "\n".join(antwort_teile)


def rag_mit_llm(frage: str, n_results: int = 3) -> str:
    """Vollstaendige RAG-Pipeline mit lokalem Qwen2.5-0.5B Modell.

    Das Modell laeuft komplett lokal ueber HuggingFace Transformers.
    Kein Server, kein API-Call, alles auf deinem Rechner.
    """
    # Schritt 1: Relevante Dokumente suchen
    print("\n  Schritt 1: Relevante Dokumente suchen...")
    docs, distances = suche(frage, n_results=n_results)

    for i, (doc, dist) in enumerate(zip(docs, distances)):
        relevanz = (1 - dist) * 100
        print(f"    Dok {i+1} [{relevanz:.0f}%]: {doc[:70]}...")

    # Schritt 2: Kontext aufbauen
    print("  Schritt 2: Prompt mit Kontext bauen...")
    context = "\n".join(f"- {doc}" for doc in docs)
    prompt = (
        f"Beantworte die Frage basierend auf dem Kontext. "
        f"Antworte auf Deutsch, kurz und praezise.\n\n"
        f"Kontext:\n{context}\n\n"
        f"Frage: {frage}\n\n"
        f"Antwort:"
    )
    print(f"    Prompt-Laenge: {len(prompt)} Zeichen")

    # Schritt 3: LLM-Generierung
    print("  Schritt 3: Lokales LLM laden (Qwen2.5-0.5B)...")
    print("    (Wird beim ersten Mal heruntergeladen, ~500MB)")
    print("    HINWEIS: Auf CPU kann das 30-60s dauern.")

    try:
        from transformers import pipeline as hf_pipeline

        t0 = time.time()
        # Qwen2.5-0.5B - klein genug fuer CPU, gross genug fuer einfache Antworten
        generator = hf_pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B",
            device_map="auto",
            torch_dtype="auto",
        )
        print(f"    Modell geladen in {time.time() - t0:.1f}s")

        t0 = time.time()
        output = generator(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            return_full_text=False,
        )
        generierter_text = output[0]["generated_text"].strip()
        print(f"    Generiert in {time.time() - t0:.1f}s")
        return generierter_text

    except Exception as e:
        print(f"\n    [!] LLM konnte nicht geladen werden: {e}")
        print("    -> Fallback: Zeige nur die gefundenen Dokumente")
        return rag_nur_suche(frage, n_results)


# ============================================================
# RAG Pipeline ausfuehren
# ============================================================

print("\n--- RAG mit reiner Suche (schnell, kein LLM) ---")
frage1 = "Wie kann ich ein grosses Modell effizient fine-tunen?"
print(f"\n  Frage: {frage1}")
antwort = rag_nur_suche(frage1)
print(f"\n  Antwort (nur Suche):\n{antwort}")

print("\n\n--- RAG mit lokalem LLM (Qwen2.5-0.5B) ---")
frage2 = "Was ist der Vorteil von RAG gegenueber Fine-Tuning?"
print(f"\n  Frage: {frage2}")
antwort = rag_mit_llm(frage2)
print(f"\n  Antwort (LLM):\n  {antwort}")


# ============================================================
# 5. Zusammenfassung und Vorteile
# ============================================================

print("\n" + "=" * 60)
print("  Zusammenfassung")
print("=" * 60)
print("""
  RAG Vorteile:
  * Kein Fine-Tuning noetig - Wissen kommt aus der Datenbank
  * Wissen ist updatebar - einfach neue Dokumente hinzufuegen
  * Antworten sind nachvollziehbar - du siehst die Quellen
  * Funktioniert mit jedem LLM (gross oder klein)
  * Reduziert Halluzinationen durch faktischen Kontext

  Diese Demo hat gezeigt:
  1. Embedding-Modell: paraphrase-multilingual-MiniLM-L12-v2
  2. Vector-DB: ChromaDB (lokal, kein Server)
  3. LLM: Qwen2.5-0.5B (lokal via HuggingFace Transformers)
  4. Alles laeuft OFFLINE auf deinem Rechner!
""")


# ============================================================
# UEBUNGEN
# ============================================================

print("=" * 60)
print("  Uebungen")
print("=" * 60)
print("""
  Uebung 1: Fuege eigene Dokumente hinzu
  -----------------------------------------
  Erweitere die WISSENSBASIS-Liste mit eigenen Texten.
  Zum Beispiel:
    - Fuege 5 Dokumente ueber Python-Bibliotheken hinzu
    - Oder ueber dein eigenes Fachgebiet
  Starte das Script neu und teste, ob die Suche deine Dokumente findet.

  Uebung 2: Aendere die Anzahl der Ergebnisse
  -----------------------------------------------
  Aendere den Parameter n_results in der suche()-Funktion.
  Was passiert bei n_results=1 vs n_results=5?
  Wann ist mehr Kontext besser, wann schlechter?

  Uebung 3: Lade Dokumente aus einer Datei
  -------------------------------------------
  Schreibe eine Funktion, die Dokumente aus einer .txt-Datei liest
  (eine Zeile = ein Dokument) und in ChromaDB indexiert.
  Tipp: with open("docs.txt", encoding="utf-8") as f: ...

  Uebung 4: Vergleiche Embedding-Modelle
  -----------------------------------------
  Ersetze das Embedding-Modell durch ein anderes, z.B.:
    - "all-MiniLM-L6-v2" (Englisch, kleiner)
    - "sentence-transformers/all-mpnet-base-v2" (Englisch, genauer)
  Vergleiche die Suchergebnisse: Welches findet bessere Treffer auf Deutsch?

  Uebung 5: Relevanz-Schwellenwert
  -----------------------------------
  Filtere Ergebnisse mit niedriger Relevanz heraus.
  Wenn kein Dokument ueber 50% Relevanz hat, sage "Keine relevanten
  Informationen gefunden" statt eine schlechte Antwort zu geben.
""")
