"""
01 - RAG (Retrieval Augmented Generation)
==========================================
RAG = Dein Modell bekommt relevante Dokumente als Kontext.
So kann ein LLM über DEINE Daten antworten, ohne Fine-Tuning.

Pipeline: Frage → Embedding → Ähnliche Docs finden → LLM mit Kontext fragen
"""

import chromadb
from sentence_transformers import SentenceTransformer
import httpx

# ============================================================
# 1. Dokumente in Vector DB speichern
# ============================================================

print("=" * 50)
print("1. Dokumente indexieren")
print("=" * 50)

# Embedding-Modell laden
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB - einfache lokale Vector-Datenbank
client = chromadb.Client()
collection = client.create_collection(
    name="wissensbasis",
    metadata={"hnsw:space": "cosine"},
)

# Beispiel-Dokumente (normalerweise: PDFs, Webseiten, Code, etc.)
documents = [
    "Python wurde 1991 von Guido van Rossum entwickelt. Es ist eine interpretierte, höhere Programmiersprache.",
    "PyTorch ist ein Deep-Learning-Framework von Meta. Es nutzt dynamische Computation Graphs.",
    "Transformer wurden 2017 im Paper 'Attention is All You Need' vorgestellt. Sie sind die Basis für GPT und BERT.",
    "LoRA (Low-Rank Adaptation) ermöglicht effizientes Fine-Tuning großer Modelle mit wenig Speicher.",
    "RLHF (Reinforcement Learning from Human Feedback) wird genutzt um LLMs an menschliche Präferenzen anzupassen.",
    "Vector-Datenbanken wie ChromaDB, Pinecone oder Weaviate speichern Embeddings für schnelle Ähnlichkeitssuche.",
    "Quantisierung reduziert die Präzision von Modell-Gewichten (z.B. float16 → int4) für schnellere Inference.",
    "Attention-Mechanismus: Q·K^T/√d berechnet, welche Teile des Inputs für jeden Output relevant sind.",
    "Ein Epoch bedeutet, dass das Modell den gesamten Trainingsdatensatz einmal durchlaufen hat.",
    "Overfitting: Das Modell lernt die Trainingsdaten auswendig statt zu generalisieren. Lösung: Dropout, Regularisierung.",
]

# Embeddings berechnen und speichern
embeddings = embedder.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))],
)
print(f"✅ {len(documents)} Dokumente indexiert")

# ============================================================
# 2. Relevante Dokumente finden
# ============================================================

print("\n" + "=" * 50)
print("2. Semantische Suche")
print("=" * 50)

def search(query: str, n_results: int = 3):
    """Finde die relevantesten Dokumente für eine Frage."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    return results["documents"][0], results["distances"][0]

# Test-Suche
queries = [
    "Wie kann ich ein Modell kleiner machen?",
    "Was ist der Transformer?",
    "Wie verhindere ich Overfitting?",
]

for query in queries:
    docs, distances = search(query)
    print(f"\n❓ '{query}'")
    for doc, dist in zip(docs, distances):
        print(f"   [{dist:.3f}] {doc[:80]}...")

# ============================================================
# 3. RAG Pipeline: Suche + LLM
# ============================================================

print("\n" + "=" * 50)
print("3. RAG: Suche + LLM Antwort")
print("=" * 50)

def rag_answer(question: str, llm_url: str = "http://192.168.178.118:11434/v1"):
    """Beantworte eine Frage mit RAG."""

    # 1. Relevante Dokumente finden
    docs, _ = search(question, n_results=3)
    context = "\n".join(f"- {doc}" for doc in docs)

    # 2. Prompt mit Kontext bauen
    prompt = f"""Beantworte die Frage basierend auf dem gegebenen Kontext.
Antworte auf Deutsch, kurz und präzise.

Kontext:
{context}

Frage: {question}

Antwort:"""

    # 3. LLM fragen
    try:
        response = httpx.post(
            f"{llm_url}/chat/completions",
            json={
                "model": "qwen2.5:14b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return f"(LLM nicht erreichbar - Kontext wäre:\n{context})"

# Test
question = "Wie kann ich ein großes Modell effizient fine-tunen?"
print(f"\n❓ {question}")
answer = rag_answer(question)
print(f"\n💬 {answer}")

print("\n" + "=" * 50)
print("💡 RAG Vorteile:")
print("  - Kein Fine-Tuning nötig")
print("  - Wissen ist updatebar (einfach neue Docs hinzufügen)")
print("  - Antworten sind nachvollziehbar (du siehst die Quellen)")
print("  - Funktioniert mit jedem LLM")
