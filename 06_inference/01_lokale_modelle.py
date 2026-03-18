"""
01 - Lokale Modelle laden und nutzen
=====================================
Verschiedene Wege, KI-Modelle lokal auszuführen.
Du hast bereits LM Studio - hier sind weitere Optionen.
"""

import httpx

# ============================================================
# 1. LM Studio / Ollama - OpenAI-kompatible API
# ============================================================

print("=" * 50)
print("1. OpenAI-kompatible API (LM Studio / Ollama)")
print("=" * 50)

# Das kennst du schon von deinen anderen Projekten!
# LM Studio: http://192.168.2.6:1234
# Ollama: http://192.168.178.118:11434

def chat_with_local_model(
    prompt: str,
    base_url: str = "http://192.168.178.118:11434/v1",
    model: str = "qwen2.5:14b",
):
    """Chat mit lokalem Modell via OpenAI-kompatible API."""
    response = httpx.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Test (nur wenn Server erreichbar)
try:
    answer = chat_with_local_model("Was ist Machine Learning in einem Satz?")
    print(f"\nAntwort: {answer}")
except Exception as e:
    print(f"\nServer nicht erreichbar: {e}")
    print("Starte LM Studio oder Ollama und versuche es erneut.")

# ============================================================
# 2. Hugging Face Transformers - direkt in Python
# ============================================================

print("\n" + "=" * 50)
print("2. Hugging Face Transformers Pipeline")
print("=" * 50)

from transformers import pipeline

# Text Generation
print("\n--- Text Generation ---")
generator = pipeline("text-generation", model="gpt2", device=-1)  # -1 = CPU
result = generator(
    "Machine Learning is",
    max_new_tokens=50,
    num_return_sequences=1,
    temperature=0.8,
)
print(f"GPT-2 sagt: {result[0]['generated_text']}")

# Sentiment Analysis
print("\n--- Sentiment Analysis ---")
classifier = pipeline("sentiment-analysis", device=-1)
texts = [
    "I love this product, it's amazing!",
    "This is terrible and broken.",
    "It's okay, nothing special.",
]
for text, result in zip(texts, classifier(texts)):
    print(f"  '{text}' → {result['label']} ({result['score']:.1%})")

# Summarization
print("\n--- Zusammenfassung ---")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
article = """
Machine learning is a branch of artificial intelligence that focuses on building
applications that learn from data and improve their accuracy over time without being
programmed to do so. In data science, an algorithm is a sequence of statistical
processing steps. In machine learning, algorithms are trained to find patterns and
features in massive amounts of data in order to make decisions and predictions based
on new data. The better the algorithm, the more accurate the decisions and predictions
will become as it processes more data.
"""
summary = summarizer(article, max_length=50, min_length=20)
print(f"  Original: {len(article.split())} Wörter")
print(f"  Summary:  {summary[0]['summary_text']}")

# ============================================================
# 3. Verschiedene Aufgaben mit einem Befehl
# ============================================================

print("\n" + "=" * 50)
print("3. Verfügbare Pipeline-Tasks")
print("=" * 50)

tasks = {
    "text-generation": "Text generieren (GPT-2, LLaMA, ...)",
    "text-classification": "Text klassifizieren (Sentiment, Spam, ...)",
    "token-classification": "Named Entity Recognition (Personen, Orte, ...)",
    "question-answering": "Fragen zu einem Text beantworten",
    "summarization": "Text zusammenfassen",
    "translation": "Text übersetzen",
    "fill-mask": "Lücken in Text füllen (BERT)",
    "image-classification": "Bilder klassifizieren",
    "object-detection": "Objekte in Bildern erkennen",
    "automatic-speech-recognition": "Audio → Text",
    "text-to-speech": "Text → Audio",
    "zero-shot-classification": "Klassifizieren ohne Training",
}

for task, desc in tasks.items():
    print(f"  pipeline('{task}') → {desc}")

print("\n💡 Jeder Task hat vortrainierte Modelle auf huggingface.co/models")
print("   Die meisten funktionieren out-of-the-box, ohne eigenes Training!")
