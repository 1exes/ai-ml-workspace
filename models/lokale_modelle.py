"""
03 - Lokale Modelle laden und nutzen
======================================
Verschiedene HuggingFace-Pipelines mit europaeischen & multilingualen Modellen.
Alles laeuft lokal auf CPU - kein Server noetig!

Modelle:
- Qwen 2.5 0.5B (Text-Generierung, winzig, laeuft auf CPU)
- nlptown/bert-base-multilingual (Sentiment, mehrsprachig)
- dslim/bert-base-NER (Named Entity Recognition)
- facebook/bart-large-cnn (Zusammenfassung)
- Helsinki-NLP/opus-mt-de-en (Uebersetzung Deutsch->Englisch)
- facebook/bart-large-mnli (Zero-Shot-Klassifikation)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline
import time


# ============================================================
# 1. TEXT-GENERIERUNG: Qwen 2.5 0.5B
# ============================================================

print("=" * 60)
print("1. TEXT-GENERIERUNG: Qwen 2.5 0.5B")
print("=" * 60)
print("   (Alibaba/Qwen - kleinstes Modell, laeuft auf CPU)")

print("\nLade Modell...")
t0 = time.time()
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B",
    device=-1,  # CPU
)
print(f"  Geladen in {time.time() - t0:.1f}s")

prompts = [
    "Machine learning is useful because",
    "The best way to learn programming is",
    "Artificial intelligence in Europe",
]

for prompt in prompts:
    result = generator(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=1,
    )
    text = result[0]["generated_text"]
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Output: {text[:200]}")

print("\n* Qwen 2.5 0.5B hat nur 500M Parameter - trotzdem brauchbar!")
print("  Fuer bessere Ergebnisse: Qwen2.5-1.5B oder Qwen2.5-7B")

# Speicher freigeben
del generator


# ============================================================
# 2. SENTIMENT-ANALYSE: Multilingual (Deutsch, Englisch, etc.)
# ============================================================

print("\n" + "=" * 60)
print("2. SENTIMENT-ANALYSE: Multilingual (1-5 Sterne)")
print("=" * 60)
print("   (nlptown - trainiert auf Reviews in 6 Sprachen)")

print("\nLade Modell...")
t0 = time.time()
sentiment = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=-1,
)
print(f"  Geladen in {time.time() - t0:.1f}s")

# Deutsche und englische Beispiele
texte = [
    ("Das Produkt ist fantastisch, ich bin begeistert!", "DE"),
    ("Totaler Schrott, funktioniert ueberhaupt nicht.", "DE"),
    ("Ganz okay, nichts Besonderes.", "DE"),
    ("I love this product, it's amazing!", "EN"),
    ("Terrible quality, waste of money.", "EN"),
    ("C'est un excellent produit, je recommande!", "FR"),
]

print(f"\n  {'Text':<52} {'Sprache':>7} {'Sterne':>7} {'Score':>6}")
print("  " + "-" * 76)

for text, lang in texte:
    result = sentiment(text)[0]
    sterne = result["label"]  # z.B. "5 stars"
    score = result["score"]
    display = text[:49] + "..." if len(text) > 52 else text
    print(f"  {display:<52} {lang:>7} {sterne:>7} {score:>5.1%}")

print("\n* Das Modell versteht Sentiment in DE, EN, FR, ES, IT, NL!")

del sentiment


# ============================================================
# 3. NAMED ENTITY RECOGNITION (NER)
# ============================================================

print("\n" + "=" * 60)
print("3. NAMED ENTITY RECOGNITION (NER)")
print("=" * 60)
print("   (dslim/bert-base-NER - erkennt Personen, Orte, Firmen)")

print("\nLade Modell...")
t0 = time.time()
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
    device=-1,
)
print(f"  Geladen in {time.time() - t0:.1f}s")

ner_texte = [
    "Angela Merkel visited the European Parliament in Strasbourg last Monday.",
    "Google and Microsoft are investing billions in artificial intelligence research.",
    "The Eiffel Tower in Paris attracts millions of tourists every year.",
]

for text in ner_texte:
    print(f"\n  Text: '{text}'")
    entities = ner(text)
    if entities:
        for ent in entities:
            print(f"    -> {ent['entity_group']:>5}: '{ent['word']}' (Score: {ent['score']:.2f})")
    else:
        print("    -> Keine Entitaeten gefunden")

print("\n* Entity-Typen: PER=Person, LOC=Ort, ORG=Organisation, MISC=Sonstiges")

del ner


# ============================================================
# 4. ZUSAMMENFASSUNG: BART-large-CNN
# ============================================================

print("\n" + "=" * 60)
print("4. ZUSAMMENFASSUNG: facebook/bart-large-cnn")
print("=" * 60)
print("   (Meta AI - trainiert auf CNN/DailyMail Artikeln)")

print("\nLade Modell...")
t0 = time.time()
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1,
)
print(f"  Geladen in {time.time() - t0:.1f}s")

artikel = """
The European Union has announced a comprehensive new framework for regulating
artificial intelligence, marking one of the most ambitious attempts by any
government to set rules for the rapidly evolving technology. The AI Act, which
was formally adopted after years of negotiation, establishes a risk-based
approach to AI regulation. High-risk AI systems, such as those used in healthcare,
law enforcement, and critical infrastructure, will face strict requirements
including mandatory risk assessments, transparency obligations, and human oversight.
The regulation also bans certain AI practices deemed unacceptable, such as social
scoring systems and real-time biometric surveillance in public spaces, with limited
exceptions for law enforcement. Companies that violate the rules could face fines
of up to 35 million euros or 7 percent of their global annual revenue. The EU hopes
the regulation will set a global standard for AI governance, similar to how the
General Data Protection Regulation influenced data privacy laws worldwide.
"""

summary = summarizer(artikel, max_length=80, min_length=30)
print(f"\n  Original: {len(artikel.split())} Woerter")
print(f"  Summary:  {summary[0]['summary_text']}")

del summarizer


# ============================================================
# 5. UEBERSETZUNG: Deutsch -> Englisch (Helsinki-NLP)
# ============================================================

print("\n" + "=" * 60)
print("5. UEBERSETZUNG: Deutsch -> Englisch")
print("=" * 60)
print("   (Helsinki-NLP/opus-mt-de-en - Open-Source Uebersetzer)")

print("\nLade Modell...")
t0 = time.time()
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-de-en",
    device=-1,
)
print(f"  Geladen in {time.time() - t0:.1f}s")

deutsche_saetze = [
    "Kuenstliche Intelligenz veraendert unsere Gesellschaft grundlegend.",
    "Deutschland investiert stark in die Erforschung erneuerbarer Energien.",
    "Der Datenschutz ist ein wichtiges Thema in der Europaeischen Union.",
    "Maschinelles Lernen hilft Aerzten bei der Diagnose von Krankheiten.",
    "Die Zukunft der Arbeit wird durch Automatisierung gepraegt sein.",
]

for satz in deutsche_saetze:
    result = translator(satz)
    print(f"\n  DE: {satz}")
    print(f"  EN: {result[0]['translation_text']}")

print("\n* Helsinki-NLP bietet Modelle fuer 1000+ Sprachpaare!")
print("  z.B. opus-mt-en-de, opus-mt-fr-de, opus-mt-de-fr, ...")

del translator


# ============================================================
# 6. ZERO-SHOT-KLASSIFIKATION: BART-large-MNLI
# ============================================================

print("\n" + "=" * 60)
print("6. ZERO-SHOT-KLASSIFIKATION (ohne Training!)")
print("=" * 60)
print("   (facebook/bart-large-mnli - klassifiziert in beliebige Kategorien)")

print("\nLade Modell...")
t0 = time.time()
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1,
)
print(f"  Geladen in {time.time() - t0:.1f}s")

# Beispiel 1: Nachrichten-Kategorisierung
print("\n--- Nachrichten-Kategorisierung ---")
nachrichten = [
    "The stock market crashed today after the central bank raised interest rates.",
    "The new smartphone features a revolutionary camera system with AI enhancement.",
    "Scientists discovered a new species of deep-sea fish near the Atlantic Ridge.",
    "The national team won the championship after a dramatic penalty shootout.",
]
kategorien = ["politics", "technology", "science", "sports", "finance"]

for text in nachrichten:
    result = classifier(text, kategorien)
    top = result["labels"][0]
    score = result["scores"][0]
    display = text[:60] + "..." if len(text) > 60 else text
    print(f"\n  Text: '{display}'")
    print(f"  Kategorie: {top} ({score:.1%})")
    # Top 3 anzeigen
    for label, s in zip(result["labels"][:3], result["scores"][:3]):
        bar = "#" * int(s * 30)
        print(f"    {label:<12} {s:.1%} {bar}")

# Beispiel 2: Intent-Erkennung (deutsch-englisch gemischt)
print("\n--- Intent-Erkennung ---")
user_inputs = [
    "How do I reset my password?",
    "I want to cancel my subscription",
    "Your product is amazing, keep it up!",
    "When will my order arrive?",
]
intents = ["technical_support", "cancellation", "feedback", "order_status", "billing"]

for text in user_inputs:
    result = classifier(text, intents)
    top = result["labels"][0]
    score = result["scores"][0]
    print(f"\n  '{text}'")
    print(f"  -> Intent: {top} ({score:.1%})")

del classifier


# ============================================================
# 7. OPTIONAL: Verbindung zu lokalem LLM-Server
# ============================================================

print("\n" + "=" * 60)
print("7. OPTIONAL: Lokaler LLM-Server (Ollama / LM Studio)")
print("=" * 60)

print("""
Falls du einen lokalen LLM-Server hast (Ollama, LM Studio, vLLM),
kannst du ihn ueber die OpenAI-kompatible API ansprechen:

  import httpx

  def chat_local(prompt, base_url="http://localhost:11434/v1", model="qwen2.5:14b"):
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
      return response.json()["choices"][0]["message"]["content"]

  # Beispiel:
  # answer = chat_local("Was ist Machine Learning?")

Typische Server-URLs:
  - Ollama:    http://localhost:11434/v1
  - LM Studio: http://localhost:1234/v1
  - vLLM:      http://localhost:8000/v1

Tipp: Das httpx-Paket ist bereits installiert (pip install httpx)
""")


# ============================================================
# UEBERSICHT: Alle genutzten Modelle
# ============================================================

print("=" * 60)
print("UEBERSICHT: Genutzte Modelle")
print("=" * 60)

modelle = [
    ("Qwen/Qwen2.5-0.5B",             "Text-Generierung",     "~1 GB",   "Alibaba (CN)"),
    ("nlptown/bert-base-multilingual",  "Sentiment (6 Sprachen)","~700 MB", "nlptown (NL)"),
    ("dslim/bert-base-NER",            "Named Entities",        "~400 MB", "dslim"),
    ("facebook/bart-large-cnn",         "Zusammenfassung",       "~1.6 GB", "Meta AI (US)"),
    ("Helsinki-NLP/opus-mt-de-en",      "DE->EN Uebersetzung",  "~300 MB", "Helsinki Uni (FI)"),
    ("facebook/bart-large-mnli",        "Zero-Shot Klassif.",    "~1.6 GB", "Meta AI (US)"),
]

print(f"\n  {'Modell':<40} {'Aufgabe':<22} {'Groesse':>8} {'Herkunft':<15}")
print("  " + "-" * 88)
for name, task, size, origin in modelle:
    print(f"  {name:<40} {task:<22} {size:>8} {origin:<15}")

print(f"\n  Gesamt ca. 5-6 GB Download (einmalig, wird in ~/.cache/huggingface/ gespeichert)")


# ============================================================
# UEBUNGEN
# ============================================================

print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)

print("""
1. EIGENE ZERO-SHOT-KATEGORIEN
   Definiere eigene Kategorien fuer den Zero-Shot-Classifier.
   z.B. E-Mail-Sortierung: ["urgent", "spam", "newsletter", "personal"]
   Teste mit verschiedenen Texten!

2. UEBERSETZUNGS-KETTE
   Nutze Helsinki-NLP Modelle fuer eine Kette:
   DE -> EN -> FR (braucht opus-mt-en-fr)
   Wie gut bleibt die Bedeutung erhalten?

3. NER AUF DEUTSCHEN TEXTEN
   Das NER-Modell ist primaer fuer Englisch trainiert.
   Teste es mit deutschen Saetzen. Funktioniert es?
   Tipp: Fuer besseres Deutsch-NER suche nach "german ner"
   auf huggingface.co/models

4. SENTIMENT MIT SARKASMUS
   Teste den Sentiment-Classifier mit sarkastischen Texten:
   "Na toll, schon wieder kaputt. Einfach wunderbar."
   Erkennt das Modell den Sarkasmus?

5. PIPELINE-KOMBINATIONEN
   Baue eine Pipeline die:
   a) Einen deutschen Text uebersetzt (DE->EN)
   b) Den englischen Text zusammenfasst
   c) Die Zusammenfassung klassifiziert
   -> Ein Mini-NLP-System!
""")

print("[OK] Script abgeschlossen!")
