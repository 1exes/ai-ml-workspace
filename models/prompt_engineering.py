import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Prompt Engineering - Die Kunst der richtigen Frage
===================================================
Wie man KI-Modelle richtig anweist.
"""

import json

# ============================================================
# 1. Prompt-Grundlagen
# ============================================================
print("=" * 60)
print("1. PROMPT PATTERNS - Ueberblick")
print("=" * 60)

patterns = {
    "Zero-Shot": {
        "beschreibung": "Direkte Anweisung ohne Beispiele",
        "wann": "Einfache, klar definierte Aufgaben",
        "beispiel": "Klassifiziere: 'Super Service!' -> Sentiment:",
    },
    "Few-Shot": {
        "beschreibung": "2-5 Beispiele vor der eigentlichen Aufgabe",
        "wann": "Wenn das Format wichtig ist oder die Aufgabe unklar",
        "beispiel": "'Tolles Essen!' -> positiv\n'Lange Wartezeit' -> negativ\n'Netter Laden' -> ?",
    },
    "Chain-of-Thought": {
        "beschreibung": "Modell soll Schritt fuer Schritt denken",
        "wann": "Mathe, Logik, komplexe Reasoning-Aufgaben",
        "beispiel": "Denke Schritt fuer Schritt: Wenn 3 Aepfel 1.50 EUR kosten...",
    },
    "ReAct": {
        "beschreibung": "Thought -> Action -> Observation Loop",
        "wann": "Agenten die Tools nutzen",
        "beispiel": "Thought: Ich muss den Preis berechnen\nAction: calculate(3*0.5)\nObs: 1.5",
    },
    "Role Prompting": {
        "beschreibung": "Modell nimmt eine bestimmte Rolle ein",
        "wann": "Wenn Expertise oder Stil wichtig ist",
        "beispiel": "Du bist ein erfahrener Python-Entwickler. Erklaere Decorators.",
    },
}

for name, info in patterns.items():
    print(f"\n  [{name}]")
    print(f"  {info['beschreibung']}")
    print(f"  Wann: {info['wann']}")
    print(f"  Beispiel: {info['beispiel'][:70]}...")

# ============================================================
# 2. PromptTemplate Klasse
# ============================================================
print("\n\n" + "=" * 60)
print("2. PROMPT-TEMPLATE SYSTEM")
print("=" * 60)

class PromptTemplate:
    """Wiederverwendbare Prompt-Templates mit Variablen."""

    def __init__(self, template: str, name: str = ""):
        self.template = template
        self.name = name

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    def __repr__(self):
        return f"PromptTemplate('{self.name}')"


# Templates definieren
TEMPLATES = {
    "classifier": PromptTemplate(
        name="Text Classifier",
        template="""Klassifiziere den folgenden Text in eine der Kategorien: {categories}

Text: "{text}"

Antworte NUR mit der Kategorie, nichts anderes.
Kategorie:""",
    ),

    "summarizer": PromptTemplate(
        name="Summarizer",
        template="""Fasse den folgenden Text in {num_sentences} Saetzen zusammen.
Schreibe auf {language}.

Text:
{text}

Zusammenfassung:""",
    ),

    "cot_math": PromptTemplate(
        name="Chain-of-Thought Math",
        template="""Loese die folgende Aufgabe Schritt fuer Schritt.
Zeige jeden Rechenschritt einzeln.

Aufgabe: {problem}

Loesung:
Schritt 1:""",
    ),

    "few_shot": PromptTemplate(
        name="Few-Shot Classifier",
        template="""Bestimme das Sentiment (positiv/negativ/neutral).

Beispiele:
{examples}

Text: "{text}"
Sentiment:""",
    ),

    "json_output": PromptTemplate(
        name="Structured JSON Output",
        template="""Extrahiere die folgenden Informationen aus dem Text und gib sie als JSON zurueck.

Felder: {fields}

Text: "{text}"

JSON:""",
    ),

    "code_review": PromptTemplate(
        name="Code Review",
        template="""Du bist ein erfahrener {language}-Entwickler.
Reviewe den folgenden Code auf:
1. Bugs und Fehler
2. Performance-Probleme
3. Best Practices

Code:
```{language}
{code}
```

Review:""",
    ),

    "role_expert": PromptTemplate(
        name="Expert Role",
        template="""Du bist ein {role} mit {years} Jahren Erfahrung.
Dein Spezialgebiet ist {specialty}.
Antworte auf {language} und passe die Erklaerung an {audience} an.

Frage: {question}

Antwort:""",
    ),
}

# Templates demonstrieren
print("\nVerfuegbare Templates:")
for key, tmpl in TEMPLATES.items():
    print(f"  {key:<15s} -> {tmpl.name}")

# Beispiel: Classifier
print("\n--- Beispiel: Classifier Template ---")
prompt = TEMPLATES["classifier"].format(
    categories="Sport, Technologie, Politik, Wirtschaft",
    text="Apple stellt das neue iPhone mit KI-Funktionen vor",
)
print(prompt)

# Beispiel: Few-Shot
print("\n--- Beispiel: Few-Shot Template ---")
examples = """- "Tolles Restaurant, super Essen!" -> positiv
- "Lange Wartezeit, kaltes Essen" -> negativ
- "War ganz okay, nichts besonderes" -> neutral"""
prompt = TEMPLATES["few_shot"].format(
    examples=examples,
    text="Der Service war freundlich aber das Essen war lau",
)
print(prompt)

# Beispiel: JSON Output
print("\n--- Beispiel: JSON Output Template ---")
prompt = TEMPLATES["json_output"].format(
    fields="name, alter, beruf, stadt",
    text="Max Mueller ist 34 Jahre alt, arbeitet als Ingenieur in Stuttgart",
)
print(prompt)
print("\nErwartete Antwort:")
print(json.dumps({"name": "Max Mueller", "alter": 34, "beruf": "Ingenieur", "stadt": "Stuttgart"}, indent=2))

# ============================================================
# 3. Temperature & Sampling Parameter
# ============================================================
print("\n" + "=" * 60)
print("3. TEMPERATURE & SAMPLING PARAMETER")
print("=" * 60)

print("""
  TEMPERATURE (0.0 - 2.0)
  ─────────────────────────────
  0.0  Deterministisch   Immer die wahrscheinlichste Antwort
  0.3  Niedrig           Fokussiert, konsistent, faktisch
  0.7  Mittel            Gute Balance (Standard bei den meisten LLMs)
  1.0  Hoch              Kreativer, vielfaeltiger
  1.5+ Sehr hoch         Chaotisch, oft unsinnig

  TOP_P (0.0 - 1.0) "Nucleus Sampling"
  ─────────────────────────────
  0.1  Nur die Top-10% wahrscheinlichsten Tokens
  0.5  Top-50%
  0.9  Top-90% (Standard)
  1.0  Alle Tokens moeglich

  EMPFEHLUNGEN:
  ─────────────────────────────
  Aufgabe                    Temp    Top_P
  Code-Generierung           0.1     0.9
  Daten-Extraktion (JSON)    0.0     1.0
  Kreatives Schreiben        0.9     0.95
  Chat / Konversation        0.7     0.9
  Mathe / Logik              0.0     1.0
  Brainstorming              1.0     0.95
""")

# ============================================================
# 4. Prompt-Optimierung (Anti-Patterns)
# ============================================================
print("=" * 60)
print("4. PROMPT DOS AND DON'TS")
print("=" * 60)

dos_donts = [
    ("SCHLECHT", "Erklaere mir ML",
     "GUT", "Erklaere Machine Learning in 3 Saetzen fuer jemanden ohne Vorkenntnisse"),
    ("SCHLECHT", "Schreib Code",
     "GUT", "Schreibe eine Python-Funktion die eine Liste von Zahlen sortiert. Nutze den Bubble-Sort Algorithmus."),
    ("SCHLECHT", "Analysiere das",
     "GUT", "Analysiere den folgenden Verkaufsbericht. Nenne die 3 wichtigsten Trends und 2 Risiken. Antworte als Bullet Points."),
    ("SCHLECHT", "Fasse zusammen",
     "GUT", "Fasse den Text in maximal 3 Saetzen zusammen. Behalte die wichtigsten Zahlen bei. Schreibe auf Deutsch."),
]

for bad_label, bad, good_label, good in dos_donts:
    print(f"\n  [X] {bad_label}: \"{bad}\"")
    print(f"  [OK] {good_label}: \"{good}\"")

print("""
  GOLDENE REGELN:
  1. Sei SPEZIFISCH (Format, Laenge, Sprache, Zielgruppe)
  2. Gib KONTEXT (Rolle, Hintergrund, Einschraenkungen)
  3. Zeige BEISPIELE (Few-Shot > Zero-Shot fuer komplexe Aufgaben)
  4. Definiere das OUTPUT-FORMAT (JSON, Liste, Tabelle)
  5. Setze GRENZEN ("maximal 3 Saetze", "nur die Top-5")
""")

# ============================================================
# 5. Prompt Chaining (Multi-Step)
# ============================================================
print("=" * 60)
print("5. PROMPT CHAINING - Komplexe Aufgaben aufteilen")
print("=" * 60)

print("""
  Statt EINE riesige Aufgabe:
    "Recherchiere, analysiere und schreibe einen Bericht ueber..."

  BESSER: Aufteilen in Schritte (Chain):

    Schritt 1: RECHERCHE
    "Nenne die 5 wichtigsten Fakten zu [Thema]"
                    |
                    v
    Schritt 2: ANALYSE
    "Gegeben diese Fakten: [Ergebnis Schritt 1]
     Was sind die 3 wichtigsten Erkenntnisse?"
                    |
                    v
    Schritt 3: SYNTHESE
    "Schreibe basierend auf diesen Erkenntnissen: [Ergebnis Schritt 2]
     einen kurzen Bericht (max 200 Woerter)."

  VORTEILE:
  - Jeder Schritt kann geprueft werden
  - Fehler werden frueh erkannt
  - Qualitaet ist hoeher
  - Einzelne Schritte sind wiederverwendbar
""")

# Demo: Chaining simulieren
print("--- Demo: Prompt Chain fuer Produktanalyse ---\n")

# Simulierte Antworten (normalerweise vom LLM)
chain_steps = [
    {
        "prompt": "Extrahiere alle Produkteigenschaften aus der Bewertung:\n'Der Laptop ist schnell, leicht und hat eine tolle Tastatur. Der Akku haelt nur 3 Stunden.'",
        "antwort": "Positiv: schnell, leicht, tolle Tastatur | Negativ: Akku nur 3 Stunden",
    },
    {
        "prompt": "Bewerte jede Eigenschaft auf einer Skala 1-5:\nPositiv: schnell, leicht, tolle Tastatur | Negativ: Akku nur 3 Stunden",
        "antwort": "Geschwindigkeit: 5/5, Gewicht: 4/5, Tastatur: 5/5, Akku: 2/5 -> Gesamt: 4.0/5",
    },
    {
        "prompt": "Schreibe eine Kaufempfehlung basierend auf: Gesamt 4.0/5, Staerke: Performance+Tastatur, Schwaeche: Akku",
        "antwort": "Empfehlung: Kaufen, wenn Performance wichtiger ist als Akkulaufzeit. Ideal fuers Buero, weniger fuer unterwegs.",
    },
]

for i, step in enumerate(chain_steps, 1):
    print(f"  Schritt {i}:")
    print(f"    Prompt: {step['prompt'][:80]}...")
    print(f"    -> {step['antwort']}")
    print()

# ============================================================
# 6. System Prompts
# ============================================================
print("=" * 60)
print("6. SYSTEM PROMPTS")
print("=" * 60)

system_prompts = {
    "Code Assistant": {
        "system": "Du bist ein Python-Experte. Antworte immer mit funktionierendem Code. Fuege Kommentare hinzu. Nutze Type Hints.",
        "nutzen": "Konsistente Code-Qualitaet",
    },
    "Daten-Analyst": {
        "system": "Du bist ein Daten-Analyst. Antworte immer mit: 1) Zusammenfassung, 2) Key Findings, 3) Empfehlungen. Nutze Zahlen wo moeglich.",
        "nutzen": "Strukturierte Analysen",
    },
    "Lehrer": {
        "system": "Du bist ein geduldiger Lehrer. Erklaere Konzepte mit Analogien aus dem Alltag. Frage am Ende ob etwas unklar ist. Nutze einfache Sprache.",
        "nutzen": "Verstaendliche Erklaerungen",
    },
    "Kritischer Reviewer": {
        "system": "Du bist ein strenger aber fairer Code-Reviewer. Finde Bugs, Security-Issues und Verbesserungen. Bewerte Schwere: CRITICAL, HIGH, MEDIUM, LOW.",
        "nutzen": "Gruendliche Code-Reviews",
    },
}

for name, info in system_prompts.items():
    print(f"\n  [{name}]")
    print(f"  System: \"{info['system'][:80]}...\"")
    print(f"  Nutzen: {info['nutzen']}")

# ============================================================
# 7. Testen mit einem lokalen Modell (optional)
# ============================================================
print("\n" + "=" * 60)
print("7. PROMPTS TESTEN")
print("=" * 60)

try:
    from transformers import pipeline
    print("Teste Prompts mit lokalem Modell (distilgpt2)...\n")

    generator = pipeline("text-generation", model="distilgpt2", device=-1)

    test_prompts = [
        ("Zero-Shot", "Question: What is machine learning?\nAnswer:"),
        ("Role", "As a Python expert, explain what a decorator is:\n"),
        ("Structured", "List the top 3 programming languages:\n1."),
    ]

    for name, prompt in test_prompts:
        result = generator(prompt, max_new_tokens=40, temperature=0.7,
                          do_sample=True, pad_token_id=50256)
        text = result[0]["generated_text"][len(prompt):].strip().split("\n")[0]
        print(f"  [{name}] {prompt[:50]}...")
        print(f"    -> {text[:80]}")
        print()

    print("  * distilgpt2 ist winzig - echte LLMs (Mistral, Llama) sind VIEL besser!")

except Exception as e:
    print(f"  Modell nicht verfuegbar: {e}")
    print("  Kein Problem - die Konzepte gelten fuer alle LLMs.")

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. EIGENES TEMPLATE
   Erstelle ein PromptTemplate fuer "Email schreiben":
   Variablen: empfaenger, betreff, ton (formell/informell), kernaussage

2. FEW-SHOT EXPERIMENT
   Schreibe einen Few-Shot Prompt fuer Named Entity Recognition:
   "Berlin ist die Hauptstadt" -> {entities: [{text: "Berlin", type: "LOCATION"}]}
   Teste mit 3, 5 und 10 Beispielen. Wird es besser?

3. PROMPT VERGLEICH
   Nimm eine Aufgabe und schreibe 3 verschiedene Prompts dafuer.
   Teste alle mit dem gleichen LLM. Welcher ist am besten?

4. CHAIN BAUEN
   Baue eine 3-Schritt Prompt Chain fuer:
   Text -> Zusammenfassung -> Stichpunkte -> Tweet
   Jeder Schritt nutzt das Ergebnis des vorherigen.

5. ANTI-HALLUZINATION
   Teste diesen Prompt-Trick: "Wenn du die Antwort nicht sicher weisst,
   sage 'Ich bin nicht sicher' statt zu raten."
   Vergleiche mit/ohne diesen Zusatz bei Wissensfragen.
""")
