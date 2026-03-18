import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
02 - KI-Agent mit Tool Use - Standalone Version
=================================================
Ein Agent = LLM + Tools + Loop
Das LLM entscheidet SELBST, welches Tool es wann aufruft.
So funktionieren Claude, ChatGPT Plugins, etc.

Funktioniert komplett OFFLINE ohne externen Server!
Benoetigt: pip install transformers torch
"""

import json
import math
import random
import re
import time
from datetime import datetime

# ============================================================
# 1. Tools definieren
# ============================================================
# Jedes Tool hat: Name, Beschreibung, Parameter, Funktion

print("=" * 60)
print("  KI-Agent mit Tool Use - Standalone")
print("=" * 60)


def tool_rechner(ausdruck: str) -> str:
    """Sichere Berechnung eines mathematischen Ausdrucks."""
    # Nur sichere Operationen erlauben
    erlaubt = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "pow": pow,
    }
    try:
        ergebnis = eval(ausdruck, {"__builtins__": {}}, erlaubt)
        return f"{ausdruck} = {ergebnis}"
    except Exception as e:
        return f"Fehler bei Berechnung '{ausdruck}': {e}"


def tool_datum_zeit(abfrage: str = "") -> str:
    """Gibt aktuelle Datum/Zeit-Informationen zurueck."""
    jetzt = datetime.now()
    wochentage = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
                  "Freitag", "Samstag", "Sonntag"]
    monate = ["Januar", "Februar", "Maerz", "April", "Mai", "Juni",
              "Juli", "August", "September", "Oktober", "November", "Dezember"]

    wochentag = wochentage[jetzt.weekday()]
    monat = monate[jetzt.month - 1]

    info = {
        "datum": f"{wochentag}, {jetzt.day}. {monat} {jetzt.year}",
        "uhrzeit": jetzt.strftime("%H:%M:%S"),
        "kalenderwoche": jetzt.isocalendar()[1],
        "tag_im_jahr": jetzt.timetuple().tm_yday,
        "unix_timestamp": int(jetzt.timestamp()),
    }

    if "woche" in abfrage.lower() or "kw" in abfrage.lower():
        return f"Aktuelle Kalenderwoche: KW {info['kalenderwoche']}"
    elif "uhr" in abfrage.lower() or "zeit" in abfrage.lower():
        return f"Aktuelle Uhrzeit: {info['uhrzeit']}"
    else:
        return (f"Datum: {info['datum']}, Uhrzeit: {info['uhrzeit']}, "
                f"KW: {info['kalenderwoche']}, Tag im Jahr: {info['tag_im_jahr']}")


def tool_einheiten(wert: float, von: str, nach: str) -> str:
    """Konvertiert zwischen gaengigen Einheiten."""
    # Alle Umrechnungen relativ zu einer Basiseinheit pro Kategorie
    umrechnungen = {
        # Laenge -> Meter
        "km": ("laenge", 1000), "m": ("laenge", 1), "cm": ("laenge", 0.01),
        "mm": ("laenge", 0.001), "meile": ("laenge", 1609.344),
        "yard": ("laenge", 0.9144), "fuss": ("laenge", 0.3048),
        "zoll": ("laenge", 0.0254),
        # Gewicht -> Kilogramm
        "t": ("gewicht", 1000), "kg": ("gewicht", 1), "g": ("gewicht", 0.001),
        "mg": ("gewicht", 0.000001), "pfund": ("gewicht", 0.453592),
        "unze": ("gewicht", 0.0283495),
        # Temperatur (Sonderbehandlung)
        "celsius": ("temperatur", None), "fahrenheit": ("temperatur", None),
        "kelvin": ("temperatur", None),
    }

    von_l = von.lower()
    nach_l = nach.lower()

    if von_l not in umrechnungen or nach_l not in umrechnungen:
        bekannt = ", ".join(sorted(umrechnungen.keys()))
        return f"Unbekannte Einheit. Bekannte Einheiten: {bekannt}"

    kat_von = umrechnungen[von_l][0]
    kat_nach = umrechnungen[nach_l][0]

    if kat_von != kat_nach:
        return f"Kann nicht von {von} ({kat_von}) nach {nach} ({kat_nach}) umrechnen."

    # Temperatur - Sonderfall
    if kat_von == "temperatur":
        # Zuerst alles nach Celsius
        if von_l == "fahrenheit":
            celsius = (wert - 32) * 5 / 9
        elif von_l == "kelvin":
            celsius = wert - 273.15
        else:
            celsius = wert

        # Dann von Celsius zum Ziel
        if nach_l == "fahrenheit":
            ergebnis = celsius * 9 / 5 + 32
        elif nach_l == "kelvin":
            ergebnis = celsius + 273.15
        else:
            ergebnis = celsius
    else:
        # Lineare Umrechnung: Wert -> Basis -> Ziel
        faktor_von = umrechnungen[von_l][1]
        faktor_nach = umrechnungen[nach_l][1]
        ergebnis = wert * faktor_von / faktor_nach

    return f"{wert} {von} = {ergebnis:.4g} {nach}"


def tool_zufallsfakt() -> str:
    """Gibt einen zufaelligen Fakt ueber KI/ML zurueck."""
    fakten = [
        "Der Begriff 'Kuenstliche Intelligenz' wurde 1956 auf der Dartmouth-Konferenz gespraegt.",
        "GPT-3 hat 175 Milliarden Parameter und wurde auf 499 Milliarden Tokens trainiert.",
        "Das menschliche Gehirn hat ca. 86 Milliarden Neuronen - aehnlich wie grosse LLMs Parameter haben.",
        "AlphaGo besiegte 2016 den Go-Weltmeister Lee Sedol 4:1. Go hat mehr moegliche Stellungen als Atome im Universum.",
        "MNIST, der klassische Handschrift-Datensatz, wurde 1998 veroeffentlicht und wird immer noch als Benchmark genutzt.",
        "Der erste Chatbot ELIZA wurde 1966 am MIT entwickelt und simulierte einen Psychotherapeuten.",
        "Transformer brauchen O(n^2) Speicher fuer die Attention - bei 100k Tokens wird das zum Problem.",
        "Das Training von GPT-4 hat geschaetzt mehrere 100 Millionen Dollar gekostet.",
        "Die Backpropagation-Methode wurde bereits 1986 von Rumelhart, Hinton und Williams populaer gemacht.",
        "Ein einzelner GPU-Trainingsrun fuer ein grosses LLM verbraucht so viel Strom wie 100 Haushalte im Jahr.",
    ]
    return random.choice(fakten)


# Tool-Registry
TOOLS = {
    "rechner": {
        "beschreibung": "Berechnet mathematische Ausdruecke (z.B. '1547 * 283', 'sqrt(144)')",
        "parameter": {"ausdruck": "str - Mathematischer Ausdruck"},
        "funktion": lambda args: tool_rechner(args.get("ausdruck", "")),
    },
    "datum_zeit": {
        "beschreibung": "Gibt aktuelle Datum/Zeit-Informationen zurueck",
        "parameter": {"abfrage": "str - Optional: 'uhrzeit', 'woche', oder leer fuer alles"},
        "funktion": lambda args: tool_datum_zeit(args.get("abfrage", "")),
    },
    "einheiten": {
        "beschreibung": "Konvertiert zwischen Einheiten (Laenge, Gewicht, Temperatur)",
        "parameter": {"wert": "float", "von": "str - Quell-Einheit", "nach": "str - Ziel-Einheit"},
        "funktion": lambda args: tool_einheiten(
            float(args.get("wert", 0)), args.get("von", ""), args.get("nach", "")
        ),
    },
    "zufallsfakt": {
        "beschreibung": "Gibt einen zufaelligen Fakt ueber KI/ML zurueck",
        "parameter": {},
        "funktion": lambda args: tool_zufallsfakt(),
    },
}


# ============================================================
# 2. ReAct Pattern - Simulierte Demonstration
# ============================================================

print("\n" + "=" * 60)
print("[1/3] Das ReAct Pattern (Reasoning + Acting)")
print("=" * 60)
print("""
  So denken moderne KI-Agents (Claude, ChatGPT, etc.):

  Der Loop: Denken -> Handeln -> Beobachten -> Denken -> ... -> Antworten
""")


def demonstriere_react():
    """Zeigt Schritt fuer Schritt, wie ein ReAct-Agent denkt."""

    print("  Beispiel: 'Wie viele Sekunden hat ein Jahr und wie viel ist das in Stunden?'")
    print("  " + "-" * 56)

    schritte = [
        ("Thought", "Ich muss zuerst berechnen, wie viele Sekunden ein Jahr hat.\n"
                     "             Ein Jahr = 365 Tage * 24 Stunden * 60 Minuten * 60 Sekunden."),
        ("Action",  'rechner(ausdruck="365 * 24 * 60 * 60")'),
        ("Result",  tool_rechner("365 * 24 * 60 * 60")),
        ("Thought", "OK, 31536000 Sekunden. Jetzt muss ich das in Stunden umrechnen.\n"
                     "             31536000 / 3600 = Stunden."),
        ("Action",  'rechner(ausdruck="31536000 / 3600")'),
        ("Result",  tool_rechner("31536000 / 3600")),
        ("Thought", "Ich habe beide Ergebnisse. Ich kann jetzt antworten."),
        ("Answer",  "Ein Jahr hat 31.536.000 Sekunden, das sind 8.760 Stunden."),
    ]

    for typ, inhalt in schritte:
        if typ == "Thought":
            print(f"\n  [Denken]    {inhalt}")
        elif typ == "Action":
            print(f"  [Handeln]   -> {inhalt}")
        elif typ == "Result":
            print(f"  [Ergebnis]  <- {inhalt}")
        elif typ == "Answer":
            print(f"\n  [Antwort]   {inhalt}")
        time.sleep(0.3)  # Kurze Pause fuer Lesbarkeit


demonstriere_react()


# ============================================================
# 3. Regelbasierter Agent (funktioniert ohne LLM)
# ============================================================

print("\n\n" + "=" * 60)
print("[2/3] Regelbasierter Agent (kein LLM noetig)")
print("=" * 60)
print("  Dieser Agent erkennt Muster in der Eingabe und waehlt das passende Tool.\n")


def regelbasierter_agent(eingabe: str) -> str:
    """Ein einfacher Agent, der Eingaben parst und Tools aufruft.

    Erkennt Muster wie:
    - Rechenaufgaben: "Was ist 5 * 3?", "Berechne sqrt(16)"
    - Datum/Zeit: "Welcher Tag ist heute?", "Wie spaet ist es?"
    - Einheiten: "Rechne 5 km in meilen um", "100 celsius in fahrenheit"
    - Fakten: "Erzaehl mir was", "Zufallsfakt"
    """
    eingabe_lower = eingabe.lower().strip()

    print(f"  Agent empfaengt: '{eingabe}'")

    # Schritt 1: Intent erkennen
    intent = None
    tool_args = {}

    # Muster: Einheiten-Umrechnung
    # "rechne 5 km in meilen um" oder "100 celsius in fahrenheit"
    einheiten_pattern = r"(?:rechne|konvertiere|wandle)?\s*(\d+(?:\.\d+)?)\s*(\w+)\s+(?:in|nach|zu)\s+(\w+)"
    match = re.search(einheiten_pattern, eingabe_lower)
    if match:
        intent = "einheiten"
        tool_args = {"wert": match.group(1), "von": match.group(2), "nach": match.group(3)}

    # Muster: Rechenaufgabe
    if not intent:
        rechen_pattern = r"(?:was ist|berechne|rechne|wie viel ist)\s+(.+?)(?:\?|$)"
        match = re.search(rechen_pattern, eingabe_lower)
        if match:
            intent = "rechner"
            tool_args = {"ausdruck": match.group(1).strip()}
        elif re.search(r"[\d]+\s*[\+\-\*\/\%\^]+\s*[\d]+", eingabe_lower):
            # Direkter mathematischer Ausdruck
            ausdruck = re.search(r"([\d\.\+\-\*\/\%\^\(\)\s]+)", eingabe_lower).group(1).strip()
            intent = "rechner"
            tool_args = {"ausdruck": ausdruck}

    # Muster: Datum/Zeit
    if not intent:
        zeit_keywords = ["uhrzeit", "zeit", "uhr", "datum", "tag", "heute",
                         "wochentag", "kalenderwoche", "kw"]
        if any(kw in eingabe_lower for kw in zeit_keywords):
            intent = "datum_zeit"
            tool_args = {"abfrage": eingabe_lower}

    # Muster: Zufallsfakt
    if not intent:
        fakt_keywords = ["fakt", "fact", "erzaehl", "wusstest", "zufaell", "interessant"]
        if any(kw in eingabe_lower for kw in fakt_keywords):
            intent = "zufallsfakt"
            tool_args = {}

    # Schritt 2: Tool aufrufen
    if intent and intent in TOOLS:
        print(f"  -> Intent erkannt: {intent}")
        print(f"  -> Argumente: {tool_args}")
        ergebnis = TOOLS[intent]["funktion"](tool_args)
        print(f"  -> Tool-Ergebnis: {ergebnis}")
        return ergebnis
    else:
        # Kein passendes Tool gefunden
        verfuegbar = ", ".join(TOOLS.keys())
        return (f"Konnte die Eingabe nicht zuordnen. "
                f"Verfuegbare Tools: {verfuegbar}. "
                f"Versuche z.B.: 'Was ist 5 * 3?', 'Welcher Tag ist heute?', "
                f"'Rechne 100 celsius in fahrenheit um', 'Erzaehl mir einen Fakt'")


# Regelbasierten Agent testen
test_eingaben = [
    "Was ist 1547 * 283?",
    "Welcher Tag ist heute?",
    "Rechne 100 celsius in fahrenheit um",
    "Erzaehl mir einen interessanten Fakt",
    "Berechne sqrt(144) + 10",
    "42 km in meilen",
    "Wie spaet ist es?",
]

for eingabe in test_eingaben:
    print()
    ergebnis = regelbasierter_agent(eingabe)
    print(f"  => {ergebnis}")
    print("  " + "-" * 56)


# ============================================================
# 4. LLM-basierter Agent (optional, mit Qwen2.5-0.5B)
# ============================================================

print("\n" + "=" * 60)
print("[3/3] LLM-basierter Agent (Qwen2.5-0.5B, optional)")
print("=" * 60)


def llm_agent(eingabe: str, max_schritte: int = 3) -> str:
    """Agent der ein lokales LLM als 'Gehirn' nutzt.

    Versucht Qwen2.5-0.5B zu laden. Wenn das nicht klappt,
    faellt er auf den regelbasierten Agent zurueck.
    """
    print(f"\n  LLM-Agent empfaengt: '{eingabe}'")

    try:
        from transformers import pipeline as hf_pipeline

        print("  Lade Qwen2.5-0.5B (beim ersten Mal ~500MB Download)...")
        t0 = time.time()

        generator = hf_pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B",
            device_map="auto",
            torch_dtype="auto",
        )
        print(f"  Modell geladen in {time.time() - t0:.1f}s")

        # Tool-Beschreibungen fuer den Prompt
        tool_desc = "\n".join(
            f"  - {name}: {info['beschreibung']}"
            for name, info in TOOLS.items()
        )

        prompt = (
            f"Du bist ein hilfreicher Assistent. Du hast folgende Tools:\n"
            f"{tool_desc}\n\n"
            f"Um ein Tool zu nutzen, antworte mit JSON: "
            f'{{"tool": "name", "args": {{"key": "value"}}}}\n'
            f"Wenn du kein Tool brauchst, antworte direkt.\n\n"
            f"Frage: {eingabe}\n"
            f"Antwort:"
        )

        t0 = time.time()
        output = generator(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.2,
            return_full_text=False,
        )
        antwort = output[0]["generated_text"].strip()
        print(f"  LLM-Antwort ({time.time() - t0:.1f}s): {antwort[:200]}")

        # Versuche Tool-Aufruf aus der Antwort zu extrahieren
        try:
            # Suche nach JSON in der Antwort
            json_match = re.search(r'\{[^}]+\}', antwort)
            if json_match:
                parsed = json.loads(json_match.group())
                if "tool" in parsed and parsed["tool"] in TOOLS:
                    tool_name = parsed["tool"]
                    tool_args = parsed.get("args", {})
                    print(f"  -> LLM will Tool nutzen: {tool_name}({tool_args})")
                    ergebnis = TOOLS[tool_name]["funktion"](tool_args)
                    print(f"  -> Tool-Ergebnis: {ergebnis}")
                    return ergebnis
        except (json.JSONDecodeError, TypeError):
            pass

        return antwort

    except Exception as e:
        print(f"  [!] LLM nicht verfuegbar: {e}")
        print("  -> Fallback auf regelbasierten Agent")
        return regelbasierter_agent(eingabe)


# LLM-Agent testen (faellt auf regelbasiert zurueck wenn kein Modell verfuegbar)
print("\n  Teste LLM-Agent mit einer Frage...")
ergebnis = llm_agent("Was ist 25 * 17?")
print(f"\n  Endergebnis: {ergebnis}")


# ============================================================
# 5. Zusammenfassung
# ============================================================

print("\n\n" + "=" * 60)
print("  Zusammenfassung")
print("=" * 60)
print("""
  Was wir gebaut haben:

  1. Tool-Registry: 4 Tools (Rechner, Datum/Zeit, Einheiten, Zufallsfakt)
     -> Jedes Tool hat Name, Beschreibung, Parameter, Funktion

  2. ReAct-Pattern (simuliert):
     -> Denken -> Handeln -> Beobachten -> Denken -> Antworten
     -> So funktionieren Claude, ChatGPT und andere Agents

  3. Regelbasierter Agent:
     -> Erkennt Muster in der Eingabe (Regex, Keywords)
     -> Waehlt passendes Tool und ruft es auf
     -> Funktioniert IMMER, kein LLM noetig

  4. LLM-basierter Agent (optional):
     -> Nutzt Qwen2.5-0.5B als "Gehirn"
     -> LLM entscheidet welches Tool aufgerufen wird
     -> Fallback auf regelbasierten Agent wenn kein LLM verfuegbar
""")


# ============================================================
# UEBUNGEN
# ============================================================

print("=" * 60)
print("  Uebungen")
print("=" * 60)
print("""
  Uebung 1: Fuege ein neues Tool hinzu
  ----------------------------------------
  Erstelle ein Tool "wuerfeln" das N Wuerfel mit S Seiten wirft.
  Schritte:
    a) Schreibe eine Funktion tool_wuerfeln(anzahl, seiten)
    b) Fuege es zur TOOLS-Registry hinzu
    c) Erweitere den regelbasierten Agent um das Muster zu erkennen
       z.B. "Wirf 3 Wuerfel mit 6 Seiten"
  Tipp: random.randint(1, seiten) fuer jeden Wurf

  Uebung 2: Erweitere den Agent um Gedaechtnis
  ------------------------------------------------
  Der Agent vergisst nach jeder Frage alles.
  Fuege eine Liste hinzu, die vorherige Fragen und Antworten speichert:
    verlauf = []
    # Nach jeder Frage:
    verlauf.append({"frage": eingabe, "antwort": ergebnis})
  Bonus: Erkenne "Was war meine letzte Frage?" als spezielles Muster.

  Uebung 3: Verkette Tools (Multi-Step)
  ----------------------------------------
  Manchmal braucht man mehrere Tools hintereinander.
  Beispiel: "Wie viele Meilen sind 1000 * 3.5 Kilometer?"
    1. Erst rechner: 1000 * 3.5 = 3500
    2. Dann einheiten: 3500 km in meilen
  Schreibe eine Funktion multi_step_agent(), die Ergebnisse
  eines Tools als Input fuer das naechste nutzt.

  Uebung 4: Fehlerbehandlung
  -----------------------------
  Was passiert wenn ein Tool fehlschlaegt?
    - "Berechne abc * xyz" -> Fehler im Rechner
    - "Rechne 5 bananen in aepfel um" -> Unbekannte Einheit
  Erweitere den Agent so, dass er bei Fehlern:
    a) Den Fehler dem Nutzer erklaert
    b) Einen Vorschlag macht, wie die Eingabe korrigiert werden kann

  Uebung 5: Interaktiver Modus
  --------------------------------
  Baue eine input()-Schleife, damit der Agent interaktiv nutzbar ist:
    while True:
        eingabe = input("Du: ")
        if eingabe.lower() in ("quit", "exit", "q"):
            break
        ergebnis = regelbasierter_agent(eingabe)
        print(f"Agent: {ergebnis}")
""")
