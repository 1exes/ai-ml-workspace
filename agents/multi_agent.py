import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Multi-Agent System
===================
3 Agenten arbeiten zusammen: Researcher -> Analyst -> Writer.
Jeder hat eine Rolle und gibt sein Ergebnis an den naechsten weiter.
"""

import time
from dataclasses import dataclass, field

# ============================================================
# 1. Agent-Infrastruktur
# ============================================================

@dataclass
class Message:
    """Nachricht zwischen Agenten."""
    sender: str
    receiver: str
    content: str
    msg_type: str = "data"  # data, request, response


class MessageBus:
    """Einfacher Message Bus fuer Agent-Kommunikation."""

    def __init__(self):
        self.messages: list[Message] = []
        self.log: list[str] = []

    def send(self, msg: Message):
        self.messages.append(msg)
        self.log.append(f"  [{msg.sender}] -> [{msg.receiver}]: {msg.content[:60]}...")

    def get_messages(self, receiver: str) -> list[Message]:
        return [m for m in self.messages if m.receiver == receiver]


class Agent:
    """Basis-Agent mit Rolle und Wissensbasis."""

    def __init__(self, name: str, role: str, bus: MessageBus):
        self.name = name
        self.role = role
        self.bus = bus
        self.memory: list[str] = []

    def think(self, input_data: str) -> str:
        raise NotImplementedError

    def process(self, input_data: str) -> str:
        print(f"\n  --- {self.name} ({self.role}) ---")
        print(f"  Input: {input_data[:80]}...")
        result = self.think(input_data)
        self.memory.append(result)
        print(f"  Output: {result[:80]}...")
        return result


# ============================================================
# 2. Spezialisierte Agenten
# ============================================================

class ResearcherAgent(Agent):
    """Sammelt Informationen aus der Wissensbasis."""

    KNOWLEDGE = {
        "python": {
            "typ": "Programmiersprache",
            "vorteile": ["Einfach zu lernen", "Riesiges Oekosystem", "ML/KI Standard", "Cross-Platform"],
            "nachteile": ["Langsam (interpretiert)", "GIL limitiert Threading", "Hoher Speicherverbrauch"],
            "einsatz": ["Web (Django/Flask)", "Data Science", "KI/ML", "Automation"],
            "performance": "10-100x langsamer als C/Rust",
            "community": "Groesste Community weltweit",
        },
        "rust": {
            "typ": "Systemprogrammiersprache",
            "vorteile": ["Extrem schnell", "Memory Safety ohne GC", "Keine Data Races", "Modernes Typsystem"],
            "nachteile": ["Steile Lernkurve", "Kleineres Oekosystem", "Laengere Kompilierzeit", "Borrow Checker frustrierend"],
            "einsatz": ["Systemprogrammierung", "WebAssembly", "CLI Tools", "Embedded"],
            "performance": "Vergleichbar mit C/C++",
            "community": "Schnell wachsend, sehr hilfreich",
        },
        "javascript": {
            "typ": "Skriptsprache",
            "vorteile": ["Laeuft ueberall (Browser)", "Riesiges NPM Oekosystem", "Async-First", "Full-Stack moeglich"],
            "nachteile": ["Quirky Typsystem", "Callback Hell", "Fragmentierung", "Security-Risiken in NPM"],
            "einsatz": ["Frontend (React/Vue)", "Backend (Node.js)", "Mobile (React Native)", "Desktop (Electron)"],
            "performance": "Schneller als Python (V8 JIT)",
            "community": "Groesste Anzahl Pakete (NPM)",
        },
        "ki_entwicklung": {
            "python_anteil": "90%+ aller KI-Projekte",
            "frameworks": ["PyTorch", "TensorFlow", "Hugging Face", "scikit-learn"],
            "trend": "Python dominiert, Rust waechst fuer Inference",
        },
    }

    def think(self, query: str) -> str:
        query_lower = query.lower()
        findings = []

        for topic, info in self.KNOWLEDGE.items():
            if any(kw in query_lower for kw in topic.split("_") + [topic]):
                findings.append(f"[{topic.upper()}] {info}")

        # Spezifische Suche
        if "python" in query_lower and "rust" in query_lower:
            findings.append("[VERGLEICH] Python vs Rust: Python fuer Prototyping/ML, Rust fuer Performance-kritische Teile")

        if not findings:
            findings.append("[ALLGEMEIN] Keine spezifischen Daten gefunden, nutze Allgemeinwissen")

        return "RECHERCHE-ERGEBNIS:\n" + "\n".join(str(f) for f in findings)


class AnalystAgent(Agent):
    """Bewertet und analysiert die gesammelten Informationen."""

    def think(self, research_data: str) -> str:
        lines = research_data.split("\n")
        pros = []
        cons = []
        scores = {}

        # Informationen extrahieren und bewerten
        for line in lines:
            if "vorteile" in line.lower():
                pros.append(line)
            if "nachteile" in line.lower():
                cons.append(line)

        # Scoring basierend auf Kriterien
        criteria = {
            "Lernkurve": {"python": 9, "rust": 4, "javascript": 7},
            "Performance": {"python": 3, "rust": 10, "javascript": 6},
            "Oekosystem": {"python": 9, "rust": 6, "javascript": 10},
            "Job-Markt": {"python": 9, "rust": 7, "javascript": 10},
            "Zukunft KI": {"python": 10, "rust": 7, "javascript": 4},
        }

        analysis = "ANALYSE:\n"
        analysis += "\nBewertungsmatrix (1-10):\n"
        analysis += f"{'Kriterium':<20s} {'Python':>8s} {'Rust':>8s} {'JS':>8s}\n"
        analysis += "-" * 46 + "\n"

        totals = {"python": 0, "rust": 0, "javascript": 0}
        for criterion, scores in criteria.items():
            analysis += f"{criterion:<20s}"
            for lang in ["python", "rust", "javascript"]:
                analysis += f" {scores[lang]:>7d}"
                totals[lang] += scores[lang]
            analysis += "\n"

        analysis += "-" * 46 + "\n"
        analysis += f"{'GESAMT':<20s}"
        for lang in ["python", "rust", "javascript"]:
            analysis += f" {totals[lang]:>7d}"
        analysis += "\n"

        winner = max(totals, key=totals.get)
        analysis += f"\nFazit: {winner.capitalize()} hat den hoechsten Gesamtscore ({totals[winner]}/50)"
        analysis += f"\nKontext: Fuer KI-Entwicklung ist Python klar vorne."
        analysis += f"\nNuance: Die 'beste' Sprache haengt vom Use-Case ab."

        return analysis


class WriterAgent(Agent):
    """Erstellt einen strukturierten Bericht."""

    def think(self, analysis: str) -> str:
        report = """
========================================
       BERICHT: Sprachvergleich
========================================

ZUSAMMENFASSUNG
Der Vergleich zeigt, dass jede Sprache ihre Staerken hat.
Fuer KI-Entwicklung ist Python der klare Standard.

EMPFEHLUNGEN
1. Einsteiger in KI/ML: Python (kein Zweifel)
2. Performance-kritische Inference: Rust
3. Web-Integration von KI: JavaScript/TypeScript
4. Ideal-Kombination: Python + Rust (Prototyp + Produktion)

DETAILS
"""
        # Analyse einbetten
        for line in analysis.split("\n"):
            if line.strip():
                report += f"  {line}\n"

        report += """
NAECHSTE SCHRITTE
- Python Grundlagen festigen (numpy, pandas)
- PyTorch / Hugging Face lernen
- Spaeter: Rust fuer Performance-Optimierung
========================================"""
        return report


# ============================================================
# 3. Multi-Agent Workflow ausfuehren
# ============================================================
print("=" * 60)
print("MULTI-AGENT SYSTEM: 3 Agenten arbeiten zusammen")
print("=" * 60)

bus = MessageBus()
researcher = ResearcherAgent("Researcher", "Informationssammlung", bus)
analyst = AnalystAgent("Analyst", "Bewertung & Scoring", bus)
writer = WriterAgent("Writer", "Berichterstellung", bus)

query = "Analysiere Python vs Rust vs JavaScript fuer KI-Entwicklung"
print(f"\nAuftrag: \"{query}\"")
print(f"Pipeline: Researcher -> Analyst -> Writer\n")

# Schritt 1: Research
research = researcher.process(query)
bus.send(Message("Researcher", "Analyst", research))

# Schritt 2: Analyse
analysis = analyst.process(research)
bus.send(Message("Analyst", "Writer", analysis))

# Schritt 3: Bericht
report = writer.process(analysis)
bus.send(Message("Writer", "User", report))

# ============================================================
# 4. Ergebnis
# ============================================================
print("\n" + "=" * 60)
print("FINALER BERICHT")
print("=" * 60)
print(report)

# ============================================================
# 5. Message-Log
# ============================================================
print("\n" + "=" * 60)
print("MESSAGE LOG (Agent-Kommunikation)")
print("=" * 60)
for entry in bus.log:
    print(entry)

# ============================================================
# 6. Konsens-Mechanismus
# ============================================================
print("\n" + "=" * 60)
print("6. KONSENS - Wenn Agenten sich nicht einig sind")
print("=" * 60)

print("""
Szenario: 3 Agenten bewerten "Beste Sprache fuer Web-Scraping"

  Agent 1 (Python-Fan):  "Python! Requests + BeautifulSoup ist perfekt."
  Agent 2 (JS-Fan):      "JavaScript! Puppeteer kann dynamische Seiten."
  Agent 3 (Pragmatiker): "Kommt drauf an: statisch -> Python, dynamisch -> JS."
""")

# Voting
votes = {
    "Python": {"Agent 1": 10, "Agent 2": 5, "Agent 3": 7},
    "JavaScript": {"Agent 1": 4, "Agent 2": 10, "Agent 3": 8},
    "Rust": {"Agent 1": 2, "Agent 2": 1, "Agent 3": 3},
}

print("Abstimmung (Confidence 1-10):")
for option, agent_votes in votes.items():
    total = sum(agent_votes.values())
    bar = "#" * (total // 2)
    print(f"  {option:<12s} {total:>3d}/30 {bar}")
    for agent, vote in agent_votes.items():
        print(f"    {agent}: {vote}")

winner = max(votes, key=lambda x: sum(votes[x].values()))
print(f"\n  -> Konsens: {winner} (gewichtete Mehrheit)")

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. KRITIKER-AGENT HINZUFUEGEN
   Erstelle einen CriticAgent der den Bericht des Writers prueft
   und Verbesserungen vorschlaegt. Pipeline wird: R -> A -> W -> C

2. WISSENSBASIS ERWEITERN
   Fuege mehr Sprachen zum KNOWLEDGE dict hinzu (Go, Java, TypeScript).
   Der Researcher findet dann automatisch mehr Infos.

3. GEWICHTETES VOTING
   Gib jedem Agent eine Expertise-Gewichtung (z.B. Python-Agent
   zaehlt doppelt bei Python-Fragen). Implementiere weighted_vote().

4. AGENT-MEMORY
   Lass Agenten sich an fruehere Analysen erinnern.
   Wenn eine aehnliche Frage kommt, nutze die alte Analyse als Basis.

5. PARALLELE AGENTEN
   Statt sequentiell (R->A->W): Lass 3 Researcher parallel arbeiten
   und den Analyst die Ergebnisse zusammenfuehren.
""")
