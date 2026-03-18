import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Chatbot mit Gedaechtnis
========================
Ein Chatbot der sich an die Konversation erinnert -
Kurzzeit- und Langzeitgedaechtnis.
"""

import re
import random
from datetime import datetime
from dataclasses import dataclass, field

# ============================================================
# 1. Memory System
# ============================================================

@dataclass
class Memory:
    """Ein gespeichertes Fakt."""
    key: str
    value: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")


class MemorySystem:
    """Kurz- und Langzeitgedaechtnis."""

    def __init__(self, short_term_limit: int = 10):
        self.short_term: list[dict] = []  # Letzte N Nachrichten
        self.long_term: list[Memory] = []  # Extrahierte Fakten
        self.limit = short_term_limit

    def add_message(self, role: str, content: str):
        """Nachricht zum Kurzzeitgedaechtnis hinzufuegen."""
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.limit:
            self.short_term.pop(0)

    def remember(self, key: str, value: str):
        """Fakt im Langzeitgedaechtnis speichern."""
        # Update wenn Key existiert
        for mem in self.long_term:
            if mem.key == key:
                mem.value = value
                return
        self.long_term.append(Memory(key, value))

    def recall(self, query: str) -> list[Memory]:
        """Relevante Erinnerungen finden."""
        query_words = set(query.lower().split())
        results = []
        for mem in self.long_term:
            mem_words = set(mem.key.lower().split() + mem.value.lower().split())
            overlap = query_words & mem_words
            if overlap:
                results.append(mem)
        return results

    def get_context(self) -> str:
        """Kontext aus Gedaechtnis zusammenbauen."""
        parts = []
        if self.long_term:
            facts = ", ".join(f"{m.key}: {m.value}" for m in self.long_term)
            parts.append(f"Bekannte Fakten: {facts}")
        if self.short_term:
            recent = self.short_term[-3:]
            history = " | ".join(f"{m['role']}: {m['content'][:50]}" for m in recent)
            parts.append(f"Letzte Nachrichten: {history}")
        return "\n".join(parts)


# ============================================================
# 2. Fakt-Extraktion
# ============================================================

def extract_facts(text: str) -> list[tuple[str, str]]:
    """Extrahiere Fakten aus User-Nachrichten."""
    facts = []

    # Name
    m = re.search(r"(?:ich bin|ich heisse|mein name ist|nennt? mich)\s+(\w+)", text.lower())
    if m:
        facts.append(("name", m.group(1).capitalize()))

    # Alter
    m = re.search(r"(?:ich bin|ich bin gerade)\s+(\d+)\s*(?:jahre?)?", text.lower())
    if m:
        facts.append(("alter", m.group(1)))

    # Beruf
    m = re.search(r"(?:ich bin|arbeite als|ich arbeite als)\s+(ein(?:e)?\s+)?(\w+[-\w]*)", text.lower())
    if m and m.group(2) not in ["bin", "gerade", "hier", "sehr", "nicht"]:
        facts.append(("beruf", m.group(2).capitalize()))

    # Stadt
    m = re.search(r"(?:aus|in|wohne in|lebe in|komme aus)\s+(\w+)", text.lower())
    if m and m.group(1) not in ["dem", "der", "den", "einem", "einer", "bin", "nicht"]:
        facts.append(("stadt", m.group(1).capitalize()))

    # Hobby/Interesse
    m = re.search(r"(?:mag|liebe|interessiere mich fuer|hobby ist|hobbys sind)\s+(.+?)(?:\.|$)", text.lower())
    if m:
        facts.append(("interesse", m.group(1).strip().capitalize()))

    return facts


# ============================================================
# 3. Response-Generator
# ============================================================

class Chatbot:
    """Chatbot mit Persoenlichkeit und Gedaechtnis."""

    PERSONALITIES = {
        "freundlich": {
            "greetings": ["Hallo! Schoen dich zu sehen!", "Hey! Wie geht's dir?", "Hi! Was kann ich fuer dich tun?"],
            "reactions": ["Das ist ja cool!", "Interessant!", "Ach wirklich?", "Super!"],
            "style": "warm und enthusiastisch",
        },
        "professionell": {
            "greetings": ["Guten Tag. Wie kann ich Ihnen helfen?", "Willkommen. Was kann ich fuer Sie tun?"],
            "reactions": ["Verstanden.", "Danke fuer die Information.", "Notiert.", "Interessanter Punkt."],
            "style": "sachlich und respektvoll",
        },
        "witzig": {
            "greetings": ["Na, wer schleicht denn da rein?", "Ah, Besuch! Und ich hab nicht mal aufgeraeumt!"],
            "reactions": ["Haha, nicht schlecht!", "Ach du meine Guete!", "Na sowas!", "Das haut mich um!"],
            "style": "humorvoll und locker",
        },
    }

    RESPONSES = {
        "wie_gehts": [
            "Mir geht's gut, danke! Und dir?",
            "Bestens! Als KI brauche ich keinen Kaffee - obwohl... waere mal interessant.",
            "Super! Ich bin bereit fuer jede Frage.",
        ],
        "was_kannst_du": [
            "Ich kann chatten, mir Sachen merken und dir bei Fragen helfen!",
            "Ich bin ein Chatbot mit Gedaechtnis - erzaehl mir was ueber dich!",
        ],
        "danke": [
            "Gerne! Dafuer bin ich da.",
            "Kein Problem! Frag jederzeit.",
            "Immer gerne!",
        ],
        "hilfe": [
            "Erzaehl mir etwas ueber dich (Name, Stadt, Hobbys) - ich merke es mir!",
            "Du kannst mich alles fragen. Ich merke mir auch Dinge aus unserem Gespraech.",
            "Sag 'was weisst du' um zu sehen was ich mir gemerkt habe.",
        ],
        "default": [
            "Hmm, darauf weiss ich keine perfekte Antwort. Erzaehl mir mehr!",
            "Interessant! Kannst du das genauer erklaeren?",
            "Das merke ich mir. Was moechtest du noch wissen?",
        ],
    }

    def __init__(self, personality: str = "freundlich"):
        self.memory = MemorySystem(short_term_limit=10)
        self.personality = self.PERSONALITIES.get(personality, self.PERSONALITIES["freundlich"])
        self.name = "ChatBot"
        self.turn_count = 0

    def respond(self, user_input: str) -> str:
        """Generiere eine Antwort."""
        self.turn_count += 1
        self.memory.add_message("user", user_input)
        text_lower = user_input.lower().strip()

        # Fakten extrahieren und merken
        facts = extract_facts(user_input)
        for key, value in facts:
            self.memory.remember(key, value)

        # Response bestimmen
        response = self._match_response(text_lower)

        # Kontext aus Gedaechtnis einbauen
        if self.turn_count > 1 and random.random() > 0.6:
            memories = self.memory.recall(text_lower)
            if memories:
                mem = random.choice(memories)
                response += f" (Ich erinnere mich: {mem.key} = {mem.value})"

        self.memory.add_message("bot", response)
        return response

    def _match_response(self, text: str) -> str:
        """Passende Antwort basierend auf Keywords finden."""

        # Begruessung
        if any(w in text for w in ["hallo", "hi", "hey", "moin", "guten tag", "servus"]):
            return random.choice(self.personality["greetings"])

        # Wie geht's
        if any(w in text for w in ["wie geht", "wie gehts", "was geht", "alles klar"]):
            return random.choice(self.RESPONSES["wie_gehts"])

        # Was kannst du
        if any(w in text for w in ["was kannst", "wer bist", "was bist"]):
            return random.choice(self.RESPONSES["was_kannst_du"])

        # Danke
        if any(w in text for w in ["danke", "thx", "thanks", "vielen dank"]):
            return random.choice(self.RESPONSES["danke"])

        # Hilfe
        if any(w in text for w in ["hilfe", "help", "was kann ich"]):
            return random.choice(self.RESPONSES["hilfe"])

        # Was weisst du (Memory abrufen)
        if any(w in text for w in ["was weisst", "erinnerst", "was merkst", "was hast du gemerkt"]):
            if self.memory.long_term:
                facts = "\n    ".join(f"- {m.key}: {m.value} (gemerkt um {m.timestamp})" for m in self.memory.long_term)
                return f"Das weiss ich ueber dich:\n    {facts}"
            return "Ich weiss noch nicht viel ueber dich. Erzaehl mir was!"

        # Fakten erkannt -> reagieren
        facts = extract_facts(text)
        if facts:
            responses = []
            for key, value in facts:
                if key == "name":
                    responses.append(f"Schoener Name, {value}!")
                elif key == "stadt":
                    responses.append(f"{value} ist eine tolle Stadt!")
                elif key == "beruf":
                    responses.append(f"{value} - das klingt spannend!")
                elif key == "interesse":
                    responses.append(f"{value}? Das finde ich auch interessant!")
                elif key == "alter":
                    responses.append(f"{value} Jahre - cooles Alter!")
            reaction = random.choice(self.personality["reactions"])
            return reaction + " " + " ".join(responses)

        # Default
        return random.choice(self.RESPONSES["default"])


# ============================================================
# 4. Demo-Konversation
# ============================================================
print("=" * 60)
print("CHATBOT MIT GEDAECHTNIS")
print("=" * 60)

bot = Chatbot(personality="freundlich")

# Simulierte Konversation
conversation = [
    "Hallo!",
    "Ich bin Max und komme aus Berlin",
    "Ich arbeite als Entwickler",
    "Ich mag Programmieren und Schach",
    "Wie geht es dir?",
    "Was weisst du ueber mich?",
    "Ich bin 28 Jahre alt",
    "Danke fuer das Gespraech!",
    "Was weisst du jetzt alles?",
]

print()
for msg in conversation:
    print(f"  Du:  {msg}")
    response = bot.respond(msg)
    print(f"  Bot: {response}")
    print()

# ============================================================
# 5. Memory Status
# ============================================================
print("=" * 60)
print("MEMORY STATUS")
print("=" * 60)

print(f"\nKurzzeitgedaechtnis: {len(bot.memory.short_term)} Nachrichten")
for m in bot.memory.short_term[-5:]:
    print(f"  [{m['role']:>4s}] {m['content'][:60]}")

print(f"\nLangzeitgedaechtnis: {len(bot.memory.long_term)} Fakten")
for m in bot.memory.long_term:
    print(f"  {m.key:<12s} = {m.value:<20s} (seit {m.timestamp})")

print(f"\nKontext:\n{bot.memory.get_context()}")

# ============================================================
# 6. Persoenlichkeiten vergleichen
# ============================================================
print("\n" + "=" * 60)
print("6. PERSOENLICHKEITEN")
print("=" * 60)

test_msg = "Hallo! Ich bin Anna."
for style in ["freundlich", "professionell", "witzig"]:
    test_bot = Chatbot(personality=style)
    response = test_bot.respond(test_msg)
    print(f"\n  [{style:>14s}] {response}")

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. INTERAKTIVER MODUS
   Baue eine while-Schleife fuer echtes Chatten:
     while True:
         user_input = input("Du: ")
         if user_input.lower() in ["quit", "exit"]: break
         print(f"Bot: {bot.respond(user_input)}")

2. MEHR FAKTEN EXTRAHIEREN
   Erweitere extract_facts() um: Lieblingsfarbe, Haustier,
   Lieblingsessen, Programmiersprache.

3. STIMMUNGS-ERKENNUNG
   Analysiere den Ton der Nachricht (positiv/negativ/neutral)
   und passe die Bot-Antwort an die Stimmung an.

4. VERGESSEN
   Implementiere ein "Vergessen" - alte Fakten im Langzeitgedaechtnis
   werden nach N Turns geloescht, wenn sie nicht referenziert werden.

5. LLM-ANBINDUNG
   Ersetze _match_response() durch einen Aufruf an ein lokales LLM
   (Ollama/LM Studio). Uebergib den Memory-Kontext als System-Prompt.
""")
