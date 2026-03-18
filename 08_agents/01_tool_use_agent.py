"""
01 - KI-Agent mit Tool Use
============================
Ein Agent = LLM + Tools + Loop
Das LLM entscheidet SELBST, welches Tool es wann aufruft.
So funktionieren Claude, ChatGPT Plugins, etc.
"""

import json
import httpx
import math

# ============================================================
# 1. Tools definieren
# ============================================================

# Jedes Tool hat: Name, Beschreibung, Parameter
TOOLS = {
    "calculate": {
        "description": "Rechne eine mathematische Aufgabe aus",
        "parameters": {"expression": "str - Mathematischer Ausdruck"},
        "function": lambda expr: str(eval(expr, {"__builtins__": {}}, {"math": math})),
    },
    "search_web": {
        "description": "Suche im Internet nach Informationen",
        "parameters": {"query": "str - Suchbegriff"},
        "function": lambda query: f"[Suchergebnis für '{query}': Beispielergebnis - in echt würde hier eine API aufgerufen]",
    },
    "get_weather": {
        "description": "Hole das aktuelle Wetter für eine Stadt",
        "parameters": {"city": "str - Stadtname"},
        "function": lambda city: json.dumps({"city": city, "temp": 12, "condition": "bewölkt"}),
    },
}

# ============================================================
# 2. Agent Loop
# ============================================================

def run_agent(
    user_message: str,
    llm_url: str = "http://192.168.178.118:11434/v1",
    model: str = "qwen2.5:14b",
    max_steps: int = 5,
):
    """Führe einen Agent-Loop aus."""

    # Tool-Beschreibungen für den Prompt
    tool_descriptions = "\n".join(
        f"- {name}: {info['description']} (Parameter: {info['parameters']})"
        for name, info in TOOLS.items()
    )

    system_prompt = f"""Du bist ein hilfreicher Assistent mit Zugang zu folgenden Tools:

{tool_descriptions}

Um ein Tool zu nutzen, antworte NUR mit einem JSON-Objekt:
{{"tool": "tool_name", "args": {{"param": "value"}}}}

Wenn du KEIN Tool brauchst, antworte normal mit Text.
Wenn du ein Tool-Ergebnis bekommst, nutze es für deine finale Antwort."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    print(f"\n🤖 Agent gestartet: '{user_message}'")
    print("-" * 50)

    for step in range(max_steps):
        # LLM aufrufen
        try:
            response = httpx.post(
                f"{llm_url}/chat/completions",
                json={"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 500},
                timeout=30,
            )
            response.raise_for_status()
            assistant_msg = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"   ⚠️ LLM nicht erreichbar: {e}")
            return _demo_mode(user_message)

        print(f"\n   Step {step + 1} - LLM sagt: {assistant_msg[:200]}")

        # Prüfe ob ein Tool aufgerufen werden soll
        try:
            parsed = json.loads(assistant_msg)
            if "tool" in parsed:
                tool_name = parsed["tool"]
                tool_args = parsed.get("args", {})

                if tool_name in TOOLS:
                    # Tool ausführen
                    result = TOOLS[tool_name]["function"](**tool_args)
                    print(f"   🔧 Tool '{tool_name}' → {result[:100]}")

                    messages.append({"role": "assistant", "content": assistant_msg})
                    messages.append({"role": "user", "content": f"Tool-Ergebnis: {result}"})
                    continue
        except (json.JSONDecodeError, TypeError):
            pass

        # Kein Tool → finale Antwort
        print(f"\n   ✅ Finale Antwort: {assistant_msg}")
        return assistant_msg

    return "Max steps erreicht"


def _demo_mode(question: str) -> str:
    """Demo ohne LLM-Server."""
    print("\n   📋 Demo-Modus (kein LLM-Server)")
    print("   So würde der Agent-Loop ablaufen:")
    print(f"   1. User fragt: '{question}'")
    print('   2. LLM antwortet: {"tool": "calculate", "args": {"expression": "2+2"}}')
    print("   3. Agent führt Tool aus → Ergebnis: 4")
    print("   4. LLM bekommt Ergebnis und antwortet: 'Das Ergebnis ist 4.'")
    return "(Demo)"


# ============================================================
# 3. Agent testen
# ============================================================

if __name__ == "__main__":
    queries = [
        "Was ist 1547 * 283?",
        "Wie ist das Wetter in Berlin?",
        "Suche nach den neuesten ML-Trends 2026",
    ]

    for q in queries:
        run_agent(q)
        print("\n" + "=" * 50)

    # ============================================================
    # 4. Das ReAct Pattern - So denken moderne Agents
    # ============================================================

    print("\n" + "=" * 50)
    print("Das ReAct Pattern (Reasoning + Acting)")
    print("=" * 50)
    print("""
    Thought: Ich muss herausfinden, wie warm es in Berlin ist.
    Action:  get_weather(city="Berlin")
    Observation: {"city": "Berlin", "temp": 12, "condition": "bewölkt"}
    Thought: Ich habe die Wetterdaten. Der User wollte das Wetter wissen.
    Answer:  In Berlin sind es 12°C und es ist bewölkt.

    Das ist GENAU so, wie Claude, ChatGPT und andere Agents funktionieren!
    Der Loop: Think → Act → Observe → Think → ... → Answer
    """)

    print("💡 Nächste Schritte:")
    print("  - Mehr Tools hinzufügen (Dateien lesen, APIs aufrufen, Code ausführen)")
    print("  - Fehlerbehandlung (was wenn ein Tool fehlschlägt?)")
    print("  - Memory (Agent merkt sich vorherige Gespräche)")
    print("  - Multi-Agent (mehrere Agents arbeiten zusammen)")
