import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Workflow Automation Framework
==============================
Ein Mini-Workflow-Engine (wie n8n/Zapier):
Tasks definieren, verketten, ausfuehren.
"""

import time
import random
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# 1. Workflow-Engine
# ============================================================

class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRYING = "RETRYING"


@dataclass
class Task:
    """Ein einzelner Task im Workflow."""
    name: str
    func: callable
    depends_on: list[str] = field(default_factory=list)
    retries: int = 0
    max_retries: int = 2
    status: TaskStatus = TaskStatus.PENDING
    result: any = None
    error: str = ""
    duration: float = 0.0


class WorkflowEngine:
    """Fuehrt Tasks in der richtigen Reihenfolge aus."""

    def __init__(self, name: str):
        self.name = name
        self.tasks: dict[str, Task] = {}
        self.log: list[str] = []

    def add_task(self, name: str, func: callable, depends_on: list[str] = None, max_retries: int = 2):
        """Task hinzufuegen."""
        self.tasks[name] = Task(
            name=name, func=func,
            depends_on=depends_on or [],
            max_retries=max_retries,
        )

    def _get_execution_order(self) -> list[str]:
        """Topologische Sortierung (DAG)."""
        visited = set()
        order = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self.tasks[name].depends_on:
                if dep in self.tasks:
                    visit(dep)
            order.append(name)

        for name in self.tasks:
            visit(name)
        return order

    def _can_run_parallel(self) -> list[list[str]]:
        """Finde Tasks die parallel laufen koennten."""
        levels = []
        assigned = set()

        while len(assigned) < len(self.tasks):
            level = []
            for name, task in self.tasks.items():
                if name in assigned:
                    continue
                deps_done = all(d in assigned for d in task.depends_on)
                if deps_done:
                    level.append(name)
            for name in level:
                assigned.add(name)
            if level:
                levels.append(level)
        return levels

    def run(self) -> bool:
        """Workflow ausfuehren."""
        self._log(f"=== Workflow '{self.name}' gestartet ===")
        order = self._get_execution_order()

        # DAG visualisieren
        self._print_dag()

        all_success = True
        for name in order:
            task = self.tasks[name]

            # Pruefen ob Dependencies erfolgreich waren
            deps_ok = all(
                self.tasks[d].status == TaskStatus.SUCCESS
                for d in task.depends_on if d in self.tasks
            )

            if not deps_ok:
                task.status = TaskStatus.SKIPPED
                self._log(f"  [SKIP] {name} (Dependency fehlgeschlagen)")
                all_success = False
                continue

            # Task ausfuehren (mit Retry)
            success = self._execute_task(task)
            if not success:
                all_success = False

        self._log(f"=== Workflow {'ERFOLGREICH' if all_success else 'MIT FEHLERN'} ===")
        return all_success

    def _execute_task(self, task: Task) -> bool:
        """Einzelnen Task ausfuehren mit Retry-Logik."""
        while task.retries <= task.max_retries:
            task.status = TaskStatus.RUNNING if task.retries == 0 else TaskStatus.RETRYING
            self._log(f"  [{'RUN' if task.retries == 0 else f'RETRY {task.retries}'}] {task.name}...")

            start = time.time()
            try:
                # Sammle Ergebnisse der Dependencies
                dep_results = {}
                for dep_name in task.depends_on:
                    if dep_name in self.tasks:
                        dep_results[dep_name] = self.tasks[dep_name].result

                task.result = task.func(dep_results)
                task.duration = time.time() - start
                task.status = TaskStatus.SUCCESS
                self._log(f"  [OK]   {task.name} ({task.duration:.2f}s)")
                return True

            except Exception as e:
                task.duration = time.time() - start
                task.error = str(e)
                task.retries += 1
                self._log(f"  [FAIL] {task.name}: {e}")

                if task.retries <= task.max_retries:
                    self._log(f"         Retry in 0.5s... ({task.retries}/{task.max_retries})")
                    time.sleep(0.5)

        task.status = TaskStatus.FAILED
        return False

    def _print_dag(self):
        """DAG als ASCII-Art anzeigen."""
        levels = self._can_run_parallel()
        self._log(f"\n  DAG (Execution Graph):")
        for i, level in enumerate(levels):
            if i > 0:
                self._log(f"  {'  |  ' * len(levels[i-1])}")
                self._log(f"  {'  v  ' * len(level)}")
            boxes = "  ".join(f"[{name}]" for name in level)
            parallel_note = " (parallel moeglich)" if len(level) > 1 else ""
            self._log(f"  {boxes}{parallel_note}")
        self._log("")

    def _log(self, msg: str):
        self.log.append(msg)
        print(msg)

    def report(self) -> str:
        """Zusammenfassenden Report generieren."""
        total_time = sum(t.duration for t in self.tasks.values())
        success = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SUCCESS)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)

        report = f"\n{'='*50}\n"
        report += f"  WORKFLOW REPORT: {self.name}\n"
        report += f"{'='*50}\n"
        report += f"  Tasks:    {len(self.tasks)} gesamt\n"
        report += f"  Erfolg:   {success}\n"
        report += f"  Fehler:   {failed}\n"
        report += f"  Skipped:  {skipped}\n"
        report += f"  Zeit:     {total_time:.2f}s\n"
        report += f"{'='*50}\n"

        for name, task in self.tasks.items():
            status_icon = {
                TaskStatus.SUCCESS: "[OK]",
                TaskStatus.FAILED: "[!!]",
                TaskStatus.SKIPPED: "[--]",
            }.get(task.status, "[??]")
            report += f"  {status_icon} {name:<25s} {task.duration:.2f}s  {task.status.value}\n"
            if task.error:
                report += f"       Error: {task.error}\n"

        return report


# ============================================================
# 2. Demo Workflow 1: Datenverarbeitung
# ============================================================
print("=" * 60)
print("WORKFLOW 1: Datenverarbeitung")
print("=" * 60)

wf1 = WorkflowEngine("Datenverarbeitung")

def load_data(deps):
    data = {"users": 1000, "fields": 15, "format": "CSV"}
    return data

def validate_data(deps):
    data = deps["load"]
    if data["users"] < 10:
        raise ValueError("Zu wenig Daten!")
    return {"valid": True, "rows": data["users"], "issues": 2}

def clean_data(deps):
    validation = deps["validate"]
    return {"cleaned_rows": validation["rows"] - validation["issues"], "removed": validation["issues"]}

def transform_data(deps):
    cleaned = deps["clean"]
    return {"features": 20, "samples": cleaned["cleaned_rows"]}

def save_results(deps):
    transformed = deps["transform"]
    return {"saved": True, "path": "output/processed.csv", "size": f"{transformed['samples']} rows"}

wf1.add_task("load", load_data)
wf1.add_task("validate", validate_data, depends_on=["load"])
wf1.add_task("clean", clean_data, depends_on=["validate"])
wf1.add_task("transform", transform_data, depends_on=["clean"])
wf1.add_task("save", save_results, depends_on=["transform"])

wf1.run()
print(wf1.report())

# ============================================================
# 3. Demo Workflow 2: ML Training (mit parallelen Tasks)
# ============================================================
print("\n" + "=" * 60)
print("WORKFLOW 2: ML Training Pipeline")
print("=" * 60)

wf2 = WorkflowEngine("ML Training")

def prepare_data(deps):
    time.sleep(0.1)
    return {"X_train": 800, "X_test": 200, "features": 10}

def train_model_a(deps):
    time.sleep(0.2)
    return {"model": "RandomForest", "accuracy": 0.87}

def train_model_b(deps):
    time.sleep(0.15)
    return {"model": "GradientBoosting", "accuracy": 0.91}

def train_model_c(deps):
    time.sleep(0.1)
    # Simulierter Fehler beim ersten Versuch
    if random.random() < 0.5:
        raise ConnectionError("Timeout beim Laden")
    return {"model": "SVM", "accuracy": 0.84}

def compare_models(deps):
    results = []
    for key in ["model_a", "model_b", "model_c"]:
        if key in deps and deps[key]:
            results.append(deps[key])
    best = max(results, key=lambda x: x["accuracy"])
    return {"best_model": best["model"], "best_accuracy": best["accuracy"]}

def deploy_model(deps):
    best = deps["compare"]
    return {"deployed": best["best_model"], "endpoint": "/api/predict", "version": "v1.0"}

wf2.add_task("prepare", prepare_data)
wf2.add_task("model_a", train_model_a, depends_on=["prepare"])
wf2.add_task("model_b", train_model_b, depends_on=["prepare"])
wf2.add_task("model_c", train_model_c, depends_on=["prepare"], max_retries=3)
wf2.add_task("compare", compare_models, depends_on=["model_a", "model_b", "model_c"])
wf2.add_task("deploy", deploy_model, depends_on=["compare"])

wf2.run()
print(wf2.report())

# ============================================================
# 4. Demo Workflow 3: Content Pipeline
# ============================================================
print("\n" + "=" * 60)
print("WORKFLOW 3: Content Pipeline")
print("=" * 60)

wf3 = WorkflowEngine("Content Pipeline")

def research_topic(deps):
    return {"topic": "KI in der Medizin", "sources": 5, "key_points": 3}

def write_draft(deps):
    r = deps["research"]
    return {"title": f"Artikel: {r['topic']}", "words": 800, "paragraphs": 5}

def review_draft(deps):
    draft = deps["draft"]
    return {"approved": True, "suggestions": 2, "quality": "gut"}

def publish(deps):
    review = deps["review"]
    if not review["approved"]:
        raise ValueError("Draft nicht genehmigt!")
    return {"published": True, "url": "/blog/ki-medizin", "status": "live"}

wf3.add_task("research", research_topic)
wf3.add_task("draft", write_draft, depends_on=["research"])
wf3.add_task("review", review_draft, depends_on=["draft"])
wf3.add_task("publish", publish, depends_on=["review"])

wf3.run()
print(wf3.report())

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. BEDINGTES ROUTING
   Fuege eine if/else Logik ein:
   Wenn accuracy > 0.9 -> deploy
   Wenn accuracy < 0.9 -> retrain mit mehr Daten

2. TIMEOUT
   Implementiere ein Timeout fuer Tasks:
   Wenn ein Task laenger als N Sekunden dauert -> abbrechen

3. EIGENER WORKFLOW
   Erstelle einen Workflow fuer dein eigenes Projekt:
   z.B. Scraping -> Cleaning -> Analyse -> Notification

4. WEBHOOK/NOTIFICATION
   Fuege einen Task hinzu der bei Erfolg/Fehler eine
   Benachrichtigung sendet (print als Simulation).

5. WORKFLOW AUS YAML
   Definiere Workflows in einer YAML-Datei und lade sie:
   tasks:
     - name: load
       depends_on: []
     - name: process
       depends_on: [load]
""")
