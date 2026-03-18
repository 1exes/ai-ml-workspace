"""
02 - LoRA Fine-Tuning (Windows-kompatibel)
============================================
Fine-Tuning = ein bestehendes Modell auf DEINE Daten anpassen.
LoRA = Low-Rank Adaptation - macht Fine-Tuning mit wenig Ressourcen moeglich.

Statt ALLE Gewichte zu aendern (teuer), fuegt LoRA kleine "Adapter" ein.
-> 7B Modell: Normales Fine-Tuning braucht ~28 GB VRAM
-> Mit LoRA: Nur ein Bruchteil davon!

Dieses Script nutzt distilgpt2 (82M Parameter) und laeuft auf CPU.
KEIN bitsandbytes noetig - funktioniert auf Windows!
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Warnungen reduzieren
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 60)
print("LoRA FINE-TUNING MIT distilgpt2 (Windows-kompatibel)")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("  -> Training auf CPU (distilgpt2 ist klein genug dafuer!)")

# ============================================================
# 1. Modell laden (float32, kein bitsandbytes noetig!)
# ============================================================

MODEL_NAME = "distilgpt2"  # 82M Parameter - klein, schnell, ueberall lauffaehig

print(f"\nLade Modell: {MODEL_NAME}")
print("  (distilgpt2 = destillierte Version von GPT-2, nur 82M Parameter)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# float32 auf CPU - funktioniert ueberall, kein Quantisierung noetig
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Modell geladen: {total_params:,} Parameter")


# ============================================================
# 2. VORHER: Modell-Output ohne Fine-Tuning
# ============================================================

print("\n" + "=" * 60)
print("VORHER: Modell-Output OHNE Fine-Tuning")
print("=" * 60)

test_prompts = [
    "Python is a programming language that",
    "A variable in Python is",
    "Machine learning helps us to",
]

def generate_text(mdl, prompt, max_new=60):
    """Generiert Text mit dem gegebenen Modell."""
    inputs = tokenizer(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        output = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("\nOriginal distilgpt2 Antworten:")
vorher_antworten = {}
for prompt in test_prompts:
    result = generate_text(model, prompt)
    vorher_antworten[prompt] = result
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Output: {result[:150]}...")


# ============================================================
# 3. LoRA Konfiguration
# ============================================================

print("\n" + "=" * 60)
print("LoRA KONFIGURATION")
print("=" * 60)

# LoRA-Adapter konfigurieren
lora_config = LoraConfig(
    r=8,                    # Rang der LoRA-Matrizen (kleiner = weniger Parameter)
    lora_alpha=16,          # Skalierungsfaktor
    lora_dropout=0.05,      # Dropout fuer Regularisierung
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # distilgpt2 nutzt "c_attn" und "c_proj" statt "q_proj"/"v_proj"
    target_modules=["c_attn"],
)

model = get_peft_model(model, lora_config)

# Zeige wie viele Parameter trainiert werden
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainierbare Parameter: {trainable:,} / {total:,}")
print(f"Das sind nur {trainable / total:.2%} aller Parameter!")
print(f"-> LoRA trainiert nur die kleinen Adapter-Matrizen")


# ============================================================
# 4. Trainingsdaten vorbereiten
# ============================================================

print("\n" + "=" * 60)
print("TRAININGSDATEN")
print("=" * 60)

# Beispiel: Modell soll Python-Konzepte erklaeren koennen
# Mehr Daten = besseres Ergebnis (hier nur Demo)
training_texts = [
    "Python is a programming language that is easy to learn and very powerful. It uses indentation instead of braces and supports multiple programming paradigms.",
    "A variable in Python is a named container that stores data. You create one with: name = 'Max'. Variables can hold numbers, strings, lists, and more.",
    "A for loop in Python repeats code for each element: for i in range(5): print(i) outputs 0 to 4. You can iterate over lists, strings, and dictionaries.",
    "A function in Python is reusable code defined with def: def greet(name): return f'Hello {name}'. Functions make code organized and maintainable.",
    "Lists in Python store multiple values: numbers = [1, 2, 3]. Access with numbers[0]. Add with append(). Lists are mutable and can hold mixed types.",
    "Machine learning helps us to find patterns in data automatically. Python has great libraries like scikit-learn, PyTorch, and TensorFlow for ML tasks.",
    "A dictionary in Python maps keys to values: person = {'name': 'Max', 'age': 25}. Access with person['name']. Dictionaries are fast for lookups.",
    "Python is a programming language that supports object-oriented, functional, and procedural styles. It has a huge ecosystem of packages on PyPI.",
    "A variable in Python is dynamically typed - you don't declare types. x = 5 makes x an integer, x = 'hello' makes it a string. Python figures it out.",
    "Machine learning helps us to predict outcomes, classify data, and discover hidden structures. It requires good data, the right algorithm, and evaluation.",
]

# Daten tokenisieren
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

dataset = Dataset.from_dict({"text": training_texts})
tokenized_dataset = dataset.map(tokenize_function, remove_columns=["text"])

print(f"Trainings-Samples: {len(tokenized_dataset)}")
print(f"Token-Laenge pro Sample: {len(tokenized_dataset[0]['input_ids'])}")


# ============================================================
# 5. Training starten (3 Epochen, laeuft auch auf CPU!)
# ============================================================

print("\n" + "=" * 60)
print("TRAINING STARTEN (3 Epochen)")
print("=" * 60)

# Output-Verzeichnis
output_dir = "./lora_output_distilgpt2"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    # Kein fp16 auf CPU
    fp16=False,
    # Weniger Speicher
    dataloader_pin_memory=False,
)

# Data Collator fuer Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Kein Masked LM, sondern Causal LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\n>> Starte Training... (distilgpt2 auf CPU: ca. 1-3 Minuten)")
print("   Beobachte den Loss - er sollte sinken!\n")

# ECHTES Training - laeuft auch auf CPU dank distilgpt2!
trainer.train()

print("\n[OK] Training abgeschlossen!")

# Adapter speichern
model.save_pretrained(f"{output_dir}/final_adapter")
print(f"[OK] LoRA-Adapter gespeichert in {output_dir}/final_adapter/")

# Adapter-Groesse anzeigen
adapter_size = 0
adapter_dir = f"{output_dir}/final_adapter"
if os.path.exists(adapter_dir):
    for f in os.listdir(adapter_dir):
        fpath = os.path.join(adapter_dir, f)
        if os.path.isfile(fpath):
            adapter_size += os.path.getsize(fpath)
    print(f"   Adapter-Groesse: {adapter_size / 1024:.1f} KB (statt {total_params * 4 / 1024 / 1024:.0f} MB fuer das volle Modell)")


# ============================================================
# 6. NACHHER: Modell-Output MIT Fine-Tuning
# ============================================================

print("\n" + "=" * 60)
print("NACHHER: Modell-Output MIT Fine-Tuning (Vergleich)")
print("=" * 60)

print("\nFine-tuned distilgpt2 Antworten:")
for prompt in test_prompts:
    result_after = generate_text(model, prompt)
    result_before = vorher_antworten[prompt]

    print(f"\n  Prompt: '{prompt}'")
    print(f"  VORHER: {result_before[:120]}...")
    print(f"  NACHHER: {result_after[:120]}...")


# ============================================================
# 7. Wie man den Adapter spaeter laedt
# ============================================================

print("\n" + "=" * 60)
print("SO LAEDST DU DEN ADAPTER SPAETER")
print("=" * 60)

print("""
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 1. Base-Modell laden
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# 2. LoRA-Adapter drauflegen
model = PeftModel.from_pretrained(base_model, "./lora_output_distilgpt2/final_adapter")

# 3. Optional: Modelle mergen (Adapter wird permanent eingebaut)
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")

# -> Merged Modell kann dann wie ein normales Modell genutzt werden
""")


# ============================================================
# UEBUNGEN
# ============================================================

print("=" * 60)
print("UEBUNGEN")
print("=" * 60)

print("""
1. MEHR TRAININGSDATEN
   Fuege 10 weitere Python-Erklaerungen zu 'training_texts' hinzu.
   Wird der Output besser? Sinkt der Loss schneller?

2. LoRA HYPERPARAMETER
   Aendere den Rang (r=4 vs r=16 vs r=32).
   Wie aendert sich die Anzahl trainierbarer Parameter?
   Wird das Ergebnis besser oder schlechter?

3. ANDERES THEMA
   Ersetze die Python-Texte durch Erklaerungen zu einem
   anderen Thema (z.B. Kochen, Geschichte, Physik).
   Kann das Modell das Thema lernen?

4. MEHR EPOCHEN
   Aendere num_train_epochs auf 10 oder 20.
   Achtung: Zu viele Epochen -> Overfitting!
   Beobachte ob der Loss irgendwann wieder steigt.

5. LEARNING RATE
   Teste learning_rate=1e-3 vs 1e-4 vs 1e-5.
   Zu hoch = instabil, zu niedrig = lernt nichts.
""")

print("[OK] Script abgeschlossen!")
