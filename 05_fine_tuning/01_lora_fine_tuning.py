"""
01 - LoRA Fine-Tuning
======================
Fine-Tuning = ein bestehendes Modell auf DEINE Daten anpassen.
LoRA = Low-Rank Adaptation - macht Fine-Tuning mit wenig VRAM möglich.

Statt ALLE Gewichte zu ändern (teuer), fügt LoRA kleine "Adapter" ein.
→ 7B Modell: Normales Fine-Tuning braucht ~28 GB VRAM
→ Mit QLoRA: Nur ~6 GB VRAM!
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cpu":
    print("\n⚠️  Fine-Tuning auf CPU ist SEHR langsam!")
    print("    Empfehlung: Google Colab (gratis GPU) oder Kaggle Notebooks")
    print("    Dieses Script zeigt den Ablauf - zum echten Training brauchst du GPU.\n")

# ============================================================
# 1. Modell laden (mit Quantisierung für weniger VRAM)
# ============================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Kleines Modell zum Testen

print(f"Lade Modell: {MODEL_NAME}")

# Quantisierung: 4-bit statt 16-bit = 4x weniger VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Auf CPU ohne Quantisierung laden (Demo-Modus)
if device == "cpu":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

# ============================================================
# 2. LoRA Konfiguration
# ============================================================

lora_config = LoraConfig(
    r=16,                  # Rang der LoRA-Matrizen (8-64, höher = mehr Kapazität)
    lora_alpha=32,         # Skalierungsfaktor
    lora_dropout=0.05,     # Dropout für Regularisierung
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # Welche Layer anpassen
)

model = get_peft_model(model, lora_config)

# Zeige wie viele Parameter trainiert werden
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainierbare Parameter: {trainable:,} / {total:,}")
print(f"Das sind nur {trainable / total:.2%} aller Parameter!")

# ============================================================
# 3. Trainingsdaten vorbereiten
# ============================================================

# Beispiel: Modell soll wie ein Python-Tutor antworten
training_data = [
    {"instruction": "Was ist eine Variable?",
     "response": "Eine Variable ist ein benannter Speicherplatz. In Python: `name = 'Max'` speichert den Text 'Max' unter dem Namen 'name'. Du kannst den Wert jederzeit ändern."},
    {"instruction": "Erkläre eine for-Schleife",
     "response": "Eine for-Schleife wiederholt Code für jedes Element: `for i in range(5): print(i)` gibt 0-4 aus. Du kannst über Listen, Strings und mehr iterieren."},
    {"instruction": "Was ist eine Funktion?",
     "response": "Eine Funktion ist wiederverwendbarer Code: `def greet(name): return f'Hallo {name}'`. Aufruf: `greet('Max')` → 'Hallo Max'. Funktionen machen Code übersichtlich."},
    {"instruction": "Was sind Listen in Python?",
     "response": "Listen speichern mehrere Werte: `zahlen = [1, 2, 3]`. Zugriff: `zahlen[0]` → 1. Hinzufügen: `zahlen.append(4)`. Listen sind veränderbar und können verschiedene Typen enthalten."},
    {"instruction": "Erkläre if/else",
     "response": "If/else prüft Bedingungen: `if alter >= 18: print('Erwachsen')` `else: print('Minderjährig')`. Python nutzt Einrückung statt Klammern."},
]

# In Chat-Format bringen
def format_example(example):
    text = (
        f"<|user|>\n{example['instruction']}\n"
        f"<|assistant|>\n{example['response']}\n"
    )
    return {"text": text}

dataset = Dataset.from_list([format_example(e) for e in training_data])

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized = dataset.map(tokenize)
print(f"\nTrainings-Samples: {len(tokenized)}")

# ============================================================
# 4. Training
# ============================================================

training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=device != "cpu",
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

print("\n🚀 Starte Training...")
print("   (Bei CPU nur als Demo - auf GPU dauert das ~2-5 Min für echte Daten)")

# Nur trainieren wenn GPU da ist (auf CPU zu langsam für Demo)
if device != "cpu":
    trainer.train()
    model.save_pretrained("./lora_output/final")
    print("\n✅ LoRA-Adapter gespeichert in ./lora_output/final/")
    print("   Adapter-Größe: ~wenige MB (statt GB für das volle Modell)")
else:
    print("\n⏭️  Training übersprungen (keine GPU)")
    print("   Auf Google Colab ausprobieren: https://colab.research.google.com")

# ============================================================
# 5. Inference mit Fine-Tuned Modell
# ============================================================

print("\n" + "=" * 50)
print("So nutzt du das fine-tuned Modell:")
print("=" * 50)
print("""
from peft import PeftModel

# Base-Modell laden
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# LoRA-Adapter drauflegen
model = PeftModel.from_pretrained(base_model, "./lora_output/final")

# Oder: Modelle mergen (Adapter wird permanent eingebaut)
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")
# → Kann dann als GGUF für llama.cpp exportiert werden
""")

print("💡 LoRA-Adapter sind klein (~5-50 MB) und stackbar.")
print("   Du kannst verschiedene Adapter für verschiedene Aufgaben trainieren")
print("   und sie dynamisch laden/wechseln!")
