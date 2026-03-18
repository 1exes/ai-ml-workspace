# AI/ML Workspace

Persönlicher Workspace zum Lernen und Experimentieren mit KI-Modellen.

## Struktur

```
basics/     - Grundlagen: NumPy, Daten-Viz, Regression, Classification, PyTorch
models/     - NLP, Fine-Tuning (LoRA), lokale Modelle (LM Studio, Transformers)
agents/     - RAG mit ChromaDB, Tool-Use Agents
```

## Lernpfad

| # | Datei | Was du lernst |
|---|-------|---------------|
| 1 | `basics/numpy_tensoren.py` | Tensoren, Dot Product, Aktivierungsfunktionen, Mini-NN |
| 2 | `basics/daten_visualisierung.py` | Pandas, Matplotlib, Train/Test Split |
| 3 | `basics/regression.py` | Zahlen vorhersagen (Linear, Random Forest, Gradient Boosting) |
| 4 | `basics/classification.py` | Kategorien vorhersagen, Confusion Matrix |
| 5 | `basics/pytorch_basics.py` | PyTorch Tensoren, Autograd, Training Loop |
| 6 | `models/tokenizer_embeddings.py` | Tokenizer, Embeddings, semantische Suche |
| 7 | `models/lora_fine_tuning.py` | LoRA/QLoRA Fine-Tuning mit Hugging Face |
| 8 | `models/lokale_modelle.py` | Modelle lokal nutzen (Pipelines, LM Studio API) |
| 9 | `agents/simple_rag.py` | RAG: Dokumente + LLM kombinieren |
| 10 | `agents/tool_use_agent.py` | Agent der selbst Tools aufruft |

## Setup

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python basics/numpy_tensoren.py
```

## Links

- [Hugging Face Hub](https://huggingface.co/) - Modelle & Datasets
- [Andrej Karpathy YouTube](https://www.youtube.com/@AndrejKarpathy) - Beste ML-Erklärungen
- [fast.ai](https://www.fast.ai/) - Kostenlose Deep Learning Kurse
