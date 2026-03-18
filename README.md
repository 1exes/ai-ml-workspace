# AI/ML Workspace

Persönlicher Workspace zum Lernen und Experimentieren mit KI-Modellen.

## Was ist hier drin?

```
01_basics/          - Python + ML Grundlagen (NumPy, Pandas, Matplotlib)
02_classical_ml/    - Klassisches ML (Scikit-learn: Regression, Classification, Clustering)
03_deep_learning/   - Neural Networks mit PyTorch
04_nlp/             - Natural Language Processing (Tokenizer, Embeddings, Transformer)
05_fine_tuning/     - Modelle fine-tunen (LoRA, QLoRA mit Hugging Face)
06_inference/       - Modelle lokal laden und nutzen (llama.cpp, vLLM, Ollama)
07_rag/             - Retrieval Augmented Generation (Embeddings + Vector DB)
08_agents/          - KI-Agenten bauen (Tool Use, ReAct Pattern)
```

## Lernpfad

| Phase | Thema | Dauer | Ordner |
|-------|-------|-------|--------|
| 1 | Python für ML (NumPy, Tensoren) | 1-2 Wochen | `01_basics/` |
| 2 | Klassisches ML verstehen | 1-2 Wochen | `02_classical_ml/` |
| 3 | Deep Learning & PyTorch | 2-3 Wochen | `03_deep_learning/` |
| 4 | NLP & Transformer-Architektur | 2-3 Wochen | `04_nlp/` |
| 5 | Fine-Tuning eigener Modelle | 1-2 Wochen | `05_fine_tuning/` |
| 6 | Lokale Inference & Deployment | 1 Woche | `06_inference/` |
| 7 | RAG-Systeme bauen | 1 Woche | `07_rag/` |
| 8 | KI-Agenten entwickeln | 1-2 Wochen | `08_agents/` |

## Setup

```bash
# Venv erstellen
python -m venv venv
venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt

# Jupyter starten
jupyter lab
```

## Hardware-Hinweise

- **CPU reicht** für Phase 1-2
- **GPU empfohlen** ab Phase 3 (dein LM Studio Server kann auch helfen)
- **Fine-Tuning** (Phase 5): QLoRA braucht ~6-8 GB VRAM für 7B Modelle
- Alternativ: Google Colab (gratis GPU) oder Kaggle Notebooks

## Nützliche Links

- [Hugging Face Hub](https://huggingface.co/) - Modelle & Datasets
- [Papers With Code](https://paperswithcode.com/) - Aktuelle Forschung
- [fast.ai](https://www.fast.ai/) - Kostenlose Deep Learning Kurse
- [Andrej Karpathy YouTube](https://www.youtube.com/@AndrejKarpathy) - Beste ML-Erklärungen
