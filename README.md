# AI/ML Workspace

Persoenlicher Workspace zum Lernen und Experimentieren mit KI-Modellen.
24 Scripts mit echten europaeischen Modellen, Uebungsaufgaben und Praxisprojekten.

## Struktur

```
basics/     - ML-Grundlagen, Daten, PyTorch, Feature Engineering, Evaluation
models/     - NLP, Computer Vision, Fine-Tuning, Prompt Engineering, Modellvergleich
agents/     - RAG, Chatbots, Multi-Agent, Pipelines, Workflow Automation
```

## Lernpfad

### Phase 1: Grundlagen (basics/)

| # | Script | Was du lernst |
|---|--------|---------------|
| 1 | `numpy_tensoren.py` | Tensoren, Dot Product, Aktivierungsfunktionen, Mini-NN von Hand |
| 2 | `daten_visualisierung.py` | Pandas, Matplotlib, Train/Test Split, Normalisierung |
| 3 | `regression.py` | Zahlen vorhersagen (Linear, Random Forest, Gradient Boosting) |
| 4 | `classification.py` | Kategorien vorhersagen, Confusion Matrix, Precision/Recall |
| 5 | `pytorch_basics.py` | PyTorch Tensoren, Autograd, Training Loop |
| 6 | `feature_engineering.py` | Encoding, PCA, Feature Selection, Polynomial Features |
| 7 | `hyperparameter_tuning.py` | GridSearch, RandomSearch, Cross-Validation, Lernkurven |
| 8 | `model_evaluation.py` | ROC-AUC, Precision-Recall Curves, Bias-Variance, 5+ Modelle |
| 9 | `daten_augmentation.py` | Bild-Augmentation, Text-Augmentation, SMOTE Konzept |

### Phase 2: Modelle & NLP (models/)

| # | Script | Was du lernst |
|---|--------|---------------|
| 10 | `tokenizer_embeddings.py` | Mistral vs German BERT Tokenizer, multilingual Embeddings |
| 11 | `lokale_modelle.py` | 6 HuggingFace Pipelines: Sentiment, NER, Translation DE->EN |
| 12 | `lora_fine_tuning.py` | LoRA Fine-Tuning auf CPU mit distilgpt2 |
| 13 | `computer_vision.py` | ResNet, Feature Extraction, Transfer Learning |
| 14 | `text_klassifikation.py` | TF-IDF vs BERT, deutsches Text-Klassifikation Training |
| 15 | `speech_und_audio.py` | Whisper, Spektrogramme, Audio-Visualisierung |
| 16 | `prompt_engineering.py` | Zero-Shot, Few-Shot, CoT, ReAct, Prompt Templates |
| 17 | `modell_vergleich.py` | 6 Modelle im Benchmark: Accuracy, Speed, Memory |

### Phase 3: Agenten & Systeme (agents/)

| # | Script | Was du lernst |
|---|--------|---------------|
| 18 | `simple_rag.py` | RAG mit ChromaDB + deutscher Wissensbasis |
| 19 | `tool_use_agent.py` | ReAct Pattern, 4 Tools, regelbasierter Agent |
| 20 | `chatbot.py` | Chatbot mit Kurz- und Langzeitgedaechtnis |
| 21 | `multi_agent.py` | 3 Agents die zusammenarbeiten (Researcher/Analyst/Writer) |
| 22 | `daten_pipeline.py` | Automatische EDA + Cleaning + Training Pipeline |
| 23 | `web_researcher.py` | Research Agent mit Quellen-Bewertung und Zitaten |
| 24 | `workflow_automation.py` | Mini Workflow Engine (DAG, Retry, Parallel Tasks) |

## Verwendete EU/Multilingual Modelle

| Modell | Herkunft | Aufgabe |
|--------|----------|---------|
| `mistralai/Mistral-7B-v0.1` | Frankreich | Tokenizer-Vergleich |
| `deepset/gbert-base` | Deutschland | German BERT |
| `paraphrase-multilingual-MiniLM-L12-v2` | EU | Multilingual Embeddings |
| `Helsinki-NLP/opus-mt-de-en` | Finnland | Deutsch->Englisch |
| `nlptown/bert-base-multilingual-uncased-sentiment` | EU | Multilingual Sentiment |
| `distilbert-base-multilingual-cased` | EU | Text Classification |
| `xlm-roberta-base` | EU (Meta) | Cross-lingual NLP |
| `openai/whisper-tiny` | OpenAI | Speech-to-Text |

## Setup

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python basics/numpy_tensoren.py
```

## Links

- [Hugging Face Hub](https://huggingface.co/) - Modelle und Datasets
- [Andrej Karpathy YouTube](https://www.youtube.com/@AndrejKarpathy) - Beste ML-Erklaerungen
- [fast.ai](https://www.fast.ai/) - Kostenlose Deep Learning Kurse
- [Papers With Code](https://paperswithcode.com/) - Aktuelle Forschung mit Code
- [Mistral AI](https://mistral.ai/) - Europaeische LLMs
- [deepset](https://www.deepset.ai/) - German NLP
