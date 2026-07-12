# Hugging Face Transformers Mini Lab

This repository is a practical mini-lab for learning the Hugging Face LLM course topics that matter for AI Infrastructure, Inference Platform, MLOps, RAG, and AI Operations roles.

The goal is not to copy the whole Hugging Face course. The goal is to build a focused hands-on foundation for:

- transformers pipelines
- model and tokenizer usage
- batching, padding, and truncation
- Hugging Face Datasets
- custom dataset loading
- tokenizer comparison and offset mapping
- semantic search with FAISS
- summarization
- question answering
- basic NLP tasks used in production AI systems

## Why this repo exists

I am learning LLM fundamentals from an AI Infrastructure / MLOps perspective. My focus is on practical model usage, inference, RAG, model serving, monitoring, and production AI systems.

## Repo structure

```text
hf-transformers-mini-lab/
├── README.md
├── requirements.txt
├── Makefile
├── data/
│   ├── sample.csv
│   ├── sample.json
│   └── sample.txt
├── scripts/
│   ├── 01_pipeline_demo.py
│   ├── 02_model_tokenizer_demo.py
│   ├── 03_batch_inference_demo.py
│   ├── 04_dataset_demo.py
│   ├── 05_custom_dataset_demo.py
│   ├── 06_tokenizer_comparison.py
│   ├── 07_fast_tokenizer_offsets.py
│   ├── 08_faiss_semantic_search_demo.py
│   ├── 09_summarization_demo.py
│   ├── 10_question_answering_demo.py
│   ├── 11_token_classification_demo.py
│   └── 12_translation_demo.py
├── notes/
└── screenshots/
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Some scripts download models from Hugging Face on the first run. After the first download, they are cached locally.

## Recommended running order

Run the scripts in this order:

```bash
python scripts/01_pipeline_demo.py
python scripts/02_model_tokenizer_demo.py
python scripts/03_batch_inference_demo.py
python scripts/04_dataset_demo.py
python scripts/05_custom_dataset_demo.py
python scripts/06_tokenizer_comparison.py
python scripts/07_fast_tokenizer_offsets.py
python scripts/08_faiss_semantic_search_demo.py
python scripts/09_summarization_demo.py
python scripts/10_question_answering_demo.py
```

Optional:

```bash
python scripts/11_token_classification_demo.py
python scripts/12_translation_demo.py
```

## What each script teaches

| Script | Topic | Why it matters |
|---|---|---|
| `01_pipeline_demo.py` | Pipelines | Quick use of pretrained models |
| `02_model_tokenizer_demo.py` | AutoTokenizer + AutoModel | Understand `input_ids`, `attention_mask`, and model outputs |
| `03_batch_inference_demo.py` | Batch inference | Needed for production inference APIs |
| `04_dataset_demo.py` | Dataset operations | `select`, `filter`, `map`, `train_test_split` |
| `05_custom_dataset_demo.py` | Local data loading | Useful for custom files, logs, PDFs, CSV/JSON |
| `06_tokenizer_comparison.py` | Tokenizer differences | Understand BPE, WordPiece-style behavior at a high level |
| `07_fast_tokenizer_offsets.py` | Offset mapping | Useful for QA, extraction, and long documents |
| `08_faiss_semantic_search_demo.py` | Embeddings + FAISS | Foundation for RAG and vector search |
| `09_summarization_demo.py` | Summarization | Useful for document processing and AI assistants |
| `10_question_answering_demo.py` | Question answering | Core foundation for RAG and document Q&A |
| `11_token_classification_demo.py` | NER / extraction | Useful for structured extraction from text |
| `12_translation_demo.py` | Translation | Optional NLP task demo |

## Key learnings to document

After completing this mini-lab, you should be able to explain:

- how text becomes tokens
- what `input_ids` and `attention_mask` mean
- why padding and truncation matter
- how batch inference works
- how to load and process datasets
- how to load CSV, JSON, and text datasets
- how embeddings enable semantic search
- how FAISS retrieves similar text
- how summarization and question answering pipelines work
- how tokenizer offset mapping helps with answer spans

## Next projects after this repo

After this mini-lab, move to these projects:

1. `local-ai-assistant-ollama`
2. `pdf-rag-chatbot-qdrant-ollama`
3. `pytorch-fastapi-model-serving`
4. `end-to-end-mlops-pipeline`
5. `ai-inference-platform-fastapi`
6. `kubernetes-ai-platform`

## Notes

This repo is intentionally small and focused. It avoids advanced topics like LLM fine-tuning, GRPO, reasoning model training, Argilla dataset curation, and training models from scratch because those are not required for the immediate AI Infrastructure path.
