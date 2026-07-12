setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run-pipelines:
	python scripts/01_pipeline_demo.py

run-core:
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
