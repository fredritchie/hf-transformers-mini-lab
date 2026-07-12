#! /bin/bash

sudo apt update -y
sudo apt install -y python3-pip
sudo apt install -y python3.12-venv
git clone https://github.com/fredritchie/hf-transformers-mini-lab.git

# Check current swap
swapon --show
free -h

# Create a 6 GB swap file
sudo fallocate -l 6G /swapfile

# Secure the file
sudo chmod 600 /swapfile

# Format it as swap
sudo mkswap /swapfile

# Enable it immediately
sudo swapon /swapfile

# Make it persistent after reboot
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab


cd hf-transformers-mini-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 scripts/01_pipeline_demo.py
python3 scripts/02_model_tokenizer_demo.py
python3 scripts/03_batch_inference_demo.py
python3 scripts/04_dataset_demo.py
python3 scripts/05_custom_dataset_demo.py
python3 scripts/06_tokenizer_comparison.py
python3 scripts/07_fast_tokenizer_offsets.py
python3 scripts/08_faiss_semantic_search_demo.py
python3 scripts/09_summarization_demo.py
python3 scripts/10_question_answering_demo.py
python3 scripts/11_token_classification_demo.py
python3 scripts/12_translation_demo.py