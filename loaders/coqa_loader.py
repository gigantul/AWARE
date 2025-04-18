# loaders/coqa_loader.py

import os
import json
import urllib.request
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from config.config import DATA_DIR

def download_raw_coqa():
    """
    Downloads the CoQA dev set if it's not already present.
    """
    url = "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json"
    file_path = os.path.join(DATA_DIR, "coqa-dev-v1.0.json")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(file_path):
        print("[INFO] Downloading CoQA dataset...")
        urllib.request.urlretrieve(url, file_path)
    else:
        print("[INFO] CoQA dataset already exists.")
    return file_path

def preprocess_coqa_dataset(model_name, save_path):
    """
    Tokenizes and formats the CoQA dev set for model inference.
    """
    print(f"[INFO] Preprocessing CoQA for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    raw_path = download_raw_coqa()
    with open(raw_path, "r") as f:
        data = json.load(f)["data"]

    samples = []
    for entry in data:
        story = entry["story"]
        questions = entry["questions"]
        answers = entry["answers"]

        for q, a in zip(questions, answers):
            question = q["input_text"]
            answer = a["input_text"]
            turn_id = q["turn_id"]

            prompt = f"Context: {story}\nQuestion: {question} Answer:"
            input_ids = tokenizer(prompt, truncation=True, padding=False)["input_ids"]

            samples.append({
                "question": question,
                "answer": answer,
                "prompt": prompt,
                "question_id": f"{entry['id']}_{turn_id}",
                "input_ids": input_ids
            })

    dataset = Dataset.from_list(samples)
    dataset.save_to_disk(save_path)
    print(f"[INFO] Saved CoQA dataset to {save_path}")

def load_coqa_dataset(path=None, model_name=None):
    """
    Loads or prepares the CoQA dataset.
    Args:
        path (str): Optional override for dataset folder.
        model_name (str): Required for tokenizer-dependent prompt encoding.
    Returns:
        HuggingFace Dataset
    """
    if model_name is None:
        raise ValueError("[ERROR] Model name must be specified for CoQA tokenizer.")

    folder_name = f"coqa_{model_name.split('/')[-1]}"
    path = path or os.path.join(DATA_DIR, folder_name)

    if not os.path.exists(path):
        preprocess_coqa_dataset(model_name, path)

    return load_from_disk(path)
