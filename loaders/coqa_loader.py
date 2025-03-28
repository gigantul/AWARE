# loaders/coqa_loader.py

from datasets import load_from_disk
from config.config import data_dir
import os
import urllib.request

def download_raw_coqa():
    """
    Downloads the CoQA raw JSON file if not already present.
    """
    url = "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json"
    file_path = f"{data_dir}/coqa-dev-v1.0.json"
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(file_path):
        print("[INFO] Downloading CoQA dataset...")
        urllib.request.urlretrieve(url, file_path)
    else:
        print("[INFO] CoQA dataset already present.")
    return file_path


def load_coqa_dataset(path=None, model_name=None):
    """
    Loads the preprocessed CoQA dataset from disk.
    If raw JSON doesn't exist, downloads it.
    Args:
        path (str): Optional override for dataset folder.
        model_name (str): Optional for model-specific dataset.
    Returns:
        HuggingFace Dataset object
    """
    path = path or f"{data_dir}/coqa_dataset"

    # Ensure raw file exists for parsing (in case you need to run parse_coqa.py)
    download_raw_coqa()

    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Parsed dataset not found at {path}. Please run parse_coqa.py first.")

    return load_from_disk(path)