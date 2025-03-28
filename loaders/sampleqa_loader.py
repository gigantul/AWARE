import json
import os
from config.config import DATA_DIR


def load_sampleqa_dataset(filename="sampleQA.json"):
    """
    Loads a local JSON file with 5–10 QA pairs for testing.
    Args:
        filename (str): File inside `data/` folder.
    Returns:
        List of dicts with keys: question, answer, id
    """
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Sample QA file not found at {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Minimal validation
    for qa in data:
        if not all(k in qa for k in ["question", "answer", "id"]):
            raise ValueError(f"Malformed QA entry: {qa}")

    return data
