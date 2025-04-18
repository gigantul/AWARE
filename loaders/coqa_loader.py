import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

DATA_DIR = "data"  # Root dir for storing dataset files

def preprocess_sciq_dataset(model_name, save_path):
    """
    Preprocesses SciQ for a given tokenizer and saves it as JSONL.
    """
    print(f"[INFO] Preprocessing SciQ for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_data = load_dataset("sciq", split="train")
    val_data = load_dataset("sciq", split="validation")

    # Few-shot context from first 10 train examples
    few_shot_prompt = "This is a bot that correctly answers questions.\n"
    for sample in train_data.select(range(10)):
        few_shot_prompt += f"Question: {sample['question']} Answer: {sample['correct_answer']} "

    def encode(example):
        prompt = few_shot_prompt.strip() + f"\n\nQuestion: {example['question']} Answer:"
        input_ids = tokenizer(prompt, truncation=False, padding=False)["input_ids"]
        return {
            "input_ids": input_ids,
            "prompt": prompt,
            "question": example["question"],
            "question_id": example.get("id", example["question"]),
            "answer": example["correct_answer"]
        }

    val_data = val_data.map(encode, remove_columns=val_data.column_names)

    os.makedirs(save_path, exist_ok=True)
    json_path = os.path.join(save_path, "sciq.json")
    val_data.to_json(json_path)

    print(f"[INFO] Saved SciQ JSONL to {json_path}")


def load_sciq_dataset(path=None, model_name=None):
    """
    Loads or prepares SciQ dataset from JSONL.
    """
    if model_name is None:
        raise ValueError("[ERROR] Model name must be specified for SciQ tokenizer.")

    folder_name = f"sciq_{model_name.split('/')[-1]}"
    path = path or os.path.join(DATA_DIR, folder_name)
    json_file = os.path.join(path, "sciq.json")

    if not os.path.exists(json_file):
        preprocess_sciq_dataset(model_name, path)

    return load_dataset("json", data_files=json_file, split="train")
