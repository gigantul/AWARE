import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from config.config import DATA_DIR


def preprocess_sciq_dataset(model_name, save_path):
    """
    Downloads and prepares the SciQ dataset for a given model's tokenizer.
    Encodes prompts for few-shot QA.
    """
    print(f"[INFO] Preprocessing SciQ for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_data = load_dataset("sciq", split="train")
    val_data = load_dataset("sciq", split="validation")

    few_shot_prompt = "This is a bot that correctly answers questions.\n"
    for sample in train_data.select(range(10)):
        few_shot_prompt += f"Question: {sample['question']} Answer: {sample['correct_answer']} "

    def encode(example):
        prompt = few_shot_prompt + f"Question: {example['question']} Answer:"
        input_ids = tokenizer(prompt, truncation=False, padding=False)["input_ids"]
        return {
            "input_ids": input_ids,
            "question": example["question"],
            "question_id": example["id"] if "id" in example else example["question"],
            "answer": example["correct_answer"]
        }

    val_data = val_data.map(encode, remove_columns=val_data.column_names)
    val_data.save_to_disk(save_path)
    print(f"[INFO] Saved SciQ to {save_path}")


def load_sciq_dataset(path=None, model_name=None):
    """
    Loads or prepares the SciQ dataset for a given model.
    Args:
        path (str): Optional override path.
        model_name (str): Required for tokenizer-dependent prompt construction.
    Returns:
        HuggingFace Dataset
    """
    if model_name is None:
        raise ValueError("[ERROR] Model name must be specified for SciQ tokenizer.")

    folder_name = f"sciq_{model_name.split('/')[-1]}"
    path = path or os.path.join(DATA_DIR, folder_name)

    if not os.path.exists(path):
        preprocess_sciq_dataset(model_name, path)

    return load_from_disk(path)
