import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from config.config import data_dir


def preprocess_triviaqa_dataset(model_name, save_path):
    """
    Prepares and saves the TriviaQA validation set for a specific model's tokenizer.
    """
    print(f"[INFO] Preprocessing TriviaQA for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")

    def encode(example):
        question = example["question"]
        prompt = f"Question: {question} Answer:"
        input_ids = tokenizer(prompt, truncation=False, padding=False)["input_ids"]
        question_id = example["question_id"]

        return {
            "input_ids": input_ids,
            "question": question,
            "question_id": question_id,
            "answer": example["answer"]["value"]
        }

    dataset = dataset.map(encode, remove_columns=dataset.column_names)
    dataset.save_to_disk(save_path)
    print(f"[INFO] Saved TriviaQA to {save_path}")


def load_triviaqa_dataset(path=None, model_name=None):
    """
    Loads or prepares the TriviaQA dataset.
    Args:
        path (str): Optional override path.
        model_name (str): Required for tokenizer-dependent prompt encoding.
    Returns:
        HuggingFace Dataset
    """
    if model_name is None:
        raise ValueError("[ERROR] Model name must be specified for TriviaQA tokenizer.")

    folder_name = f"trivia_qa_{model_name.split('/')[-1]}"
    path = path or os.path.join(data_dir, folder_name)

    if not os.path.exists(path):
        preprocess_triviaqa_dataset(model_name, path)

    return load_from_disk(path)
