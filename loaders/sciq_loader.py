import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

def preprocess_sciq_dataset(model_name, save_path):
    """
    Downloads and prepares the SciQ dataset for a given model's tokenizer.
    Encodes prompts for few-shot QA and saves to disk.
    """
    print(f"[INFO] Preprocessing SciQ for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_data = load_dataset("sciq", split="train")
    val_data = load_dataset("sciq", split="validation")

    # ✅ Few-shot setup: Build reusable prefix with correct answers
    few_shot_prompt = "This is a bot that correctly answers questions.\n"
    for sample in train_data.select(range(10)):
        few_shot_prompt += f"Question: {sample['question']} Answer: {sample['correct_answer']} "

    def encode(example):
        # ✅ Clean separation between few-shot context and actual prompt
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

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    val_data.save_to_disk(save_path)
    print(f"[INFO] Saved SciQ to {save_path}")
