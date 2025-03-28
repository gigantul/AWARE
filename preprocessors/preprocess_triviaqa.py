# preprocessors/preprocess_triviaqa.py

import os
import tqdm
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from config.config import DATA_DIR

def preprocess_triviaqa(save_name="trivia_qa_cleaned"):
    dataset_raw = load_dataset("trivia_qa", "unfiltered.nocontext")

    def extract_fields(sample):
        return {
            'question': sample['question'],
            'answer': sample['answer']['value'],
            'id': sample['question_id']
        }

    processed = {}
    for split in ['train', 'validation']:
        samples = [extract_fields(x) for x in tqdm.tqdm(dataset_raw[split])]
        df = pd.DataFrame(samples)
        processed_split = Dataset.from_pandas(df)
        processed[split] = processed_split

    final_dataset = DatasetDict(processed)
    save_path = os.path.join(DATA_DIR, save_name)
    final_dataset.save_to_disk(save_path)
    print(f"TriviaQA dataset saved to {save_path}")

if __name__ == "__main__":
    preprocess_triviaqa()