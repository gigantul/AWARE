# preprocessors/preprocess_coqa.py

import os
import json
import evaluate
import pandas as pd
import torch
import tqdm
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config.config import data_dir

def preprocess_coqa(raw_path=f"{data_dir}/coqa-dev-v1.0.json", save_name="coqa_dataset"):
    with open(raw_path, 'r') as infile:
        data = json.load(infile)['data']

    rouge = evaluate.load('rouge')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()

    dataset = {
        'story': [], 'question': [], 'answer': [], 'additional_answers': [],
        'rouge1': [], 'rouge2': [], 'rougeL': [], 'semantic_variability': [], 'id': []
    }

    for sample_id, sample in enumerate(tqdm.tqdm(data)):
        story = sample['story']
        questions = sample['questions']
        answers = sample['answers']
        additional_answers = sample['additional_answers']

        for question_index, question in enumerate(questions):
            q_text = question['input_text']
            a_text = answers[question_index]['input_text']
            dataset['story'].append(story)
            dataset['question'].append(q_text)
            dataset['answer'].append({
                'text': a_text,
                'answer_start': answers[question_index]['span_start']
            })
            dataset['id'].append(sample['id'] + '_' + str(question_index))

            additional_answers_list = [
                additional_answers[str(i)][question_index]['input_text'] for i in range(3)
            ]
            dataset['additional_answers'].append(additional_answers_list)

            # Update story context
            story = story + ' Q: ' + q_text + ' A: ' + a_text
            if not story.endswith('.'):
                story += '.'

            all_answers = [a_text] + additional_answers_list
            inputs, a1_list, a2_list = [], [], []
            for i, a1 in enumerate(all_answers):
                for j, a2 in enumerate(all_answers):
                    if i != j:
                        qa1 = q_text + ' ' + a1
                        qa2 = q_text + ' ' + a2
                        inputs.append(qa1 + ' [SEP] ' + qa2)
                        a1_list.append(a1)
                        a2_list.append(a2)

            encoded = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors='pt').to('cuda')
            logits = model(**encoded).logits
            predictions = torch.argmax(logits, dim=1)
            has_different = (predictions == 0).any().item()

            rouge_scores = rouge.compute(predictions=a1_list, references=a2_list)

            dataset['semantic_variability'].append(has_different)
            dataset['rouge1'].append(rouge_scores['rouge1'])
            dataset['rouge2'].append(rouge_scores['rouge2'])
            dataset['rougeL'].append(rouge_scores['rougeL'])

    df = pd.DataFrame(dataset)
    hf_dataset = Dataset.from_pandas(df)
    save_path = os.path.join(data_dir, save_name)
    hf_dataset.save_to_disk(save_path)
    print(f"CoQA dataset saved to {save_path}")

if __name__ == "__main__":
    preprocess_coqa()
