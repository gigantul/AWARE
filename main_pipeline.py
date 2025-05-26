import argparse
import csv
import torch
from config.config import MODEL_NAME, DATASET_PATH
from loaders.sciq_loader import load_sciq_dataset
from loaders.coqa_loader import load_coqa_dataset
from loaders.triviaqa_loader import load_triviaqa_dataset
from loaders.sampleqa_loader import load_sampleqa_dataset
from models.generator import run_generation
from analysis.likelihoods import compute_likelihoods
from analysis.similarity import compute_similarity
from analysis.correctness import evaluate_response
from utils.logger import setup_logger

logger = setup_logger("sar_logs/main.log")

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main(args):
    # Load dataset
    print("[Step 1] Loading dataset...")
    if args.dataset == "sampleqa":
        dataset = load_sampleqa_dataset()
    elif args.dataset == "sciq":
        dataset = load_sciq_dataset(DATASET_PATH, model_name=args.model)
    elif args.dataset == "coqa":
        dataset = load_coqa_dataset()
    elif args.dataset == "triviaqa":
        dataset = load_triviaqa_dataset(model_name=args.model)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    results = []
    save_path = f"results_{args.dataset}.csv"
    header_written = False

    # List of uncertainty methods to compute in one shot
    uncertainty_methods = ["entropy", "lastde", "lastn_entropy", "logit_gap", "bertsar", "aware"]

    print(f"[Step 2] Processing QA pairs in batches of {args.batch_size}...")
    processed_count = 0
    max_items = 10 if args.testrun else float('inf')

    for batch_idx, batch in enumerate(batchify(list(dataset), args.batch_size)):
        # Trim batch if it would exceed max_items
        if processed_count + len(batch) > max_items:
            batch = batch[:max_items - processed_count]

        # 1. Run generation (now handles uncertainty computation internally)
        outputs = run_generation(
            batch,
            model_name=args.model,
            return_logits=True,
            return_attentions=(args.similarity_method == 'attention'),
            uncertainty_methods=uncertainty_methods
        )

        for i, (sample, output) in enumerate(zip(batch, outputs)):
            # 2. Compute correctness
            correctness = evaluate_response(sample, output)

            # 3. Compute similarity (optional)
            similarity_score = None
            if args.similarity_method == 'attention':
                similarity_score = compute_similarity(
                    model_outputs=output,
                    method='attention'
                )

            # 4. Store full row with multiple uncertainty scores
            row = {
                'id': batch_idx * args.batch_size + i,
                'question': sample['question'],
                'generated_answer': output['generated_text'],
                'correct': correctness,
                'similarity_score': similarity_score if similarity_score is not None else 'NA'
            }

            # Add all uncertainty scores to the row
            uncertainty_scores = output.get("uncertainty_scores", {})
            for method in uncertainty_methods:
                row[f"uncertainty_{method}"] = uncertainty_scores.get(method, "NA")

            results.append(row)

            # 5. Periodically save to CSV
            with open(save_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not header_written:
                    writer.writeheader()
                    header_written = True
                writer.writerow(row)

            processed_count += 1

        # 6. Free up memory
        del outputs, batch
        torch.cuda.empty_cache()


        if (batch_idx + 1) * args.batch_size % 100 == 0:
            print(f"Processed {(batch_idx + 1) * args.batch_size} questions...")

    print(f"\nAll QA pairs processed. Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--similarity_method", type=str, default="sbert")
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, choices=["sciq", "coqa", "triviaqa", "sampleqa"], default="sciq")
    args = parser.parse_args()
    main(args)