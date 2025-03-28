#!/bin/bash

# run_sar_sample.sh
# Lightweight test run of SAR pipeline on sampleQA.json

set -e
set -o pipefail

MODEL="facebook/opt-1.3b"
BATCH_SIZE=1
DATASET="sampleqa"
METHODS=("sar" "token-sar" "semantic-entropy" "predictive-entropy")
METRICS=("rougeL_to_target" "sentsim")
TEMP=0.001

echo -e "\n🚀 [TEST RUN] Running SAR pipeline on sample QA pairs"

cd ..

# Run main pipeline
python main_pipeline.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --uncertainty_method "sar" \
  --similarity_method "attention" \
  --batch_size "$BATCH_SIZE"

# Run correctness eval (if needed for AUC comparison)
python analysis/correctness.py --dataset "$DATASET"

# Run uncertainty metrics
RUN_NAME="${MODEL//\//-}/$DATASET"
python analysis/uncertainty.py \
  --run-name "$RUN_NAME" \
  --methods "${METHODS[@]}" \
  --metrics "${METRICS[@]}" \
  --temperature "$TEMP"

echo -e "\n✅ Sample run completed. Check output_dir/ and results.csv for results."
