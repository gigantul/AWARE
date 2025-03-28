#!/bin/bash

# run_sar_pipeline.sh
# Automates SAR benchmark across datasets and methods for Duan et al. replication

set -e

MODEL="meta-llama/Llama-2-13b-hf"
BATCH_SIZE=32

DATASETS=("sciq" "triviaqa")
METHODS=("sar" "semantic-entropy" "predictive-entropy" "len-normed-predictive-entropy" "token-sar" "sentence-sar" "lexical-similarity")
METRICS=("rougeL_to_target" "bertscore_f1" "sentsim")
TEMP=0.001

# Step 1: Generate results via main_pipeline
for DATASET in "${DATASETS[@]}"; do
  echo "\n[Running main pipeline on $DATASET]"
  python main_pipeline.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --uncertainty_method sar \
    --batch_size $BATCH_SIZE

  echo "[Generating correctness labels for $DATASET]"
  python analysis/correctness.py --dataset "$DATASET"
done

# Step 2: Uncertainty Estimation (Tables & AUCs)
for DATASET in "${DATASETS[@]}"; do
  RUN_NAME="facebook-opt-1.3b/$DATASET"
  echo "\n[Computing uncertainty scores for $RUN_NAME]"
  python analysis/uncertainty.py \
    --run-name "$RUN_NAME" \
    --methods ${METHODS[@]} \
    --metrics ${METRICS[@]} \
    --temperature $TEMP

done

echo "\n✅ All experiments completed. Results in output dir + CSVs."
