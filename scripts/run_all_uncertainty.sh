#!/bin/bash

# run_all_uncertainty.sh — Run SAR and baseline uncertainty methods on SciQ & TriviaQA
set -e
set -o pipefail

MODEL="facebook/opt-1.3b"
BATCH_SIZE=1
SIM_METHOD="sbert"  # or "attention" if you want for attentionsar

DATASETS=("sciq" "triviaqa")
METHODS=("lastde" "entropy" "lastn_entropy" "sequence_entropy" "bertsar" "attentionsar")

echo -e "\n🚀 Running uncertainty pipeline for: ${DATASETS[*]}"
cd ..

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo -e "\n📊 Running: dataset=$dataset | method=$method"

    python main_pipeline.py \
      --dataset "$dataset" \
      --model "$MODEL" \
      --uncertainty_method "$method" \
      --similarity_method "$SIM_METHOD" \
      --batch_size "$BATCH_SIZE"
  done
done

echo -e "\n✅ All methods run on all datasets complete. Check results_*.csv"
