#!/bin/bash

# run_all_uncertainty.sh — Run SAR and baseline uncertainty methods on SciQ & TriviaQA
set -e
set -o pipefail

MODEL="facebook/opt-1.3b"
BATCH_SIZE=1
SIM_METHOD="sbert"  # or "attention" if you want attentionsar features

DATASETS=("sciq" "triviaqa")

echo -e "\n🚀 Running uncertainty pipeline for: ${DATASETS[*]}"
cd ..

for dataset in "${DATASETS[@]}"; do
  echo -e "\n📊 Running all methods on: dataset=$dataset"

  python main_pipeline.py \
    --dataset "$dataset" \
    --model "$MODEL" \
    --similarity_method "$SIM_METHOD" \
    --batch_size "$BATCH_SIZE"
done

echo -e "\n✅ All datasets processed. Check results_*.csv"
