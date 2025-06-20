#!/bin/bash

# Define models, metrics, and decoding methods
models=("alexnet" "convnext_tiny" "densenet121" "efficientnet_vs_2" "inception_v3" "resnet18" "swin_v2_t" "vgg11" "vit_b16")
metrics=("Accuracy" "Hamming Loss" "hF_1" "hF_2" "hF_0_5" "Mistake Severity" "Wu-Palmer" "Zhao")
decodings=("Optimal" "Thresholding 0.5" "(Karthik et al., 2021)" "Exp. Information" "Hie-Self (Jain et al., 2023)" "information_threshold" "Plurality" "Top-down argmax" "Argmax leaves")
blurr_levels=$(seq 0 10)

# Settings
dataset="tieredimagenet"
split="test"
batch_size=64
gpu=0

# Inference
mkdir -p logs/inference
for model in "${models[@]}"; do
  for blurr in $blurr_levels; do
    echo "Running inference: model=$model, blurr_level=$blurr, dataset=$dataset, split=$split"
    python run_inference.py \
      --dataset "$dataset" \
      --model_name "$model" \
      --split "$split" \
      --batch_size "$batch_size" \
      --blurr_level "$blurr" \
      --gpu "$gpu"
  done
done

# Evaluation
mkdir -p logs/evaluation
for model in "${models[@]}"; do
  for blurr in $blurr_levels; do
    for metric in "${metrics[@]}"; do
      for decoding in "${decodings[@]}"; do
        echo "Evaluating: model=$model, blurr_level=$blurr, metric=$metric, decoding=$decoding"
        python run_evaluation.py \
          --dataset "$dataset" \
          --model_name "$model" \
          --split "$split" \
          --blurr_level "$blurr" \
          --metric_name "$metric" \
          --decoding_method "$decoding" \
          --path_save "results/${dataset}/${model}/evaluation_${metric}_${decoding// /_}_blurr${blurr}.json"
      done
    done
  done
done
