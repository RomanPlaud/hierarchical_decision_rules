#!/bin/bash

# Define models, metrics, and decoding methods
models=("alexnet" "convnext_tiny" "densenet121" "efficientnet_vs_2" "inception_v3" "resnet18" "swin_v2_t" "vgg11" "vit_b16")
models=("alexnet")
metrics=("Accuracy" "Hamming Loss" "hF_1" "hF_2" "hF_0_5" "Mistake Severity" "Wu-Palmer" "Zhao")
metrics=("Accuracy")
decodings=("Optimal" "Thresholding 0.5" "(Karthik et al., 2021)" "Exp. Information" "Hie-Self (Jain et al., 2023)" "information_threshold" "Plurality" "Top-down argmax" "Argmax leaves")
decodings=("Optimal")
blurr_levels=$(seq 0 10)
blurr_levels=("0")

# Settings
dataset="tieredimagenet"
split="test"
batch_size=64
gpu=0

# Inference
for model in "${models[@]}"; do
  for blurr in $blurr_levels; do
    echo "Running inference: model=$model, blurr_level=$blurr, dataset=$dataset, split=$split"
    python scripts/run_inference.py \
      --dataset "$dataset" \
      --model_name "$model" \
      --split "$split" \
      --batch_size "$batch_size" \
      --blurr_level "$blurr" \
      --gpu "$gpu"
  done
done

# Evaluation
for model in "${models[@]}"; do
  for blurr in $blurr_levels; do
    for metric in "${metrics[@]}"; do
      for decoding in "${decodings[@]}"; do
        echo "Evaluating: model=$model, blurr_level=$blurr, metric=$metric, decoding=$decoding"
        python scripts/run_evaluation.py \
          --dataset "$dataset" \
          --model_name "$model" \
          --split "$split" \
          --blurr_level "$blurr" \
          --metric_name "$metric" \
          --decoding_method "$decoding" \
          --path_save "results"
      done
    done
  done
done
