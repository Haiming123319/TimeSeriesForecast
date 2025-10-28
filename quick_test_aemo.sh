#!/bin/bash

# Quick Test Script for AEMO Data
# This script runs a quick test with DLinear on NSW 30min data
# to verify everything is working correctly

echo "========================================"
echo "AEMO Data Quick Test"
echo "========================================"
echo "Model: DLinear"
echo "Data: NSW 30min"
echo "Prediction: 24 steps (12 hours)"
echo "========================================"
echo ""

cd /Users/yhm/Desktop/Time-Series-Library

# Disable CUDA to force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Run a single quick test (using CPU)
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/AEMO/ \
  --data_path NSW_30min.csv \
  --model_id NSW_30min_test_24_12 \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 12 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Quick_Test' \
  --itr 1 \
  --train_epochs 3 \
  --batch_size 8 \
  --target Price

echo ""
echo "========================================"
echo "Test completed!"
echo "========================================"
echo "If you see results above, the data format is correct"
echo "and you can proceed to run the full training scripts."
echo ""
echo "Next steps:"
echo "1. Check the results in ./results/ directory"
echo "2. Run full training scripts in scripts/AEMO_forecast/"
echo "3. Compare different models' performance"
echo "========================================"

