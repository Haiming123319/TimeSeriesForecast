#!/bin/bash

# DLinear model for AEMO QLD 5min data
# Features: Price, Demand, Scheduled_Gen, Semi_Scheduled_Gen, Net_Import (5 features)
# 5min intervals: 288 steps = 1 day

export CUDA_VISIBLE_DEVICES=""

model_name=DLinear

# Predict 12 hours ahead (144 steps)
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/AEMO/ \
  --data_path QLD_5min.csv \
  --model_id QLD_5min_288_144 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 144 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --itr 1 \
  --target Price

# Predict 1 day ahead (288 steps)
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/AEMO/ \
  --data_path QLD_5min.csv \
  --model_id QLD_5min_288_288 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 288 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --itr 1 \
  --target Price
