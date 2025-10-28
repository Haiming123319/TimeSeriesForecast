#!/bin/bash

# TimesNet model for AEMO TAS 30min data
# Features: Price, Demand, Scheduled_Gen, Semi_Scheduled_Gen, Net_Import (5 features)
# 30min intervals: 48 steps = 1 day

export CUDA_VISIBLE_DEVICES=""

model_name=TimesNet

# Predict 12 hours ahead (24 steps)
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/AEMO/ \
  --data_path TAS_30min.csv \
  --model_id TAS_30min_48_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \  --des 'Exp' \
  --itr 1 \
  --target Price

# Predict 1 day ahead (48 steps)
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/AEMO/ \
  --data_path TAS_30min.csv \
  --model_id TAS_30min_48_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \  --des 'Exp' \
  --itr 1 \
  --target Price

# Predict 2 days ahead (96 steps)
python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/AEMO/ \
  --data_path TAS_30min.csv \
  --model_id TAS_30min_48_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \  --des 'Exp' \
  --itr 1 \
  --target Price
