"""
Generate Training Scripts for All AEMO Regions
This script generates training scripts for all regions and models automatically.
"""

import os

def generate_script(model_name, region, interval, script_dir):
    """
    Generate a training script for a specific model, region, and interval.
    
    Args:
        model_name: Model name (DLinear, PatchTST, TimesNet)
        region: Region code (NSW, QLD, VIC, SA, TAS)
        interval: Time interval (30min, 5min)
        script_dir: Directory to save scripts
    """
    
    # Set parameters based on interval
    if interval == '30min':
        seq_len = 48
        label_len = 24
        pred_lens = [24, 48, 96]  # 12h, 1d, 2d
    else:  # 5min
        seq_len = 288
        label_len = 144
        pred_lens = [144, 288]  # 12h, 1d
    
    # Model-specific parameters
    model_params = {
        'DLinear': {
            'batch_size': '',
            'extra_params': ''
        },
        'PatchTST': {
            'batch_size': '  --batch_size 16 \\',
            'extra_params': ''
        },
        'TimesNet': {
            'batch_size': '',
            'extra_params': '''  --d_model 64 \\
  --d_ff 128 \\
  --top_k 5 \\'''
        }
    }
    
    params = model_params[model_name]
    
    # Generate script content
    script_content = f"""#!/bin/bash

# {model_name} model for AEMO {region} {interval} data
# Features: Price, Demand, Scheduled_Gen, Semi_Scheduled_Gen, Net_Import (5 features)
# {interval} intervals: {seq_len} steps = 1 day

export CUDA_VISIBLE_DEVICES=0

model_name={model_name}
"""
    
    # Add experiments for different prediction lengths
    for pred_len in pred_lens:
        pred_hours = pred_len / (288 if interval == '5min' else 48) * 24
        pred_desc = f"{int(pred_hours)} hours" if pred_hours < 24 else f"{int(pred_hours/24)} day{'s' if pred_hours > 24 else ''}"
        
        script_content += f"""
# Predict {pred_desc} ahead ({pred_len} steps)
python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path ./data/AEMO/ \\
  --data_path {region}_{interval}.csv \\
  --model_id {region}_{interval}_{seq_len}_{pred_len} \\
  --model $model_name \\
  --data custom \\
  --features M \\
  --seq_len {seq_len} \\
  --label_len {label_len} \\
  --pred_len {pred_len} \\
  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --enc_in 5 \\
  --dec_in 5 \\
  --c_out 5 \\
{params['extra_params']}  --des 'Exp' \\
{params['batch_size']}  --itr 1 \\
  --target Price
"""
    
    # Save script
    filename = f"{model_name}_{region}_{interval}.sh"
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(filepath, 0o755)
    
    return filename

def main():
    """Generate all training scripts."""
    
    # Configuration
    models = ['DLinear', 'PatchTST', 'TimesNet']
    regions = ['NSW', 'QLD', 'VIC', 'SA', 'TAS']
    intervals = ['30min', '5min']
    
    # Script directory
    script_dir = '/Users/yhm/Desktop/Time-Series-Library/scripts/AEMO_forecast'
    os.makedirs(script_dir, exist_ok=True)
    
    print("="*60)
    print("Generating AEMO Training Scripts")
    print("="*60)
    print(f"Models: {', '.join(models)}")
    print(f"Regions: {', '.join(regions)}")
    print(f"Intervals: {', '.join(intervals)}")
    print()
    
    generated_scripts = []
    
    # Generate scripts for each combination
    for model in models:
        for region in regions:
            for interval in intervals:
                filename = generate_script(model, region, interval, script_dir)
                generated_scripts.append(filename)
                print(f"✓ Generated: {filename}")
    
    print()
    print("="*60)
    print(f"Successfully generated {len(generated_scripts)} scripts!")
    print("="*60)
    print()
    print("Scripts saved to:", script_dir)
    print()
    print("Example usage:")
    print(f"  bash {script_dir}/DLinear_NSW_30min.sh")
    print(f"  bash {script_dir}/PatchTST_QLD_5min.sh")
    print(f"  bash {script_dir}/TimesNet_VIC_30min.sh")
    print()
    print("To run all scripts:")
    print(f"  for script in {script_dir}/*.sh; do bash \"$script\"; done")
    print()

if __name__ == '__main__':
    main()

