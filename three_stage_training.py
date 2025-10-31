#!/usr/bin/env python3
"""
三阶段优化训练策略
阶段1: 模型筛选（2州，30min，pred_len=24，4模型）
阶段2: 扩展验证（5州，30min+15min，pred_len=24，前2模型）
阶段3: 步长扩展（最优模型，所有组合）
"""

import os
import subprocess
import time
import json
import pandas as pd
import glob
import re
from pathlib import Path


# ============================================================
# 轻量级模型配置（加速训练）
# ============================================================
MODELS_CONFIG = {
    'DLinear': {
        'params': '--e_layers 2 --d_layers 1',
        'batch_size': 64
    },
    'PatchTST': {
        'params': '--e_layers 2 --n_heads 4 --d_model 128 --d_ff 128 --patch_len 16',
        'batch_size': 32
    },
    'TimesNet': {
        # 轻量级配置：减少e_layers, d_model, num_kernels，适合T4 GPU
        'params': '--e_layers 1 --d_model 128 --d_ff 128 --num_kernels 4 --top_k 3',
        'batch_size': 16
    },
    'iTransformer': {
        # 轻量级配置：减少layers和hidden size
        'params': '--e_layers 2 --d_model 128 --d_ff 256 --n_heads 4',
        'batch_size': 16
    }
}


def get_model_specific_params(model_name, device_type, seq_len, label_len, batch_size):
    """
    获取模型特定参数，针对不同模型和设备优化
    
    特殊处理：
    - TimesNet: 关闭AMP（避免cuFFT FP16错误），使用轻量配置
    - 其他模型: 开启AMP加速
    """
    name = model_name.lower()
    
    # 基础参数
    num_workers = 2 if device_type == 'cuda' else (4 if device_type == 'mps' else 2)
    
    params = {
        'num_workers': num_workers,
        'seq_len': seq_len,
        'label_len': label_len,
        'batch_size': batch_size,
        'use_amp': True,  # 默认开启AMP
    }
    
    # TimesNet特殊处理：关闭AMP避免cuFFT FP16错误
    if name == 'timesnet':
        params['use_amp'] = False  # 关键：TimesNet不用AMP
        params['batch_size'] = 16  # 轻量配置
        print(f"  ⚠️  TimesNet关闭AMP (避免cuFFT FP16错误)，使用FP32训练")
    
    return params


def run_experiment(data_path, model_name, model_id, seq_len, label_len, pred_len, 
                   batch_size, epochs=10, patience=2, use_gpu=True):
    """运行单个实验"""
    
    config = MODELS_CONFIG[model_name]
    
    # 检测设备类型
    import torch
    device_type = 'cuda'
    if use_gpu:
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
        else:
            use_gpu = False
    
    # 获取模型特定参数（特别处理TimesNet）
    model_params = get_model_specific_params(model_name, device_type, seq_len, label_len, batch_size)
    
    # 构建AMP参数（TimesNet不用，其他模型用）
    amp_flag = '--use_amp' if model_params['use_amp'] else ''
    
    # 构建GPU参数
    if device_type == 'mps':
        gpu_params = '--use_gpu 1 --gpu_type mps'
    else:
        gpu_params = f"--use_gpu {1 if use_gpu else 0} --gpu 0"
    
    cmd = f"""python3 run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/AEMO_optimized/ \
      --data_path {data_path} \
      --model_id {model_id} \
      --model {model_name} \
      --data custom \
      --features M \
      --target Price \
      --seq_len {model_params['seq_len']} \
      --label_len {model_params['label_len']} \
      --pred_len {pred_len} \
      --enc_in 5 \
      --dec_in 5 \
      --c_out 5 \
      --train_epochs {epochs} \
      --batch_size {model_params['batch_size']} \
      --patience {patience} \
      --learning_rate 0.001 \
      --num_workers {model_params['num_workers']} \
      {gpu_params} \
      {amp_flag} \
      {config['params']}"""
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start_time
    
    return result.returncode == 0, elapsed


def extract_metrics(model_id):
    """从结果文件中提取指标"""
    result_files = glob.glob(f'./results/*{model_id}*/result*.txt')
    
    if not result_files:
        return None
    
    try:
        with open(result_files[0], 'r') as f:
            content = f.read()
        
        mse_match = re.search(r'mse:([\d.]+)', content)
        mae_match = re.search(r'mae:([\d.]+)', content)
        
        if mse_match and mae_match:
            return {
                'mse': float(mse_match.group(1)),
                'mae': float(mae_match.group(1))
            }
    except:
        pass
    
    return None


# ============================================================
# 阶段1: 模型筛选
# ============================================================
def stage1_model_selection(use_gpu=True):
    """
    阶段1: 模型筛选
    - 2个代表性州 (NSW=体量大, SA=波动大)
    - 只用30分钟数据
    - pred_len=24
    - 4个模型快速对比
    """
    
    print("\n" + "="*80)
    print("🎯 阶段1: 模型筛选（快速对比4个模型）")
    print("="*80)
    print("配置:")
    print("  - 州: NSW (体量大), SA (波动大)")
    print("  - 频率: 30min")
    print("  - 预测步长: 24 (12小时)")
    print("  - 训练轮数: 8 epochs, 早停patience=2")
    print("  - 目标: 选出前2个最优模型")
    print("="*80)
    print()
    
    states = ['NSW', 'SA']
    models = ['DLinear', 'PatchTST', 'TimesNet', 'iTransformer']
    pred_len = 24
    seq_len = 192  # 4天历史（30min间隔）
    label_len = 96
    
    results = []
    total = len(states) * len(models)
    count = 0
    
    for state in states:
        for model in models:
            count += 1
            
            data_file = f'{state}_30min.csv'
            model_id = f'stage1_{state}_{model}'
            
            print(f"[{count}/{total}] 🚀 {state} | {model}")
            print("-"*80)
            
            success, elapsed = run_experiment(
                data_path=data_file,
                model_name=model,
                model_id=model_id,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                batch_size=MODELS_CONFIG[model]['batch_size'],
                epochs=8,
                patience=2,
                use_gpu=use_gpu
            )
            
            metrics = extract_metrics(model_id) if success else None
            
            result = {
                'stage': 1,
                'state': state,
                'freq': '30min',
                'model': model,
                'pred_len': pred_len,
                'success': success,
                'time_minutes': elapsed / 60,
                'mse': metrics['mse'] if metrics else None,
                'mae': metrics['mae'] if metrics else None
            }
            
            results.append(result)
            
            status = "✅" if success else "❌"
            if metrics:
                print(f"{status} 完成 | MAE: {metrics['mae']:.4f} | 用时: {elapsed/60:.1f}分钟\n")
            else:
                print(f"{status} 完成 | 用时: {elapsed/60:.1f}分钟\n")
    
    # 分析结果，选出前2个模型
    df = pd.DataFrame(results)
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) > 0 and df_success['mae'].notna().any():
        # 按MAE排序
        df_success = df_success.sort_values('mae')
        
        print("\n" + "="*80)
        print("📊 阶段1结果汇总")
        print("="*80)
        print(df_success[['state', 'model', 'mae', 'mse', 'time_minutes']].to_string(index=False))
        
        # 选出前2个模型（按平均MAE）
        model_avg = df_success.groupby('model')['mae'].mean().sort_values()
        top2_models = model_avg.head(2).index.tolist()
        
        print(f"\n✨ 选中的前2个模型:")
        for i, model in enumerate(top2_models, 1):
            avg_mae = model_avg[model]
            print(f"  {i}. {model:15s} - 平均MAE: {avg_mae:.4f}")
        
        return top2_models, results
    else:
        print("\n⚠️  阶段1没有成功的实验")
        return [], results


# ============================================================
# 阶段2: 扩展验证
# ============================================================
def stage2_expansion(top2_models, use_gpu=True):
    """
    阶段2: 扩展验证
    - 所有5个州
    - 30min + 15min 两个频率
    - pred_len=24
    - 只用阶段1选出的前2个模型
    """
    
    if not top2_models:
        print("\n⚠️  阶段1未选出模型，跳过阶段2")
        return []
    
    print("\n" + "="*80)
    print("🎯 阶段2: 扩展验证（所有州+两个频率）")
    print("="*80)
    print("配置:")
    print(f"  - 模型: {', '.join(top2_models)}")
    print("  - 州: 全部5个 (NSW, QLD, VIC, SA, TAS)")
    print("  - 频率: 30min, 15min")
    print("  - 预测步长: 24")
    print("  - 训练轮数: 10 epochs, 早停patience=2")
    print("="*80)
    print()
    
    states = ['NSW', 'QLD', 'VIC', 'SA', 'TAS']
    freqs = [
        ('30min', 192, 96),  # (freq, seq_len, label_len)
        ('15min', 384, 192)
    ]
    pred_len = 24
    
    results = []
    total = len(states) * len(freqs) * len(top2_models)
    count = 0
    
    for state in states:
        for freq, seq_len, label_len in freqs:
            for model in top2_models:
                count += 1
                
                data_file = f'{state}_{freq}.csv'
                model_id = f'stage2_{state}_{freq}_{model}'
                
                print(f"[{count}/{total}] 🚀 {state} | {freq} | {model}")
                print("-"*80)
                
                success, elapsed = run_experiment(
                    data_path=data_file,
                    model_name=model,
                    model_id=model_id,
                    seq_len=seq_len,
                    label_len=label_len,
                    pred_len=pred_len,
                    batch_size=MODELS_CONFIG[model]['batch_size'],
                    epochs=10,
                    patience=2,
                    use_gpu=use_gpu
                )
                
                metrics = extract_metrics(model_id) if success else None
                
                result = {
                    'stage': 2,
                    'state': state,
                    'freq': freq,
                    'model': model,
                    'pred_len': pred_len,
                    'success': success,
                    'time_minutes': elapsed / 60,
                    'mse': metrics['mse'] if metrics else None,
                    'mae': metrics['mae'] if metrics else None
                }
                
                results.append(result)
                
                status = "✅" if success else "❌"
                if metrics:
                    print(f"{status} 完成 | MAE: {metrics['mae']:.4f} | 用时: {elapsed/60:.1f}分钟\n")
                else:
                    print(f"{status} 完成 | 用时: {elapsed/60:.1f}分钟\n")
    
    # 分析结果
    df = pd.DataFrame(results)
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) > 0 and df_success['mae'].notna().any():
        print("\n" + "="*80)
        print("📊 阶段2结果汇总")
        print("="*80)
        
        # 按频率分组显示
        for freq in ['30min', '15min']:
            df_freq = df_success[df_success['freq'] == freq]
            if len(df_freq) > 0:
                print(f"\n{freq} 数据:")
                print(df_freq[['state', 'model', 'mae', 'mse']].sort_values('mae').to_string(index=False))
        
        # 选出最优模型
        model_avg = df_success.groupby('model')['mae'].mean().sort_values()
        best_model = model_avg.index[0]
        
        print(f"\n✨ 最优模型: {best_model}")
        print(f"   平均MAE: {model_avg[best_model]:.4f}")
        
        return best_model, results
    else:
        print("\n⚠️  阶段2没有成功的实验")
        return None, results


# ============================================================
# 阶段3: 步长扩展
# ============================================================
def stage3_pred_len_expansion(best_model, use_gpu=True):
    """
    阶段3: 步长扩展
    - 只用最优模型
    - 所有州
    - 30min频率（主要）
    - pred_len = 24, 48, 96
    """
    
    if not best_model:
        print("\n⚠️  未找到最优模型，跳过阶段3")
        return []
    
    print("\n" + "="*80)
    print("🎯 阶段3: 预测步长扩展")
    print("="*80)
    print("配置:")
    print(f"  - 模型: {best_model}")
    print("  - 州: 全部5个")
    print("  - 频率: 30min")
    print("  - 预测步长: 24, 48, 96")
    print("  - 训练轮数: 15 epochs, 早停patience=3")
    print("="*80)
    print()
    
    states = ['NSW', 'QLD', 'VIC', 'SA', 'TAS']
    pred_lens = [24, 48, 96]
    seq_len = 192
    label_len = 96
    
    results = []
    total = len(states) * len(pred_lens)
    count = 0
    
    for state in states:
        for pred_len in pred_lens:
            count += 1
            
            data_file = f'{state}_30min.csv'
            model_id = f'stage3_{state}_{best_model}_pl{pred_len}'
            
            print(f"[{count}/{total}] 🚀 {state} | pred_len={pred_len}")
            print("-"*80)
            
            success, elapsed = run_experiment(
                data_path=data_file,
                model_name=best_model,
                model_id=model_id,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                batch_size=MODELS_CONFIG[best_model]['batch_size'],
                epochs=15,
                patience=3,
                use_gpu=use_gpu
            )
            
            metrics = extract_metrics(model_id) if success else None
            
            result = {
                'stage': 3,
                'state': state,
                'freq': '30min',
                'model': best_model,
                'pred_len': pred_len,
                'success': success,
                'time_minutes': elapsed / 60,
                'mse': metrics['mse'] if metrics else None,
                'mae': metrics['mae'] if metrics else None
            }
            
            results.append(result)
            
            status = "✅" if success else "❌"
            if metrics:
                print(f"{status} 完成 | MAE: {metrics['mae']:.4f} | 用时: {elapsed/60:.1f}分钟\n")
            else:
                print(f"{status} 完成 | 用时: {elapsed/60:.1f}分钟\n")
    
    # 分析结果
    df = pd.DataFrame(results)
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) > 0 and df_success['mae'].notna().any():
        print("\n" + "="*80)
        print("📊 阶段3结果汇总")
        print("="*80)
        
        # 按预测步长分组
        for pred_len in pred_lens:
            df_pred = df_success[df_success['pred_len'] == pred_len]
            if len(df_pred) > 0:
                print(f"\n预测{pred_len}步 ({'12小时' if pred_len==24 else '24小时' if pred_len==48 else '48小时'}):")
                print(df_pred[['state', 'mae', 'mse']].sort_values('mae').to_string(index=False))
        
        return results
    else:
        print("\n⚠️  阶段3没有成功的实验")
        return results


# ============================================================
# 主流程
# ============================================================
def main():
    """主流程：执行三阶段训练"""
    
    print("="*80)
    print("🚀 AEMO时序预测 - 三阶段优化训练")
    print("="*80)
    print("策略说明:")
    print("  阶段1: 模型筛选（2州 × 4模型 = 8个实验，约1-2小时）")
    print("  阶段2: 扩展验证（5州 × 2频率 × 2模型 = 20个实验，约3-4小时）")
    print("  阶段3: 步长扩展（5州 × 3步长 × 1模型 = 15个实验，约2-3小时）")
    print("  总计: 约43个实验，相比原方案（120个）减少64%")
    print("="*80)
    
    # 检查数据目录
    if not os.path.exists('./data/AEMO_optimized'):
        print("\n❌ 错误: 数据目录不存在")
        print("请先运行: python3 generate_optimized_aemo_data.py")
        return
    
    # 检测设备并优化
    import torch
    
    # 启用cuDNN benchmark优化（对TimesNet特别有效）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("✅ cuDNN benchmark 已启用")
    
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    
    # 打印详细设备信息
    if torch.cuda.is_available():
        device_name = f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
        print(f"\n🖥️  设备: {device_name}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"⚡ AMP: 已启用 (FP16混合精度)")
        print(f"👷 Workers: 2 (优化Colab性能)")
    elif torch.backends.mps.is_available():
        device_name = "MPS (Apple Silicon)"
        print(f"\n🖥️  设备: {device_name}")
        print(f"⚡ AMP: 已启用")
        print(f"👷 Workers: 4")
    else:
        device_name = "CPU"
        print(f"\n🖥️  设备: {device_name}")
        print(f"👷 Workers: 2")
    
    # 创建结果目录
    os.makedirs('./three_stage_results', exist_ok=True)
    
    start_time = time.time()
    all_results = []
    
    # 阶段1: 模型筛选
    top2_models, stage1_results = stage1_model_selection(use_gpu=use_gpu)
    all_results.extend(stage1_results)
    
    # 保存阶段1结果
    with open('./three_stage_results/stage1_results.json', 'w') as f:
        json.dump(stage1_results, f, indent=2)
    
    if not top2_models:
        print("\n❌ 阶段1失败，停止训练")
        return
    
    # 阶段2: 扩展验证
    best_model, stage2_results = stage2_expansion(top2_models, use_gpu=use_gpu)
    all_results.extend(stage2_results)
    
    # 保存阶段2结果
    with open('./three_stage_results/stage2_results.json', 'w') as f:
        json.dump(stage2_results, f, indent=2)
    
    if not best_model:
        print("\n❌ 阶段2失败，停止训练")
        return
    
    # 阶段3: 步长扩展
    stage3_results = stage3_pred_len_expansion(best_model, use_gpu=use_gpu)
    all_results.extend(stage3_results)
    
    # 保存阶段3结果
    with open('./three_stage_results/stage3_results.json', 'w') as f:
        json.dump(stage3_results, f, indent=2)
    
    # 总结
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print("\n" + "="*80)
    print("🎉 三阶段训练完成！")
    print("="*80)
    print(f"⏱️  总耗时: {hours}小时 {minutes}分钟")
    print(f"✅ 总实验数: {len(all_results)}")
    print(f"✅ 成功: {len([r for r in all_results if r['success']])}")
    print(f"❌ 失败: {len([r for r in all_results if not r['success']])}")
    
    # 保存完整结果
    df_all = pd.DataFrame(all_results)
    df_all.to_csv('./three_stage_results/all_results.csv', index=False)
    
    print("\n📊 结果已保存到: ./three_stage_results/")
    print("  - stage1_results.json")
    print("  - stage2_results.json")
    print("  - stage3_results.json")
    print("  - all_results.csv")


if __name__ == '__main__':
    main()

