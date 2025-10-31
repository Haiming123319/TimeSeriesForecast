#!/usr/bin/env python3
"""
快速验证脚本 - 确保优化设置正常工作
在Colab中运行，5-10分钟快速测试
"""

import os
import sys


def check_environment():
    """检查环境"""
    print("="*80)
    print("🔍 步骤1: 检查环境")
    print("="*80)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas未安装")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy未安装")
        return False
    
    print()
    return True


def verify_code_fixes():
    """验证代码修复"""
    print("="*80)
    print("🔧 步骤2: 验证代码修复")
    print("="*80)
    
    # 检查 sktime 修复
    try:
        with open('data_provider/data_loader.py', 'r') as f:
            content = f.read()
        
        if 'try:\n    from sktime.datasets' in content or 'except ImportError:' in content:
            print("✅ data_loader.py 已修复（sktime条件导入）")
        else:
            print("⚠️  data_loader.py 可能需要修复")
    except FileNotFoundError:
        print("❌ data_provider/data_loader.py 不存在")
        return False
    
    # 检查 patoolib 修复
    try:
        with open('data_provider/m4.py', 'r') as f:
            content = f.read()
        
        if 'try:' in content and 'import patoolib' in content:
            print("✅ m4.py 已修复（patoolib条件导入）")
        else:
            print("⚠️  m4.py 可能需要修复")
    except FileNotFoundError:
        print("❌ data_provider/m4.py 不存在")
        return False
    
    print()
    return True


def test_data_generation():
    """测试数据生成"""
    print("="*80)
    print("📊 步骤3: 测试数据生成（小样本）")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        # 生成一个小测试数据
        os.makedirs('./data/AEMO_optimized', exist_ok=True)
        
        periods = 1000  # 只生成1000条测试
        dates = pd.date_range('2024-01-01', periods=periods, freq='30min')
        
        np.random.seed(42)
        price = 80 + 20*np.sin(np.arange(periods)/48*2*np.pi) + np.random.normal(0, 15, periods)
        price = np.maximum(price, 10)
        
        demand = price * 12 + np.random.normal(1500, 100, periods)
        
        df = pd.DataFrame({
            'date': dates,
            'Price': price,
            'Demand': demand,
            'Scheduled_Gen': demand * 0.7,
            'Semi_Scheduled_Gen': demand * 0.2,
            'Net_Import': demand * 0.1
        })
        
        test_file = './data/AEMO_optimized/NSW_30min.csv'
        df.to_csv(test_file, index=False)
        
        print(f"✅ 生成测试数据: {test_file}")
        print(f"   行数: {len(df)}")
        print(f"   列: {list(df.columns)}")
        
        # 验证能读取
        df_read = pd.read_csv(test_file)
        print(f"✅ 数据可读取，形状: {df_read.shape}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 数据生成失败: {e}")
        return False


def test_model_import():
    """测试模型导入"""
    print("="*80)
    print("🤖 步骤4: 测试模型导入")
    print("="*80)
    
    try:
        from data_provider.data_loader import Dataset_Custom
        print("✅ Dataset_Custom 导入成功")
    except Exception as e:
        print(f"❌ Dataset_Custom 导入失败: {e}")
        return False
    
    try:
        from data_provider.data_factory import data_provider
        print("✅ data_provider 导入成功")
    except Exception as e:
        print(f"❌ data_provider 导入失败: {e}")
        return False
    
    try:
        from models import DLinear
        print("✅ DLinear 模型导入成功")
    except Exception as e:
        print(f"❌ DLinear 导入失败: {e}")
        return False
    
    print()
    return True


def run_mini_experiment():
    """运行一个超小型实验"""
    print("="*80)
    print("🚀 步骤5: 运行微型实验（1 epoch，验证流程）")
    print("="*80)
    
    cmd = """python3 run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/AEMO_optimized/ \
      --data_path NSW_30min.csv \
      --model_id verify_test \
      --model DLinear \
      --data custom \
      --features M \
      --target Price \
      --seq_len 48 \
      --label_len 24 \
      --pred_len 12 \
      --enc_in 5 \
      --dec_in 5 \
      --c_out 5 \
      --e_layers 2 \
      --d_layers 1 \
      --train_epochs 1 \
      --batch_size 16 \
      --learning_rate 0.001 \
      --use_gpu 1 \
      --gpu 0"""
    
    print("执行命令...")
    result = os.system(cmd)
    
    if result == 0:
        print("\n✅ 微型实验成功！环境配置正确")
        return True
    else:
        print("\n❌ 微型实验失败，请检查错误信息")
        return False


def main():
    """主流程"""
    print("\n" + "="*80)
    print("🎯 快速验证 - AEMO优化设置")
    print("="*80)
    print("此脚本将快速验证环境和配置是否正确")
    print("预计时间: 5-10分钟")
    print("="*80)
    print()
    
    results = []
    
    # 步骤1: 检查环境
    results.append(("环境检查", check_environment()))
    
    # 步骤2: 验证代码修复
    results.append(("代码修复", verify_code_fixes()))
    
    # 步骤3: 测试数据生成
    results.append(("数据生成", test_data_generation()))
    
    # 步骤4: 测试模型导入
    results.append(("模型导入", test_model_import()))
    
    # 步骤5: 运行微型实验
    results.append(("微型实验", run_mini_experiment()))
    
    # 总结
    print("\n" + "="*80)
    print("📋 验证结果汇总")
    print("="*80)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 所有验证通过！")
        print("="*80)
        print("\n✨ 下一步:")
        print("  1. 运行完整数据生成:")
        print("     python3 generate_optimized_aemo_data.py")
        print("  2. 运行三阶段训练:")
        print("     python3 three_stage_training.py")
    else:
        print("❌ 部分验证失败")
        print("="*80)
        print("\n请检查上面的错误信息，修复后重新运行此脚本")
    print("="*80)


if __name__ == '__main__':
    main()

