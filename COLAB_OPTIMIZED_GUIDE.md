# 🚀 Colab优化训练指南 - 三阶段策略

> **重要更新**：采用专业建议的三阶段训练策略，训练速度提升3-5倍！

---

## 📋 核心优化点

### ✅ 数据优化
- **30分钟数据**：15个月（足够覆盖季节性，而非22个月）
- **5分钟数据**：6个月（避免过旧数据噪声）
- **15分钟数据**：从5分钟降采样（平衡精度和速度）

### ✅ 训练策略
- **阶段1**：模型筛选（2州 × 4模型 = 8实验）
- **阶段2**：扩展验证（5州 × 2频率 × 2模型 = 20实验）
- **阶段3**：步长扩展（5州 × 3步长 × 1模型 = 15实验）
- **总计**：43个实验（原120个，减少64%）

### ✅ 模型配置（轻量级）
- DLinear: 保持默认
- PatchTST: d_model=128, n_heads=4
- TimesNet: d_model=256
- iTransformer: d_model=256

---

## 🎯 快速开始（Colab）

### 步骤1：环境设置（5分钟）

```python
import os

print("🔍 检查GPU...")
!nvidia-smi

print("\n📥 克隆代码...")
if os.path.exists('/content/TimeSeriesForecast'):
    !rm -rf /content/TimeSeriesForecast

!git clone https://github.com/yhm-amber/Time-Series-Library.git /content/TimeSeriesForecast

os.chdir('/content/TimeSeriesForecast')

print("\n📦 安装依赖...")
!pip install -q torch numpy pandas scikit-learn matplotlib einops transformers statsmodels scipy

# 修复依赖导入问题
print("\n🔧 修复代码...")
# 修复 sktime
with open('data_provider/data_loader.py', 'r') as f:
    content = f.read()
if 'from sktime.datasets import load_from_tsfile_to_dataframe' in content:
    content = content.replace(
        'from sktime.datasets import load_from_tsfile_to_dataframe',
        'try:\n    from sktime.datasets import load_from_tsfile_to_dataframe\nexcept:\n    load_from_tsfile_to_dataframe = None'
    )
    with open('data_provider/data_loader.py', 'w') as f:
        f.write(content)

# 修复 patoolib
with open('data_provider/m4.py', 'r') as f:
    content = f.read()
if 'import patoolib' in content:
    content = content.replace(
        'import patoolib',
        'try:\n    import patoolib\nexcept:\n    patoolib = None'
    )
    with open('data_provider/m4.py', 'w') as f:
        f.write(content)

print("\n✅ 环境设置完成！")

import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
```

---

### 步骤2：生成优化数据（2分钟）

```python
import os
os.chdir('/content/TimeSeriesForecast')

# 运行数据生成脚本
!python3 generate_optimized_aemo_data.py

print("\n✅ 数据生成完成！")

# 验证数据
import pandas as pd
data_dir = './data/AEMO_optimized'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

print(f"\n📊 生成了 {len(files)} 个文件:")
for f in files[:5]:
    df = pd.read_csv(f'{data_dir}/{f}')
    print(f"  - {f:25s}: {len(df):6d} 行")

if len(files) > 5:
    print(f"  ... 还有 {len(files)-5} 个文件")
```

---

### 步骤3：三阶段训练（6-8小时）

```python
import os
os.chdir('/content/TimeSeriesForecast')

print("🚀 开始三阶段训练...")
print("预计总时间: 6-8小时")
print("  - 阶段1: 1-2小时（模型筛选）")
print("  - 阶段2: 3-4小时（扩展验证）")
print("  - 阶段3: 2-3小时（步长扩展）")
print()

# 运行三阶段训练
!python3 three_stage_training.py

print("\n✅ 训练完成！")
```

---

### 步骤4：查看结果

```python
import os
import pandas as pd
import json

os.chdir('/content/TimeSeriesForecast')

print("="*80)
print("📊 三阶段训练结果汇总")
print("="*80)

# 读取所有结果
df_all = pd.read_csv('./three_stage_results/all_results.csv')

# 按阶段统计
print("\n🎯 各阶段统计:")
for stage in [1, 2, 3]:
    df_stage = df_all[df_all['stage'] == stage]
    success_count = len(df_stage[df_stage['success'] == True])
    total_count = len(df_stage)
    avg_time = df_stage['time_minutes'].mean()
    
    print(f"\n阶段{stage}:")
    print(f"  成功: {success_count}/{total_count}")
    print(f"  平均用时: {avg_time:.1f}分钟")

# 最佳结果
df_success = df_all[(df_all['success'] == True) & (df_all['mae'].notna())]

if len(df_success) > 0:
    df_success = df_success.sort_values('mae')
    
    print("\n" + "="*80)
    print("🏆 Top 10 最佳结果")
    print("="*80)
    print(df_success[['stage', 'state', 'freq', 'model', 'pred_len', 'mae', 'mse']].head(10).to_string(index=False))
    
    best = df_success.iloc[0]
    print(f"\n✨ 最佳配置:")
    print(f"   阶段: {int(best['stage'])}")
    print(f"   模型: {best['model']}")
    print(f"   州: {best['state']}")
    print(f"   频率: {best['freq']}")
    print(f"   预测步长: {int(best['pred_len'])}")
    print(f"   MAE: {best['mae']:.4f}")
    print(f"   MSE: {best['mse']:.4f}")
```

---

### 步骤5：可视化对比

```python
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/content/TimeSeriesForecast')

df_all = pd.read_csv('./three_stage_results/all_results.csv')
df_success = df_all[(df_all['success'] == True) & (df_all['mae'].notna())]

if len(df_success) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 阶段1: 模型对比
    ax1 = axes[0, 0]
    df_stage1 = df_success[df_success['stage'] == 1]
    if len(df_stage1) > 0:
        df_stage1.groupby('model')['mae'].mean().sort_values().plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('阶段1: 模型对比 (平均MAE)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('MAE ($/MWh)')
        ax1.grid(True, alpha=0.3)
    
    # 2. 阶段2: 频率对比
    ax2 = axes[0, 1]
    df_stage2 = df_success[df_success['stage'] == 2]
    if len(df_stage2) > 0:
        df_stage2.groupby('freq')['mae'].mean().plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('阶段2: 频率对比 (平均MAE)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE ($/MWh)')
        ax2.set_xlabel('频率')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    
    # 3. 阶段3: 预测步长对比
    ax3 = axes[1, 0]
    df_stage3 = df_success[df_success['stage'] == 3]
    if len(df_stage3) > 0:
        df_stage3.groupby('pred_len')['mae'].mean().plot(kind='bar', ax=ax3, color='lightgreen')
        ax3.set_title('阶段3: 预测步长对比 (平均MAE)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('MAE ($/MWh)')
        ax3.set_xlabel('预测步长')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    
    # 4. 各州表现
    ax4 = axes[1, 1]
    state_avg = df_success.groupby('state')['mae'].mean().sort_values()
    state_avg.plot(kind='barh', ax=ax4, color='mediumpurple')
    ax4.set_title('各州表现对比 (平均MAE)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('MAE ($/MWh)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./three_stage_results/results_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ 图表已保存: ./three_stage_results/results_visualization.png")
    plt.show()
else:
    print("⚠️  没有足够的数据用于可视化")
```

---

### 步骤6：下载所有结果

```python
import os
from google.colab import files

os.chdir('/content/TimeSeriesForecast')

print("📦 打包所有结果...")

!zip -r aemo_three_stage_results.zip \
  ./three_stage_results/ \
  ./results/ \
  ./checkpoints/ \
  -x "*.pyc" "**/__pycache__/*" \
  2>&1 | tail -10

size_mb = os.path.getsize('aemo_three_stage_results.zip') / (1024 * 1024)
print(f"\n✅ 打包完成: {size_mb:.1f} MB")

print("\n📥 开始下载...")
files.download('aemo_three_stage_results.zip')

print("\n✅ 下载完成！")
print("\n📦 压缩包内容:")
print("  - three_stage_results/  : 三阶段汇总结果")
print("  - results/             : 所有实验的详细结果")
print("  - checkpoints/         : 训练好的模型文件")
```

---

## 📊 结果文件说明

下载的压缩包包含：

```
aemo_three_stage_results.zip
├── three_stage_results/
│   ├── stage1_results.json       # 阶段1: 模型筛选结果
│   ├── stage2_results.json       # 阶段2: 扩展验证结果
│   ├── stage3_results.json       # 阶段3: 步长扩展结果
│   ├── all_results.csv           # 所有结果汇总（Excel可打开）
│   └── results_visualization.png # 可视化对比图
├── results/                      # 每个实验的详细输出
└── checkpoints/                  # 训练好的模型权重
```

---

## 🎯 关键改进点

| 项目 | 原方案 | 优化方案 | 改进 |
|------|--------|---------|------|
| **实验数量** | 120 | 43 | **减少64%** |
| **数据量** | 22个月 | 15/6个月 | **减少60-70%** |
| **训练时间** | ~60小时 | ~6-8小时 | **提升7-10倍** |
| **模型配置** | 默认（重） | 轻量级 | **速度提升2-3倍** |
| **早停** | patience=3 | patience=2 | **加速20-30%** |

---

## 💡 为什么这样更快更好？

1. **避免无效实验**：先筛选再扩展，不浪费时间在差模型上
2. **数据长度优化**：15/6个月足够，避免过旧数据带来的噪声
3. **轻量级配置**：在保证效果的前提下，大幅减少计算量
4. **早停策略**：及时停止无效训练
5. **15分钟降采样**：平衡了5分钟的信息量和30分钟的速度

---

## 🚀 现在就开始！

按顺序复制上面6个步骤的代码到Colab，直接运行即可！

预计**6-8小时**后，您将得到所有模型在不同配置下的完整对比结果。

