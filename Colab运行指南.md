# Google Colab 运行指南

## 📋 快速开始

在 Google Colab 的代码单元格中依次运行以下命令：

### 1️⃣ 检查环境和GPU

```python
# 检查是否有GPU
!nvidia-smi

# 检查Python版本
!python --version

# 查看当前目录
!pwd
!ls
```

### 2️⃣ 进入项目目录

```python
# 如果项目在根目录
%cd /content/Time-Series-Library

# 或者如果在其他位置（替换为你的实际路径）
# %cd /content/drive/MyDrive/Time-Series-Library
```

### 3️⃣ 安装依赖

```python
# 安装所有必要的依赖包
!pip install torch numpy pandas matplotlib scikit-learn tqdm patool sktime einops PyWavelets reformer-pytorch local-attention -q

# 验证安装
import torch
import pandas as pd
import numpy as np
print("✓ PyTorch版本:", torch.__version__)
print("✓ CUDA可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ GPU设备:", torch.cuda.get_device_name(0))
```

### 4️⃣ 查看数据

```python
# 查看处理后的数据
!head -5 data/AEMO/NSW_30min.csv

# 查看数据统计
!python3 verify_aemo_data.py
```

### 5️⃣ 运行快速测试（推荐）

```python
# 运行快速测试（约5-10分钟，使用GPU会更快）
!bash quick_test_aemo.sh
```

### 6️⃣ 运行完整训练

```python
# 运行单个模型（DLinear，最快）
!bash scripts/AEMO_forecast/DLinear_NSW_30min.sh
```

---

## 🎯 完整的Colab代码单元格

直接复制以下完整代码块到一个Colab单元格中运行：

```python
# ============================================
# AEMO 时间序列预测 - Google Colab 完整运行
# ============================================

# 1. 检查环境
print("="*50)
print("1. 检查环境")
print("="*50)
!nvidia-smi
print("\nPython版本:")
!python --version

# 2. 进入项目目录
print("\n" + "="*50)
print("2. 进入项目目录")
print("="*50)
%cd /content/Time-Series-Library
!pwd

# 3. 安装依赖
print("\n" + "="*50)
print("3. 安装依赖")
print("="*50)
!pip install torch numpy pandas matplotlib scikit-learn tqdm patool sktime einops PyWavelets reformer-pytorch local-attention -q

# 4. 验证安装
print("\n" + "="*50)
print("4. 验证安装")
print("="*50)
import torch
import pandas as pd
import numpy as np
print("✓ PyTorch:", torch.__version__)
print("✓ CUDA可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ GPU:", torch.cuda.get_device_name(0))

# 5. 查看数据
print("\n" + "="*50)
print("5. 查看数据")
print("="*50)
!head -5 data/AEMO/NSW_30min.csv
print("\n数据文件:")
!ls -lh data/AEMO/*.csv | grep -v backup | wc -l
print("个处理后的数据文件")

# 6. 运行快速测试
print("\n" + "="*50)
print("6. 运行快速测试")
print("="*50)
print("开始训练（约5-10分钟）...\n")
!bash quick_test_aemo.sh

# 7. 查看结果
print("\n" + "="*50)
print("7. 查看结果")
print("="*50)
!ls -lh results/
!ls -lh checkpoints/
```

---

## 🚀 分步运行（推荐新手）

如果你想一步步运行，可以将上面的代码分成多个单元格：

### 单元格 1: 环境检查

```python
!nvidia-smi
!python --version
```

### 单元格 2: 切换目录

```python
%cd /content/Time-Series-Library
!ls
```

### 单元格 3: 安装依赖

```python
!pip install torch numpy pandas matplotlib scikit-learn tqdm patool sktime einops PyWavelets reformer-pytorch local-attention -q
```

### 单元格 4: 快速测试

```python
!bash quick_test_aemo.sh
```

### 单元格 5: 查看结果

```python
# 查看结果目录
!ls results/

# 读取和显示评估指标
import numpy as np
import glob

result_dirs = glob.glob("results/*")
if result_dirs:
    metrics_file = f"{result_dirs[0]}/metrics.npy"
    metrics = np.load(metrics_file)
    print(f"MSE: {metrics[0]:.4f}")
    print(f"MAE: {metrics[1]:.4f}")
```

---

## 🎨 运行不同的模型

### DLinear（最快，推荐开始）

```python
!bash scripts/AEMO_forecast/DLinear_NSW_30min.sh
```

### PatchTST（效果好）

```python
!bash scripts/AEMO_forecast/PatchTST_NSW_30min.sh
```

### TimesNet（最先进）

```python
!bash scripts/AEMO_forecast/TimesNet_NSW_30min.sh
```

### 其他区域

```python
# QLD（昆士兰）
!bash scripts/AEMO_forecast/DLinear_QLD_30min.sh

# VIC（维多利亚）
!bash scripts/AEMO_forecast/DLinear_VIC_30min.sh

# 5分钟数据
!bash scripts/AEMO_forecast/DLinear_NSW_5min.sh
```

---

## 📊 可视化结果（可选）

在Colab中运行以下代码来可视化预测结果：

```python
import numpy as np
import matplotlib.pyplot as plt
import glob

# 找到最新的结果目录
result_dirs = sorted(glob.glob("results/*"))
if result_dirs:
    latest_result = result_dirs[-1]
    
    # 加载数据
    pred = np.load(f"{latest_result}/pred.npy")
    true = np.load(f"{latest_result}/true.npy")
    metrics = np.load(f"{latest_result}/metrics.npy")
    
    # 绘图
    plt.figure(figsize=(15, 5))
    
    # 显示前5个样本的预测
    for i in range(min(5, len(pred))):
        plt.subplot(1, 5, i+1)
        plt.plot(true[i, :, 0], label='True', marker='o')
        plt.plot(pred[i, :, 0], label='Pred', marker='x')
        plt.title(f'Sample {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 结果保存到: prediction_results.png")
    print(f"MSE: {metrics[0]:.4f}")
    print(f"MAE: {metrics[1]:.4f}")
else:
    print("还没有结果，请先运行训练脚本")
```

---

## 💾 保存结果到Google Drive（可选）

如果你想保存结果，可以先挂载Google Drive：

```python
# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 复制结果到Drive
!mkdir -p /content/drive/MyDrive/AEMO_Results
!cp -r results/* /content/drive/MyDrive/AEMO_Results/
!cp -r checkpoints/* /content/drive/MyDrive/AEMO_Results/

print("✓ 结果已保存到 Google Drive/AEMO_Results/")
```

---

## 🔧 修改训练参数

如果你想修改参数（比如增加训练轮次），可以直接运行Python命令：

```python
!python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/AEMO/ \
  --data_path NSW_30min.csv \
  --model_id NSW_custom_experiment \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 48 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --target Price \
  --des 'Custom_Experiment' \
  --itr 1
```

---

## ⏱️ 预计运行时间（在Colab GPU上）

- **安装依赖**: 2-3分钟
- **快速测试**: 2-5分钟
- **完整训练**: 10-30分钟（取决于模型）
  - DLinear: ~10分钟
  - PatchTST: ~20分钟
  - TimesNet: ~30分钟

---

## 🐛 常见问题

### Q: 找不到项目目录
**A**: 确认你上传的位置，然后修改 `%cd` 命令的路径

### Q: GPU内存不足
**A**: 减小 batch_size：
```python
# 编辑脚本，或者直接用命令指定
!python3 run.py ... --batch_size 8
```

### Q: 训练中断了
**A**: Colab有运行时间限制。可以：
1. 减少训练轮次（`--train_epochs`）
2. 保存中间结果到Drive
3. 使用Colab Pro获得更长运行时间

### Q: 想要批量运行
**A**: 使用循环：
```python
models = ['DLinear', 'PatchTST', 'TimesNet']
regions = ['NSW', 'QLD', 'VIC']

for model in models:
    for region in regions:
        script = f"scripts/AEMO_forecast/{model}_{region}_30min.sh"
        print(f"\n{'='*50}")
        print(f"运行: {model} - {region}")
        print('='*50)
        !bash {script}
```

---

## 📚 更多信息

- 详细使用说明：`AEMO使用指南.md`
- 技术文档：`AEMO_DATA_SUMMARY.md`
- 数据验证：运行 `!python3 verify_aemo_data.py`

---

## 🎉 开始运行

最简单的方式：复制"完整的Colab代码单元格"到一个Colab单元格中，然后运行！

祝训练顺利！🚀

