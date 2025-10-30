# Google Colab 完整设置指南

## 🚨 数据文件缺失问题解决

由于CSV文件较大，没有上传到GitHub。请使用以下方法之一：

---

## 🎯 方法1：使用示例数据（最快，推荐测试）

在Colab中运行：

```python
# ============================================
# 完整的Colab设置 - 包含数据生成
# ============================================

# 1. 克隆项目
%cd /content
!git clone https://github.com/Haiming123319/TimeSeriesForecast.git
%cd TimeSeriesForecast

# 2. 安装依赖
!pip install torch numpy pandas matplotlib scikit-learn tqdm patool sktime einops PyWavelets reformer-pytorch local-attention -q

# 3. 生成示例AEMO数据
!python3 setup_colab_data.py

# 4. 验证数据
!ls -lh data/AEMO/*.csv | head -5
!head -5 data/AEMO/NSW_30min.csv

# 5. 修复脚本格式问题（如果有）
import glob
for script in glob.glob('scripts/AEMO_forecast/*.sh'):
    with open(script, 'r') as f:
        lines = f.readlines()
    with open(script, 'w') as f:
        f.writelines([line.rstrip() + '\n' for line in lines])
print("✓ 脚本已修复")

# 6. 运行快速测试
!bash quick_test_aemo.sh
```

---

## 📤 方法2：从本地上传真实数据

### 步骤1：在Colab中准备上传

```python
from google.colab import files
import os

%cd /content/TimeSeriesForecast
os.makedirs('data/AEMO', exist_ok=True)

print("请选择10个CSV文件上传（NSW/QLD/VIC/SA/TAS，各30min和5min）...")
uploaded = files.upload()

# 移动文件
for filename in uploaded.keys():
    !mv {filename} data/AEMO/
    print(f"✓ {filename}")

print("\n✓ 数据上传完成！")
!ls -lh data/AEMO/
```

### 步骤2：继续运行

```python
# 验证数据
!head -5 data/AEMO/NSW_30min.csv

# 运行测试
!bash quick_test_aemo.sh
```

---

## 💾 方法3：使用Google Drive

### 步骤1：上传数据到Drive

1. 在Google Drive中创建文件夹 `AEMO_Data`
2. 上传所有CSV文件到该文件夹

### 步骤2：在Colab中使用

```python
# 挂载Drive
from google.colab import drive
drive.mount('/content/drive')

# 进入项目
%cd /content/TimeSeriesForecast

# 复制数据
!mkdir -p data/AEMO
!cp /content/drive/MyDrive/AEMO_Data/*.csv data/AEMO/

# 验证
!ls -lh data/AEMO/
!head -5 data/AEMO/NSW_30min.csv

# 运行测试
!bash quick_test_aemo.sh
```

---

## 🔄 方法4：从本地机器推送到GitHub

在你的本地机器 `/Users/yhm/Desktop/Time-Series-Library` 运行：

```bash
# 1. 检查哪些文件被忽略
git status

# 2. 查看.gitignore
cat .gitignore

# 3. 如果CSV被忽略，有两个选择：

# 选择A：强制添加CSV文件（如果文件不太大）
git add -f data/AEMO/*.csv
git commit -m "Add AEMO CSV data files"
git push

# 选择B：使用Git LFS（大文件存储）
git lfs install
git lfs track "data/AEMO/*.csv"
git add .gitattributes
git add data/AEMO/*.csv
git commit -m "Add AEMO data with LFS"
git push
```

---

## 📊 方法5：使用process_aemo_data.py重新生成

如果你的仓库中有原始的备份文件：

```python
%cd /content/TimeSeriesForecast

# 检查备份文件
!ls -la data/AEMO/*backup*.csv

# 如果有备份，恢复数据
import shutil
import glob

backups = glob.glob('data/AEMO/*_backup.csv')
if backups:
    for backup in backups:
        original = backup.replace('_backup', '')
        shutil.copy(backup, original)
        print(f"✓ 恢复: {original}")
else:
    print("没有找到备份文件，运行数据处理脚本...")
    !python3 process_aemo_data.py

# 验证
!ls -lh data/AEMO/*.csv | grep -v backup
```

---

## 🎯 推荐流程（最简单）

**复制以下完整代码到一个Colab单元格**：

```python
# ============================================
# AEMO项目完整设置 - Google Colab
# ============================================

import os
import glob

print("1️⃣  克隆项目...")
%cd /content
!git clone https://github.com/Haiming123319/TimeSeriesForecast.git 2>/dev/null || echo "项目已存在"
%cd TimeSeriesForecast

print("\n2️⃣  安装依赖...")
!pip install torch numpy pandas matplotlib scikit-learn tqdm patool sktime einops PyWavelets reformer-pytorch local-attention -q

print("\n3️⃣  生成示例数据...")
!python3 setup_colab_data.py

print("\n4️⃣  修复脚本...")
for script in glob.glob('scripts/AEMO_forecast/*.sh') + ['quick_test_aemo.sh']:
    if os.path.exists(script):
        with open(script, 'r') as f:
            lines = f.readlines()
        with open(script, 'w') as f:
            f.writelines([line.rstrip() + '\n' for line in lines])
print("✓ 脚本格式已修复")

print("\n5️⃣  验证设置...")
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA可用: {torch.cuda.is_available()}")
!ls -lh data/AEMO/ | head -5

print("\n6️⃣  运行测试...")
print("="*60)
!bash quick_test_aemo.sh

print("\n✅ 设置完成！现在可以运行完整训练：")
print("   !bash scripts/AEMO_forecast/DLinear_NSW_30min.sh")
```

---

## 📝 验证清单

运行后检查：
- [ ] ✓ 10个CSV文件存在（5个区域 × 2个时间间隔）
- [ ] ✓ 每个CSV有5列：date, Price, Demand, Scheduled_Gen, Semi_Scheduled_Gen, Net_Import
- [ ] ✓ 快速测试成功运行
- [ ] ✓ 生成了results/和checkpoints/目录

```python
# 验证命令
!ls -lh data/AEMO/*.csv | wc -l  # 应该是10
!head -2 data/AEMO/NSW_30min.csv
!ls results/ 2>/dev/null && echo "✓ 有结果" || echo "✗ 还没结果"
```

---

## 🆘 故障排除

### 问题1：FileNotFoundError
**解决**：确保运行了 `setup_colab_data.py` 或手动上传了CSV

### 问题2：脚本参数错误
**解决**：运行脚本修复代码（上面的步骤4）

### 问题3：GPU内存不足
**解决**：
```python
# 在脚本中添加或修改：
# --batch_size 8
# --seq_len 24
```

### 问题4：数据格式错误
**解决**：
```python
# 验证数据格式
import pandas as pd
df = pd.read_csv('data/AEMO/NSW_30min.csv')
print(df.columns)
print(df.head())
# 应该有：date, Price, Demand, Scheduled_Gen, Semi_Scheduled_Gen, Net_Import
```

---

## 🎉 开始使用

最简单的方式：**复制"推荐流程"中的完整代码块**到Colab并运行！

一次性解决所有问题：数据、依赖、脚本修复、测试运行。

需要真实数据？选择方法2或方法3上传你的本地CSV文件。

