# TSL环境依赖安装说明

## 问题说明

你的 `tsl` conda环境中缺少必要的Python包。所有脚本已经修改为：
- ✅ 使用 `python3` 命令
- ✅ 强制使用CPU（不需要GPU）

## 安装步骤

### 方法一：在tsl环境中安装依赖（推荐）

```bash
# 1. 激活tsl环境
conda activate tsl

# 2. 安装核心依赖
pip install torch numpy pandas matplotlib scikit-learn tqdm

# 3. 安装其他依赖
pip install patool sktime einops reformer-pytorch local-attention PyWavelets

# 4. 验证安装
python3 -c "import torch; import pandas; import numpy; print('✓ 依赖安装成功！')"

# 5. 运行测试
cd /Users/yhm/Desktop/Time-Series-Library
bash quick_test_aemo.sh
```

### 方法二：使用requirements.txt（如果方法一有问题）

```bash
# 1. 激活tsl环境
conda activate tsl

# 2. 安装所有依赖（可能需要一些时间）
cd /Users/yhm/Desktop/Time-Series-Library
pip install torch numpy pandas matplotlib scikit-learn tqdm patool sktime einops PyWavelets

# 注意：reformer-pytorch 和 local-attention 可能需要特定版本
pip install reformer-pytorch==1.4.4 local-attention

# 3. 运行测试
bash quick_test_aemo.sh
```

### 方法三：使用base环境（快速方案）

如果你不想在tsl环境中安装依赖，可以直接使用base环境：

```bash
# 1. 退出tsl环境
conda deactivate

# 2. 激活base环境（已经安装了依赖）
conda activate base

# 3. 运行脚本
cd /Users/yhm/Desktop/Time-Series-Library
bash quick_test_aemo.sh
```

## 已完成的修复

✅ **所有31个脚本已修改**：
- `scripts/AEMO_forecast/*.sh` (30个训练脚本)
- `quick_test_aemo.sh` (快速测试脚本)

修改内容：
1. `python` → `python3` (适配你的环境)
2. `CUDA_VISIBLE_DEVICES=0` → `CUDA_VISIBLE_DEVICES=""` (强制使用CPU)

## 快速测试

安装完依赖后，运行快速测试验证：

```bash
conda activate tsl
cd /Users/yhm/Desktop/Time-Series-Library
bash quick_test_aemo.sh
```

如果看到类似以下输出，说明成功：

```
Epoch: 1, Steps: 24 | Train Loss: 0.5457467 Vali Loss: 0.6820247
...
test shape: (52, 12, 5) (52, 12, 5)
mse:1.9711657, mae:0.8326349
```

## 常见问题

### Q: pip install 报错
**A**: 尝试升级pip：`pip install --upgrade pip`

### Q: 依然提示找不到模块
**A**: 确认你在正确的conda环境中：`conda env list` 查看当前环境

### Q: 训练太慢
**A**: 这是正常的，CPU训练比GPU慢。可以：
- 减小 `batch_size`
- 减小 `seq_len` 和 `pred_len`
- 减少 `train_epochs`

## 推荐的运行顺序

```bash
# 1. 安装依赖
conda activate tsl
pip install torch numpy pandas matplotlib scikit-learn tqdm patool sktime einops reformer-pytorch local-attention PyWavelets

# 2. 快速测试（约5-10分钟）
bash quick_test_aemo.sh

# 3. 完整训练（约30-60分钟）
bash scripts/AEMO_forecast/DLinear_NSW_30min.sh

# 4. 尝试其他模型
bash scripts/AEMO_forecast/PatchTST_NSW_30min.sh
bash scripts/AEMO_forecast/TimesNet_NSW_30min.sh
```

## 需要帮助？

如果遇到其他问题，请查看：
- `AEMO使用指南.md` - 中文使用指南
- `AEMO_DATA_SUMMARY.md` - 详细技术文档

