# AEMO Data Time Series Forecasting Scripts

## 数据说明

AEMO（Australian Energy Market Operator）数据已经处理成标准格式，可以直接用于时间序列预测模型。

### 处理后的数据格式

每个CSV文件包含以下列：
- `date`: 时间戳
- `Price`: 电价 ($/MWh)
- `Demand`: 需求 (MW)
- `Scheduled_Gen`: 计划发电量 (MW)
- `Semi_Scheduled_Gen`: 半计划发电量 (MW)
- `Net_Import`: 净进口量 (MW)

### 可用数据文件

**30分钟间隔数据：**
- `NSW_30min.csv` - 新南威尔士州 (319条记录)
- `QLD_30min.csv` - 昆士兰州 (319条记录)
- `SA_30min.csv` - 南澳大利亚州 (319条记录)
- `TAS_30min.csv` - 塔斯马尼亚州 (319条记录)
- `VIC_30min.csv` - 维多利亚州 (319条记录)

**5分钟间隔数据：**
- `NSW_5min.csv` - 新南威尔士州 (576条记录)
- `QLD_5min.csv` - 昆士兰州 (576条记录)
- `SA_5min.csv` - 南澳大利亚州 (576条记录)
- `TAS_5min.csv` - 塔斯马尼亚州 (576条记录)
- `VIC_5min.csv` - 维多利亚州 (576条记录)

## 使用说明

### 1. 基本用法

本目录提供了三个模型的示例脚本：
- **DLinear**: 线性模型，训练快速，适合快速验证
- **PatchTST**: 基于Transformer的patch方法，性能优秀
- **TimesNet**: 基于时间序列分解的模型，捕捉多周期性

### 2. 运行示例脚本

给脚本添加执行权限并运行：

```bash
# 进入项目根目录
cd /Users/yhm/Desktop/Time-Series-Library

# 添加执行权限
chmod +x scripts/AEMO_forecast/*.sh

# 运行DLinear模型（NSW 30分钟数据）
bash scripts/AEMO_forecast/DLinear_NSW_30min.sh

# 运行PatchTST模型（NSW 30分钟数据）
bash scripts/AEMO_forecast/PatchTST_NSW_30min.sh

# 运行TimesNet模型（NSW 30分钟数据）
bash scripts/AEMO_forecast/TimesNet_NSW_30min.sh

# 运行5分钟数据
bash scripts/AEMO_forecast/DLinear_NSW_5min.sh
```

### 3. 自定义脚本参数

#### 关键参数说明：

- `--root_path`: 数据目录路径 (./data/AEMO/)
- `--data_path`: 具体的CSV文件名
- `--model`: 模型名称 (DLinear, PatchTST, TimesNet)
- `--data`: 数据集类型 (custom - 使用自定义数据集加载器)
- `--features`: 
  - `M`: 多变量预测（使用所有5个特征）
  - `S`: 单变量预测（仅使用target指定的特征）
  - `MS`: 多变量输入，单变量输出
- `--target`: 目标变量名称 (Price, Demand, 等)
- `--seq_len`: 输入序列长度（历史数据长度）
- `--label_len`: 标签序列长度（用于decoder的起始序列）
- `--pred_len`: 预测长度
- `--enc_in`: 编码器输入特征数 (5)
- `--dec_in`: 解码器输入特征数 (5)
- `--c_out`: 输出通道数 (5)

#### 30分钟数据的时间步长参考：
- 24 steps = 12小时
- 48 steps = 1天
- 96 steps = 2天
- 144 steps = 3天

#### 5分钟数据的时间步长参考：
- 144 steps = 12小时
- 288 steps = 1天
- 576 steps = 2天

### 4. 创建新的区域脚本

要为其他区域（如QLD, VIC等）创建脚本，只需复制现有脚本并修改：

```bash
# 复制NSW脚本
cp scripts/AEMO_forecast/DLinear_NSW_30min.sh scripts/AEMO_forecast/DLinear_QLD_30min.sh

# 编辑新脚本，替换所有 "NSW" 为 "QLD"
# 修改 --data_path 和 --model_id 参数
```

或使用sed命令批量创建：

```bash
# 为QLD创建脚本
sed 's/NSW/QLD/g' scripts/AEMO_forecast/DLinear_NSW_30min.sh > scripts/AEMO_forecast/DLinear_QLD_30min.sh
sed 's/NSW/QLD/g' scripts/AEMO_forecast/PatchTST_NSW_30min.sh > scripts/AEMO_forecast/PatchTST_QLD_30min.sh
sed 's/NSW/QLD/g' scripts/AEMO_forecast/TimesNet_NSW_30min.sh > scripts/AEMO_forecast/TimesNet_QLD_30min.sh
```

### 5. 单变量预测示例

如果只想预测电价（Price），可以使用单变量模式：

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/AEMO/ \
  --data_path NSW_30min.csv \
  --model_id NSW_30min_price_only \
  --model DLinear \
  --data custom \
  --features S \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 48 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target Price \
  --des 'Exp' \
  --itr 1
```

注意：单变量预测时，`enc_in`, `dec_in`, `c_out` 都设为 1。

### 6. 结果查看

训练完成后，结果会保存在 `./results/` 目录下：
- 模型检查点：`./checkpoints/`
- 预测结果：`./results/{model_id}/`
- 日志文件：包含MSE、MAE等评估指标

### 7. 测试模式

要对已训练的模型进行测试，设置 `--is_training 0`：

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./data/AEMO/ \
  --data_path NSW_30min.csv \
  --model_id NSW_30min_48_48 \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 48 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --target Price
```

## 数据量考虑

**重要提示：** 当前AEMO数据集较小（30分钟数据319条，5分钟数据576条）。

对于深度学习模型：
- **建议数据量**：至少数千到数万条记录
- **当前数据量**：可能导致过拟合

### 解决方案：

1. **使用简单模型**：DLinear等线性模型对小数据集效果较好
2. **调整参数**：
   - 减小 `seq_len` 和 `pred_len`
   - 减小模型大小（`d_model`, `d_ff`等）
   - 增加正则化
3. **数据增强**：如果可能，获取更多历史数据
4. **使用预训练模型**：某些模型支持迁移学习

### 推荐配置（小数据集）：

对于当前数据量，建议使用较小的窗口：

```bash
# 30分钟数据 - 推荐配置
--seq_len 24   # 12小时历史
--pred_len 12  # 预测6小时

# 5分钟数据 - 推荐配置
--seq_len 72   # 6小时历史
--pred_len 36  # 预测3小时
```

## 常见问题

### Q1: 训练时报错 "not enough data"
A: 数据量不足以支持当前的seq_len和pred_len设置。减小这些参数。

### Q2: GPU内存不足
A: 减小batch_size或使用CPU训练（不设置CUDA_VISIBLE_DEVICES）

### Q3: 如何比较不同模型的效果？
A: 查看输出的MSE和MAE指标，数值越小越好。也可以查看生成的预测图表。

### Q4: 如何修改频率参数？
A: 添加 `--freq` 参数：
- 5分钟数据：`--freq 5min`
- 30分钟数据：`--freq 30min`

## 进阶使用

### 批量运行所有模型

创建批量运行脚本：

```bash
#!/bin/bash
# run_all_models.sh

models=("DLinear" "PatchTST" "TimesNet")
regions=("NSW" "QLD" "VIC" "SA" "TAS")

for model in "${models[@]}"; do
  for region in "${regions[@]}"; do
    echo "Running ${model} for ${region}..."
    bash scripts/AEMO_forecast/${model}_${region}_30min.sh
  done
done
```

### 自定义训练参数

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/AEMO/ \
  --data_path NSW_30min.csv \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len 48 \
  --pred_len 48 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --target Price \
  --train_epochs 50 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --patience 5 \
  --itr 3
```

## 参考

- 原始AEMO数据备份位于：`data/AEMO/*_backup.csv`
- 数据处理脚本：`process_aemo_data.py`
- 模型代码位于：`models/` 目录
- 更多示例：参考 `scripts/long_term_forecast/` 目录

