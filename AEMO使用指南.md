# AEMO 数据时间序列预测 - 使用指南

## ✅ 已完成的工作

### 1. 数据处理 ✓
- ✅ 处理了 **10个CSV文件** (5个州 × 2个时间间隔)
- ✅ 转换为模型所需的标准格式
- ✅ 列名：`date, Price, Demand, Scheduled_Gen, Semi_Scheduled_Gen, Net_Import`
- ✅ 所有数据已验证，可以直接使用
- ✅ 原始数据已备份（*_backup.csv）

### 2. 训练脚本 ✓
- ✅ 生成了 **30个训练脚本**
- ✅ 3个模型：DLinear, PatchTST, TimesNet
- ✅ 5个区域：NSW, QLD, VIC, SA, TAS
- ✅ 2个时间间隔：30分钟, 5分钟
- ✅ 所有脚本都已设置好参数，可以直接运行

### 3. 辅助工具 ✓
- ✅ 数据处理脚本：`process_aemo_data.py`
- ✅ 数据验证脚本：`verify_aemo_data.py`
- ✅ 脚本生成工具：`generate_all_scripts.py`
- ✅ 快速测试脚本：`quick_test_aemo.sh`

## 🚀 如何使用

### 方法一：快速测试（推荐新手）

运行快速测试，验证环境和数据：

```bash
cd /Users/yhm/Desktop/Time-Series-Library
bash quick_test_aemo.sh
```

这会用最小的参数快速训练一个DLinear模型（只需几分钟）。

### 方法二：运行单个模型

选择一个脚本运行完整训练：

```bash
cd /Users/yhm/Desktop/Time-Series-Library

# 运行 DLinear（最简单，推荐开始）
bash scripts/AEMO_forecast/DLinear_NSW_30min.sh

# 运行 PatchTST
bash scripts/AEMO_forecast/PatchTST_NSW_30min.sh

# 运行 TimesNet
bash scripts/AEMO_forecast/TimesNet_NSW_30min.sh
```

### 方法三：运行所有模型（批量）

运行某个模型的所有区域：

```bash
# 运行所有 DLinear 脚本
for script in scripts/AEMO_forecast/DLinear_*.sh; do 
    bash "$script"
done

# 运行所有 30分钟 数据的脚本
for script in scripts/AEMO_forecast/*_30min.sh; do 
    bash "$script"
done

# 运行所有脚本（需要很长时间！）
for script in scripts/AEMO_forecast/*.sh; do 
    bash "$script"
done
```

## 📊 可用的数据

### 数据文件位置
所有处理好的数据在：`data/AEMO/`

### 30分钟数据（319条记录）
```
NSW_30min.csv  - 新南威尔士州
QLD_30min.csv  - 昆士兰州
VIC_30min.csv  - 维多利亚州
SA_30min.csv   - 南澳大利亚州
TAS_30min.csv  - 塔斯马尼亚州
```

### 5分钟数据（576条记录）
```
NSW_5min.csv  - 新南威尔士州
QLD_5min.csv  - 昆士兰州
VIC_5min.csv  - 维多利亚州
SA_5min.csv   - 南澳大利亚州
TAS_5min.csv  - 塔斯马尼亚州
```

### 数据特征（5个）
1. **Price** - 电价 ($/MWh)
2. **Demand** - 需求 (MW)
3. **Scheduled_Gen** - 计划发电 (MW)
4. **Semi_Scheduled_Gen** - 半计划发电 (MW)
5. **Net_Import** - 净进口 (MW)

## 📁 文件结构

```
Time-Series-Library/
├── data/AEMO/                    # 数据目录
│   ├── NSW_30min.csv            # 处理后的数据（可用）
│   ├── NSW_30min_backup.csv     # 原始备份
│   └── ... (共20个文件)
│
├── scripts/AEMO_forecast/        # 训练脚本
│   ├── README.md                # 详细说明（英文）
│   ├── DLinear_NSW_30min.sh     # 训练脚本
│   └── ... (共30个脚本)
│
├── process_aemo_data.py         # 数据处理（已完成）
├── verify_aemo_data.py          # 数据验证
├── generate_all_scripts.py      # 脚本生成（已完成）
├── quick_test_aemo.sh           # 快速测试
├── AEMO_DATA_SUMMARY.md         # 详细总结（英文）
└── AEMO使用指南.md              # 本文档
```

## 🎯 模型说明

### DLinear（推荐开始）
- **优点**：简单快速，适合小数据集
- **训练时间**：最快
- **适用场景**：快速验证，基准模型

### PatchTST
- **优点**：基于Transformer，效果较好
- **训练时间**：中等
- **适用场景**：追求更好的预测性能

### TimesNet
- **优点**：捕捉多周期性，最先进
- **训练时间**：最慢
- **适用场景**：数据充足时使用

## ⚠️ 重要提示

### 数据量较小
你的数据集比较小（30分钟319条，5分钟576条），这可能导致：
- 模型容易过拟合
- 深度学习模型效果可能不如简单模型

### 建议：
1. **先用 DLinear** - 最适合小数据集
2. **使用较短的预测长度** - 预测12小时比预测2天更可靠
3. **观察训练损失** - 如果训练损失很小但验证损失很大，说明过拟合了
4. **比较不同模型** - 看哪个模型在你的数据上效果最好

## 📈 查看结果

训练完成后，结果在：

```bash
# 模型检查点
ls checkpoints/

# 预测结果
ls results/

# 查看具体实验的结果
cat results/long_term_forecast_custom_NSW_30min_*_DLinear_*/result.txt
```

结果包含以下指标（数值越小越好）：
- **MSE** (均方误差)
- **MAE** (平均绝对误差)

## 🔧 自定义训练

如果你想修改参数，可以直接编辑脚本或运行：

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
  --pred_len 24 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --target Price \
  --train_epochs 10 \
  --batch_size 16
```

### 常用参数：
- `--seq_len`: 输入长度（历史数据）
- `--pred_len`: 预测长度
- `--train_epochs`: 训练轮数（默认10）
- `--batch_size`: 批次大小（默认32）
- `--learning_rate`: 学习率（默认0.0001）
- `--target`: 目标变量（Price, Demand等）
- `--features`: 
  - `M` - 多变量预测（用所有5个特征）
  - `S` - 单变量预测（只用target指定的特征）

## 💡 使用技巧

### 1. 从简单开始
```bash
# 第一步：快速测试
bash quick_test_aemo.sh

# 第二步：完整训练一个简单模型
bash scripts/AEMO_forecast/DLinear_NSW_30min.sh

# 第三步：尝试其他模型
bash scripts/AEMO_forecast/PatchTST_NSW_30min.sh
```

### 2. 比较不同区域
```bash
# 训练同一个模型在不同区域的表现
bash scripts/AEMO_forecast/DLinear_NSW_30min.sh
bash scripts/AEMO_forecast/DLinear_QLD_30min.sh
bash scripts/AEMO_forecast/DLinear_VIC_30min.sh
```

### 3. 比较不同时间间隔
```bash
# 比较30分钟 vs 5分钟数据
bash scripts/AEMO_forecast/DLinear_NSW_30min.sh
bash scripts/AEMO_forecast/DLinear_NSW_5min.sh
```

## ❓ 常见问题

### Q: 训练报错 "not enough data"
**A**: 数据量不够。解决方法：
- 减小 `seq_len` 和 `pred_len`
- 例如改为 `--seq_len 24 --pred_len 12`

### Q: GPU 内存不足
**A**: 解决方法：
- 在脚本中注释掉 `export CUDA_VISIBLE_DEVICES=0` 这行（使用CPU）
- 或减小 `batch_size`

### Q: 如何只预测电价？
**A**: 使用单变量模式：
```bash
python -u run.py \
  --features S \
  --target Price \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  ...其他参数
```

### Q: 如何知道模型训练好了？
**A**: 查看输出：
- 训练完成会显示 MSE 和 MAE
- 数值越小越好
- 可以在 `results/` 目录查看详细结果

## 📚 更多信息

- **详细说明**：`scripts/AEMO_forecast/README.md`
- **完整总结**：`AEMO_DATA_SUMMARY.md`
- **模型代码**：`models/` 目录
- **更多示例**：`scripts/long_term_forecast/` 目录

## ✅ 检查清单

开始前确认：
- [ ] 已安装所有依赖（见 requirements.txt）
- [ ] 数据在 `data/AEMO/` 目录
- [ ] 脚本在 `scripts/AEMO_forecast/` 目录
- [ ] 已进入项目根目录

开始训练：
- [ ] 运行快速测试验证环境
- [ ] 选择一个模型和区域
- [ ] 运行训练脚本
- [ ] 查看结果

## 🎉 开始使用

推荐的第一步：

```bash
cd /Users/yhm/Desktop/Time-Series-Library

# 1. 验证数据（可选）
python verify_aemo_data.py

# 2. 快速测试
bash quick_test_aemo.sh

# 3. 完整训练（DLinear最简单）
bash scripts/AEMO_forecast/DLinear_NSW_30min.sh
```

祝你训练顺利！如果遇到问题，可以查看 `scripts/AEMO_forecast/README.md` 获取更多帮助。

