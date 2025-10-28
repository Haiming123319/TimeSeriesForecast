# AEMO 数据处理与模型训练总结

## 📊 完成内容

### 1. 数据处理

已成功处理 **10 个 AEMO CSV 文件**，转换为模型所需的标准格式：

#### 处理的数据文件：
- **30分钟间隔数据** (5个州，每个319条记录)
  - NSW_30min.csv - 新南威尔士州
  - QLD_30min.csv - 昆士兰州
  - VIC_30min.csv - 维多利亚州
  - SA_30min.csv - 南澳大利亚州
  - TAS_30min.csv - 塔斯马尼亚州

- **5分钟间隔数据** (5个州，每个576条记录)
  - NSW_5min.csv - 新南威尔士州
  - QLD_5min.csv - 昆士兰州
  - VIC_5min.csv - 维多利亚州
  - SA_5min.csv - 南澳大利亚州
  - TAS_5min.csv - 塔斯马尼亚州

#### 数据格式：
```
date, Price, Demand, Scheduled_Gen, Semi_Scheduled_Gen, Net_Import
```

- ✅ 第一列：`date` (标准日期时间格式)
- ✅ 5个数值特征列（价格、需求、发电量等）
- ✅ 按时间排序（从早到晚）
- ✅ 无缺失值
- ✅ 备份文件已保存（*_backup.csv）

### 2. 生成的训练脚本

已生成 **30 个训练脚本**，覆盖：
- **3个模型**: DLinear, PatchTST, TimesNet
- **5个区域**: NSW, QLD, VIC, SA, TAS
- **2个时间间隔**: 30分钟, 5分钟

所有脚本位于：`scripts/AEMO_forecast/`

#### 脚本列表示例：
```bash
DLinear_NSW_30min.sh    # DLinear模型 - NSW 30分钟数据
DLinear_NSW_5min.sh     # DLinear模型 - NSW 5分钟数据
PatchTST_QLD_30min.sh   # PatchTST模型 - QLD 30分钟数据
TimesNet_VIC_5min.sh    # TimesNet模型 - VIC 5分钟数据
... (共30个脚本)
```

### 3. 辅助工具

创建了3个实用Python脚本：

1. **process_aemo_data.py** - 数据处理脚本
   - 重命名列为标准格式
   - 删除非数值列
   - 转换日期格式
   - 按时间排序

2. **verify_aemo_data.py** - 数据验证脚本
   - 检查数据格式是否正确
   - 显示数据统计信息
   - 推荐模型参数
   - 验证所有10个文件 ✅

3. **generate_all_scripts.py** - 批量生成训练脚本
   - 自动生成所有区域和模型的脚本
   - 参数自动适配不同时间间隔
   - 可执行权限自动设置

### 4. 文档

- **scripts/AEMO_forecast/README.md** - 详细使用说明
  - 数据格式说明
  - 模型参数解释
  - 运行示例
  - 常见问题解答
  - 进阶使用技巧

## 🚀 快速开始

### 验证数据（可选）
```bash
cd /Users/yhm/Desktop/Time-Series-Library
python verify_aemo_data.py
```

### 运行单个模型
```bash
# 运行 DLinear 模型（NSW 30分钟数据）
bash scripts/AEMO_forecast/DLinear_NSW_30min.sh

# 运行 PatchTST 模型（QLD 5分钟数据）
bash scripts/AEMO_forecast/PatchTST_QLD_5min.sh

# 运行 TimesNet 模型（VIC 30分钟数据）
bash scripts/AEMO_forecast/TimesNet_VIC_30min.sh
```

### 批量运行所有模型
```bash
# 运行所有30个脚本（需要较长时间）
for script in scripts/AEMO_forecast/*.sh; do 
    bash "$script"
done
```

### 运行特定模型的所有区域
```bash
# 运行所有 DLinear 脚本
for script in scripts/AEMO_forecast/DLinear_*.sh; do 
    bash "$script"
done
```

## 📈 模型参数说明

### 数据特征
- **特征数量**: 5个
  - Price (电价)
  - Demand (需求)
  - Scheduled_Gen (计划发电)
  - Semi_Scheduled_Gen (半计划发电)
  - Net_Import (净进口)

### 预测配置

#### 30分钟数据
- **seq_len**: 48 (输入1天历史数据)
- **label_len**: 24 (decoder起始序列)
- **pred_len**: 24/48/96 (预测12小时/1天/2天)
- **参数**: enc_in=5, dec_in=5, c_out=5

#### 5分钟数据
- **seq_len**: 288 (输入1天历史数据)
- **label_len**: 144 (decoder起始序列)
- **pred_len**: 144/288 (预测12小时/1天)
- **参数**: enc_in=5, dec_in=5, c_out=5

## ⚠️ 重要提示

### 数据量限制
当前数据集较小：
- 30分钟数据：319条记录（约1.5天）
- 5分钟数据：576条记录（约2天）

**建议**：
- ✅ DLinear：最适合小数据集，线性模型训练快速
- ⚠️ PatchTST：可能会过拟合，建议调小模型参数
- ⚠️ TimesNet：深度模型，小数据集上可能效果不佳

### 优化建议
1. **减小预测长度**: 使用较短的pred_len（如12小时而非2天）
2. **增加正则化**: 添加dropout或权重衰减
3. **减小模型大小**: 降低d_model, d_ff等参数
4. **使用简单模型**: 优先尝试DLinear
5. **获取更多数据**: 如果可能，收集更长时间的历史数据

## 📁 文件结构

```
Time-Series-Library/
├── data/
│   └── AEMO/
│       ├── NSW_30min.csv          # 处理后的数据
│       ├── NSW_5min.csv
│       ├── QLD_30min.csv
│       ├── ... (共10个CSV文件)
│       ├── NSW_30min_backup.csv   # 原始备份
│       └── ... (共10个备份文件)
├── scripts/
│   └── AEMO_forecast/
│       ├── README.md              # 详细使用说明
│       ├── DLinear_NSW_30min.sh   # 训练脚本
│       ├── ... (共30个.sh文件)
├── process_aemo_data.py          # 数据处理脚本
├── verify_aemo_data.py           # 数据验证脚本
├── generate_all_scripts.py       # 脚本生成工具
└── AEMO_DATA_SUMMARY.md          # 本文档
```

## 🎯 查看结果

训练完成后，结果保存在：
- **模型检查点**: `checkpoints/`
- **预测结果**: `results/`
- **日志文件**: 包含MSE、MAE等评估指标

查看结果示例：
```bash
# 查看训练日志
ls -lh checkpoints/

# 查看预测结果
ls -lh results/

# 查看特定实验的结果
cat results/long_term_forecast_custom_NSW_30min_48_48_DLinear_custom_ftM_*/result.txt
```

## 📖 更多信息

- 详细使用说明：`scripts/AEMO_forecast/README.md`
- 模型文档：`models/` 目录中的各个模型文件
- 更多示例：`scripts/long_term_forecast/` 目录

## ✅ 验证清单

- [x] 处理所有AEMO CSV文件
- [x] 转换数据为标准格式
- [x] 创建备份文件
- [x] 验证数据格式正确性
- [x] 生成DLinear训练脚本（所有区域）
- [x] 生成PatchTST训练脚本（所有区域）
- [x] 生成TimesNet训练脚本（所有区域）
- [x] 创建使用说明文档
- [x] 创建辅助工具脚本

**状态**: ✅ 所有数据已处理完毕，可以直接运行模型！

## 🚀 下一步

1. 选择一个区域和模型开始训练
2. 观察训练过程和结果
3. 根据结果调整参数
4. 比较不同模型的预测效果
5. 如需要，收集更多历史数据以提高模型性能

祝训练顺利！🎉

