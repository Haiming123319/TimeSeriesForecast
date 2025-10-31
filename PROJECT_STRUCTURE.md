# 📁 AEMO时序预测项目 - 文件结构

## 🎯 核心文件（优化版）

### 数据生成
- **`generate_optimized_aemo_data.py`** - 生成优化长度的AEMO数据
  - 30分钟数据：15个月
  - 5分钟数据：6个月
  - 自动生成15分钟降采样版本

### 训练脚本
- **`three_stage_training.py`** - 三阶段渐进式训练
  - 阶段1: 模型筛选（8个实验）
  - 阶段2: 扩展验证（20个实验）
  - 阶段3: 步长扩展（15个实验）
  - 总计：43个实验（原120个，减少64%）

### 验证脚本
- **`quick_verify_setup.py`** - 快速环境验证（5-10分钟）

### 主程序
- **`run.py`** - 训练入口程序

---

## 📚 文档

### Colab使用
- **`COLAB_OPTIMIZED_GUIDE.md`** - 完整的Colab运行指南
- **`COLAB_QUICKSTART.txt`** - 快速启动（直接复制运行）

### 项目说明
- **`UPDATE_NOTES.md`** - 优化更新说明
- **`OPTIMIZATION_SUMMARY.md`** - 完整的优化总结
- **`PROJECT_STRUCTURE.md`** - 本文件
- **`README.md`** - 项目总览

---

## 📊 数据目录

```
data/
├── AEMO_optimized/        # 优化后的数据（自动生成）
│   ├── NSW_30min.csv      # 15个月，21,936行
│   ├── NSW_15min.csv      # 6个月，17,568行
│   ├── NSW_5min.csv       # 6个月，52,704行
│   ├── ... (其他4个地区)
```

---

## 📈 结果目录

```
three_stage_results/       # 三阶段训练结果（自动生成）
├── stage1_results.json    # 阶段1: 模型筛选
├── stage2_results.json    # 阶段2: 扩展验证
├── stage3_results.json    # 阶段3: 步长扩展
├── all_results.csv        # 汇总结果（Excel可打开）
└── results_visualization.png  # 可视化对比图

results/                   # 每个实验的详细输出
checkpoints/               # 训练好的模型权重
```

---

## 🔧 核心代码目录

### 数据加载
```
data_provider/
├── data_loader.py         # Dataset实现（高效滑窗）
├── data_factory.py        # DataLoader工厂
├── m4.py                  # M4数据集
└── uea.py                 # UEA分类数据集
```

### 实验管理
```
exp/
├── exp_basic.py           # 基础实验类
├── exp_long_term_forecasting.py  # 长期预测
├── exp_short_term_forecasting.py # 短期预测
├── exp_classification.py # 分类任务
├── exp_imputation.py      # 缺失值填充
└── exp_anomaly_detection.py      # 异常检测
```

### 模型
```
models/
├── DLinear.py             # DLinear模型（最快）
├── PatchTST.py            # PatchTST模型
├── TimesNet.py            # TimesNet模型
├── iTransformer.py        # iTransformer模型
└── ... (其他20+模型)
```

### 工具
```
utils/
├── tools.py               # 训练工具（早停、学习率等）
├── metrics.py             # 评估指标
├── timefeatures.py        # 时间特征编码
└── losses.py              # 损失函数
```

---

## 🚀 快速开始

### 本地Mac运行
```bash
# 1. 生成数据
python3 generate_optimized_aemo_data.py

# 2. 运行训练（后台）
nohup python3 three_stage_training.py > training.log 2>&1 &

# 3. 查看进度
tail -f training.log
```

### Colab运行
参考 `COLAB_QUICKSTART.txt`，复制代码直接运行。

---

## 📊 优化亮点

| 指标 | 原方案 | 优化方案 | 改进 |
|------|--------|----------|------|
| 实验数量 | 120 | 43 | ↓ 64% |
| 训练时间 | ~60小时 | ~6-8小时(Colab) / ~35-40小时(Mac) | ↑ 7-10倍 / ↑ 1.5倍 |
| 数据量 | 22个月 | 15/6个月 | ↓ 60-70% |
| 模型配置 | 重量级 | 轻量级 | 速度提升2-4倍 |

---

## 🎯 三阶段策略

### 阶段1: 模型筛选（1-2小时 Colab / 5-6小时 Mac）
- 2个代表性州（NSW, SA）
- 只用30分钟数据
- pred_len=24
- 4个模型快速对比
- **选出前2个最优模型**

### 阶段2: 扩展验证（3-4小时 Colab / 14-16小时 Mac）
- 所有5个州
- 30min + 15min两个频率
- pred_len=24
- 只用阶段1选出的前2个模型
- **选出最优模型**

### 阶段3: 步长扩展（2-3小时 Colab / 16-18小时 Mac）
- 所有5个州
- 只用30分钟数据
- pred_len = 24, 48, 96
- 只用最优模型
- **获得完整性能矩阵**

---

## 💾 Git仓库

GitHub: `https://github.com/Haiming123319/TimeSeriesForecast`

---

**更新时间**: 2025-10-31  
**版本**: 2.0 (优化版)  
**状态**: ✅ 已清理并整理

