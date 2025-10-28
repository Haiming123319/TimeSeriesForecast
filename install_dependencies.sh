#!/bin/bash

echo "=========================================="
echo "AEMO 项目依赖安装脚本"
echo "=========================================="
echo ""

# 检查Python版本
echo "检查 Python 环境..."
python3 --version
echo ""

# 安装核心依赖
echo "步骤 1/3: 安装核心依赖 (torch, numpy, pandas)..."
pip3 install torch numpy pandas matplotlib scikit-learn tqdm
echo ""

# 安装额外依赖
echo "步骤 2/3: 安装额外依赖..."
pip3 install patool sktime einops PyWavelets
echo ""

# 安装特定版本的包
echo "步骤 3/3: 安装 reformer-pytorch 和 local-attention..."
pip3 install reformer-pytorch local-attention
echo ""

# 验证安装
echo "=========================================="
echo "验证安装..."
echo "=========================================="
python3 -c "
import sys
try:
    import torch
    print('✓ torch:', torch.__version__)
except ImportError as e:
    print('✗ torch: 未安装')
    sys.exit(1)

try:
    import numpy
    print('✓ numpy:', numpy.__version__)
except ImportError:
    print('✗ numpy: 未安装')

try:
    import pandas
    print('✓ pandas:', pandas.__version__)
except ImportError:
    print('✗ pandas: 未安装')

try:
    import sklearn
    print('✓ scikit-learn:', sklearn.__version__)
except ImportError:
    print('✗ scikit-learn: 未安装')

try:
    from reformer_pytorch import LSHSelfAttention
    print('✓ reformer-pytorch: 已安装')
except ImportError:
    print('✗ reformer-pytorch: 未安装')

print('')
print('========================================')
print('✓ 核心依赖安装完成！')
print('========================================')
print('')
print('现在可以运行测试：')
print('  bash quick_test_aemo.sh')
print('')
print('或运行完整训练：')
print('  bash scripts/AEMO_forecast/DLinear_NSW_30min.sh')
"

