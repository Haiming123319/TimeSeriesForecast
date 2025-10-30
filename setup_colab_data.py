"""
在Google Colab中设置AEMO数据
如果CSV文件没有在GitHub仓库中，可以使用这个脚本重新生成
"""

import pandas as pd
import os

def create_sample_aemo_data():
    """创建示例AEMO数据用于测试"""
    
    print("="*60)
    print("创建AEMO示例数据")
    print("="*60)
    
    # 确保目录存在
    os.makedirs('data/AEMO', exist_ok=True)
    
    # 创建示例数据
    import numpy as np
    from datetime import datetime, timedelta
    
    regions = ['NSW', 'QLD', 'VIC', 'SA', 'TAS']
    intervals = [('30min', 30, 319), ('5min', 5, 576)]
    
    for region in regions:
        for interval_name, interval_mins, n_records in intervals:
            filename = f'data/AEMO/{region}_{interval_name}.csv'
            
            # 生成时间序列
            start_date = datetime(2025, 10, 26, 12, 35)
            dates = [start_date + timedelta(minutes=i*interval_mins) for i in range(n_records)]
            
            # 生成随机数据（模拟真实模式）
            np.random.seed(hash(region + interval_name) % 2**32)
            
            data = {
                'date': dates,
                'Price': np.random.uniform(-20, 500, n_records),
                'Demand': np.random.uniform(1000, 10000, n_records),
                'Scheduled_Gen': np.random.uniform(500, 5000, n_records),
                'Semi_Scheduled_Gen': np.random.uniform(100, 4000, n_records),
                'Net_Import': np.random.uniform(-1000, 1500, n_records)
            }
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"✓ 创建: {filename} ({n_records} 条记录)")
    
    print("\n" + "="*60)
    print("✓ 所有数据文件创建完成！")
    print("="*60)
    print("\n现在可以运行训练脚本了：")
    print("  !bash quick_test_aemo.sh")
    print("  !bash scripts/AEMO_forecast/DLinear_NSW_30min.sh")

if __name__ == '__main__':
    create_sample_aemo_data()

