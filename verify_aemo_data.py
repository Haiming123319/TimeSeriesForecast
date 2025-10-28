"""
Verify AEMO Data Format
Quick script to verify that processed AEMO data is ready for model training.
"""

import pandas as pd
import os
from glob import glob

def verify_csv(filepath):
    """Verify a single CSV file."""
    print(f"\n{'='*60}")
    print(f"Checking: {os.path.basename(filepath)}")
    print('='*60)
    
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Check columns
        required_cols = ['date', 'Price', 'Demand', 'Scheduled_Gen', 
                        'Semi_Scheduled_Gen', 'Net_Import']
        has_all_cols = all(col in df.columns for col in required_cols)
        
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Required columns present: {has_all_cols}")
        
        # Check date column
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Date format valid")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠ Missing values found:")
            for col, count in missing[missing > 0].items():
                print(f"    {col}: {count}")
        else:
            print(f"✓ No missing values")
        
        # Check data shape
        print(f"✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check if sorted by date
        is_sorted = df['date'].is_monotonic_increasing
        print(f"✓ Sorted by date: {is_sorted}")
        
        # Show basic statistics
        print(f"\nBasic Statistics:")
        print(df[required_cols[1:]].describe().loc[['min', 'max', 'mean']].round(2))
        
        # Estimate suitable parameters
        n_rows = df.shape[0]
        if '5min' in filepath:
            interval = '5min'
            steps_per_day = 288
        else:
            interval = '30min'
            steps_per_day = 48
        
        # Calculate max possible seq_len and pred_len
        # Using 70% for training, 10% for validation, 20% for test
        train_size = int(n_rows * 0.7)
        
        print(f"\n📊 Recommended Parameters:")
        print(f"  Interval: {interval}")
        print(f"  Steps per day: {steps_per_day}")
        print(f"  Total records: {n_rows}")
        print(f"  Training size (70%): {train_size}")
        
        if train_size < 100:
            print(f"  ⚠ WARNING: Training set is very small!")
            print(f"  ⚠ Deep learning models may overfit.")
            print(f"  ⚠ Consider using simpler models like DLinear.")
            max_seq = max(12, train_size // 10)
            max_pred = max(6, train_size // 20)
        else:
            max_seq = min(steps_per_day * 2, train_size // 5)
            max_pred = min(steps_per_day, train_size // 10)
        
        print(f"\n  Suggested seq_len: {max_seq} ({max_seq/steps_per_day:.1f} days)")
        print(f"  Suggested pred_len: {max_pred} ({max_pred/steps_per_day:.1f} days)")
        print(f"  enc_in/dec_in/c_out: {len(required_cols)-1}")
        
        return True, None
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False, str(e)

def main():
    """Verify all AEMO CSV files."""
    
    data_dir = '/Users/yhm/Desktop/Time-Series-Library/data/AEMO'
    
    # Find all processed CSV files (exclude backups)
    csv_files = [f for f in glob(os.path.join(data_dir, '*.csv'))
                 if 'backup' not in f]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print("\n" + "="*60)
    print("AEMO Data Verification")
    print("="*60)
    print(f"Found {len(csv_files)} CSV files\n")
    
    results = {}
    for csv_file in sorted(csv_files):
        success, error = verify_csv(csv_file)
        results[os.path.basename(csv_file)] = (success, error)
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    passed = sum(1 for s, _ in results.values() if s)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All files are ready for model training!")
        print("\nNext steps:")
        print("1. Review the recommended parameters above")
        print("2. Run the example scripts in scripts/AEMO_forecast/")
        print("3. Check README.md for detailed usage instructions")
    else:
        print("\n✗ Some files have issues:")
        for filename, (success, error) in results.items():
            if not success:
                print(f"  - {filename}: {error}")
    
    print()

if __name__ == '__main__':
    main()

