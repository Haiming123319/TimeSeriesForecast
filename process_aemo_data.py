"""
AEMO Data Processing Script
This script processes AEMO CSV files to match the format required by time series models.

Required format:
- First column: 'date' (datetime)
- Remaining columns: numeric features
- Data sorted by time (earliest to latest)
"""

import pandas as pd
import os
from glob import glob

def process_aemo_csv(input_file, output_file):
    """
    Process a single AEMO CSV file to match model requirements.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    print(f"Processing: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Rename the date column to 'date'
    if 'Settlement Date' in df.columns:
        df.rename(columns={'Settlement Date': 'date'}, inplace=True)
    
    # Standardize price column name
    if 'Dispatch Price ($/MWh)' in df.columns:
        df.rename(columns={'Dispatch Price ($/MWh)': 'Price'}, inplace=True)
    elif 'Spot Price ($/MWh)' in df.columns:
        df.rename(columns={'Spot Price ($/MWh)': 'Price'}, inplace=True)
    
    # Rename other columns to simpler names
    column_mapping = {
        'Scheduled Demand (MW)': 'Demand',
        'Scheduled Generation (MW)': 'Scheduled_Gen',
        'Semi Scheduled Generation (MW)': 'Semi_Scheduled_Gen',
        'Net Import (MW)': 'Net_Import'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Remove non-numeric columns
    if 'Type' in df.columns:
        df.drop('Type', axis=1, inplace=True)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
    
    # Sort by date (earliest to latest)
    df.sort_values('date', inplace=True)
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    # Reorder columns: date first, then all features
    cols = ['date'] + [col for col in df.columns if col != 'date']
    df = df[cols]
    
    # Save to output file
    df.to_csv(output_file, index=False)
    
    print(f"  ✓ Saved to: {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    return df

def main():
    """Process all AEMO CSV files."""
    
    # Define paths
    data_dir = '/Users/yhm/Desktop/Time-Series-Library/data/AEMO'
    
    # Find all CSV files in the AEMO directory
    csv_files = glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print("="*60)
    print("AEMO Data Processing")
    print("="*60)
    print(f"Found {len(csv_files)} CSV files\n")
    
    # Process each CSV file
    for csv_file in sorted(csv_files):
        # Get filename
        filename = os.path.basename(csv_file)
        
        # Create output filename (overwrite original or create new)
        # For safety, we'll create a backup and overwrite
        backup_file = csv_file.replace('.csv', '_backup.csv')
        
        # Backup original file
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy2(csv_file, backup_file)
            print(f"Backup created: {backup_file}")
        
        # Process the file (overwrite original)
        try:
            df = process_aemo_csv(csv_file, csv_file)
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")
            print()
            continue
    
    print("="*60)
    print("Processing complete!")
    print("="*60)
    print("\nProcessed files are ready for use with models.")
    print("\nExample usage:")
    print("  - For multivariate forecasting (features='M'):")
    print("    All numeric columns will be used as features")
    print("  - For univariate forecasting (features='S'):")
    print("    Specify target='Price' or target='Demand', etc.")
    print("\nBackup files (*_backup.csv) have been created for safety.")

if __name__ == '__main__':
    main()

