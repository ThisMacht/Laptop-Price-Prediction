import pandas as pd
import os
import sys

# Add the project root to the sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the data cleaning functions from src.data
from src.data.data_cleaning import (
    clean_newlines,
    clean_manufacturer,
    clean_ram,
    clean_storage,
    clean_resolution,
    clean_screen_size,
    clean_weight,
    clean_cpu_speed,
    clean_dimensions,
    clean_gpu,
    format_prices,
    fill_missing,
    drop_unnecessary_columns
)

def process_all_features(df):
    """Orchestrates all cleaning steps on a pandas DataFrame."""
    df = clean_newlines(df)
    df = clean_manufacturer(df)
    df = clean_ram(df)
    df = clean_storage(df)
    df = clean_resolution(df)
    df = clean_screen_size(df)
    df = clean_weight(df)
    df = clean_cpu_speed(df)
    df = clean_dimensions(df)
    df = clean_gpu(df)
    df = format_prices(df)
    df = fill_missing(df)
    df = drop_unnecessary_columns(df)
    return df

def run_cleaning_pipeline(input_path, output_path):
    print(f"Reading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print("Initial shape:", df.shape)
    
    # Process the data using the orchestrating function
    df = process_all_features(df)
    
    # Validation step
    print("Missing values in shop prices after cleaning:")
    print(df[['shop_1_price', 'shop_2_price', 'shop_3_price']].isnull().sum())
    print("Cleaned shape:", df.shape)

    # Save to processed directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved cleaned data to {output_path}")
