import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Get the directory of the current script to build robust paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TEXT_SENTIMENT_FILE = "weighted_textpes.csv"
PHOTO_SENTIMENT_FILE = "weighted_photopes_deduped.csv"
INDEX_FILE = "2025Q2_stock.xlsx"
OUTPUT_FILE = "merged_sentiment_and_returns.csv"

# Define the index columns to process and their new names
# Note: Assuming 'SSE Composite.1' is the SZSE Composite Index based on typical data ordering.
INDEX_COLUMNS_MAP = {
    'CSI 300': 'csi300',
    'SSE Composite': 'sse_composite',
    'SZSE Composite': 'szse_composite', # This is likely 'SSE Composite.1' in the source file, we will rename it.
    'ChiNext Composite': 'chinext_composite',
    'CSI 500': 'csi500'
}

def calculate_log_returns(df, columns):
    """Calculates log returns for specified columns."""
    for col in columns:
        # Ensure column is numeric and handle potential non-numeric values
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Calculate log return: ln(price_t / price_{t-1})
        df[f'{col}_return'] = np.log(df[col] / df[col].shift(1))
    return df

def main():
    """Main function to load, process, and merge data."""
    # --- 1. Load and Process Index Data ---
    print("Loading and processing stock index data...")
    index_df = pd.read_excel(os.path.join(DATA_DIR, INDEX_FILE))
    
    # Rename 'Date' to 'news_date' for consistency and 'SSE Composite.1' to 'SZSE Composite'
    index_df = index_df.rename(columns={'Date': 'news_date', 'SSE Composite.1': 'SZSE Composite'})
    index_df['news_date'] = pd.to_datetime(index_df['news_date'])

    # Select only the necessary columns
    required_cols = ['news_date'] + list(INDEX_COLUMNS_MAP.keys())
    missing_cols = [col for col in required_cols if col not in index_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in index file: {missing_cols}")
    index_df = index_df[required_cols]

    # Filter for dates between 2014 and 2024
    index_df = index_df[(index_df['news_date'].dt.year >= 2013) & (index_df['news_date'].dt.year <= 2025)]
    print(f"Index data loaded for 2014-2024, containing {len(index_df)} records.")

    # --- 2. Calculate Log Returns ---
    print("Calculating log returns for indices...")
    # We keep original price columns and add return columns
    index_with_returns_df = calculate_log_returns(index_df.copy(), list(INDEX_COLUMNS_MAP.keys()))

    # --- 3. Load and Merge Sentiment Data ---
    print("Loading and merging sentiment data...")
    text_df = pd.read_csv(os.path.join(DATA_DIR, TEXT_SENTIMENT_FILE))
    photo_df = pd.read_csv(os.path.join(DATA_DIR, PHOTO_SENTIMENT_FILE))

    # Convert date columns to datetime objects
    text_df['news_date'] = pd.to_datetime(text_df['news_date'])
    photo_df['news_date'] = pd.to_datetime(photo_df['news_date'])

    # Merge sentiment data using an outer join to keep all dates from both
    sentiment_df = pd.merge(text_df, photo_df, on='news_date', how='outer')
    print(f"Merged sentiment data contains {len(sentiment_df)} unique dates.")

    # --- 4. Merge All Data ---
    print("Merging index data with sentiment data...")
    # Left join sentiment data onto the index data to preserve all index dates
    final_df = pd.merge(index_with_returns_df, sentiment_df, on='news_date', how='left')

    # --- 5. Save Final Data ---
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Successfully merged data and saved to {output_path}")
    print(f"Final dataset has {len(final_df)} rows.")
    print("Final data head:")
    print(final_df.head())

if __name__ == '__main__':
    main()
