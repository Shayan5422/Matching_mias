import pandas as pd
import numpy as np
import sys

# Function to calculate duration within a dataframe
def calculate_duration(df, id_col='id', entree_col='entree_date', sortie_col='sortie_date'):
    df_copy = df[[id_col, entree_col, sortie_col]].copy()
    df_copy[entree_col] = pd.to_datetime(df_copy[entree_col], format='%m/%d/%y', errors='coerce')
    df_copy[sortie_col] = pd.to_datetime(df_copy[sortie_col], format='%m/%d/%y', errors='coerce')
    df_copy.dropna(subset=[entree_col, sortie_col], inplace=True)
    if df_copy.empty:
        return None # Return None if no valid date pairs
    df_copy['duration'] = df_copy[sortie_col] - df_copy[entree_col]
    # Keep only id and duration, drop duplicates based on id
    df_duration = df_copy[[id_col, 'duration']].drop_duplicates(subset=[id_col], keep='first')
    return df_duration

try:
    # Define file paths
    log_file = 'log_sortie1.csv'
    entree_file = 'export_entree.csv'
    sortie_file = 'export_sortie1.csv'

    print(f"Loading {log_file}...")
    # log_sortie1.csv has unnamed columns at the beginning, get header to find correct cols
    log_cols = pd.read_csv(log_file, nrows=0).columns.tolist()
    # Assuming 'entre_id' and 'sortie_id' are the last two columns
    log_df = pd.read_csv(log_file, usecols=[log_cols[-2], log_cols[-1]], low_memory=False)
    log_df = log_df.rename(columns={log_cols[-2]: 'entre_id', log_cols[-1]: 'sortie_id'})
    log_df.dropna(subset=['entre_id', 'sortie_id'], inplace=True) # Drop rows with missing IDs
    # Convert IDs to integer type if possible, handle errors
    log_df['entre_id'] = pd.to_numeric(log_df['entre_id'], errors='coerce')
    log_df['sortie_id'] = pd.to_numeric(log_df['sortie_id'], errors='coerce')
    log_df.dropna(subset=['entre_id', 'sortie_id'], inplace=True)
    log_df['entre_id'] = log_df['entre_id'].astype(int)
    log_df['sortie_id'] = log_df['sortie_id'].astype(int)
    log_df = log_df.drop_duplicates(subset=['entre_id', 'sortie_id'], keep='first') # Ensure unique links

    print(f"Loading {entree_file}...")
    entree_df_full = pd.read_csv(entree_file, low_memory=False)
    # Ensure ID is int for merging
    entree_df_full['id'] = pd.to_numeric(entree_df_full['id'], errors='coerce').dropna().astype(int)
    entree_df_full = entree_df_full.drop_duplicates(subset=['id'], keep='first')


    print(f"Loading {sortie_file}...")
    sortie_df_full = pd.read_csv(sortie_file, low_memory=False)
    # Ensure ID is int for merging
    sortie_df_full['id'] = pd.to_numeric(sortie_df_full['id'], errors='coerce').dropna().astype(int)
    sortie_df_full = sortie_df_full.drop_duplicates(subset=['id'], keep='first')

    print("Merging data...")
    # Merge log with entree data
    # Suffixes=['_entree', '_sortie'] might be clearer than default _x, _y
    merged_df = pd.merge(log_df, entree_df_full, left_on='entre_id', right_on='id', how='inner')
    # Merge result with sortie data
    merged_df = pd.merge(merged_df, sortie_df_full, left_on='sortie_id', right_on='id', how='inner', suffixes=['_entree', '_sortie'])

    # Check if merge resulted in any rows
    if merged_df.empty:
        print('No matching records found between the files after merging.')
        sys.exit()

    print("Comparing common columns...")

    # Identify common columns to compare (excluding IDs and dates)
    cols_to_exclude = ['id_entree', 'id_sortie', 'entre_id', 'sortie_id',
                       'entree_date_entree', 'sortie_date_entree',
                       'entree_date_sortie', 'sortie_date_sortie',
                       'offset_entree', 'offset_sortie'] # Also exclude offset
    common_cols_base = [col for col in entree_df_full.columns if col in sortie_df_full.columns and col not in ['id', 'entree_date', 'sortie_date', 'offset']]

    comparison_results = {}
    total_records_compared = len(merged_df)

    for col in common_cols_base:
        col_entree = col + '_entree'
        col_sortie = col + '_sortie'

        # Check if columns exist after merge (should always be true here)
        if col_entree in merged_df.columns and col_sortie in merged_df.columns:
            # Compare values, considering NaNs as equal only if both are NaN
            # Convert to string first to handle mixed types gracefully, though this might impact performance
            # A more robust approach might involve type-specific comparison or fillna with a non-colliding placeholder
            equal_mask = (merged_df[col_entree].astype(str) == merged_df[col_sortie].astype(str))
            # Alternative considering NaN == NaN:
            # equal_mask = (merged_df[col_entree] == merged_df[col_sortie]) | (merged_df[col_entree].isna() & merged_df[col_sortie].isna())

            equal_count = equal_mask.sum()
            percentage = (equal_count / total_records_compared) * 100 if total_records_compared > 0 else 0
            comparison_results[col] = percentage
        else:
            print(f"Warning: Columns {col_entree} or {col_sortie} not found in merged data for base column '{col}'.")

    print("\n--- Column Value Comparison Results ---")
    print(f'Total linked records compared: {total_records_compared}')
    print("Percentage of records with identical values for each common column:")

    # Sort results alphabetically by column name
    sorted_results = sorted(comparison_results.items())

    for col, percentage in sorted_results:
        print(f'  - {col}: {percentage:.2f}%')

except FileNotFoundError as e:
    print(f'Error: File not found - {e.filename}')
except KeyError as e:
    print(f'Error: Column not found - {e}. Please check column names in CSV files.')
except Exception as e:
    print(f'An unexpected error occurred: {type(e).__name__} - {e}') 