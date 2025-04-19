import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

# --- Configuration ---
ENTREE_FILE = 'export_entree.csv'
SORTIE_FILE = 'export_sortie1.csv'
OUTPUT_FILE = 'mapping.csv'
# !!! IMPORTANT: Please verify and set the correct ID column names for your files below !!!
ENTREE_ID_COL = 'id'  # Replace with the actual ID column name in export_entree.csv
SORTIE_ID_COL = 'id'  # Replace with the actual ID column name in export_sortie1.csv

DATE_COLS = ['entree_date', 'sortie_date']
# !!! IMPORTANT: Adjust the date format if it differs from DD/MM/YYYY HH:MM:SS !!!
# Examples: '%Y-%m-%d', '%d/%m/%Y', etc.
DATE_FORMAT = '%m/%d/%Y' # Set to None if formats might vary wildly or are unknown

NUMERIC_COLS = ['annee', 'age', 'nb_rea', 'nb_si']
# Use appropriate encoding if needed, e.g., 'latin1' or 'utf-8-sig'
ENCODING = 'utf-8'

# --- Helper Functions ---

def calculate_annee_naiss(df):
    """Calculates year of birth (annee - age), handling potential errors."""
    annee = pd.to_numeric(df['annee'], errors='coerce')
    age = pd.to_numeric(df['age'], errors='coerce')
    return annee - age

def calculate_delta_days(df):
    """Calculates the difference in days between sortie_date and entree_date."""
    # Ensure columns are datetime objects
    if pd.api.types.is_datetime64_any_dtype(df['sortie_date']) and pd.api.types.is_datetime64_any_dtype(df['entree_date']):
        delta = (df['sortie_date'] - df['entree_date'])
        # Check if delta is NaT before accessing .days
        return delta.dt.days if not pd.isna(delta).all() else pd.NA
    return pd.NA

def safe_str_comparison(val1, val2):
    """Compare strings, treating NaN/None as matching anything (skip)."""
    if pd.isna(val1):
        return True
    # Ensure comparison is between strings, handle potential non-string types
    return str(val1) == str(val2)

def safe_non_negative_check(val_sortie, val_entree):
    """Check if both values are >= 0, skipping if sortie value is NaN."""
    if pd.isna(val_sortie):
        return True
    # Convert to numeric, coercing errors. Check if both are valid numbers >= 0.
    num_sortie = pd.to_numeric(val_sortie, errors='coerce')
    num_entree = pd.to_numeric(val_entree, errors='coerce')
    if pd.isna(num_sortie) or pd.isna(num_entree):
        # If conversion failed or original was NaN, outcome depends on rule interpretation.
        # Current rule: skip if sortie value is NaN originally.
        # If sortie *was* a value but failed numeric conversion, treat as non-match.
        # If entree value failed conversion, treat as non-match.
        return pd.isna(val_sortie) # True only if original sortie value was NaN
    return num_sortie >= 0 and num_entree >= 0


# --- Main Logic ---

print(f"Loading data...")
try:
    # Load DataFrames, explicitly handling potential dtype issues
    entree_df = pd.read_csv(ENTREE_FILE, encoding=ENCODING, low_memory=False)
    sortie_df = pd.read_csv(SORTIE_FILE, encoding=ENCODING, low_memory=False)
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure '{ENTREE_FILE}' and '{SORTIE_FILE}' are in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred during file loading: {e}")
    exit()


print("Preprocessing data...")
# Convert date columns
for col in DATE_COLS:
    print(f"Converting column '{col}' to datetime...") # Added for debugging
    try:
        entree_df[col] = pd.to_datetime(entree_df[col], format=DATE_FORMAT, errors='coerce')
        sortie_df[col] = pd.to_datetime(sortie_df[col], format=DATE_FORMAT, errors='coerce')
    except ValueError as e:
        print(f"Warning: Could not parse dates in column '{col}' with format '{DATE_FORMAT}'. Check format string or data. Error: {e}")
        # Fallback or alternative handling if needed
        entree_df[col] = pd.to_datetime(entree_df[col], errors='coerce') # Attempt without format
        sortie_df[col] = pd.to_datetime(sortie_df[col], errors='coerce')

# Calculate derived columns
entree_df['annee_naiss'] = calculate_annee_naiss(entree_df)
sortie_df['annee_naiss'] = calculate_annee_naiss(sortie_df)

entree_df['ghm_prefix'] = entree_df['ghm'].astype(str).str[:3].replace('nan', None)
sortie_df['ghm_prefix'] = sortie_df['ghm'].astype(str).str[:3].replace('nan', None)

entree_df['delta_days'] = calculate_delta_days(entree_df)
sortie_df['delta_days'] = calculate_delta_days(sortie_df)

# Ensure ID columns are suitable for matching and use configured names
print(f"Checking ID columns: Entree='{ENTREE_ID_COL}', Sortie='{SORTIE_ID_COL}'")
if ENTREE_ID_COL not in entree_df.columns:
    print(f"Error: Entree ID column '{ENTREE_ID_COL}' not found in {ENTREE_FILE}. Available columns: {list(entree_df.columns)}")
    exit()
if SORTIE_ID_COL not in sortie_df.columns:
    print(f"Error: Sortie ID column '{SORTIE_ID_COL}' not found in {SORTIE_FILE}. Available columns: {list(sortie_df.columns)}")
    exit()

entree_df[ENTREE_ID_COL] = entree_df[ENTREE_ID_COL].astype(str)
sortie_df[SORTIE_ID_COL] = sortie_df[SORTIE_ID_COL].astype(str)

print(f"Starting matching process (Sortie: {len(sortie_df)}, Entree: {len(entree_df)})...")
matches = []

# Use tqdm for a progress bar
for sortie_row in tqdm(sortie_df.itertuples(index=False), total=len(sortie_df), desc="Matching Records"):
    sortie_id = getattr(sortie_row, SORTIE_ID_COL)
    potential_matches = []

    # --- Pre-filtering entree_df (Optional Optimization) ---
    # Filter based on exact match columns first if performance is an issue
    # For now, iterate through all entree rows for simplicity and correctness
    # filtered_entree_df = entree_df[
    #     (entree_df['sexe'] == sortie_row.sexe) &
    #     (entree_df['annee_naiss'] == sortie_row.annee_naiss)
    # ]
    # Use filtered_entree_df in the inner loop if implementing optimization

    for entree_row in entree_df.itertuples(index=False):
        # --- Apply Matching Criteria ---

        # 1. Sexe (Mandatory Exact Match)
        if sortie_row.sexe != entree_row.sexe:
            continue

        # 2. Year of Birth (annee - age) (Mandatory Exact Match, handles relative shifts)
        # Check for NaN before comparison
        if pd.isna(sortie_row.annee_naiss) or pd.isna(entree_row.annee_naiss) or sortie_row.annee_naiss != entree_row.annee_naiss:
            continue

        # 3. GHM Prefix (First 3 chars)
        if not safe_str_comparison(sortie_row.ghm_prefix, entree_row.ghm_prefix):
             continue

        # 4. Entree Mode/Provenance
        if not safe_str_comparison(sortie_row.entree_mode, entree_row.entree_mode):
            continue
        if not safe_str_comparison(sortie_row.entree_provenance, entree_row.entree_provenance):
            continue

        # 5. Sortie Mode/Destination
        if not safe_str_comparison(sortie_row.sortie_mode, entree_row.sortie_mode):
            continue
        if not safe_str_comparison(sortie_row.sortie_destination, entree_row.sortie_destination):
            continue

        # 6. Nb Rea (>= 0)
        if not safe_non_negative_check(sortie_row.nb_rea, entree_row.nb_rea):
            continue

        # 7. Nb SI (>= 0)
        if not safe_non_negative_check(sortie_row.nb_si, entree_row.nb_si):
            continue

        # 8. Date Interval (delta_days difference <= 2)
        # Skip if sortie delta is NaN (due to missing dates in sortie record)
        if not pd.isna(sortie_row.delta_days):
            # If sortie delta exists, entree delta must also exist for comparison
            if pd.isna(entree_row.delta_days):
                continue # Entree dates missing, cannot compare interval
            # Compare interval difference
            if abs(sortie_row.delta_days - entree_row.delta_days) > 2:
                continue

        # If all checks passed, it's a match
        potential_matches.append(getattr(entree_row, ENTREE_ID_COL))

    # Store matches for this sortie_id
    if potential_matches:
        for entree_id in potential_matches:
            matches.append({'sortie_id': sortie_id, 'entree_id': entree_id})

print("Matching complete.")

# --- Output Results ---
if matches:
    results_df = pd.DataFrame(matches)
    print(f"Found {len(results_df)} matches.")
    try:
        results_df.to_csv(OUTPUT_FILE, index=False, encoding=ENCODING)
        print(f"Results saved to '{OUTPUT_FILE}'.")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
else:
    print("No matches found.") 