import pandas as pd
import numpy as np
from datetime import datetime

# --- Constants copied from app.py ---
ENCODING = 'utf-8'
DATE_FORMAT = '%m/%d/%y'
UPLOADED_ID_COL = 'id'
ALLOWED_COLUMNS = {
    "annee_naiss": "Année de Naissance (calculée)",
    "ghm_prefix": "GHM (3 premiers car.)",
    "sexe": "Sexe",
    "age_gestationnel_weeks": "Âge Gestationnel (semaines)",
    "entree_mode": "Mode d'Entrée",
    "entree_provenance": "Provenance d'Entrée",
    "sortie_mode": "Mode de Sortie",
    "sortie_destination": "Destination de Sortie",
    "nb_rea": "Nb Réa (≥ 0)",
    "nb_si": "Nb SI (≥ 0)",
    "delta_days": "Durée Séjour (± 2 jours)"
}
AUTO_MATCH_DISPLAY_COLS = [UPLOADED_ID_COL, 'annee_naiss', 'sexe', 'ghm_prefix', 'delta_days'] # Needed by find_confident_matches

# --- Helper Functions copied from app.py (Streamlit dependencies removed) ---

def calculate_annee_naiss(annee_series, age_series):
    # تبدیل به عدد
    annee = pd.to_numeric(annee_series, errors='coerce')
    age   = pd.to_numeric(age_series, errors='coerce')
    # سال تولد
    birth_year = annee - age
    # محاسبه دهه (مثلاً 1983 → 1980)
    return (birth_year // 10) * 10


def calculate_delta_days(entree_dates, sortie_dates):
    entree_dt = pd.to_datetime(entree_dates, errors='coerce')
    sortie_dt = pd.to_datetime(sortie_dates, errors='coerce')
    delta = sortie_dt - entree_dt
    return delta.dt.days

def extract_gestational_weeks(age_gest_series):
    def parse_weeks(age_gest_str):
        if pd.isna(age_gest_str):
            return pd.NA
        try:
            s = str(age_gest_str)
            weeks_part = ''.join(filter(str.isdigit, s.split('+')[0].split()[0]))
            return int(weeks_part) if weeks_part else pd.NA
        except (ValueError, TypeError, IndexError):
            return pd.NA
    return age_gest_series.apply(parse_weeks)

# --- Preprocessing Function copied from app.py (Streamlit dependencies removed) ---
def preprocess_dataframe(df, file_name):
    """Adds calculated/derived columns needed for matching to the DataFrame."""
    print(f"Preprocessing {file_name}...")
    processed_df = df.copy()

    base_cols_needed = ['annee', 'age', 'ghm', 'entree_date', 'sortie_date', 'age_gestationnel',
                          'sexe', 'entree_mode', 'entree_provenance', 'sortie_mode',
                          'sortie_destination', 'nb_rea', 'nb_si', UPLOADED_ID_COL]
    for col in base_cols_needed:
        if col not in processed_df.columns:
            print(f"WARNING: File '{file_name}' missing expected column: '{col}'. Adding with empty values.")
            if col in ['annee', 'age', 'nb_rea', 'nb_si']:
                processed_df[col] = pd.NA
            elif col in ['entree_date', 'sortie_date']:
                 processed_df[col] = pd.NaT
            else:
                processed_df[col] = None

    processed_df['annee_naiss'] = calculate_annee_naiss(processed_df.get('annee'), processed_df.get('age'))
    processed_df['ghm_prefix'] = processed_df.get('ghm', pd.Series(dtype=str)).astype(str).str[:3].replace('nan', None)
    processed_df['age_gestationnel_weeks'] = extract_gestational_weeks(processed_df.get('age_gestationnel', pd.Series(dtype=object)))

    processed_df['entree_date'] = pd.to_datetime(processed_df.get('entree_date'), format=DATE_FORMAT, errors='coerce')
    processed_df['sortie_date'] = pd.to_datetime(processed_df.get('sortie_date'), format=DATE_FORMAT, errors='coerce')
    processed_df['delta_days'] = calculate_delta_days(processed_df['entree_date'], processed_df['sortie_date'])

    if UPLOADED_ID_COL in processed_df.columns:
        processed_df[UPLOADED_ID_COL] = processed_df[UPLOADED_ID_COL].astype(str)
    else:
         print(f"ERROR: ID column '{UPLOADED_ID_COL}' is missing in file '{file_name}' and cannot be created.")
         pass # Let it proceed but ID search might fail

    for key in ALLOWED_COLUMNS.keys():
         if key not in processed_df.columns:
             processed_df[key] = pd.NA

    print(f"Preprocessing for {file_name} complete.")
    return processed_df

# --- Automatic Matching Function copied from app.py (Streamlit dependencies removed) ---
def find_confident_matches(_comparison_df, _entree_df, _comparison_filename, _entree_filename):
    """
    Finds rows in comparison_df that match exactly one row in entree_df
    based on available ALLOWED_COLUMNS.
    """
    print(f"Finding automatic matches between '{_comparison_filename}' and '{_entree_filename}'...")

    if UPLOADED_ID_COL not in _comparison_df.columns or UPLOADED_ID_COL not in _entree_df.columns:
        print(f"WARNING: ID column '{UPLOADED_ID_COL}' needed in both files for automatic matching.")
        return pd.DataFrame()

    confident_matches_list = []
    comparison_cols = _comparison_df.columns
    entree_cols = _entree_df.columns

    match_cols = [col for col in ALLOWED_COLUMNS.keys() if col in comparison_cols and col in entree_cols]
    if not match_cols:
        print("WARNING: No common columns found among allowed columns for automatic matching.")
        return pd.DataFrame()

    # Pre-convert columns in entree_df for faster comparison
    entree_numeric_cols = {}
    entree_string_cols = {}
    for col_key in match_cols:
        if col_key in ['annee_naiss', 'age_gestationnel_weeks', 'nb_rea', 'nb_si', 'delta_days']:
             entree_numeric_cols[col_key] = pd.to_numeric(_entree_df[col_key], errors='coerce')
        elif col_key == 'sexe':
             entree_string_cols[col_key] = _entree_df[col_key].astype(str).str.upper().fillna('')
        else: # ghm_prefix, modes, prov, dest
             entree_string_cols[col_key] = _entree_df[col_key].astype(str).str.strip().fillna('')

    # Iterate through comparison dataframe rows
    for i, comp_row in _comparison_df.iterrows():
        current_filter = pd.Series(True, index=_entree_df.index)

        for col_key in match_cols:
            comp_val = comp_row[col_key]
            if pd.isna(comp_val):
                continue

            try:
                if col_key == 'delta_days':
                    comp_num = pd.to_numeric(comp_val, errors='coerce')
                    if pd.notna(comp_num):
                         lower_bound = comp_num - 2
                         upper_bound = comp_num + 2
                         entree_vals = entree_numeric_cols[col_key]
                         col_filter = entree_vals.notna() & (entree_vals >= lower_bound) & (entree_vals <= upper_bound)
                         current_filter &= col_filter
                    else:
                         current_filter &= pd.Series(False, index=_entree_df.index)

                elif col_key in ['annee_naiss', 'age_gestationnel_weeks', 'nb_rea', 'nb_si']:
                     comp_num = pd.to_numeric(comp_val, errors='coerce')
                     if pd.notna(comp_num):
                          entree_vals = entree_numeric_cols[col_key]
                          current_filter &= (entree_vals == comp_num)
                     else:
                          current_filter &= pd.Series(False, index=_entree_df.index)

                elif col_key == 'sexe':
                     comp_str = str(comp_val).upper()
                     current_filter &= (entree_string_cols[col_key] == comp_str)

                else: # Generic string comparison
                     comp_str = str(comp_val).strip()
                     current_filter &= (entree_string_cols[col_key] == comp_str)

            except Exception as e:
                 print(f"ERROR applying filter for '{col_key}' on comparison row index {i}: {e}")
                 current_filter &= pd.Series(False, index=_entree_df.index)
                 break

        entree_indices = _entree_df.index[current_filter]
        if len(entree_indices) == 1:
            entree_match_index = entree_indices[0]
            comp_id = comp_row[UPLOADED_ID_COL]
            entree_id = _entree_df.loc[entree_match_index, UPLOADED_ID_COL]

            match_data = {
                f"comparison_{UPLOADED_ID_COL}": comp_id,
                f"entree_{UPLOADED_ID_COL}": entree_id
            }
            # Optionally add display columns if needed for debugging matches later
            # for disp_col in AUTO_MATCH_DISPLAY_COLS:
            #     if disp_col in comp_row.index and disp_col != UPLOADED_ID_COL:
            #         match_data[f"comparison_{disp_col}"] = comp_row[disp_col]

            confident_matches_list.append(match_data)

    print("Automatic matching search complete.")
    if confident_matches_list:
        return pd.DataFrame(confident_matches_list)
    else:
        return pd.DataFrame()

# --- Evaluation Script ---
def evaluate_accuracy(comparison_file, entree_file, log_file):
    """Loads data, runs matching, compares to log, and prints accuracy."""
    print("\n--- Starting Evaluation ---")
    try:
        # 1. Load Data
        print(f"Loading comparison file: {comparison_file}")
        comparison_df_raw = pd.read_csv(comparison_file, encoding=ENCODING, low_memory=False)
        print(f"Loading entree file: {entree_file}")
        entree_df_raw = pd.read_csv(entree_file, encoding=ENCODING, low_memory=False)
        print(f"Loading log file (ground truth): {log_file}")
        # *** Corrected log file reading based on actual header and column names ***
        # *** Delimiter: ',', Has header, Use 'sortie_id' and 'entre_id' columns ***
        log_df = pd.read_csv(
            log_file,
            encoding=ENCODING,
            low_memory=False,
            delimiter=',', # Use comma delimiter
            # header=None, usecols=[1, 3], names=[...] removed - read header now
        )
        # Rename columns to match expected names in the rest of the script
        if 'sortie_id' in log_df.columns and 'entre_id' in log_df.columns:
            log_df = log_df.rename(columns={
                'sortie_id': 'comparison_id',
                'entre_id': 'entree_id'
            })
            print(f"Loaded {len(log_df)} ground truth matches.")
        else:
            print(f"ERROR: Log file '{log_file}' is missing required columns 'sortie_id' or 'entre_id'. Aborting evaluation.")
            return

        # Ensure IDs are strings for comparison
        log_df['comparison_id'] = log_df['comparison_id'].astype(str)
        log_df['entree_id'] = log_df['entree_id'].astype(str)

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}. Aborting evaluation.")
        return
    except Exception as e:
        print(f"ERROR loading files: {e}. Aborting evaluation.")
        return

    # 2. Preprocess DataFrames
    comparison_df_processed = preprocess_dataframe(comparison_df_raw, comparison_file)
    entree_df_processed = preprocess_dataframe(entree_df_raw, entree_file)

    # Check if ID columns exist after preprocessing before matching
    if UPLOADED_ID_COL not in comparison_df_processed.columns or UPLOADED_ID_COL not in entree_df_processed.columns:
         print(f"ERROR: ID column '{UPLOADED_ID_COL}' missing in preprocessed dataframes. Cannot perform matching.")
         return

    # 3. Run Automatic Matching
    predicted_matches_df = find_confident_matches(
        comparison_df_processed,
        entree_df_processed,
        comparison_file,
        entree_file
    )
    total_predicted = len(predicted_matches_df)
    print(f"Found {total_predicted} automatic unique matches.")

    if total_predicted == 0:
        print("No automatic matches found by the algorithm.")
        # Depending on the log file, accuracy might be 0 or undefined.
        # If the log file also has 0 matches, it's 100% correct in finding none.
        # If the log file has matches, accuracy is 0%.
        if len(log_df) == 0:
            print("Log file also contains 0 matches. Accuracy: 100% (correctly found no matches)")
        else:
            print(f"Log file contains {len(log_df)} matches. Accuracy: 0.00%")
        return

    # 4. Prepare Ground Truth Set
    true_matches_set = set(zip(log_df['comparison_id'], log_df['entree_id']))

    # 5. Evaluate Predictions
    correct_matches = 0
    predicted_comparison_col = f"comparison_{UPLOADED_ID_COL}"
    predicted_entree_col = f"entree_{UPLOADED_ID_COL}"

    # Ensure predicted columns exist
    if predicted_comparison_col not in predicted_matches_df.columns or predicted_entree_col not in predicted_matches_df.columns:
        print(f"ERROR: Predicted match dataframe missing required ID columns: '{predicted_comparison_col}' or '{predicted_entree_col}'. Cannot evaluate.")
        return

    for _, row in predicted_matches_df.iterrows():
        pred_comp_id = str(row[predicted_comparison_col])
        pred_entree_id = str(row[predicted_entree_col])
        if (pred_comp_id, pred_entree_id) in true_matches_set:
            correct_matches += 1

    # 6. Calculate and Print Accuracy
    accuracy = (correct_matches / total_predicted) * 100
    print("\n--- Evaluation Results ---")
    print(f"Total Automatic Matches Predicted: {total_predicted}")
    print(f"Correct Matches (found in log): {correct_matches}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("--------------------------")


# --- Main Execution ---
if __name__ == "__main__":
    # Define your file paths here
    comparison_csv = "export_sortie1.csv"
    entree_csv = "export_entree.csv"
    log_csv = "log_sortie1.csv" # Ground truth file

    evaluate_accuracy(comparison_csv, entree_csv, log_csv) 