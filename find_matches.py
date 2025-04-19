import pandas as pd
import numpy as np
import time
import ast # For safely evaluating string representation of list later

# Define core identifiers that should not be skipped even if missing in sortie
CORE_IDENTIFIERS = ['birth_year', 'ghm_prefix']
# Placeholder for invalid/missing calculated values
INVALID_STAY = -999
INVALID_AGE_GEST = -1
INVALID_BIRTH_YEAR = -1

def preprocess_df(df):
    """Preprocesses the DataFrame for matching."""
    processed = pd.DataFrame(index=df.index)
    # 1. Calculate birth year
    processed['birth_year'] = pd.to_numeric(df['annee'], errors='coerce') - pd.to_numeric(df['age'], errors='coerce')
    processed['birth_year'] = processed['birth_year'].fillna(INVALID_BIRTH_YEAR).astype(int)

    # 2. GHM prefix
    processed['ghm_prefix'] = df['ghm'].astype(str).str[:3]
    # 3. Sex
    processed['sexe'] = df['sexe']

    # --- Date Processing for Length of Stay ---
    entree_dt = pd.to_datetime(df['entree_date'], format='%m/%d/%y', errors='coerce')
    sortie_dt = pd.to_datetime(df['sortie_date'], format='%m/%d/%y', errors='coerce')
    processed['length_of_stay'] = (sortie_dt - entree_dt).dt.days
    processed['length_of_stay'] = processed['length_of_stay'].fillna(INVALID_STAY).astype(int)

    # --- Other Columns ---
    processed['entree_mode'] = df['entree_mode']
    processed['entree_provenance'] = df['entree_provenance'].fillna('').astype(str)
    processed['sortie_mode'] = df['sortie_mode']
    processed['sortie_destination'] = df['sortie_destination'].fillna('').astype(str)
    processed['age_gestationnel'] = pd.to_numeric(df['age_gestationnel'], errors='coerce').fillna(INVALID_AGE_GEST).astype(int)
    processed['nb_rea_bool'] = pd.to_numeric(df['nb_rea'], errors='coerce').fillna(0) > 0
    processed['nb_si_bool'] = pd.to_numeric(df['nb_si'], errors='coerce').fillna(0) > 0

    return processed

def run_matching_strategy(strategy_columns, entree_processed, sortie_processed, entree_ids, sortie_ids):
    """Runs the sequential matching logic for a given strategy.
       Returns a DataFrame with sortie_id, status, matched_entree_id, multiple_matches_list.
    """
    print(f"  Running strategy: {strategy_columns}")
    start_time = time.time()
    results_list = []

    for index in sortie_processed.index:
        sortie_row = sortie_processed.loc[index]
        sortie_id_val = sortie_ids.loc[index]

        potential_matches_df = entree_processed.copy()
        status = 'pending'
        matched_entree_id_val = np.nan
        multiple_matches_list = [] # Store list of IDs if multiple
        num_matches = len(potential_matches_df)

        for col in strategy_columns:
            if num_matches <= 1:
                 break

            value = sortie_row[col]

            is_missing = False
            if pd.isna(value):
                is_missing = True
            elif isinstance(value, str) and value == '':
                 is_missing = True
            elif col == 'age_gestationnel' and value == INVALID_AGE_GEST:
                 is_missing = True
            elif col == 'length_of_stay' and value == INVALID_STAY:
                 is_missing = True
            elif col == 'birth_year' and value == INVALID_BIRTH_YEAR:
                 is_missing = True

            if is_missing and col not in CORE_IDENTIFIERS:
                continue

            if col == 'length_of_stay':
                if value == INVALID_STAY:
                    potential_matches_df = potential_matches_df[potential_matches_df[col] == INVALID_STAY]
                else:
                    valid_entree_stays = potential_matches_df[potential_matches_df[col] != INVALID_STAY]
                    potential_matches_df = valid_entree_stays[abs(valid_entree_stays[col] - value) <= 2]
            elif col == 'birth_year':
                 if value == INVALID_BIRTH_YEAR:
                      potential_matches_df = potential_matches_df[potential_matches_df[col] == INVALID_BIRTH_YEAR]
                 else:
                      valid_entree_births = potential_matches_df[potential_matches_df[col] != INVALID_BIRTH_YEAR]
                      potential_matches_df = valid_entree_births[valid_entree_births[col] == value]
            elif col == 'age_gestationnel':
                 if value == INVALID_AGE_GEST:
                     potential_matches_df = potential_matches_df[potential_matches_df[col] == INVALID_AGE_GEST]
                 else:
                     valid_entree_ages = potential_matches_df[potential_matches_df[col] != INVALID_AGE_GEST]
                     potential_matches_df = valid_entree_ages[valid_entree_ages[col] == value]
            else:
                try:
                    potential_matches_df = potential_matches_df[potential_matches_df[col] == value]
                except TypeError:
                     potential_matches_df = potential_matches_df[potential_matches_df[col].astype(str) == str(value)]

            num_matches = len(potential_matches_df)
            if num_matches == 1:
                status = 'unique'
                matched_entree_index = potential_matches_df.index[0]
                matched_entree_id_val = entree_ids.loc[matched_entree_index]
                multiple_matches_list = [] # Clear list
                break
            elif num_matches == 0:
                status = 'none'
                matched_entree_id_val = np.nan
                multiple_matches_list = [] # Clear list
                break

        if status == 'pending':
            if num_matches > 1:
                status = 'multiple'
                # Get the list of IDs for multiple matches
                multiple_matches_list = entree_ids.loc[potential_matches_df.index].tolist()
                matched_entree_id_val = np.nan # No single matched ID
            else: # Should be 0
                status = 'none'
                matched_entree_id_val = np.nan
                multiple_matches_list = []

        results_list.append({
            'sortie_id': sortie_id_val,
            'status': status,
            'matched_entree_id': matched_entree_id_val, # Will be NaN if multiple or none
            'multiple_matches_list': multiple_matches_list # List of IDs if status is 'multiple' initially
        })

    end_time = time.time()
    print(f"  Strategy finished in {end_time - start_time:.2f} seconds.")
    return pd.DataFrame(results_list)

def evaluate_strategy(match_results_df, ground_truth_df):
    """Evaluates the post-processed strategy results against ground truth."""
    total_records = len(match_results_df)
    if total_records == 0:
        return {
            'accuracy': 0, 'unique_correct_rate': 0, 'correct_in_multiple_rate': 0,
            'unique_incorrect_rate': 0, 'none_rate': 1, 'multiple_incorrect_rate': 0
        }

    # --- Calculate Counts Based on Final Status --- 
    # Note: Assumes match_results_df already has the final statuses after post-processing

    # Unique and Correct
    unique_correct_matches = match_results_df[
        (match_results_df['status'] == 'unique') &
        (match_results_df['matched_entree_id'].notna()) &
        (match_results_df['entre_id'].notna()) &
        (match_results_df['matched_entree_id'] == match_results_df['entre_id'])
    ].shape[0]

    # Unique but Incorrect
    unique_incorrect_matches = match_results_df[
        (match_results_df['status'] == 'unique') &
        (
            (match_results_df['matched_entree_id'].isna()) |
            (match_results_df['entre_id'].isna()) |
            (match_results_df['matched_entree_id'] != match_results_df['entre_id'])
        )
    ].shape[0]

    # Correct found within Multiple
    correct_in_multiple_matches = match_results_df[match_results_df['status'] == 'correct_in_multiple'].shape[0]

    # None matches
    total_none_matches = match_results_df[match_results_df['status'] == 'none'].shape[0]

    # Multiple matches where correct ID was NOT found
    multiple_incorrect_matches = match_results_df[match_results_df['status'] == 'multiple'].shape[0]

    # --- Calculate Rates --- 
    accuracy = (unique_correct_matches + correct_in_multiple_matches) / total_records if total_records > 0 else 0
    unique_correct_rate = unique_correct_matches / total_records if total_records > 0 else 0
    correct_in_multiple_rate = correct_in_multiple_matches / total_records if total_records > 0 else 0
    unique_incorrect_rate = unique_incorrect_matches / total_records if total_records > 0 else 0
    none_rate = total_none_matches / total_records if total_records > 0 else 0
    multiple_incorrect_rate = multiple_incorrect_matches / total_records if total_records > 0 else 0

    # Sanity check
    # total_rate = unique_correct_rate + correct_in_multiple_rate + unique_incorrect_rate + none_rate + multiple_incorrect_rate
    # print(f"Debug: Total Rate Sum = {total_rate}")

    return {
        'accuracy': accuracy, # Overall accuracy (unique correct + correct_in_multiple)
        'unique_correct_rate': unique_correct_rate,
        'correct_in_multiple_rate': correct_in_multiple_rate,
        'unique_incorrect_rate': unique_incorrect_rate,
        'none_rate': none_rate,
        'multiple_incorrect_rate': multiple_incorrect_rate
    }

# --- Main Script ---
print("Loading data...")
try:
    df_entree = pd.read_csv('export_entree.csv', sep=',', low_memory=False)
    df_sortie = pd.read_csv('export_sortie1.csv', sep=',', low_memory=False)
    df_ground_truth = pd.read_csv('log_sortie1.csv', sep=',', usecols=['entre_id', 'sortie_id'])
    # Ensure IDs in ground truth are numeric, coercing errors
    df_ground_truth['entre_id'] = pd.to_numeric(df_ground_truth['entre_id'], errors='coerce')
    df_ground_truth['sortie_id'] = pd.to_numeric(df_ground_truth['sortie_id'], errors='coerce')
    df_ground_truth.dropna(subset=['entre_id', 'sortie_id'], inplace=True) # Remove rows where IDs couldn't be parsed
    df_ground_truth['entre_id'] = df_ground_truth['entre_id'].astype(int)
    df_ground_truth['sortie_id'] = df_ground_truth['sortie_id'].astype(int)

    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure all CSV files are in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred during file loading: {e}")
    exit()

print("Preprocessing data...")
try:
    entree_processed = preprocess_df(df_entree.copy())
    sortie_processed = preprocess_df(df_sortie.copy())
    entree_ids = df_entree['id']
    sortie_ids = df_sortie['id']
    print("Preprocessing complete.")
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")
    exit()

# --- Define Strategies (using length_of_stay) ---
strategies = {
    "Strategy_A_LOS": ['birth_year', 'ghm_prefix', 'sexe', 'length_of_stay', 'entree_mode', 'entree_provenance', 'sortie_mode', 'sortie_destination', 'age_gestationnel', 'nb_rea_bool', 'nb_si_bool'],
    "Strategy_B": ['birth_year', 'ghm_prefix', 'sexe'],
    "Strategy_C_LOS": ['birth_year', 'ghm_prefix', 'sexe', 'length_of_stay'],
    "Strategy_D_LOS": ['ghm_prefix', 'birth_year', 'sexe', 'length_of_stay', 'entree_mode', 'sortie_mode'],
    "Strategy_E_LOS": ['birth_year', 'ghm_prefix', 'sexe', 'length_of_stay', 'entree_mode', 'entree_provenance', 'sortie_mode', 'sortie_destination'],
    "Strategy_F_LOS": ['birth_year', 'ghm_prefix', 'sexe', 'length_of_stay', 'entree_mode', 'entree_provenance', 'sortie_mode', 'sortie_destination', 'age_gestationnel', 'nb_rea_bool', 'nb_si_bool'],
}

# --- Run, Post-Process, and Evaluate Strategies ---
evaluation_results = {}
all_strategy_results = {} # Store the post-processed results df for each strategy
best_strategy_name = None
best_accuracy = -1

print("\nRunning and evaluating strategies...")
for name, strategy_cols in strategies.items():
    print(f"\n--- Evaluating {name} ---")
    missing_cols = [col for col in strategy_cols if col not in entree_processed.columns or col not in sortie_processed.columns]
    if missing_cols:
        print(f"  Skipping strategy {name} due to missing columns: {missing_cols}")
        continue

    # 1. Run the matching strategy
    initial_match_results = run_matching_strategy(
        strategy_cols,
        entree_processed,
        sortie_processed,
        entree_ids,
        sortie_ids
    )

    # Ensure base ID columns are correct type before merge/processing
    initial_match_results['sortie_id'] = initial_match_results['sortie_id'].astype(int)
    initial_match_results['matched_entree_id'] = pd.to_numeric(initial_match_results['matched_entree_id'], errors='coerce')

    # 2. Merge with ground truth
    merged_results = pd.merge(
        initial_match_results,
        df_ground_truth,
        on='sortie_id',
        how='left' # Keep all strategy results
    )

    # 3. Post-process 'multiple' status
    final_results = merged_results.copy()
    for index, row in final_results.iterrows():
        if row['status'] == 'multiple':
            true_entree_id = row['entre_id'] # Ground truth ID
            multiple_list = row['multiple_matches_list']
            if isinstance(multiple_list, list) and not pd.isna(true_entree_id) and int(true_entree_id) in multiple_list:
                final_results.loc[index, 'status'] = 'correct_in_multiple'
                # Store the list as a string in the matched_id column
                final_results.loc[index, 'matched_entree_id'] = str(multiple_list)
            else:
                # Keep status as 'multiple' (incorrect), ensure matched_id is NaN
                final_results.loc[index, 'matched_entree_id'] = np.nan

    # Convert matched_entree_id back to appropriate type (object now due to strings)
    # final_results['matched_entree_id'] = pd.to_numeric(final_results['matched_entree_id'], errors='coerce') # This won't work for the lists-as-strings

    # Store post-processed results
    all_strategy_results[name] = final_results

    # 4. Evaluate the final post-processed results
    metrics = evaluate_strategy(final_results, df_ground_truth) # Pass final_results
    evaluation_results[name] = metrics
    print(f"  Metrics for {name}: {metrics}")

    if metrics['accuracy'] > best_accuracy:
        best_accuracy = metrics['accuracy']
        best_strategy_name = name

# --- Print Summary and Recommendation ---
print("\n--- Evaluation Summary ---")
summary_df = pd.DataFrame.from_dict(evaluation_results, orient='index')
summary_df.index.name = 'Strategy'
summary_df = summary_df.sort_values(by='accuracy', ascending=False)
# Update formatting for new metrics
print(summary_df.to_string(formatters={
    'accuracy': '{:,.2%}'.format,
    'unique_correct_rate': '{:,.2%}'.format,
    'correct_in_multiple_rate': '{:,.2%}'.format,
    'unique_incorrect_rate': '{:,.2%}'.format,
    'none_rate': '{:,.2%}'.format,
    'multiple_incorrect_rate': '{:,.2%}'.format
}))

if best_strategy_name:
    print(f"\nRecommended Strategy (Highest Accuracy): {best_strategy_name}")
    print(f"Columns: {strategies[best_strategy_name]}")

    print(f"\nGenerating final results using {best_strategy_name}...")
    # Retrieve the already post-processed results for the best strategy
    best_final_results_df = all_strategy_results[best_strategy_name]

    output_filename = 'best_strategy_matches.csv'
    try:
        # Select and save relevant columns
        output_columns = ['sortie_id', 'status', 'matched_entree_id', 'entre_id'] # Include ground truth ID for reference
        # Ensure columns exist before selecting
        output_columns = [col for col in output_columns if col in best_final_results_df.columns]
        save_df = best_final_results_df[output_columns]

        # Convert sortie_id and entre_id to Int64
        save_df['sortie_id'] = save_df['sortie_id'].astype('Int64')
        if 'entre_id' in save_df.columns:
            save_df['entre_id'] = save_df['entre_id'].astype('Int64')
        # matched_entree_id remains object/string due to lists

        save_df.to_csv(output_filename, index=False)
        print(f"Detailed results for the best strategy saved to {output_filename}")
    except Exception as e:
        print(f"Error saving best strategy results to CSV: {e}")

else:
    print("\nNo strategies were successfully evaluated.")

print("\nScript finished.")

