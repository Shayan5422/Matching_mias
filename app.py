import streamlit as st
import pandas as pd
import numpy as np

# --- Configuration ---
ENTREE_FILE = 'export_entree.csv'
# Use appropriate encoding if needed, e.g., 'latin1' or 'utf-8-sig'
ENCODING = 'utf-8'
DATE_FORMAT = '%m/%d/%Y' # Adjust if needed, or set to None
ENTREE_ID_COL = 'id' # Assuming 'id' is the ID column in entree file

# Allowed columns for filtering
ALLOWED_COLUMNS = {
    "annee_naiss": "Year of Birth (calculated)",
    "ghm_prefix": "GHM (First 3 chars)",
    "sexe": "Sex",
    "entree_mode": "Entry Mode",
    "entree_provenance": "Entry Source",
    "sortie_mode": "Exit Mode",
    "sortie_destination": "Exit Destination",
    "nb_rea": "Nb Rea (>= 0)",
    "nb_si": "Nb SI (>= 0)",
    "delta_days": "Stay Duration (within +/- 2 days)"
}

# --- Helper Functions (Adapted from reverse_anonymization.py) ---

def calculate_annee_naiss(df):
    """Calculates year of birth (annee - age), handling potential errors."""
    annee = pd.to_numeric(df['annee'], errors='coerce')
    age = pd.to_numeric(df['age'], errors='coerce')
    return annee - age

def calculate_delta_days(df):
    """Calculates the difference in days between sortie_date and entree_date."""
    for col in ['entree_date', 'sortie_date']:
        if col not in df.columns:
            return pd.NA # Column missing
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
             # Attempt conversion if not already datetime
             df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors='coerce')

    # Proceed only if both columns are now datetime
    if pd.api.types.is_datetime64_any_dtype(df['sortie_date']) and pd.api.types.is_datetime64_any_dtype(df['entree_date']):
        delta = (df['sortie_date'] - df['entree_date'])
        return delta.dt.days
    return pd.NA

def safe_str_comparison(val_sortie, val_entree, is_sortie_nan):
    """Compare strings, treating NaN/None in sortie as matching anything (skip)."""
    if is_sortie_nan:
        return True
    # If sortie value is not NaN, entree must match exactly (as strings)
    return str(val_sortie) == str(val_entree)

def safe_non_negative_check(val_sortie, val_entree, is_sortie_nan):
    """Check if both values are >= 0, skipping if sortie value is NaN."""
    if is_sortie_nan:
        return True

    # If sortie value exists, convert both to numeric and check non-negativity
    num_sortie = pd.to_numeric(val_sortie, errors='coerce')
    num_entree = pd.to_numeric(val_entree, errors='coerce')

    # Both must be valid non-negative numbers
    if pd.isna(num_sortie) or pd.isna(num_entree):
        return False # Cannot compare if conversion fails
    return num_sortie >= 0 and num_entree >= 0

# --- Data Loading and Caching ---
@st.cache_data # Cache the loaded entree data
def load_entree_data():
    try:
        df = pd.read_csv(ENTREE_FILE, encoding=ENCODING, low_memory=False)
        df['entree_date'] = pd.to_datetime(df['entree_date'], format=DATE_FORMAT, errors='coerce')
        df['sortie_date'] = pd.to_datetime(df['sortie_date'], format=DATE_FORMAT, errors='coerce')
        df['annee_naiss'] = calculate_annee_naiss(df)
        df['ghm_prefix'] = df['ghm'].astype(str).str[:3].replace('nan', None)
        df['delta_days'] = calculate_delta_days(df.copy()) # Pass copy to avoid modifying cache
        # Ensure mandatory columns exist
        if 'sexe' not in df.columns:
             st.error(f"Mandatory column 'sexe' not found in {ENTREE_FILE}.")
             return None
        if ENTREE_ID_COL not in df.columns:
             st.error(f"ID column '{ENTREE_ID_COL}' not found in {ENTREE_FILE}.")
             return None
        df[ENTREE_ID_COL] = df[ENTREE_ID_COL].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{ENTREE_FILE}' was not found. Please place it in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading {ENTREE_FILE}: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Find Matching Records in Original Data")

# Load entree data once
entree_df = load_entree_data()

if entree_df is None:
    st.stop()

# File Uploader for Sortie Data
st.header("1. Upload Your 'Sortie' CSV File")
uploaded_file = st.file_uploader("Choose a CSV file (e.g., export_sortie1.csv)", type="csv")

if uploaded_file is not None:
    try:
        sortie_df_uploaded = pd.read_csv(uploaded_file, encoding=ENCODING, low_memory=False)
        st.success("File uploaded successfully!")

        # --- Preprocess Uploaded Sortie Data ---
        # Check for necessary columns used in calculations
        required_cols_for_calc = ['annee', 'age', 'ghm', 'entree_date', 'sortie_date']
        missing_calc_cols = [col for col in required_cols_for_calc if col not in sortie_df_uploaded.columns]
        if missing_calc_cols:
            st.warning(f"Uploaded file is missing columns needed for some calculations: {', '.join(missing_calc_cols)}. Corresponding filters might not work correctly.")

        # Calculate derived columns for uploaded data (handle missing columns gracefully)
        if 'annee' in sortie_df_uploaded.columns and 'age' in sortie_df_uploaded.columns:
             sortie_df_uploaded['annee_naiss'] = calculate_annee_naiss(sortie_df_uploaded)
        else: 
             sortie_df_uploaded['annee_naiss'] = pd.NA
        
        if 'ghm' in sortie_df_uploaded.columns:
             sortie_df_uploaded['ghm_prefix'] = sortie_df_uploaded['ghm'].astype(str).str[:3].replace('nan', None)
        else:
            sortie_df_uploaded['ghm_prefix'] = None
        
        # Convert dates and calculate delta days
        if 'entree_date' in sortie_df_uploaded.columns:
            sortie_df_uploaded['entree_date'] = pd.to_datetime(sortie_df_uploaded['entree_date'], format=DATE_FORMAT, errors='coerce')
        if 'sortie_date' in sortie_df_uploaded.columns:
            sortie_df_uploaded['sortie_date'] = pd.to_datetime(sortie_df_uploaded['sortie_date'], format=DATE_FORMAT, errors='coerce')
        
        sortie_df_uploaded['delta_days'] = calculate_delta_days(sortie_df_uploaded.copy())

        # Display uploaded data
        st.header("2. Select a Record from Uploaded Data")
        st.dataframe(sortie_df_uploaded)

        # Select row from uploaded data
        max_index = len(sortie_df_uploaded) - 1
        selected_index = st.number_input(f"Enter the index of the row to match (0 to {max_index})", min_value=0, max_value=max_index, step=1)
        selected_sortie_row = sortie_df_uploaded.iloc[selected_index]

        st.subheader("Selected Row Data:")
        st.dataframe(pd.DataFrame(selected_sortie_row).T) # Display selected row horizontally

        # --- Filter Selection ---
        st.header("3. Select Matching Criteria")
        st.markdown("Check the boxes for the criteria you want to use for matching. Filters based on blank values in the *selected* row will be skipped (except for Sex and Year of Birth).")

        active_filters = {}
        cols = st.columns(3) # Arrange checkboxes in columns
        i = 0
        for col_key, col_label in ALLOWED_COLUMNS.items():
            # Check if the column exists in the selected sortie row
            if col_key not in selected_sortie_row.index:
                with cols[i % 3]:
                     st.checkbox(f"{col_label} (Missing in uploaded data)", value=False, disabled=True, key=f"filter_{col_key}")
            else:
                 # Get the value from the selected row to display it
                 sortie_val = selected_sortie_row[col_key]
                 display_val = "(Blank)" if pd.isna(sortie_val) else sortie_val
                 # Special handling for date interval display
                 if col_key == 'delta_days' and not pd.isna(sortie_val):
                     display_val = f"{int(sortie_val)} days"
                 
                 # Only enable checkbox if column exists
                 is_enabled = True
                 # Default checked state could be set here if needed, e.g., check all by default
                 default_checked = True 
                 
                 # Use Streamlit columns for better layout
                 with cols[i % 3]:
                     active_filters[col_key] = st.checkbox(f"{col_label}: `{display_val}`", value=default_checked, disabled=not is_enabled, key=f"filter_{col_key}")
            i += 1

        # --- Perform Matching ---
        st.header("4. Matching Results")
        if st.button("Find Matches"):
            with st.spinner("Searching for matches..."):
                # Start with the full entree dataframe
                potential_matches_df = entree_df.copy()
                match_criteria_used = []

                # Apply selected filters sequentially
                for filter_key, is_active in active_filters.items():
                    if is_active and filter_key in selected_sortie_row.index:
                        sortie_value = selected_sortie_row[filter_key]
                        is_sortie_nan = pd.isna(sortie_value)
                        match_criteria_used.append(ALLOWED_COLUMNS[filter_key])

                        # Apply filter based on the key
                        if filter_key == 'sexe':
                            # Mandatory exact match, even if sortie is NaN (though unlikely for sex)
                             potential_matches_df = potential_matches_df[potential_matches_df['sexe'] == sortie_value]
                        elif filter_key == 'annee_naiss':
                            # Mandatory exact match, cannot match if either is NaN
                             if is_sortie_nan:
                                 potential_matches_df = potential_matches_df[pd.isna(potential_matches_df['annee_naiss'])] # Match only NaN if sortie is NaN
                             else:
                                potential_matches_df = potential_matches_df[potential_matches_df['annee_naiss'] == sortie_value]
                        elif filter_key == 'delta_days':
                            if not is_sortie_nan:
                                # Compare interval only if sortie has valid dates
                                lower_bound = sortie_value - 2
                                upper_bound = sortie_value + 2
                                # Filter entree rows where delta_days is within the tolerance range
                                # Exclude entree rows where delta_days is NaN
                                potential_matches_df = potential_matches_df[
                                    potential_matches_df['delta_days'].notna() & 
                                    (potential_matches_df['delta_days'] >= lower_bound) & 
                                    (potential_matches_df['delta_days'] <= upper_bound)
                                ]
                        elif filter_key in ['nb_rea', 'nb_si']:
                            if not is_sortie_nan:
                                # Apply non-negative check only if sortie value exists
                                # Vectorized check: Both must be numbers >= 0
                                entree_num = pd.to_numeric(potential_matches_df[filter_key], errors='coerce')
                                sortie_num = pd.to_numeric(sortie_value, errors='coerce') # Should be a number already, but coerce just in case
                                
                                # Keep rows where entree is non-negative and sortie is non-negative (implicit)
                                potential_matches_df = potential_matches_df[
                                     entree_num.notna() & (entree_num >= 0) & (not pd.isna(sortie_num)) & (sortie_num >=0)
                                ]
                                
                                # Simplified logic from original: if sortie_val is not NaN and >=0, 
                                # then entree_val must also be not NaN and >=0.
                                # Note: safe_non_negative_check was row-wise, this is vectorized.
                                # Let's refine the vectorized logic to match the description:
                                # Keep if sortie is NaN OR (sortie is >= 0 AND entree is >= 0)
                                # This is slightly complex to vectorize directly with the NaN skip rule.
                                # Reverting to apply for simplicity, although slower:
                                # potential_matches_df = potential_matches_df[potential_matches_df.apply(
                                #     lambda row: safe_non_negative_check(sortie_value, row[filter_key], is_sortie_nan),
                                #     axis=1
                                # )] 
                                # Let's stick to the faster vectorized approach: keep rows where entree is >= 0, 
                                # since the check is only active when sortie_value is not NaN (and assumed >=0 by rule)
                                potential_matches_df = potential_matches_df[pd.to_numeric(potential_matches_df[filter_key], errors='coerce') >= 0]
                                
                        else: # Generic string comparison columns (ghm_prefix, modes, etc.)
                            if not is_sortie_nan:
                                potential_matches_df = potential_matches_df[potential_matches_df[filter_key].astype(str) == str(sortie_value)]

                # Display Results
                st.subheader("Results")
                if not potential_matches_df.empty:
                    st.info(f"Found {len(potential_matches_df)} potential match(es) in '{ENTREE_FILE}' based on criteria: {', '.join(match_criteria_used)}")
                    # Display relevant columns + ID
                    display_cols = [ENTREE_ID_COL] + [key for key, active in active_filters.items() if active and key in potential_matches_df.columns]
                    # Ensure sexe and annee_naiss are always shown if they exist, as they are mandatory filters
                    if 'sexe' in potential_matches_df.columns and 'sexe' not in display_cols: display_cols.append('sexe')
                    if 'annee_naiss' in potential_matches_df.columns and 'annee_naiss' not in display_cols: display_cols.append('annee_naiss')
                    st.dataframe(potential_matches_df[list(dict.fromkeys(display_cols))]) # Use dict.fromkeys to keep order and remove duplicates
                else:
                    st.warning(f"No matches found in '{ENTREE_FILE}' for the selected criteria: {', '.join(match_criteria_used)}")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("Please upload a CSV file to begin.") 