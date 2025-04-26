import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- Configuration ---
# No longer assumes a fixed entree file
ENCODING = 'utf-8'
DATE_FORMAT = '%m/%d/%y' # Adjusted based on previous interaction
# !!! IMPORTANT: Set the name of the ID column in the files users will upload !!!
UPLOADED_ID_COL = 'id' # Assuming same ID column name for both files

# Allowed columns for filtering and derived columns
# Base columns expected for calculations: annee, age, ghm, entree_date, sortie_date, age_gestationnel, sexe, ... (and modes/dest/prov)
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

# --- Helper Functions --- (Unchanged)

def calculate_annee_naiss(annee_series, age_series):
    annee = pd.to_numeric(annee_series, errors='coerce')
    age = pd.to_numeric(age_series, errors='coerce')
    return annee - age

def calculate_delta_days(entree_dates, sortie_dates):
    entree_dt = pd.to_datetime(entree_dates, errors='coerce')
    sortie_dt = pd.to_datetime(sortie_dates, errors='coerce')
    delta = sortie_dt - entree_dt
    return delta.dt.days

def extract_gestational_weeks(age_gest_series):
    # Optimized for applying to a Series
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

# --- Function to preprocess the entire uploaded DataFrame ---
# Renamed cache function slightly to allow caching both files separately
@st.cache_data
def preprocess_dataframe(df, file_name):
    """Adds calculated/derived columns needed for matching to the DataFrame."""
    st.info(f"Preprocessing {file_name}...") # Add info message
    processed_df = df.copy()

    # --- Ensure Base Columns Exist ---
    base_cols_needed = ['annee', 'age', 'ghm', 'entree_date', 'sortie_date', 'age_gestationnel',
                          'sexe', 'entree_mode', 'entree_provenance', 'sortie_mode',
                          'sortie_destination', 'nb_rea', 'nb_si', UPLOADED_ID_COL]
    for col in base_cols_needed:
        if col not in processed_df.columns:
            # Using warning consistently
            st.warning(f"Fichier '{file_name}' manquant la colonne attendue : '{col}'. Elle sera ajoutée avec des valeurs vides.")
            processed_df[col] = pd.NA if col in ['annee', 'age', 'nb_rea', 'nb_si'] else None

    # --- Calculate Derived Columns ---
    # Using .get() for safety in case base columns were missing despite the check above
    processed_df['annee_naiss'] = calculate_annee_naiss(processed_df.get('annee'), processed_df.get('age'))
    processed_df['ghm_prefix'] = processed_df.get('ghm', pd.Series(dtype=str)).astype(str).str[:3].replace('nan', None)
    processed_df['age_gestationnel_weeks'] = extract_gestational_weeks(processed_df.get('age_gestationnel', pd.Series(dtype=object)))

    # Convert dates before calculating delta
    processed_df['entree_date'] = pd.to_datetime(processed_df.get('entree_date'), format=DATE_FORMAT, errors='coerce')
    processed_df['sortie_date'] = pd.to_datetime(processed_df.get('sortie_date'), format=DATE_FORMAT, errors='coerce')
    processed_df['delta_days'] = calculate_delta_days(processed_df['entree_date'], processed_df['sortie_date'])

    # Ensure ID column is string
    if UPLOADED_ID_COL in processed_df.columns:
        processed_df[UPLOADED_ID_COL] = processed_df[UPLOADED_ID_COL].astype(str)
    else:
         st.error(f"La colonne ID '{UPLOADED_ID_COL}' est manquante dans le fichier '{file_name}' et ne peut pas être créée.")
         # Return empty df or handle differently? For now, let it proceed but ID search won't work.
         pass

    # Ensure all allowed columns exist, even if derived as NA
    for key in ALLOWED_COLUMNS.keys():
         if key not in processed_df.columns:
             processed_df[key] = pd.NA

    st.info(f"Preprocessing for {file_name} complete.")
    return processed_df

# --- Function to preprocess a single manual input record --- (Unchanged)
def preprocess_manual_record(record_dict):
    """Converts dict from manual input into a processed Series with derived fields."""
    manual_series = pd.Series(record_dict, dtype=object)
    base_cols = ['annee', 'age', 'ghm', 'entree_date', 'sortie_date', 'age_gestationnel']
    for col in base_cols:
        if col not in manual_series.index:
             manual_series[col] = None
    manual_series['entree_date'] = pd.to_datetime(manual_series.get('entree_date'), errors='coerce')
    manual_series['sortie_date'] = pd.to_datetime(manual_series.get('sortie_date'), errors='coerce')
    combined_index = list(ALLOWED_COLUMNS.keys()) + list(record_dict.keys())
    processed = pd.Series(index=list(dict.fromkeys(combined_index)), dtype=object)
    processed.update(manual_series)
    annee = pd.to_numeric(processed.get('annee'), errors='coerce')
    age = pd.to_numeric(processed.get('age'), errors='coerce')
    processed['annee_naiss'] = annee - age if pd.notna(annee) and pd.notna(age) else pd.NA
    processed['ghm_prefix'] = str(processed['ghm'])[:3] if pd.notna(processed['ghm']) else None
    age_gest_val = processed.get('age_gestationnel')
    processed['age_gestationnel_weeks'] = extract_gestational_weeks_scalar(age_gest_val) # Assuming scalar version exists
    if pd.notna(processed['entree_date']) and pd.notna(processed['sortie_date']):
        processed['delta_days'] = (processed['sortie_date'] - processed['entree_date']).days
    else:
        processed['delta_days'] = pd.NA
    for key in ALLOWED_COLUMNS.keys():
        if key not in processed.index:
            processed[key] = pd.NA
    return processed

# --- Scalar version of extract_gestational_weeks --- (Unchanged)
def extract_gestational_weeks_scalar(age_gest_str):
    """Extracts whole weeks from a single gestational age string."""
    if pd.isna(age_gest_str):
        return pd.NA
    try:
        s = str(age_gest_str)
        weeks_part = ''.join(filter(str.isdigit, s.split('+')[0].split()[0]))
        return int(weeks_part) if weeks_part else pd.NA
    except (ValueError, TypeError, IndexError):
        return pd.NA

# --- Helper Functions --- (Potentially reusable for auto-matching)
# Need to slightly adjust filter logic for reuse
def apply_filters(target_df, filters_dict, allowed_cols_map):
    """Applies a dictionary of filters to a DataFrame.

    Args:
        target_df: The DataFrame to filter.
        filters_dict: A dictionary where keys are column names (like 'ghm_prefix')
                      and values are the values to filter by.
        allowed_cols_map: The ALLOWED_COLUMNS dictionary for display names.

    Returns:
        A filtered DataFrame.
    """
    results_df = target_df.copy()
    for filter_key, search_value in filters_dict.items():
        is_search_nan = pd.isna(search_value) or search_value is None

        # Skip filter if search value is NaN, except for specific fields allowed to match NaN
        if is_search_nan and filter_key not in ['sexe', 'annee_naiss']:
            continue

        # Ensure the column exists in the dataframe being filtered
        if filter_key not in results_df.columns:
             # In auto-matching, we might just skip instead of warning for every row
             # st.warning(f"Auto-match: Colonne '{filter_key}' manquante dans le fichier cible.")
             continue

        # Apply filter based on the key (reusing logic from button handler)
        try:
            if filter_key == 'sexe':
                if is_search_nan:
                     results_df = results_df[results_df['sexe'].isna() | (results_df['sexe'] == '')]
                else:
                    results_df = results_df[results_df['sexe'].astype(str).str.upper() == str(search_value).upper()]
            elif filter_key == 'annee_naiss':
                if is_search_nan:
                     results_df = results_df[results_df['annee_naiss'].isna()]
                else:
                     search_num = pd.to_numeric(search_value, errors='coerce')
                     if pd.notna(search_num):
                          results_df = results_df[pd.to_numeric(results_df['annee_naiss'], errors='coerce') == search_num]
                     else: results_df = results_df.iloc[0:0]

            elif filter_key == 'delta_days':
                search_num = pd.to_numeric(search_value, errors='coerce')
                if pd.notna(search_num):
                    lower_bound = search_num - 2
                    upper_bound = search_num + 2
                    results_df = results_df[results_df['delta_days'].notna() &
                                          (results_df['delta_days'] >= lower_bound) &
                                          (results_df['delta_days'] <= upper_bound)]
                else: results_df = results_df.iloc[0:0]

            elif filter_key in ['nb_rea', 'nb_si']:
                 search_num = pd.to_numeric(search_value, errors='coerce')
                 if pd.notna(search_num) and search_num >= 0:
                     df_num = pd.to_numeric(results_df[filter_key], errors='coerce')
                     # Match exact value
                     results_df = results_df[df_num == search_num]
                 else: results_df = results_df.iloc[0:0]

            elif filter_key == 'age_gestationnel_weeks':
                search_num = pd.to_numeric(search_value, errors='coerce')
                if pd.notna(search_num):
                     results_df = results_df[pd.to_numeric(results_df['age_gestationnel_weeks'], errors='coerce') == search_num]
                else: results_df = results_df.iloc[0:0]

            else: # Generic comparison
                if not is_search_nan:
                    results_df = results_df[results_df[filter_key].astype(str).str.strip().fillna('') == str(search_value).strip()]

            # Optimization: if results become empty, stop applying more filters for this row
            if results_df.empty:
                break

        except Exception as filter_ex:
            # Log error or handle differently for auto-matching? Maybe just return empty.
            st.error(f"Erreur auto-match filtre '{allowed_cols_map.get(filter_key, filter_key)}': {filter_ex}")
            return target_df.iloc[0:0] # Return empty df on error

    return results_df

# --- Function to perform automatic confident matching ---
@st.cache_data # Cache the result of auto-matching
def find_confident_matches(_comparison_df, _entree_df, _allowed_cols_keys, _id_col):
    """
    Iterates through comparison_df, uses all valid allowed columns to filter entree_df,
    and returns rows where exactly one match is found.
    """
    confident_matches_list = []
    if _id_col not in _comparison_df.columns or _id_col not in _entree_df.columns:
        st.warning(f"La colonne ID '{_id_col}' est nécessaire dans les deux fichiers pour le matching automatique.")
        return pd.DataFrame(columns=[f'Comparison_{_id_col}', f'Entree_{_id_col}'])

    total_rows = len(_comparison_df)
    progress_bar = st.progress(0, text="Recherche automatique des correspondances...")

    for i, (_, comparison_row) in enumerate(_comparison_df.iterrows()):
        filters = {}
        # Build filters from the current comparison row using ALLOWED_COLUMNS
        for col_key in _allowed_cols_keys:
            if col_key in comparison_row and pd.notna(comparison_row[col_key]):
                filters[col_key] = comparison_row[col_key]

        if not filters: # Skip if no valid filters can be built for this row
            continue

        # Apply the filters to the entree dataframe
        matched_entree_rows = apply_filters(_entree_df, filters, ALLOWED_COLUMNS) # Pass ALLOWED_COLUMNS for context if needed by apply_filters

        # Check if exactly one match was found
        if len(matched_entree_rows) == 1:
            comparison_id = comparison_row[_id_col]
            entree_id = matched_entree_rows.iloc[0][_id_col]
            confident_matches_list.append({
                f'Comparison_{_id_col}': comparison_id,
                f'Entree_{_id_col}': entree_id
            })

        # Update progress bar
        progress_bar.progress((i + 1) / total_rows, text=f"Recherche automatique: Ligne {i+1}/{total_rows}")

    progress_bar.empty() # Clear progress bar
    if confident_matches_list:
        return pd.DataFrame(confident_matches_list)
    else:
        return pd.DataFrame(columns=[f'Comparison_{_id_col}', f'Entree_{_id_col}'])

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Comparer des Fichiers CSV contre un Fichier 'Entree' de Référence")

# --- Initialize Session State ---
# Separate state for entree file and comparison file
if 'entree_data' not in st.session_state:
    st.session_state.entree_data = None
    st.session_state.entree_filename = None
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
    st.session_state.comparison_filename = None
if 'search_record' not in st.session_state:
    st.session_state.search_record = None
if 'selected_id_value' not in st.session_state: # Index refers to the comparison file now
     st.session_state.selected_id_value = None
if 'confident_matches' not in st.session_state: # Add state for confident matches
     st.session_state.confident_matches = None

# --- File Upload Section ---
st.header("1. Téléverser les Fichiers")
col1_up, col2_up = st.columns(2)

with col1_up:
    st.subheader("Fichier Entree (Référence)")
    uploaded_entree_file = st.file_uploader("Choisir le fichier CSV 'Entree'", type="csv", key="entree_uploader")
    if uploaded_entree_file is not None:
        # Process entree file if it's new
        if uploaded_entree_file.name != st.session_state.entree_filename:
            with st.spinner("Traitement du fichier Entree..."):
                try:
                    entree_df_raw = pd.read_csv(uploaded_entree_file, encoding=ENCODING, low_memory=False)
                    st.session_state.entree_data = preprocess_dataframe(entree_df_raw, uploaded_entree_file.name)
                    st.session_state.entree_filename = uploaded_entree_file.name
                    # Reset comparison if entree file changes? Maybe not necessary, depends on workflow.
                    # st.session_state.comparison_data = None
                    # st.session_state.comparison_filename = None
                    # st.session_state.search_record = None
                    st.success(f"Fichier Entree '{uploaded_entree_file.name}' traité.")
                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier Entree : {e}")
                    st.session_state.entree_data = None
                    st.session_state.entree_filename = None

with col2_up:
    st.subheader("Fichier à Comparer")
    uploaded_comparison_file = st.file_uploader("Choisir le fichier CSV à comparer", type="csv", key="comparison_uploader")
    if uploaded_comparison_file is not None:
        # Process comparison file if it's new
        if uploaded_comparison_file.name != st.session_state.comparison_filename:
             with st.spinner("Traitement du fichier de comparaison..."):
                try:
                    comparison_df_raw = pd.read_csv(uploaded_comparison_file, encoding=ENCODING, low_memory=False)
                    st.session_state.comparison_data = preprocess_dataframe(comparison_df_raw, uploaded_comparison_file.name)
                    st.session_state.comparison_filename = uploaded_comparison_file.name
                    st.session_state.search_record = None # Reset search record
                    st.session_state.selected_id_value = None
                    st.session_state.confident_matches = None # Reset confident matches on new file upload
                    st.success(f"Fichier de comparaison '{uploaded_comparison_file.name}' traité.")

                    # --- Perform Automatic Confident Matching ---
                    if st.session_state.entree_data is not None:
                         st.info("Lancement de la recherche automatique des correspondances uniques...")
                         comp_df = st.session_state.comparison_data
                         ent_df = st.session_state.entree_data
                         allowed_keys = list(ALLOWED_COLUMNS.keys())
                         id_col = UPLOADED_ID_COL

                         # Run the matching function (will be cached)
                         st.session_state.confident_matches = find_confident_matches(comp_df, ent_df, allowed_keys, id_col)
                         st.info("Recherche automatique terminée.")
                    else:
                         st.warning("Le fichier 'Entree' n'est pas chargé. Impossible d'effectuer la recherche automatique.")

                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier de comparaison : {e}")
                    st.session_state.comparison_data = None
                    st.session_state.comparison_filename = None
                    st.session_state.confident_matches = None # Reset on error

# --- Display Confident Matches (if available) ---
if st.session_state.get('confident_matches') is not None:
     st.subheader("Correspondances Automatiques Uniques Trouvées")
     confident_df = st.session_state.confident_matches
     if not confident_df.empty:
         st.success(f"{len(confident_df)} correspondances uniques trouvées automatiquement en utilisant tous les critères disponibles.")
         st.dataframe(confident_df)
     else:
         st.info("Aucune correspondance unique n'a été trouvée automatiquement avec les critères stricts.")

# --- Main Application Logic (Requires both files) ---
if st.session_state.entree_data is not None and st.session_state.comparison_data is not None:
    entree_df = st.session_state.entree_data
    comparison_df = st.session_state.comparison_data
    entree_filename = st.session_state.entree_filename
    comparison_filename = st.session_state.comparison_filename

    st.header("2. Définir l'Enregistrement de Recherche (depuis le fichier à comparer)")
    search_method = st.radio("Comment voulez-vous définir l'enregistrement à rechercher ?",
                           ("Sélectionner une Ligne du Fichier à Comparer", "Saisir les Critères Manuellement"), horizontal=True,
                           key="search_method")

    current_search_record = None
    # current_selected_index = None # No longer storing index, just the record
    current_selected_id = None # Store the ID instead

    if search_method == "Sélectionner une Ligne du Fichier à Comparer":
        st.subheader(f"2a. Sélectionner l'ID dans le Fichier '{comparison_filename}'")
        
        # Check if ID column exists
        if UPLOADED_ID_COL not in comparison_df.columns:
            st.error(f"La colonne ID '{UPLOADED_ID_COL}' est manquante dans le fichier de comparaison '{comparison_filename}'. Impossible de sélectionner par ID.")
        elif not comparison_df.empty:
            # Ensure IDs are unique enough for selection, convert to list for selectbox
            # Handle potential NaN/None values in ID column before getting unique values
            valid_ids = comparison_df[UPLOADED_ID_COL].dropna().unique().tolist()
            if not valid_ids:
                st.warning(f"Aucun ID valide trouvé dans la colonne '{UPLOADED_ID_COL}' du fichier '{comparison_filename}'.")
            else:
                # Display the dataframe for context
                st.dataframe(comparison_df)
                
                # Use selectbox for ID selection
                selected_id = st.selectbox(
                    f"Sélectionnez l'ID de la ligne du fichier '{comparison_filename}' à utiliser comme base de recherche",
                    options=valid_ids,
                    key="id_selector",
                    index=None, # Default to no selection
                    placeholder="Choisir un ID..."
                )
                
                if selected_id is not None:
                    # Find the first row matching the selected ID
                    # Use .loc with boolean indexing. Ensure comparison handles data types (IDs are string)
                    matching_rows = comparison_df.loc[comparison_df[UPLOADED_ID_COL] == str(selected_id)]
                    if not matching_rows.empty:
                        current_search_record = matching_rows.iloc[0] # Get the first match as a Series
                        current_selected_id = selected_id # Store the selected ID
                    else:
                        # This case should theoretically not happen with selectbox using valid IDs
                        st.warning(f"ID '{selected_id}' sélectionné mais non trouvé dans le DataFrame. Veuillez réessayer.")
        else:
            st.warning(f"Le fichier de comparaison '{comparison_filename}' ne contient aucune ligne de données à sélectionner.")


    elif search_method == "Saisir les Critères Manuellement":
        st.subheader("2a. Saisir les Critères de Recherche")
        st.markdown("Entrez les détails de l'enregistrement que vous souhaitez rechercher dans le fichier Entree. **Les champs peuvent être laissés vides.**")
        manual_data = {}
        col1, col2, col3 = st.columns(3)
        # Manual input fields (unchanged logic, keys might need adjustment if conflicts)
        with col1:
            manual_data['annee'] = st.number_input("Année (annee)", value=None, step=1, key="man_annee")
            manual_data['age'] = st.number_input("Âge (age)", value=None, step=1, key="man_age")
            manual_data['sexe'] = st.text_input("Sexe (sexe)", key="man_sexe")
            manual_data['ghm'] = st.text_input("GHM", key="man_ghm")
            manual_data['age_gestationnel'] = st.text_input("Âge Gestationnel (ex: 38+3)", key="man_ag")
        with col2:
            manual_data['entree_date'] = st.date_input("Date d'Entrée", value=None, format="MM/DD/YYYY", key="man_edate")
            manual_data['entree_mode'] = st.text_input("Mode d'Entrée", key="man_emode")
            manual_data['entree_provenance'] = st.text_input("Provenance d'Entrée", key="man_eprov")
            manual_data['nb_rea'] = st.number_input("Nb Réa", value=None, min_value=0, step=1, key="man_rea")
        with col3:
            manual_data['sortie_date'] = st.date_input("Date de Sortie", value=None, format="MM/DD/YYYY", key="man_sdate")
            manual_data['sortie_mode'] = st.text_input("Mode de Sortie", key="man_smode")
            manual_data['sortie_destination'] = st.text_input("Destination de Sortie", key="man_sdest")
            manual_data['nb_si'] = st.number_input("Nb SI", value=None, min_value=0, step=1, key="man_si")

        for key, value in manual_data.items():
             if isinstance(value, str) and not value: manual_data[key] = None
             if key.endswith('_date') and value is not None: manual_data[key] = pd.to_datetime(value)

        current_search_record = preprocess_manual_record(manual_data)
        current_selected_id = None # No specific ID for manual entry

    # --- Store search record in session state if it changed ---
    # Update state if a record was found (either by selection or manual input)
    if current_search_record is not None:
        st.session_state.search_record = current_search_record
        st.session_state.selected_id_value = current_selected_id # Store the ID
    # If method changed back from selection or manual input cleared, reset
    # This logic might need refinement depending on desired state persistence
    # elif search_method != st.session_state.get('last_search_method'):
    #     st.session_state.search_record = None
    #     st.session_state.selected_id_value = None

    # Store the method for comparison next time (optional)
    # st.session_state.last_search_method = search_method


    # --- Display Selected Record & Filters --- 
    search_record_to_use = st.session_state.search_record
    if search_record_to_use is not None:
        st.header("3. Enregistrement de Recherche & Filtres")
        st.subheader("Données de Base pour la Recherche (issues de la sélection/saisie) :")
        st.dataframe(pd.DataFrame(search_record_to_use).T)

        st.subheader(f"Sélectionner les Critères pour Rechercher dans '{entree_filename}'")
        st.markdown("Cochez les cases des critères à utiliser pour filtrer le fichier Entree. Les filtres basés sur des valeurs vides sont ignorés (sauf Sexe/Année Naissance).")

        active_filters = {}
        cols = st.columns(3)
        i = 0
        for col_key, col_label in ALLOWED_COLUMNS.items():
             # Check if key exists in the *search record*
             if col_key not in search_record_to_use.index:
                 # Display as N/A if not in the search record itself
                 with cols[i % 3]: st.checkbox(f"{col_label} (N/A dans recherche)", value=False, disabled=True, key=f"filt_{col_key}")
             else:
                 search_val = search_record_to_use[col_key]
                 display_val = "(Vide)" if pd.isna(search_val) or search_val is None else search_val
                 # Formatting for display
                 if col_key == 'delta_days' and pd.notna(search_val): display_val = f"{int(search_val)} jours"
                 elif col_key == 'annee_naiss' and pd.notna(search_val): display_val = f"{int(search_val)}"
                 elif col_key == 'age_gestationnel_weeks' and pd.notna(search_val): display_val = f"{int(search_val)} semaines"

                 # Check if the corresponding column exists in the *entree_data* to enable the checkbox
                 is_col_in_entree = col_key in entree_df.columns
                 default_checked = pd.notna(search_val) and search_val is not None and is_col_in_entree

                 with cols[i % 3]:
                     active_filters[col_key] = st.checkbox(
                         f"{col_label}: `{display_val}`",
                         value=default_checked,
                         key=f"filt_{col_key}",
                         disabled=not is_col_in_entree # Disable if column missing in entree file
                     )
                     if not is_col_in_entree:
                         st.caption(f" (colonne manquante dans '{entree_filename}')") # Add caption if disabled
             i += 1

        # --- Perform Matching ---
        st.header("4. Résultats de la Correspondance")
        if st.button(f"Rechercher les Correspondances dans '{entree_filename}'"):
            with st.spinner(f"Recherche des correspondances dans '{entree_filename}'..."):
                # Start with the full ENTREE dataframe
                results_df = entree_df.copy()
                match_criteria_used = []
                # selected_row_index_to_exclude = st.session_state.selected_index_value # No longer applicable when comparing files

                # Apply selected filters sequentially TO THE ENTREE DF
                for filter_key, is_active in active_filters.items():
                    # Check if filter is active AND the key exists in the search record
                    if is_active and filter_key in search_record_to_use.index:
                        search_value = search_record_to_use[filter_key]
                        is_search_nan = pd.isna(search_value) or search_value is None

                        # Skip filter if search value is NaN, except for specific fields
                        if is_search_nan and filter_key not in ['sexe', 'annee_naiss']:
                            continue

                        # Ensure the column exists in the dataframe being filtered (entree_df)
                        if filter_key not in results_df.columns:
                             st.warning(f"Impossible de filtrer par '{ALLOWED_COLUMNS[filter_key]}' car la colonne '{filter_key}' est manquante dans '{entree_filename}'.")
                             # Skip this filter, don't clear results
                             continue

                        match_criteria_used.append(f"{ALLOWED_COLUMNS[filter_key]} = '{search_value}'")

                        # Apply filter based on the key
                        try: # Add try-except for robustness during filtering
                            if filter_key == 'sexe':
                                if is_search_nan:
                                     results_df = results_df[results_df['sexe'].isna() | (results_df['sexe'] == '')]
                                else:
                                    # Ensure comparison is robust (e.g., handle case, string conversion)
                                    results_df = results_df[results_df['sexe'].astype(str).str.upper() == str(search_value).upper()]
                            elif filter_key == 'annee_naiss':
                                if is_search_nan:
                                     results_df = results_df[results_df['annee_naiss'].isna()]
                                else:
                                     # Convert both sides to numeric for safe comparison
                                     search_num = pd.to_numeric(search_value, errors='coerce')
                                     if pd.notna(search_num):
                                          results_df = results_df[pd.to_numeric(results_df['annee_naiss'], errors='coerce') == search_num]
                                     else: # If search value is not numeric, no match possible
                                          results_df = results_df.iloc[0:0]

                            elif filter_key == 'delta_days':
                                search_num = pd.to_numeric(search_value, errors='coerce')
                                if pd.notna(search_num):
                                    lower_bound = search_num - 2
                                    upper_bound = search_num + 2
                                    results_df = results_df[results_df['delta_days'].notna() &
                                                          (results_df['delta_days'] >= lower_bound) &
                                                          (results_df['delta_days'] <= upper_bound)]
                                else: results_df = results_df.iloc[0:0] # No match if search delta is invalid

                            elif filter_key in ['nb_rea', 'nb_si']:
                                 search_num = pd.to_numeric(search_value, errors='coerce')
                                 # Match if search value is >= 0 and entree value is >= 0 (as per original logic)
                                 if pd.notna(search_num) and search_num >= 0:
                                     df_num = pd.to_numeric(results_df[filter_key], errors='coerce')
                                     # Filter entree_df where the number is >= 0 (original logic seemed to just check existence)
                                     # Let's refine: Match EXACTLY if search_num is provided? Or >= search_num? Assuming exact for now.
                                     # results_df = results_df[df_num.notna() & (df_num >= 0)] # Old logic?
                                     results_df = results_df[df_num == search_num] # Match exact value
                                 else:
                                     results_df = results_df.iloc[0:0] # No match if search value invalid or negative

                            elif filter_key == 'age_gestationnel_weeks':
                                search_num = pd.to_numeric(search_value, errors='coerce')
                                if pd.notna(search_num):
                                     results_df = results_df[pd.to_numeric(results_df['age_gestationnel_weeks'], errors='coerce') == search_num]
                                else: results_df = results_df.iloc[0:0]

                            else: # Generic comparison (assume string for flexibility)
                                if not is_search_nan:
                                    results_df = results_df[results_df[filter_key].astype(str).str.strip().fillna('') == str(search_value).strip()]

                        except Exception as filter_ex:
                            st.error(f"Erreur lors de l'application du filtre '{ALLOWED_COLUMNS[filter_key]}': {filter_ex}")
                            results_df = entree_df.iloc[0:0] # Clear results on filter error
                            break # Stop filtering

                # Exclude the selected row itself - NO LONGER NEEDED as comparing across files
                # if selected_row_index_to_exclude is not None and selected_row_index_to_exclude in results_df.index:
                #      results_df = results_df.drop(index=selected_row_index_to_exclude)

                # Display Results
                st.subheader("Résultats Trouvés dans le Fichier Entree")
                if not results_df.empty:
                    st.info(f"{len(results_df)} enregistrement(s) correspondant(s) trouvé(s) dans '{entree_filename}' basé(s) sur : {', '.join(match_criteria_used)}")
                    # Display columns: ID + columns used in filter + maybe basic demographics?
                    display_cols_keys = [UPLOADED_ID_COL] + [key for key, active in active_filters.items() if active]
                    # Add sexe/annee_naiss if they exist in entree_df and weren't already added
                    if 'sexe' in entree_df.columns and 'sexe' not in display_cols_keys: display_cols_keys.append('sexe')
                    if 'annee_naiss' in entree_df.columns and 'annee_naiss' not in display_cols_keys: display_cols_keys.append('annee_naiss')
                    # Ensure columns actually exist in the results_df before trying to display
                    final_display_cols = [col for col in dict.fromkeys(display_cols_keys) if col in results_df.columns]
                    st.dataframe(results_df[final_display_cols])
                else:
                    st.warning(f"Aucune correspondance trouvée dans '{entree_filename}' pour les critères sélectionnés : {', '.join(match_criteria_used)}")
    else:
         st.info("Définissez un enregistrement de recherche en utilisant l'une des méthodes ci-dessus (basées sur le fichier à comparer).")

elif st.session_state.entree_data is None:
    st.warning("Veuillez téléverser le fichier 'Entree' de référence pour commencer.")
elif st.session_state.comparison_data is None:
    st.warning("Veuillez téléverser un fichier à comparer contre le fichier 'Entree'.")

# Error handling outside the main block (optional, as file loading has try-except)
# try:
#     # Main app logic could be wrapped if needed, but specific try-excepts are better
#     pass
# except Exception as e:
#     st.error(f"Une erreur générale est survenue dans l'application : {e}")
#     # Optionally reset state on major error
#     # st.session_state.entree_data = None
#     # st.session_state.comparison_data = None
#     # st.session_state.search_record = None
#     # st.session_state.selected_index_value = None

# Error handling outside the main block (optional, as file loading has try-except)
# try:
#     # Main app logic could be wrapped if needed, but specific try-excepts are better
#     pass
# except Exception as e:
#     st.error(f"Une erreur générale est survenue dans l'application : {e}")
#     # Optionally reset state on major error
#     # st.session_state.entree_data = None
#     # st.session_state.comparison_data = None
#     # st.session_state.search_record = None
#     # st.session_state.selected_index_value = None

# Error handling outside the main block (optional, as file loading has try-except)
# try:
#     # Main app logic could be wrapped if needed, but specific try-excepts are better
#     pass
# except Exception as e:
#     st.error(f"Une erreur générale est survenue dans l'application : {e}")
#     # Optionally reset state on major error
#     # st.session_state.entree_data = None
#     # st.session_state.comparison_data = None
#     # st.session_state.search_record = None
#     # st.session_state.selected_index_value = None

# Error handling outside the main block (optional, as file loading has try-except)
# try:
#     # Main app logic could be wrapped if needed, but specific try-excepts are better
#     pass
# except Exception as e:
#     st.error(f"Une erreur générale est survenue dans l'application : {e}")
#     # Optionally reset state on major error
#     # st.session_state.entree_data = None
#     # st.session_state.comparison_data = None
#     # st.session_state.search_record = None
#     # st.session_state.selected_index_value = None 