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
    "decennie_naiss": "Decennie de Naissance (calculée)",
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

# Columns to potentially display in the automatic match results
AUTO_MATCH_DISPLAY_COLS = [UPLOADED_ID_COL, 'decennie_naiss', 'sexe', 'ghm_prefix', 'delta_days']

# --- Helper Functions ---

def calculate_decennie_naiss(annee_series, age_series):
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
            # Initialize numeric columns with NA, others with None
            if col in ['annee', 'age', 'nb_rea', 'nb_si']:
                processed_df[col] = pd.NA
            elif col in ['entree_date', 'sortie_date']:
                 processed_df[col] = pd.NaT # Use NaT for datetime columns
            else:
                processed_df[col] = None # For string/object columns like ghm, sexe, etc.

    # --- Calculate Derived Columns ---
    # Using .get() for safety in case base columns were missing despite the check above
    processed_df['decennie_naiss'] = calculate_decennie_naiss(processed_df.get('annee'), processed_df.get('age'))
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
         # Let it proceed but ID search won't work.
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
    processed['decennie_naiss'] = annee - age if pd.notna(annee) and pd.notna(age) else pd.NA
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

# --- Function for Automatic Matching ---
@st.cache_data # Cache the result of this potentially expensive operation
def find_confident_matches(_comparison_df, _entree_df, _comparison_filename, _entree_filename):
    """
    Finds rows in comparison_df that match exactly one row in entree_df
    based on all available ALLOWED_COLUMNS.
    """
    st.info(f"Recherche de correspondances automatiques entre '{_comparison_filename}' et '{_entree_filename}'...")
    
    # Ensure ID columns exist in both dataframes
    if UPLOADED_ID_COL not in _comparison_df.columns or UPLOADED_ID_COL not in _entree_df.columns:
        st.warning(f"La colonne ID '{UPLOADED_ID_COL}' est nécessaire dans les deux fichiers pour la recherche automatique.")
        return pd.DataFrame() # Return empty dataframe

    confident_matches_list = []
    comparison_cols = _comparison_df.columns
    entree_cols = _entree_df.columns
    
    # Use only columns present in BOTH dataframes and in ALLOWED_COLUMNS
    match_cols = [col for col in ALLOWED_COLUMNS.keys() if col in comparison_cols and col in entree_cols]
    if not match_cols:
        st.warning("Aucune colonne commune trouvée parmi les colonnes autorisées pour le matching automatique.")
        return pd.DataFrame()

    # Pre-convert columns in entree_df for faster comparison inside the loop
    entree_numeric_cols = {}
    entree_string_cols = {}
    for col_key in match_cols:
        if col_key in ['decennie_naiss', 'age_gestationnel_weeks', 'nb_rea', 'nb_si', 'delta_days']:
             entree_numeric_cols[col_key] = pd.to_numeric(_entree_df[col_key], errors='coerce')
        elif col_key == 'sexe':
             entree_string_cols[col_key] = _entree_df[col_key].astype(str).str.upper().fillna('') # Handle NaN before upper
        else: # ghm_prefix, modes, prov, dest
             entree_string_cols[col_key] = _entree_df[col_key].astype(str).str.strip().fillna('')

    # Iterate through comparison dataframe rows
    for i, comp_row in _comparison_df.iterrows():
        # Start with all entree rows as potential matches
        current_filter = pd.Series(True, index=_entree_df.index)
        
        # Apply filters for each column
        for col_key in match_cols:
            comp_val = comp_row[col_key]

            # Skip filter if comparison value is NaN/None/NaT
            if pd.isna(comp_val):
                continue

            # Apply filter based on column type
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
                         current_filter &= pd.Series(False, index=_entree_df.index) # No match if comp_val invalid

                elif col_key in ['decennie_naiss', 'age_gestationnel_weeks', 'nb_rea', 'nb_si']:
                     comp_num = pd.to_numeric(comp_val, errors='coerce')
                     if pd.notna(comp_num):
                          entree_vals = entree_numeric_cols[col_key]
                          # Ensure exact match for these numeric fields
                          current_filter &= (entree_vals == comp_num)
                     else:
                          current_filter &= pd.Series(False, index=_entree_df.index)

                elif col_key == 'sexe':
                     # Case-insensitive comparison, handle NaN fill
                     comp_str = str(comp_val).upper()
                     current_filter &= (entree_string_cols[col_key] == comp_str)

                else: # Generic string comparison (ghm_prefix, modes, etc.)
                     comp_str = str(comp_val).strip()
                     current_filter &= (entree_string_cols[col_key] == comp_str)
                     
            except Exception as e:
                 st.error(f"Erreur lors de l'application du filtre automatique pour '{col_key}' sur la ligne {i} du fichier comparaison: {e}")
                 current_filter &= pd.Series(False, index=_entree_df.index) # Exclude on error
                 break # Stop processing this comp_row if a filter fails

        # Check if exactly one match was found
        entree_indices = _entree_df.index[current_filter]
        if len(entree_indices) == 1:
            entree_match_index = entree_indices[0]
            comp_id = comp_row[UPLOADED_ID_COL]
            entree_id = _entree_df.loc[entree_match_index, UPLOADED_ID_COL]

            match_data = {
                f"comparison_{UPLOADED_ID_COL}": comp_id,
                f"entree_{UPLOADED_ID_COL}": entree_id
            }
            # Add display columns from the comparison row for context
            for disp_col in AUTO_MATCH_DISPLAY_COLS:
                if disp_col in comp_row.index and disp_col != UPLOADED_ID_COL: # Avoid duplicating ID
                    match_data[f"comparison_{disp_col}"] = comp_row[disp_col]
                elif disp_col == UPLOADED_ID_COL:
                     continue # Already added
                else:
                     match_data[f"comparison_{disp_col}"] = pd.NA # Indicate if column missing

            confident_matches_list.append(match_data)

    st.info("Recherche de correspondances automatiques terminée.")
    if confident_matches_list:
        return pd.DataFrame(confident_matches_list)
    else:
        return pd.DataFrame() # Return empty dataframe if no matches

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
if 'auto_matches' not in st.session_state: # State for automatic matches
    st.session_state.auto_matches = None

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
                    st.session_state.auto_matches = None # Reset auto-matches if entree changes
                    st.session_state.search_record = None # Reset manual search
                    st.session_state.selected_id_value = None
                    st.success(f"Fichier Entree '{uploaded_entree_file.name}' traité.")
                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier Entree : {e}")
                    st.session_state.entree_data = None
                    st.session_state.entree_filename = None
                    st.session_state.auto_matches = None

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
                    st.session_state.search_record = None # Reset search record if comparison file changes
                    st.session_state.selected_id_value = None
                    st.session_state.auto_matches = None # Reset auto-matches if comparison changes
                    st.success(f"Fichier de comparaison '{uploaded_comparison_file.name}' traité.")
                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier de comparaison : {e}")
                    st.session_state.comparison_data = None
                    st.session_state.comparison_filename = None
                    st.session_state.auto_matches = None

# --- Automatic Matching Section ---
if st.session_state.entree_data is not None and st.session_state.comparison_data is not None:
    # Trigger auto-match calculation only if it hasn't been done yet for these files
    if st.session_state.auto_matches is None:
        st.session_state.auto_matches = find_confident_matches(
            st.session_state.comparison_data,
            st.session_state.entree_data,
            st.session_state.comparison_filename,
            st.session_state.entree_filename
        )

    # Display auto-match results
    if st.session_state.auto_matches is not None and not st.session_state.auto_matches.empty:
        st.header("Correspondances Automatiques Potentielles (Unique)")
        st.markdown(f"""
        Les lignes suivantes du fichier `{st.session_state.comparison_filename}` correspondent **exactement à une seule ligne**
        dans le fichier `{st.session_state.entree_filename}` en utilisant toutes les colonnes disponibles suivantes pour la comparaison :
        `{', '.join(list(ALLOWED_COLUMNS.values()))}`.
        *(La tolérance de ±2 jours est appliquée pour la durée de séjour)*.
        """)
        st.info(f"{len(st.session_state.auto_matches)} correspondance(s) automatique(s) unique(s) trouvée(s).")

        # Prepare display dataframe
        display_auto_matches = st.session_state.auto_matches.copy()
        # Rename columns for clarity
        rename_map = {f"comparison_{UPLOADED_ID_COL}": f"ID ({st.session_state.comparison_filename})",
                      f"entree_{UPLOADED_ID_COL}": f"ID ({st.session_state.entree_filename})"}
        for col in AUTO_MATCH_DISPLAY_COLS:
             if col != UPLOADED_ID_COL and f"comparison_{col}" in display_auto_matches.columns:
                  rename_map[f"comparison_{col}"] = f"{ALLOWED_COLUMNS.get(col, col)} (Comparer)"

        display_auto_matches.rename(columns=rename_map, inplace=True)
        
        # Select only the renamed columns for display
        final_auto_display_cols = list(rename_map.values())
        
        st.dataframe(display_auto_matches[final_auto_display_cols])
    elif st.session_state.auto_matches is not None: # Checked but empty
        st.info("Aucune correspondance automatique unique trouvée entre les deux fichiers en utilisant tous les critères.")

    st.divider() # Add a visual separator


# --- Main Application Logic (Requires both files) ---
if st.session_state.entree_data is not None and st.session_state.comparison_data is not None:
    entree_df = st.session_state.entree_data
    comparison_df = st.session_state.comparison_data
    entree_filename = st.session_state.entree_filename
    comparison_filename = st.session_state.comparison_filename

    st.header("2. Définir l'Enregistrement de Recherche (depuis le fichier à comparer)")
    search_method = st.radio("Comment voulez-vous définir l'enregistrement à rechercher ?",
                           ("Sélectionner une Ligne du Fichier à Comparer", "Saisir les Critères Manuellement"), horizontal=True,
                           key="search_method", help="Ceci définit la ligne 'source' pour laquelle vous chercherez des correspondances dans le fichier Entree.")

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
                # Display the dataframe for context (optional, can be large)
                # Consider using st.expander for the full dataframe display
                with st.expander(f"Voir les données du fichier '{comparison_filename}' pour sélectionner un ID"):
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
            manual_data['entree_date'] = st.date_input("Date d'Entrée", value=None, format="MM/DD/YYYY", key="man_edate") # Ensure format matches DATE_FORMAT expectation if needed elsewhere
            manual_data['entree_mode'] = st.text_input("Mode d'Entrée", key="man_emode")
            manual_data['entree_provenance'] = st.text_input("Provenance d'Entrée", key="man_eprov")
            manual_data['nb_rea'] = st.number_input("Nb Réa", value=None, min_value=0, step=1, key="man_rea")
        with col3:
            manual_data['sortie_date'] = st.date_input("Date de Sortie", value=None, format="MM/DD/YYYY", key="man_sdate") # Ensure format matches DATE_FORMAT expectation
            manual_data['sortie_mode'] = st.text_input("Mode de Sortie", key="man_smode")
            manual_data['sortie_destination'] = st.text_input("Destination de Sortie", key="man_sdest")
            manual_data['nb_si'] = st.number_input("Nb SI", value=None, min_value=0, step=1, key="man_si")

        for key, value in manual_data.items():
             if isinstance(value, str) and not value: manual_data[key] = None
             # Convert date inputs (which are datetime.date objects) to pandas Timestamps
             if key.endswith('_date') and value is not None: manual_data[key] = pd.to_datetime(value)

        current_search_record = preprocess_manual_record(manual_data)
        current_selected_id = None # No specific ID for manual entry

    # --- Store search record in session state if it changed ---
    # Update state if a record was found (either by selection or manual input)
    # Avoid overwriting if the user deselects the ID
    if current_search_record is not None:
        if not current_search_record.equals(st.session_state.search_record): # Check if it actually changed
             st.session_state.search_record = current_search_record
             st.session_state.selected_id_value = current_selected_id # Store the ID
    elif search_method == "Sélectionner une Ligne du Fichier à Comparer" and selected_id is None:
         # If using selection method and ID is deselected, clear the search record
         st.session_state.search_record = None
         st.session_state.selected_id_value = None


    # --- Display Selected Record & Filters ---
    search_record_to_use = st.session_state.search_record
    if search_record_to_use is not None:
        st.header("3. Enregistrement de Recherche & Filtres")
        st.subheader("Données de Base pour la Recherche (issues de la sélection/saisie) :")
        # Display the search record transposed for better readability
        st.dataframe(pd.DataFrame(search_record_to_use).T)

        st.subheader(f"Sélectionner les Critères pour Rechercher dans '{entree_filename}'")
        st.markdown("Cochez les cases des critères à utiliser pour filtrer le fichier Entree. Les filtres basés sur des valeurs vides ou manquantes dans l'enregistrement de recherche sont ignorés (sauf pour Sexe et Année Naissance qui peuvent matcher des vides).")

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
                 is_search_val_missing = pd.isna(search_val) or search_val is None

                 display_val = "(Vide/Manquant)" if is_search_val_missing else search_val
                 # Formatting for display
                 if col_key == 'delta_days' and not is_search_val_missing: display_val = f"{int(search_val)} jours (±2)"
                 elif col_key == 'decennie_naiss' and not is_search_val_missing: display_val = f"{int(search_val)}"
                 elif col_key == 'age_gestationnel_weeks' and not is_search_val_missing: display_val = f"{int(search_val)} semaines"

                 # Check if the corresponding column exists in the *entree_data* to enable the checkbox
                 is_col_in_entree = col_key in entree_df.columns
                 # Default checked only if value is present in search record AND column exists in entree file
                 # Exception: Sexe/Annee Naiss can be checked even if search_val is missing
                 default_checked = is_col_in_entree and (not is_search_val_missing or col_key in ['sexe', 'decennie_naiss'])


                 with cols[i % 3]:
                     is_active = st.checkbox(
                         f"{col_label}: `{display_val}`",
                         value=default_checked,
                         key=f"filt_{col_key}",
                         disabled=not is_col_in_entree # Disable if column missing in entree file
                     )
                     active_filters[col_key] = is_active
                     if not is_col_in_entree:
                         st.caption(f" (colonne manquante dans '{entree_filename}')") # Add caption if disabled
             i += 1

        # --- Perform Matching ---
        st.header("4. Résultats de la Correspondance Manuelle")
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

                        # Skip filter if search value is NaN, EXCEPT for specific fields where matching NaN is intended
                        if is_search_nan and filter_key not in ['sexe', 'decennie_naiss']:
                            continue

                        # Ensure the column exists in the dataframe being filtered (entree_df)
                        if filter_key not in results_df.columns:
                             st.warning(f"Impossible de filtrer par '{ALLOWED_COLUMNS[filter_key]}' car la colonne '{filter_key}' est manquante dans '{entree_filename}'.")
                             # Skip this filter, don't clear results
                             continue

                        # Add criterion description (handle NaN display)
                        search_display = "(Vide/Manquant)" if is_search_nan else search_value
                        match_criteria_used.append(f"{ALLOWED_COLUMNS[filter_key]} = '{search_display}'")

                        # Apply filter based on the key
                        try: # Add try-except for robustness during filtering
                            if filter_key == 'sexe':
                                entree_sexe_col = results_df['sexe'].astype(str).str.upper().fillna('') # Handle NaNs
                                if is_search_nan:
                                     results_df = results_df[entree_sexe_col == ''] # Match empty strings after fillna
                                else:
                                    search_str = str(search_value).upper()
                                    results_df = results_df[entree_sexe_col == search_str]
                            elif filter_key == 'decennie_naiss':
                                entree_annee_col = pd.to_numeric(results_df['decennie_naiss'], errors='coerce')
                                if is_search_nan:
                                     results_df = results_df[entree_annee_col.isna()]
                                else:
                                     search_num = pd.to_numeric(search_value, errors='coerce')
                                     if pd.notna(search_num):
                                          results_df = results_df[entree_annee_col == search_num]
                                     else: # If search value is not numeric, no match possible
                                          results_df = results_df.iloc[0:0]

                            elif filter_key == 'delta_days':
                                # This filter should only be active if search_value is not NaN (handled above)
                                search_num = pd.to_numeric(search_value, errors='coerce')
                                if pd.notna(search_num):
                                    lower_bound = search_num - 2
                                    upper_bound = search_num + 2
                                    entree_delta_col = pd.to_numeric(results_df['delta_days'], errors='coerce')
                                    results_df = results_df[entree_delta_col.notna() &
                                                          (entree_delta_col >= lower_bound) &
                                                          (entree_delta_col <= upper_bound)]
                                else: results_df = results_df.iloc[0:0] # Should not happen if Nan check works

                            elif filter_key in ['nb_rea', 'nb_si']:
                                 # This filter should only be active if search_value is not NaN (handled above)
                                 search_num = pd.to_numeric(search_value, errors='coerce')
                                 if pd.notna(search_num): # Allow search for 0
                                     entree_num_col = pd.to_numeric(results_df[filter_key], errors='coerce')
                                     # Match exact value provided in the search record
                                     results_df = results_df[entree_num_col == search_num]
                                 else:
                                     results_df = results_df.iloc[0:0] # No match if search value invalid

                            elif filter_key == 'age_gestationnel_weeks':
                                # This filter should only be active if search_value is not NaN (handled above)
                                search_num = pd.to_numeric(search_value, errors='coerce')
                                if pd.notna(search_num):
                                     entree_weeks_col = pd.to_numeric(results_df['age_gestationnel_weeks'], errors='coerce')
                                     results_df = results_df[entree_weeks_col == search_num]
                                else: results_df = results_df.iloc[0:0]

                            else: # Generic comparison (assume string for flexibility)
                                # This filter should only be active if search_value is not NaN (handled above)
                                search_str = str(search_value).strip()
                                entree_str_col = results_df[filter_key].astype(str).str.strip().fillna('')
                                results_df = results_df[entree_str_col == search_str]

                        except Exception as filter_ex:
                            st.error(f"Erreur lors de l'application du filtre '{ALLOWED_COLUMNS[filter_key]}': {filter_ex}")
                            results_df = entree_df.iloc[0:0] # Clear results on filter error
                            break # Stop filtering

                # Exclude the selected row itself - NO LONGER NEEDED as comparing across files

                # Display Results
                st.subheader("Résultats Trouvés dans le Fichier Entree")
                if not results_df.empty:
                    st.info(f"{len(results_df)} enregistrement(s) correspondant(s) trouvé(s) dans '{entree_filename}' basé(s) sur : {', '.join(match_criteria_used)}")
                    # Display columns: ID + columns used in filter + maybe basic demographics?
                    display_cols_keys = [UPLOADED_ID_COL] + [key for key, active in active_filters.items() if active and key in entree_df.columns] # Ensure used keys exist
                    # Add sexe/decennie_naiss if they exist in entree_df and weren't already added by filter selection
                    if 'sexe' in entree_df.columns and 'sexe' not in display_cols_keys: display_cols_keys.append('sexe')
                    if 'decennie_naiss' in entree_df.columns and 'decennie_naiss' not in display_cols_keys: display_cols_keys.append('decennie_naiss')
                    # Ensure columns actually exist in the results_df before trying to display
                    final_display_cols = [col for col in dict.fromkeys(display_cols_keys) if col in results_df.columns]
                    st.dataframe(results_df[final_display_cols])
                else:
                    st.warning(f"Aucune correspondance trouvée dans '{entree_filename}' pour les critères sélectionnés : {', '.join(match_criteria_used)}")
    else:
         st.info("Définissez un enregistrement de recherche en utilisant l'une des méthodes ci-dessus (basées sur le fichier à comparer) pour activer la recherche manuelle.")

elif st.session_state.entree_data is None:
    st.warning("Veuillez téléverser le fichier 'Entree' de référence pour commencer.")
elif st.session_state.comparison_data is None:
    st.warning("Veuillez téléverser un fichier à comparer contre le fichier 'Entree'.")

# --- Footer or final messages ---
st.markdown("---")
st.caption("Application de comparaison de fichiers CSV") 