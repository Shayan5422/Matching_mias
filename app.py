import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- Configuration ---
# No longer assumes a fixed entree file
ENCODING = 'utf-8'
DATE_FORMAT = '%m/%d/%Y' # Adjust if needed, or set to None
# !!! IMPORTANT: Set the name of the ID column in the files users will upload !!!
UPLOADED_ID_COL = 'id'

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

# --- Helper Functions --- (Mostly unchanged, but adapted for single record/dataframe)

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
@st.cache_data # Cache the processed dataframe based on the uploaded file
def preprocess_uploaded_df(df):
    """Adds calculated/derived columns needed for matching to the DataFrame."""
    processed_df = df.copy()

    # --- Ensure Base Columns Exist (add if missing, fill with NaN/None) ---
    # Add more columns here if they are essential for filtering but might be missing
    base_cols_needed = ['annee', 'age', 'ghm', 'entree_date', 'sortie_date', 'age_gestationnel', 
                          'sexe', 'entree_mode', 'entree_provenance', 'sortie_mode', 
                          'sortie_destination', 'nb_rea', 'nb_si', UPLOADED_ID_COL]
    for col in base_cols_needed:
        if col not in processed_df.columns:
            st.warning(f"Fichier téléversé manquant la colonne attendue : '{col}'. Elle sera ajoutée avec des valeurs vides.")
            processed_df[col] = pd.NA if col in ['annee', 'age', 'nb_rea', 'nb_si'] else None

    # --- Calculate Derived Columns ---
    processed_df['annee_naiss'] = calculate_annee_naiss(processed_df['annee'], processed_df['age'])
    processed_df['ghm_prefix'] = processed_df['ghm'].astype(str).str[:3].replace('nan', None)
    processed_df['age_gestationnel_weeks'] = extract_gestational_weeks(processed_df['age_gestationnel'])
    
    # Convert dates before calculating delta
    processed_df['entree_date'] = pd.to_datetime(processed_df['entree_date'], format=DATE_FORMAT, errors='coerce')
    processed_df['sortie_date'] = pd.to_datetime(processed_df['sortie_date'], format=DATE_FORMAT, errors='coerce')
    processed_df['delta_days'] = calculate_delta_days(processed_df['entree_date'], processed_df['sortie_date'])

    # Ensure ID column is string
    processed_df[UPLOADED_ID_COL] = processed_df[UPLOADED_ID_COL].astype(str)
    
    # Ensure all columns used for filtering exist, even if derived as NA
    for key in ALLOWED_COLUMNS.keys():
         if key not in processed_df.columns:
             processed_df[key] = pd.NA # Add derived cols if somehow missing
             
    return processed_df

# --- Function to preprocess a single manual input record ---
def preprocess_manual_record(record_dict):
    """Converts dict from manual input into a processed Series with derived fields."""
    # Create Series ensure all expected keys exist, use object dtype
    manual_series = pd.Series(record_dict, dtype=object)
    
    # Ensure base columns exist for calculation
    base_cols = ['annee', 'age', 'ghm', 'entree_date', 'sortie_date', 'age_gestationnel']
    for col in base_cols:
        if col not in manual_series.index:
             manual_series[col] = None
             
    # Convert dates first if they exist
    manual_series['entree_date'] = pd.to_datetime(manual_series.get('entree_date'), errors='coerce')
    manual_series['sortie_date'] = pd.to_datetime(manual_series.get('sortie_date'), errors='coerce')

    # Calculate derived fields
    # Initialize processed Series with combined index and object dtype
    combined_index = list(ALLOWED_COLUMNS.keys()) + list(record_dict.keys())
    processed = pd.Series(index=list(dict.fromkeys(combined_index)), dtype=object) # Use dict.fromkeys to remove duplicates and keep order
    processed.update(manual_series) # Add original values
    
    annee = pd.to_numeric(processed.get('annee'), errors='coerce')
    age = pd.to_numeric(processed.get('age'), errors='coerce')
    processed['annee_naiss'] = annee - age if pd.notna(annee) and pd.notna(age) else pd.NA

    processed['ghm_prefix'] = str(processed['ghm'])[:3] if pd.notna(processed['ghm']) else None
    
    # Ensure extract_gestational_weeks handles single value correctly or adjust call
    age_gest_val = processed.get('age_gestationnel')
    # Pass the single value directly instead of creating a single-element Series
    processed['age_gestationnel_weeks'] = extract_gestational_weeks_scalar(age_gest_val)

    if pd.notna(processed['entree_date']) and pd.notna(processed['sortie_date']):
        processed['delta_days'] = (processed['sortie_date'] - processed['entree_date']).days
    else:
        processed['delta_days'] = pd.NA
        
    # Ensure all allowed columns exist in the final series
    for key in ALLOWED_COLUMNS.keys():
        if key not in processed.index:
            processed[key] = pd.NA

    return processed

# --- Add a scalar version of extract_gestational_weeks --- 
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

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Rechercher des Enregistrements Correspondants dans le Fichier Téléversé")

# --- File Upload --- 
st.header("1. Téléverser le Fichier CSV")
st.markdown("Téléversez le fichier CSV dans lequel vous souhaitez rechercher.")
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

# Initialize session state for processed data and search record
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'search_record' not in st.session_state:
    st.session_state.search_record = None
if 'selected_index_value' not in st.session_state:
     st.session_state.selected_index_value = None

if uploaded_file is not None:
    try:
        # Check if the uploaded file is new or the same as before
        # This simple check helps avoid reprocessing the same file unnecessarily
        # More robust checks might involve hashing the file content
        if st.session_state.processed_data is None or uploaded_file.name != st.session_state.get('uploaded_filename'):
            with st.spinner("Traitement du fichier téléversé..."):
                 st.session_state.uploaded_filename = uploaded_file.name
                 sortie_df_uploaded = pd.read_csv(uploaded_file, encoding=ENCODING, low_memory=False)
                 st.session_state.processed_data = preprocess_uploaded_df(sortie_df_uploaded)
                 st.session_state.search_record = None # Reset search record if file changes
                 st.session_state.selected_index_value = None
            st.success(f"Fichier '{uploaded_file.name}' traité avec succès !")
        
        processed_df = st.session_state.processed_data

        if processed_df is None or processed_df.empty:
             st.error("Échec du traitement du fichier téléversé ou le fichier est vide.")
             st.stop()

        st.header("2. Définir l'Enregistrement de Recherche")
        search_method = st.radio("Comment voulez-vous définir l'enregistrement à rechercher ?",
                               ("Sélectionner une Ligne du Fichier Téléversé", "Saisir les Critères Manuellement"), horizontal=True,
                               key="search_method") # Use key to prevent state issues on rerun

        current_search_record = None
        current_selected_index = None

        if search_method == "Sélectionner une Ligne du Fichier Téléversé":
            st.subheader("2a. Sélectionner l'Index de la Ligne")
            st.dataframe(processed_df) # Show processed data for selection
            max_index = len(processed_df) - 1
            if max_index >= 0:
                selected_index = st.number_input(f"Entrez l'index de la ligne à utiliser comme base de recherche (0 à {max_index})",
                                                  min_value=0, max_value=max_index, step=1, key="row_selector")
                current_search_record = processed_df.iloc[selected_index]
                current_selected_index = selected_index # Store the index
            else:
                st.warning("Le fichier traité ne contient aucune ligne de données à sélectionner.")

        elif search_method == "Saisir les Critères Manuellement":
            st.subheader("2a. Saisir les Critères de Recherche")
            st.markdown("Entrez les détails de l'enregistrement que vous souhaitez faire correspondre. **Les champs peuvent être laissés vides.**")
            manual_data = {}
            col1, col2, col3 = st.columns(3)
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
            
            # Clean blank strings
            for key, value in manual_data.items():
                 if isinstance(value, str) and not value: manual_data[key] = None
                 # Convert date objects to datetime for preprocessing
                 if key.endswith('_date') and value is not None: manual_data[key] = pd.to_datetime(value)
            
            current_search_record = preprocess_manual_record(manual_data)
            current_selected_index = None # No specific index for manual entry

        # --- Store search record in session state if it changed ---
        # Comparing Series can be tricky, maybe just update if method changes or index changes?
        st.session_state.search_record = current_search_record
        st.session_state.selected_index_value = current_selected_index
        
        # --- Display Selected Record & Filters --- 
        search_record_to_use = st.session_state.search_record
        if search_record_to_use is not None:
            st.header("3. Enregistrement de Recherche & Filtres")
            st.subheader("Données de Base pour la Recherche :")
            st.dataframe(pd.DataFrame(search_record_to_use).T)

            st.subheader("Sélectionner les Critères de Correspondance")
            st.markdown("Cochez les cases des critères à utiliser. Les filtres basés sur des valeurs vides dans l'enregistrement de recherche sont ignorés (sauf Sexe/Année de Naissance).")

            active_filters = {}
            cols = st.columns(3)
            i = 0
            for col_key, col_label in ALLOWED_COLUMNS.items():
                if col_key not in search_record_to_use.index:
                     with cols[i % 3]: st.checkbox(f"{col_label} (N/A)", value=False, disabled=True, key=f"filt_{col_key}")
                else:
                     sortie_val = search_record_to_use[col_key]
                     display_val = "(Vide)" if pd.isna(sortie_val) or sortie_val is None else sortie_val
                     if col_key == 'delta_days' and pd.notna(sortie_val): display_val = f"{int(sortie_val)} jours"
                     elif col_key == 'annee_naiss' and pd.notna(sortie_val): display_val = f"{int(sortie_val)}"
                     elif col_key == 'age_gestationnel_weeks' and pd.notna(sortie_val): display_val = f"{int(sortie_val)} semaines"
                     
                     default_checked = pd.notna(sortie_val) and sortie_val is not None
                     with cols[i % 3]:
                         active_filters[col_key] = st.checkbox(f"{col_label}: `{display_val}`", value=default_checked, key=f"filt_{col_key}")
                i += 1

            # --- Perform Matching --- 
            st.header("4. Résultats de la Correspondance")
            if st.button("Rechercher les Correspondances dans le Fichier Téléversé"):
                with st.spinner("Recherche des correspondances..."):
                    # Start with the full processed dataframe
                    results_df = processed_df.copy()
                    match_criteria_used = []
                    selected_row_index_to_exclude = st.session_state.selected_index_value

                    # Apply selected filters sequentially
                    for filter_key, is_active in active_filters.items():
                        if is_active and filter_key in search_record_to_use.index:
                            search_value = search_record_to_use[filter_key]
                            is_search_nan = pd.isna(search_value) or search_value is None
                            
                            # Skip filter if search value is NaN, except for mandatory fields handled below
                            if is_search_nan and filter_key not in ['sexe', 'annee_naiss']:
                                continue
                                
                            match_criteria_used.append(ALLOWED_COLUMNS[filter_key])
                            
                            # Ensure the column exists in the dataframe being filtered
                            if filter_key not in results_df.columns:
                                 st.warning(f"Impossible de filtrer par '{ALLOWED_COLUMNS[filter_key]}' car la colonne '{filter_key}' est manquante dans les données.")
                                 results_df = results_df.iloc[0:0] # No results possible if filter col missing
                                 break
                                 
                            # Apply filter based on the key
                            if filter_key == 'sexe':
                                if is_search_nan:
                                     results_df = results_df[results_df['sexe'].isna() | (results_df['sexe'] == '')]
                                else:
                                    results_df = results_df[results_df['sexe'] == search_value]
                            elif filter_key == 'annee_naiss':
                                if is_search_nan:
                                     results_df = results_df[results_df['annee_naiss'].isna()]
                                else:
                                    results_df = results_df[pd.to_numeric(results_df['annee_naiss'], errors='coerce') == float(search_value)]
                            elif filter_key == 'delta_days':
                                if not is_search_nan:
                                    lower_bound = search_value - 2
                                    upper_bound = search_value + 2
                                    results_df = results_df[results_df['delta_days'].notna() & 
                                                          (results_df['delta_days'] >= lower_bound) & 
                                                          (results_df['delta_days'] <= upper_bound)]
                            elif filter_key in ['nb_rea', 'nb_si']:
                                if not is_search_nan:
                                     search_num = pd.to_numeric(search_value, errors='coerce')
                                     if pd.notna(search_num) and search_num >= 0:
                                         df_num = pd.to_numeric(results_df[filter_key], errors='coerce')
                                         results_df = results_df[df_num.notna() & (df_num >= 0)]
                                     else:
                                         results_df = results_df.iloc[0:0]
                            elif filter_key == 'age_gestationnel_weeks':
                                if not is_search_nan:
                                    results_df = results_df[pd.to_numeric(results_df['age_gestationnel_weeks'], errors='coerce') == float(search_value)]
                            else: # Generic string comparison
                                if not is_search_nan:
                                    results_df = results_df[results_df[filter_key].astype(str) == str(search_value)]
                    
                    # Exclude the selected row itself if search was based on selection
                    if selected_row_index_to_exclude is not None and selected_row_index_to_exclude in results_df.index:
                         results_df = results_df.drop(index=selected_row_index_to_exclude)

                    # Display Results
                    st.subheader("Résultats")
                    if not results_df.empty:
                        st.info(f"{len(results_df)} enregistrement(s) correspondant(s) trouvé(s) dans '{uploaded_file.name}' basé(s) sur les critères : {', '.join(match_criteria_used)}")
                        display_cols = [UPLOADED_ID_COL] + [key for key, active in active_filters.items() if active and key in results_df.columns]
                        if 'sexe' in results_df.columns and 'sexe' not in display_cols: display_cols.append('sexe')
                        if 'annee_naiss' in results_df.columns and 'annee_naiss' not in display_cols: display_cols.append('annee_naiss')
                        st.dataframe(results_df[list(dict.fromkeys(display_cols))])
                    else:
                        st.warning(f"Aucune correspondance trouvée dans '{uploaded_file.name}' pour les critères sélectionnés : {', '.join(match_criteria_used)}")
        else:
             st.info("Définissez un enregistrement de recherche en utilisant l'une des méthodes ci-dessus.")

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
        # Optionally reset state on error
        # st.session_state.processed_data = None
        # st.session_state.search_record = None
        # st.session_state.selected_index_value = None

else:
    st.info("Téléversez un fichier CSV pour commencer le processus de recherche.") 