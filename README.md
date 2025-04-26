# Application de Recherche de Correspondances dans un Fichier CSV

Cette application Streamlit permet de rechercher des enregistrements similaires au sein d'un même fichier CSV téléversé. L'utilisateur peut définir l'enregistrement de référence soit en sélectionnant une ligne dans le fichier téléversé, soit en saisissant manuellement les critères.

## Fonctionnalités

*   Téléversement d'un fichier CSV comme source de données.
*   Calcul automatique de champs dérivés pour la comparaison (Année de naissance, Préfixe GHM, Durée de séjour, Âge gestationnel en semaines).
*   Deux méthodes pour définir l'enregistrement de recherche :
    *   Sélectionner une ligne existante dans le fichier téléversé.
    *   Saisir manuellement les valeurs pour les critères souhaités.
*   Sélection interactive des critères à utiliser pour la recherche de correspondances.
*   Application des règles de correspondance spécifiques :
    *   Comparaison de l'année de naissance (Année - Âge).
    *   Comparaison des 3 premiers caractères du GHM.
    *   Comparaison exacte pour le Sexe, les Modes/Provenances/Destinations d'Entrée/Sortie.
    *   Vérification que Nb Réa et Nb SI sont ≥ 0 (si une valeur est fournie).
    *   Comparaison de la durée de séjour (Date de Sortie - Date d'Entrée) avec une tolérance de ± 2 jours.
    *   Comparaison de l'âge gestationnel en semaines entières.
    *   Les filtres basés sur des valeurs vides (non renseignées) dans l'enregistrement de recherche sont ignorés (sauf pour Sexe et Année de Naissance qui nécessitent une correspondance exacte, même pour les valeurs vides).
*   Affichage des enregistrements correspondants trouvés dans le fichier téléversé (en excluant la ligne source si elle a été sélectionnée).

## Prérequis

*   Python 3.7+
*   Pip (gestionnaire de paquets Python)

## Installation

1.  **Clonez ou téléchargez le projet :**
    Si vous utilisez git :
    ```bash
    git clone <url_du_repository>
    cd <nom_du_dossier>
    ```
    Sinon, téléchargez et décompressez les fichiers (`app.py`, `requirements.txt`).

2.  **Installez les dépendances :**
    Ouvrez un terminal ou une invite de commande dans le dossier du projet et exécutez :
    ```bash
    pip install -r requirements.txt
    ```

## Configuration (Optionnelle)

*   **Nom de la Colonne ID :** Ouvrez le fichier `app.py` et modifiez la ligne suivante si la colonne identifiant unique dans vos fichiers CSV ne s'appelle pas `id` :
    ```python
    UPLOADED_ID_COL = 'votre_nom_de_colonne_id'
    ```
*   **Format des Dates :** Si vos dates ne sont pas au format `MM/DD/YYYY`, ajustez la variable `DATE_FORMAT` dans `app.py`.
    ```python
    DATE_FORMAT = '%d/%m/%Y' # Exemple pour Jour/Mois/Année
    ```
*   **Encodage :** Si vos fichiers CSV utilisent un encodage différent de UTF-8, modifiez la variable `ENCODING` :
    ```python
    ENCODING = 'latin1' # Exemple
    ```

## Exécution

1.  Assurez-vous d'être dans le dossier du projet dans votre terminal.
2.  Lancez l'application Streamlit avec la commande :
    ```bash
    streamlit run app.py
    ```
3.  L'application devrait s'ouvrir automatiquement dans votre navigateur web.

## Utilisation

1.  **Téléverser le Fichier CSV :** Cliquez sur "Choisir un fichier CSV" et sélectionnez le fichier que vous souhaitez analyser.
2.  **Définir l'Enregistrement de Recherche :**
    *   Choisissez "Sélectionner une Ligne du Fichier Téléversé" pour voir le contenu du fichier et entrer l'index (numéro de ligne commençant à 0) de l'enregistrement qui servira de base à la recherche.
    *   Choisissez "Saisir les Critères Manuellement" pour entrer directement les valeurs des champs que vous souhaitez utiliser pour la recherche. Laissez les champs vides si la valeur est inconnue.
3.  **Sélectionner les Critères de Correspondance :**
    *   Une fois l'enregistrement de recherche défini (par sélection ou saisie), une section "Sélectionner les Critères de Correspondance" apparaît.
    *   Les cases à cocher correspondent aux critères de recherche disponibles. Par défaut, seules les cases correspondant aux champs *non vides* de votre enregistrement de recherche sont cochées.
    *   Cochez ou décochez les cases pour activer/désactiver les filtres correspondants.
4.  **Lancer la Recherche :** Cliquez sur le bouton "Rechercher les Correspondances dans le Fichier Téléversé".
5.  **Consulter les Résultats :** Les enregistrements du fichier téléversé qui correspondent aux critères sélectionnés s'affichent dans la section "Résultats".

## Format Attendu du Fichier CSV

L'application s'attend à trouver certaines colonnes pour pouvoir effectuer tous les calculs et comparaisons. Si une colonne attendue est manquante, un avertissement sera affiché et la colonne sera ajoutée avec des valeurs vides.

Colonnes importantes :

*   La colonne identifiant (configurée par `UPLOADED_ID_COL`, par défaut `id`).
*   `annee`, `age` (pour calculer l'année de naissance).
*   `sexe`.
*   `ghm` (pour extraire le préfixe).
*   `entree_date`, `sortie_date` (pour calculer la durée de séjour).
*   `age_gestationnel` (pour extraire les semaines).
*   `entree_mode`, `entree_provenance`, `sortie_mode`, `sortie_destination`.
*   `nb_rea`, `nb_si`.

Assurez-vous que les formats de date correspondent à `DATE_FORMAT` si vous le modifiez. 