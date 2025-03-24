import pandas as pd
import numpy as np

# Lecture du fichier CSV avec le bon séparateur et en gérant les valeurs manquantes
# Ajout de low_memory=False pour éviter l'avertissement sur les types mixtes
df = pd.read_csv('not_cleaned_security_data.csv', sep=';', na_values=['NA'], low_memory=False)

# Conversion des colonnes numériques
numeric_columns = ['tauxpourmille', 'complementinfoval', 'complementinfotaux', 'POP', 'millPOP', 'LOG', 'millLOG']
for col in numeric_columns:
    # Vérification si la colonne est de type object (string)
    if df[col].dtype == 'object':
        # Remplacement des virgules par des points seulement si c'est une chaîne de caractères
        df[col] = df[col].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
    # Conversion en float avec gestion des erreurs
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Conversion de la colonne 'faits' en numérique
df['faits'] = pd.to_numeric(df['faits'], errors='coerce')

# Nettoyage des codes géographiques et années
df['CODGEO_2024'] = df['CODGEO_2024'].astype(str)
df['annee'] = '20' + df['annee'].astype(str)  # Conversion en année complète (16 -> 2016)

# Création d'un DataFrame nettoyé
df_clean = df.copy()

# Remplacement des valeurs manquantes par 0 pour les colonnes de faits et taux
df_clean['faits'] = df_clean['faits'].fillna(0)
df_clean['tauxpourmille'] = df_clean['tauxpourmille'].fillna(0)

# Création de statistiques descriptives par type de crime
stats_by_crime = df_clean.groupby('classe').agg({
    'faits': ['count', 'sum', 'mean', 'std'],
    'tauxpourmille': ['mean', 'std']
}).round(2)

# Sauvegarde des données nettoyées
df_clean.to_csv('cleaned_security_data.csv', sep=';', index=False)

# Sauvegarde des statistiques
stats_by_crime.to_csv('stats_by_crime.csv', sep=';')

# Affichage des statistiques
print("\nStatistiques générales :")
print(f"Nombre total d'enregistrements : {len(df_clean)}")
print(f"Nombre de communes uniques : {df_clean['CODGEO_2024'].nunique()}")
print(f"Années couvertes : {df_clean['annee'].unique()}")
print("\nLes fichiers nettoyés ont été sauvegardés :")
print("- cleaned_security_data.csv : données nettoyées")
print("- stats_by_crime.csv : statistiques par type de crime")