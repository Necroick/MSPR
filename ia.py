import pandas as pd
df = pd.read_csv('final_data.csv')

### TRAITEMENT DES DONNES SOCIO-ECONOMIQUES ###

# Récupération de l'année 2022
annee_recente = 22
annee_ancienne = 17
df_election_base = df[df['Année'] == annee_recente].copy()

# Récupération des variables à suivre
variables_a_suivre = [
    'Population',
    'Nb Faits Divers',
    'Taux de chômage',
    'Créations',
    'Inscrits',
    'Nb_Votant'
]

# Calcul des tendances pour les variables socio-économiques
nouvelles_features_diff = [] 
for var in variables_a_suivre:
    print(f"  Calcul pour la variable : {var}")
    data_var = df[['Code Commune', 'Année', var]].copy()
    data_var_grouped = data_var.groupby(['Code Commune', 'Année'])[var].sum().reset_index()

    # Pivoter la table pour avoir les années en colonnes
    pivot_var = data_var_grouped.pivot(index='Code Commune', columns='Année', values=var)

    # Calculer la différence entre l'année récente et l'année ancienne
    col_diff_name = f'{var}_Diff_{annee_recente}_{annee_ancienne}'
    nouvelles_features_diff.append(col_diff_name)
    
    # Vérifier que les colonnes années existent avant de soustraire
    if annee_recente in pivot_var.columns and annee_ancienne in pivot_var.columns:
         pivot_var[col_diff_name] = pivot_var[annee_recente] - pivot_var[annee_ancienne]
    else:
        nouvelles_features_diff.remove(col_diff_name)
        continue

    # Sélectionner uniquement la colonne de différence et l'identifiant
    diff_data_to_merge = pivot_var[[col_diff_name]].reset_index()

    # Fusionner avec le DataFrame de base
    # (Utiliser 'left' pour garder toutes les communes de 2022, même si elles manquaient en 2017)
    df_election_base = df_election_base.merge(diff_data_to_merge, on='Code Commune', how='left')
    print(f"    Colonne '{col_diff_name}' ajoutée.")

# Gérer les NaN introduits par le merge ou les calculs de différence
fill_values = {col: 0 for col in nouvelles_features_diff if col in df_election_base.columns}
df_election_base.fillna(value=fill_values, inplace=True)
