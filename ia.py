import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

df = pd.read_csv('final_data.csv')

### TRAITEMENT DES DONNES SOCIO-ECONOMIQUES ###

# Récupération de l'année 2022
annee_recente = 22
annee_ancienne = 17
df_election_base = df[df['Année'] == annee_recente].copy()

# Récupération des variables à suivre
variables_pour_calcul_diff = [
    'Population', 'Nb Faits Divers', 'Taux de chômage',
    'Créations', 'Inscrits', 'Nb_Votant'
]
variables_communales_a_traiter = [
    'Population', 'Nb Faits Divers',
    'Inscrits', 'Nb_Votant'
]
variables_nationales_regionales = ['Taux de chômage', 'Créations']

# Calcul des tendances pour les variables socio-économiques
nouvelles_features_diff_all = {}
for var in variables_pour_calcul_diff:
    data_var = df[['Code Commune', 'Année', var]].copy()
    data_var_grouped = data_var.groupby(['Code Commune', 'Année'])[var].sum().reset_index()
    col_diff_name = f'{var}_Diff_{annee_recente}_{annee_ancienne}'

    # Pivoter la table pour avoir les années en colonnes
    pivot_var = data_var_grouped.pivot(index='Code Commune', columns='Année', values=var)

    # Calculer la différence entre l'année récente et l'année ancienne
    if annee_recente in pivot_var.columns and annee_ancienne in pivot_var.columns:
        pivot_var[col_diff_name] = pivot_var[annee_recente] - pivot_var[annee_ancienne]
        diff_data_to_merge = pivot_var[[col_diff_name]].reset_index()
        df_election_base = df_election_base.merge(diff_data_to_merge, on='Code Commune', how='left')
        nouvelles_features_diff_all[var] = col_diff_name # Garder trace du nom

# Gérer les NaN introduits par le merge ou les calculs de différence
fill_values_diff = {name: 0 for name in nouvelles_features_diff_all.values() if name in df_election_base.columns}
df_election_base.fillna(value=fill_values_diff, inplace=True)

# Traitement des données catégorielles
df_election_base['Région'] = df_election_base['Région'].astype(str)
df_election_base['Code Departement'] = df_election_base['Code Departement'].astype(str)

### CREATION DES ENSEMBLES D'ENTRAINEMENT ###

# Encodage de la cible (y)
label_encoder = LabelEncoder()
df_election_base = df_election_base.dropna(subset=['Nom_Prenom_Gagnant'])
y = label_encoder.fit_transform(df_election_base['Nom_Prenom_Gagnant'])
# On garde le mapping de côté pour l'interprétation future
gagnant_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Définition des variables pour l'encodage des features
cols_communales_num_impute_scale = variables_communales_a_traiter + [
    nouvelles_features_diff_all[var] for var in variables_communales_a_traiter if var in nouvelles_features_diff_all
]
cols_diff_national_passthrough = [
    nouvelles_features_diff_all[var] for var in variables_nationales_regionales if var in nouvelles_features_diff_all
]
categorical_features = ['Région', 'Code Departement']
features_for_X = cols_communales_num_impute_scale + cols_diff_national_passthrough + categorical_features
features_for_X = [col for col in features_for_X if col in df_election_base.columns]

# Définition des features (x)
X = df_election_base[features_for_X].copy()

# Creation de la pipeline numeric
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Identifier les colonnes qui seront effectivement traitées par chaque transformateur
cols_num_in_X = [col for col in cols_communales_num_impute_scale if col in X.columns]
cols_cat_in_X = [col for col in categorical_features if col in X.columns]

# Identifier les colonnes restantes (celles qui étaient en passthrough)
processed_cols = cols_num_in_X + cols_cat_in_X
remainder_cols = [col for col in features_for_X if col not in processed_cols]

# Définition du processus d'encodage de X - en imputant aussi le reste
preprocessor = ColumnTransformer(
    transformers=[
        ('num_communal', numeric_pipeline, cols_num_in_X),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cols_cat_in_X),
        ('remainder_impute', SimpleImputer(strategy='median'), remainder_cols)
    ],
    remainder='drop'
)


# Séparer les données en ensemble d'entrainement et leur appliqué l'encodage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Application du préprocesseur...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("Préprocessing terminé.")

### ENTRAINEMENT DU MODELE - Random Forest ###

# Définition du modèle - Random Forest
model = RandomForestClassifier(n_estimators=125,
                               random_state=42,
                               class_weight='balanced',
                               n_jobs=-1)

# Entrainement du modèle - Random Forest
print("Entraînement du modèle Random Forest...")
model.fit(X_train_processed, y_train)
print("Entraînement terminé.")

# Test du model et calcul de l'accuracy - Random Forest
print("Prédiction et évaluation du modèle RandomForest...")
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: (Random Forest): {accuracy:.4f}")

# Définition du modèle - Gradient Boosting
model_gb = GradientBoostingClassifier(n_estimators=100,
                                     learning_rate=0.1,
                                     max_depth=3,
                                     random_state=42)

# Entrainement du modèle - Gradient Boosting
print("Entraînement Gradient Boosting...")
model_gb.fit(X_train_processed, y_train)
print("Entraînement terminé.")

# Test du model et calcul de l'accuracy - Gradient Boosting
print("Prédiction et évaluation (Gradient Boosting)...")
y_pred_gb = model_gb.predict(X_test_processed)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Accuracy (Gradient Boosting): {accuracy_gb:.4f}")

# Définition du modèle - KMeans
n_cluster = 250
model_kmeans = KMeans(n_clusters=n_cluster,
                      random_state=42,
                      n_init=10)

# Entrainement du modèle - KMeans
print("Entraînement K-Means (fit sur X_train_processed)...")
model_kmeans.fit(X_train_processed)
print("K-Means entraîné.")

cluster_labels_train = model_kmeans.labels_
cluster_winner_map = {}
for i in range(n_cluster):
    indices_in_cluster = np.where(cluster_labels_train == i)[0]
    if len(indices_in_cluster) > 0:
        winners_in_cluster = y_train[indices_in_cluster]
        most_common_winner_encoded = Counter(winners_in_cluster).most_common(1)[0][0]
        cluster_winner_map[i] = most_common_winner_encoded
        winner_name = label_encoder.inverse_transform([most_common_winner_encoded])[0]
        print(f"  Cluster {i}: Gagnant majoritaire = {winner_name} ({len(indices_in_cluster)} communes)")
    else:
        print(f"  Cluster {i}: Vide (aucune commune d'entraînement)")
        most_common_global = Counter(y_train).most_common(1)[0][0]
        cluster_winner_map[i] = most_common_global

# Test du model et calcul de l'accuracy - KMeans
print("Prédiction et évaluation (KMeans)...")
cluster_labels_test = model_kmeans.predict(X_test_processed)
y_pred_kmeans_mapped = np.array([cluster_winner_map.get(label, -1) for label in cluster_labels_test]) # -1 si label inconnu
valid_preds_mask = y_pred_kmeans_mapped != -1
accuracy_kmeans = accuracy_score(y_test[valid_preds_mask], y_pred_kmeans_mapped[valid_preds_mask])
print(f"Accuracy (K-Means via mapping): {accuracy_kmeans:.4f}")

### EXTRAPOLATION DES DONNEES POUR 2027 EN AUVERGNE RHONES ALPES ###

# Identification des communes de la région cible
annee_prediction = 27
target_region = 'Auvergne-Rhône-Alpes'
df_aura_2022 = df_election_base[df_election_base['Région'] == target_region].copy()
print(f"Prédiction pour {len(df_aura_2022)} communes en {target_region}.")
codes_communes_aura = df_aura_2022['Code Commune'].unique()

# Init. du dataframe contenant les estimations pour 2027
df_estimations_2027 = df_aura_2022[['Code Commune', 'Nom Commune', 'Région', 'Code Departement']].copy()
df_estimations_2027 = df_estimations_2027.set_index('Code Commune') 

# Sélectionner l'historique uniquement pour les communes AURA
df_historique_aura = df[df['Code Commune'].isin(codes_communes_aura)].copy()
variables_a_extrapoler = variables_communales_a_traiter + variables_nationales_regionales

# Estimation des données socio-économiques pour 2027 par extrapolation
print("Estimation des données socio-économiques pour 2027 par extrapolation...")
for var in variables_a_extrapoler:
    print(f"  Extrapolation pour : {var}")
    estimations_var = {}

    for code_commune in codes_communes_aura:
        serie_commune = df_historique_aura[df_historique_aura['Code Commune'] == code_commune][['Année', var]].dropna().sort_values('Année')

        if len(serie_commune) >= 2:
            coeffs = np.polyfit(serie_commune['Année'], serie_commune[var], 1)
            valeur_estimee = np.polyval(coeffs, annee_prediction)

            if var in ['Population', 'Inscrits', 'Nb_Votant']:
                valeur_estimee = max(0, round(valeur_estimee))
            elif var == 'Taux de chômage':
                 valeur_estimee = max(0, min(100, valeur_estimee))
            elif var == 'Nb Faits Divers':
                valeur_estimee = max(0, round(valeur_estimee))
            estimations_var[code_commune] = valeur_estimee

        elif len(serie_commune) == 1:
            estimations_var[code_commune] = serie_commune[var].iloc[0]
        else:
            estimations_var[code_commune] = 0

    # Ajouter la colonne estimée pour 2027
    df_estimations_2027[var] = pd.Series(estimations_var)

print("Extrapolation terminée.")
df_estimations_2027 = df_estimations_2027.reset_index()

# Init. du dataframe pour le calcul les différences 2027 vs 2022
print("Calcul des différences 2027 vs 2022...")
df_predict_2027 = df_estimations_2027.copy()
df_predict_2027 = df_predict_2027.merge(
    df_aura_2022[['Code Commune'] + variables_a_extrapoler],
    on='Code Commune',
    suffixes=('_2027', '_2022')
)

# Calcul et ajout des nouvelles colonnes de différence (_Diff_27_22)
nouvelles_features_diff_27_22 = {}
for var in variables_a_extrapoler:
    col_diff_name_27_22 = f'{var}_Diff_{annee_prediction}_{annee_recente}'
    df_predict_2027[col_diff_name_27_22] = df_predict_2027[f'{var}_2027'] - df_predict_2027[f'{var}_2022']
    nouvelles_features_diff_27_22[var] = col_diff_name_27_22

# Recréer les listes de colonnes mais avec les noms _Diff_27_22
cols_communales_2027 = variables_communales_a_traiter
cols_diff_communales_27_22 = [nouvelles_features_diff_27_22[var] for var in variables_communales_a_traiter if var in nouvelles_features_diff_27_22]
cols_diff_national_27_22 = [nouvelles_features_diff_27_22[var] for var in variables_nationales_regionales if var in nouvelles_features_diff_27_22]

# Mapper les noms _Diff_22_17 aux noms _Diff_27_22
map_diff_names = {nouvelles_features_diff_all[k]: nouvelles_features_diff_27_22[k] for k in nouvelles_features_diff_all if k in nouvelles_features_diff_27_22}
final_feature_list_2027 = []
for col in features_for_X:
    if col in variables_communales_a_traiter:
        final_feature_list_2027.append(f'{col}_2027')
    elif col in map_diff_names:
        final_feature_list_2027.append(map_diff_names[col])
    elif col in cols_diff_national_passthrough:
        if col in map_diff_names:
            final_feature_list_2027.append(map_diff_names[col])
        else:
            print(f"Attention: Colonne {col} attendue mais non trouvée dans map_diff_names")
    elif col in categorical_features:
        final_feature_list_2027.append(col)
    else:
        print(f"Attention: Colonne {col} non reconnue lors de la construction de X_2027")

# Créer X_2027_aura avec les bonnes colonnes renommées et dans le bon ordre
X_2027_aura_raw = df_predict_2027.copy()
rename_map_2027 = {f'{col}_2027': col for col in variables_communales_a_traiter}
X_2027_aura_raw.rename(columns=rename_map_2027, inplace=True)

# Renommer les colonnes _Diff_27_22 en _Diff_22_17
rename_map_diff = {v: k for k, v in map_diff_names.items()}
X_2027_aura_raw.rename(columns=rename_map_diff, inplace=True)


# Sélectionner uniquement les colonnes nécessaires dans le bon ordre
try:
    X_2027_aura = X_2027_aura_raw[features_for_X]
except KeyError as e:
    print(f"Erreur de clé lors de la sélection des colonnes pour X_2027_aura: {e}")
    print("Colonnes attendues:", features_for_X)
    print("Colonnes disponibles:", X_2027_aura_raw.columns.tolist())
    exit()
    
# Appliquer le Préprocesseur
print("Application du préprocesseur aux données 2027...")
X_2027_aura_processed = preprocessor.transform(X_2027_aura)

### PREDICTION DES RESULTATS POUR 2027 EN AUVERGNE RHONES ALPES ###

# Prédiction modèle - Random Forest
print("Prédiction pour 2027 (Random Forest)...")
predictions_2027_encoded_rf = model.predict(X_2027_aura_processed)
predictions_2027_noms_rf = label_encoder.inverse_transform(predictions_2027_encoded_rf)

# Prédiction modèle - Gradient Boosting
print("Prédiction avec Gradient Boosting...")
predictions_2027_encoded_gb = model_gb.predict(X_2027_aura_processed)
predictions_2027_noms_gb = label_encoder.inverse_transform(predictions_2027_encoded_gb)

# Prédiction modèle - KMeans
print("Assignation aux clusters K-Means et mapping vers gagnant...")
clusters_2027 = model_kmeans.predict(X_2027_aura_processed)
predictions_2027_encoded_kmeans = np.array([cluster_winner_map.get(label, -1) for label in clusters_2027])
valid_kmeans_preds_mask = predictions_2027_encoded_kmeans != -1
predictions_2027_noms_kmeans = np.full(predictions_2027_encoded_kmeans.shape, "Mapping_Inconnu", dtype=object)
if np.any(valid_kmeans_preds_mask): # S'il y a au moins une prédiction valide
    predictions_2027_noms_kmeans[valid_kmeans_preds_mask] = label_encoder.inverse_transform(
        predictions_2027_encoded_kmeans[valid_kmeans_preds_mask]
    )

# Créer un DataFrame de résultats
df_results_2027 = df_predict_2027[['Code Commune', 'Nom Commune']].copy()
df_results_2027['Gagnant_Predit_2027_RF'] = predictions_2027_noms_rf
df_results_2027['Gagnant_Predit_2027_GB'] = predictions_2027_noms_gb
df_results_2027['Gagnant_Predit_2027_KM'] = predictions_2027_noms_kmeans

# Ajout du nombre de votant
votes_2027_df = df_predict_2027[['Code Commune', 'Nb_Votant_2027']].copy()
df_results_2027 = pd.merge(df_results_2027, votes_2027_df, on='Code Commune', how='left')
df_results_2027['Nb_Votant_2027'] = df_results_2027['Nb_Votant_2027'].fillna(0).astype(int)

# Afficher la répartition (nb de commune)
print("\nRépartition des communes prédites par candidat en AURA en 2017 (Random Forest):")
print(df_results_2027['Gagnant_Predit_2027_RF'].value_counts())
print("\nRépartition des communes prédites par candidat en AURA en 2017 (Gradient Boosting):")
print(df_results_2027['Gagnant_Predit_2027_GB'].value_counts())
print("\nRépartition des communes prédites par candidat en AURA en 2017 (KMeans):")
print(df_results_2027['Gagnant_Predit_2027_KM'].value_counts())
df_results_2027.to_csv('predictions_aura_2027.csv', index=False)

# Calcul et affichage du gagnant pour Random Forest
print("\nTotal des votes prédits par candidat en AURA en 2027 (Random Forest):")
print(df_results_2027.groupby('Gagnant_Predit_2027_RF')['Nb_Votant_2027'].sum().sort_values(ascending=False))

# Calcul et affichage du gagnant pour Gradient Boosting
print("\nTotal des votes prédits par candidat en AURA en 2027 (Gradient Boosting):")
print(df_results_2027.groupby('Gagnant_Predit_2027_GB')['Nb_Votant_2027'].sum().sort_values(ascending=False))

# Calcul et affichage du gagnant pour KMeans
print("\nTotal des votes prédits par candidat en AURA en 2027 (KMeans):")
print(df_results_2027[df_results_2027['Gagnant_Predit_2027_KM'] != "Mapping_Inconnu"].groupby('Gagnant_Predit_2027_KM')['Nb_Votant_2027'].sum().sort_values(ascending=False))
