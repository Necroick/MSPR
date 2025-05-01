import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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

# Définiton du processus d'encodage de X
preprocessor = ColumnTransformer(
    transformers=[
        ('num_communal', numeric_pipeline, cols_communales_num_impute_scale),   # Appliquer Imputer+Scaler aux colonnes communales (base + diff)
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # OneHotEncoder les catégorielles
    ],
    remainder='passthrough' # Les colonnes non listées (cols_diff_national_passthrough) passeront ici
)


# Séparer les données en ensemble d'entrainement et leur appliqué l'encodage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Application du préprocesseur...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("Préprocessing terminé.")

print(f"NaNs dans X_train_processed: {np.isnan(X_train_processed).sum()}")
print(f"NaNs dans X_test_processed: {np.isnan(X_test_processed).sum()}")

### ENTRAINEMENT DES MODELES ###

# Définition du modèle - Random Forest
model = RandomForestClassifier(n_estimators=125,
                               random_state=42,
                               class_weight='balanced',
                               n_jobs=-1)

# Entrainement du modèle - Random Forest
print("Entraînement du modèle RandomForest...")
model.fit(X_train_processed, y_train)
print("Entraînement terminé.")

# Test du model et calcul de l'accuracy - Random Forest
print("Prédiction et évaluation...")
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")