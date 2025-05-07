import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

fichier = 'cleaned_criminal_data.csv'

if not os.path.exists(fichier):
    print("⚠️ Fichier introuvable :", os.path.abspath(fichier))
else:
    print("✅ Fichier trouvé :", os.path.abspath(fichier))


# Liste des départements d'Auvergne-Rhône-Alpes
ara_departements = {"01", "03", "07", "15", "26", "43", "42", "74", "38", "69", "63", "73"}

# Chargement du fichier CSV
df = pd.read_csv("cleaned_criminal_data.csv", sep=';')

# Extraction du département à partir du code géographique
df['departement'] = df['CODGEO_2024'].astype(str).str[:2]

# Catégorisation région / hors région
df['region'] = df['departement'].apply(lambda x: 'Auvergne-Rhône-Alpes' if x in ara_departements else 'Reste de la France')

# Agrégation du nombre de faits par année et par région
df_grouped = df.groupby(['annee', 'region'])['faits'].sum().reset_index()

# Population en millions
population = {
    'Auvergne-Rhône-Alpes': 8.2,
    'Reste de la France': 60.0
}

# Ajout du taux de faits par million d'habitants
df_grouped['faits_par_million'] = df_grouped.apply(
    lambda row: row['faits'] / population[row['region']], axis=1
)


# Visualisation
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_grouped, x='annee', y='faits_par_million', hue='region', marker='o')
plt.title("Taux de faits criminel par million d'habitants")
plt.xlabel("Année")
plt.ylabel("Faits pour 1 million d'habitants")
plt.grid(True)
plt.tight_layout()
plt.show()
