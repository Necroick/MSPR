
import os
import pandas as pd


script_dir = os.path.dirname(os.path.abspath(__file__))  # Chemin absolu du script
input_file = os.path.join(script_dir, "not_cleaned_criminal_data.csv")  # Chemin absolu du fichier CSV

df = pd.read_csv(input_file, sep=';', low_memory=False)

# Nettoyage de la colonne "année"
df["annee"] = df["annee"].astype(int)
df = df[df["annee"] != 16]

df["faits"] = df["faits"].fillna(0).astype(int)

df_cleaned = df[["annee", "CODGEO_2024", "faits"]]
df_grouped = df_cleaned.groupby(["annee", "CODGEO_2024"], as_index=False)["faits"].sum()

print(df_grouped.head(100))  # Vérifie les premières lignes du DataFrame regroupé
print(df_grouped.describe())  # Résumé statistique pour vérifier les totaux

# TODO : Fusion avec les infos complémentaires géographiques

df_grouped.to_csv("cleaned_criminal_data.csv", sep=";", index=False)

