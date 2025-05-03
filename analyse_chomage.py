import pandas as pd

# Charger le fichier Excel
df = pd.read_excel("serie_001739986_24032025.xlsx")

# Supprimer les 3 premières lignes inutiles
df = df.drop(index=[0, 1, 2]).reset_index(drop=True)

# Renommer les colonnes
df.columns = ["Période", "Taux de chômage"]

# Afficher les 10 premières lignes
print(df.head(10))

# Convertir la colonne 'Taux de chômage' en float
df['Taux de chômage'] = pd.to_numeric(df['Taux de chômage'], errors='coerce')

# Extraire l'année depuis la colonne 'Période'
df['Année'] = df['Période'].str[:4]

# Calculer la moyenne annuelle
df_moyenne = df.groupby('Année')['Taux de chômage'].mean().reset_index()

# Afficher le résultat
print(df_moyenne)

import matplotlib.pyplot as plt

# Enregistrer la table des moyennes en CSV
df_moyenne.to_csv("taux_chomage_moyen.csv", index=False, encoding='utf-8')
print(" Fichier CSV créé : taux_chomage_moyen.csv")
