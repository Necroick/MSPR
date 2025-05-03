import pandas as pd

# Charger le fichier Excel téléchargé depuis l'INSEE
fichier = "TAB_SIDE_CREA_ENT_COM_HISTO_fr.xlsx"

# Lire la deuxieme feuille du fichier (où sont les vraies données)
donnees = pd.read_excel(fichier, sheet_name=1, skiprows=3)

# Donner des noms clairs aux colonnes
donnees.columns = [
    "Code_Commune", "Nom_Commune",
    "2012", "2013", "2014", "2015", "2016",
    "2017", "2018", "2019", "2020", "2021", "2022", "2023"
]

# Garder uniquement les colonnes de 2012 à 2022
donnees_filtrees = donnees[["Code_Commune", "Nom_Commune",
                            "2012", "2013", "2014", "2015", "2016",
                            "2017", "2018", "2019", "2020", "2021", "2022"]]

# Supprimer les lignes où le code INSEE est vide 
donnees_filtrees = donnees_filtrees.dropna(subset=["Code_Commune"])

# Afficher les 10 premières lignes du tableau
print("Voici un aperçu des données :")
print(donnees_filtrees.head(10))

# enregistrer les données nettoyées dans un nouveau fichier CSV
donnees_filtrees.to_csv("creations_entreprises_2012_2022.csv", index=False)
print("Données enregistrées dans 'creations_entreprises_2012_2022.csv'")
