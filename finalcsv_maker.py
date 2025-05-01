import pandas as pd

df_chomage = pd.read_csv('cleaned_data\\data_chomage.csv')
df_creat_entreprise = pd.read_csv('cleaned_data\\data_creations_entreprises.csv')
df_criminalite = pd.read_csv('cleaned_data\\data_criminalite.csv', sep=';')
df_election = pd.read_csv('cleaned_data\\data_election.csv')
df_population = pd.read_csv('cleaned_data\\data_population.csv')
df_final = pd.DataFrame()

# Mise en commun du nom  et format des colonnes
df_chomage['Année'] = df_chomage['Année'].astype(str).str[-2:].astype(int)
df_creat_entreprise['Année'] = df_creat_entreprise['Année'].astype(str).str[-2:].astype(int)
df_criminalite.rename(columns={'annee': 'Année', 'CODGEO_2024': 'Code Commune', 'faits': 'Nb Faits Divers'}, inplace=True)

# Concaténation des différents dataframe
df_final = df_population.copy()

df_final = pd.merge(
    df_final,                       # DataFrame de gauche (celui qu'on veut compléter)
    df_criminalite,                 # DataFrame de droite (celui dont on prend les données)
    on=['Code Commune', 'Année'],   # Colonnes sur lesquelles baser la fusion
    how='left'                      # Type de fusion: 'left' garde toutes les lignes de df_final
)

df_final = pd.merge(
    df_final,
    df_chomage,
    on=['Année'],
    how='left'
)

df_final = pd.merge(
    df_final,
    df_creat_entreprise,
    on=['Année'],
    how='left'
)

df_final = pd.merge(
    df_final,
    df_election,
    on=['Code Commune', 'Année'],
    how='left'
)


df_final['Nom_Prenom_Gagnant'] = df_final['Nom_Prenom_Gagnant'].str.replace('Jean-Luc Mï¿½LENCHON', 'Jean-Luc MELENCHON', regex=False)
df_final['Nom_Prenom_Gagnant'] = df_final['Nom_Prenom_Gagnant'].str.replace('Franï¿½ois FILLON', 'Francois FILLON', regex=False)
df_final['Nom_Prenom_Gagnant'] = df_final['Nom_Prenom_Gagnant'].str.replace('Benoï¿½t HAMON', 'Benoit HAMON', regex=False)
df_final['Nom_Prenom_Gagnant'] = df_final['Nom_Prenom_Gagnant'].str.replace('Franï¿½ois ASSELINEAU', 'Francois ASSELINEAU', regex=False)
df_final['Nom_Prenom_Gagnant'] = df_final['Nom_Prenom_Gagnant'].str.replace('ï¿½ric ZEMMOUR', 'Eric ZEMMOUR', regex=False)
df_final['Nom_Prenom_Gagnant'] = df_final['Nom_Prenom_Gagnant'].str.replace('Valï¿½rie Pï¿½CRESSE', 'Valerie PECRESSE', regex=False)


print(df_final.head())

df_final.to_csv('final_data.csv', index=False)