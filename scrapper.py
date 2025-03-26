from scrapper_function import createDepcom, resElectionClean2017
import pandas as pd

# Ouverture des données de population de 2022 // Renommage de certaines colonnes pour une cohérence entre les datasets
df_pop_2022 = pd.read_csv('raw_data/donnees_population_communes_2022.csv', sep=';')
df_pop_2022.rename(columns={'COM':'DEPCOM', 'REG':'CODREG', 'DEP':'CODDEP', 'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)

# Ouverture des données de population de 2021 // Renommage de certaines colonnes pour une cohérence entre les datasets
df_pop_2021 = pd.read_csv('raw_data/donnees_population_communes_2021.csv', sep=';')
df_pop_2021.rename(columns={'COM':'DEPCOM', 'REG':'CODREG', 'DEP':'CODDEP', 'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)

# Ouverture des données de population de 2020 // Renommage de certaines colonnes pour une cohérence entre les datasets
df_pop_2020 = pd.read_csv('raw_data/donnees_population_communes_2020.csv', sep=';')
df_pop_2020.rename(columns={'REG':'Région', 'COM': 'Commune', 'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
df_pop_2020 = createDepcom(df_pop_2020)

# Ouverture des données de population de 2019 // Renommage de certaines colonnes pour une cohérence entre les datasets
df_pop_2019 = pd.read_csv('raw_data/donnees_population_communes_2019.csv', sep=';')
df_pop_2019.rename(columns={'REG':'Région', 'COM': 'Commune', 'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
df_pop_2019 = createDepcom(df_pop_2019)

# Ouverture des données de population de 2018 // Renommage de certaines colonnes pour une cohérence entre les datasets
df_pop_2018 = pd.read_csv('raw_data/donnees_population_communes_2018.csv', sep=';')
df_pop_2018.rename(columns={'REG':'Région', 'COM': 'Commune', 'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
df_pop_2018 = createDepcom(df_pop_2018)

# Ouverture des données de population de 2017 // Ajout et renommage de certaines colonnes pour une cohérence entre les datasets
df_pop_2017 = pd.read_csv('raw_data/donnees_population_communes_2017.csv', sep=';')
df_pop_2017.rename(columns={'COM': 'Commune', 'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
df_pop_2017['CODDEP'] = df_pop_2017['DEPCOM'].str[:2]
df_pop_2017['CODCOM'] = df_pop_2017['DEPCOM'].str[2:]

# Creation de df_pop_2017['CODREG'], df_pop_2017['Région'], df_pop_2017['CODARR'], df_pop_2017['CODCAN'] pour df_pop_2017['DEPCOM'] == df_pop_2021['DEPCOM']
df_pop_2021_subset = df_pop_2021[['DEPCOM', 'CODREG', 'Région', 'CODARR', 'CODCAN']]
df_pop_2017 = pd.merge(df_pop_2017, df_pop_2021_subset, on='DEPCOM', how='left')

# Ouverture des données d'élections de 2017 \\ Traitement de .csv pour avoir un candidat et une commun par ligne
df_elec_2017 = pd.read_csv('raw_data/resultat_election_2017_burvot.csv', sep=';', encoding='latin-1', header=None)
df_elec_2017 = resElectionClean2017(df_elec_2017) 


# Suppresion de CORARR, CODCAN et CODREG dans les df de populations
df_pop_2017.drop('CODARR', axis=1, inplace=True)
df_pop_2017.drop('CODCAN', axis=1, inplace=True)
df_pop_2017.drop('CODREG', axis=1, inplace=True)

df_pop_2018.drop('CODARR', axis=1, inplace=True)
df_pop_2018.drop('CODCAN', axis=1, inplace=True)
df_pop_2018.drop('CODREG', axis=1, inplace=True)

df_pop_2019.drop('CODARR', axis=1, inplace=True)
df_pop_2019.drop('CODCAN', axis=1, inplace=True)
df_pop_2019.drop('CODREG', axis=1, inplace=True)

df_pop_2020.drop('CODARR', axis=1, inplace=True)
df_pop_2020.drop('CODCAN', axis=1, inplace=True)
df_pop_2020.drop('CODREG', axis=1, inplace=True)

df_pop_2021.drop('CODARR', axis=1, inplace=True)
df_pop_2021.drop('CODCAN', axis=1, inplace=True)
df_pop_2021.drop('CODREG', axis=1, inplace=True)

df_pop_2022.drop('CODARR', axis=1, inplace=True)
df_pop_2022.drop('CODCAN', axis=1, inplace=True)
df_pop_2022.drop('CODREG', axis=1, inplace=True)

print(df_pop_2017.columns)