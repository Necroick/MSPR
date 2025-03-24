from scrapper_function import createDepcom
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

# Création des entête manuellement
base_columns = "Code du département;Libellé du département;Code de la circonscription;Libellé de la circonscription;Code de la commune;Libellé de la commune;Code du b.vote;Inscrits;Abstentions;% Abs/Ins;Votants;% Vot/Ins;Blancs;% Blancs/Ins;% Blancs/Vot;Nuls;% Nuls/Ins;% Nuls/Vot;Exprimés;% Exp/Ins;% Exp/Vot".split(';')
candidat_columns = "N°Panneau;Sexe;Nom;Prénom;Voix;% Voix/Ins;% Voix/Exp".split(';')
candidat_columns_renamed = []
for i in range(11):
    candidat_columns_renamed.extend([f'{col}_candi{i+1}' for col in candidat_columns])
df_elec_2017.columns = base_columns+candidat_columns_renamed

# Suppression des entête inutiles
df_elec_2017.drop(columns=['Code de la circonscription','Libellé de la circonscription','Code du b.vote','% Abs/Ins', '% Vot/Ins', '% Blancs/Ins', '% Blancs/Vot', '% Nuls/Ins', '% Nuls/Vot', '% Exp/Ins', '% Exp/Vot', '% Voix/Ins_candi1', '% Voix/Exp_candi1', '% Voix/Ins_candi2', '% Voix/Exp_candi2', '% Voix/Ins_candi3', '% Voix/Exp_candi3', '% Voix/Ins_candi4', '% Voix/Exp_candi4', '% Voix/Ins_candi5', '% Voix/Exp_candi5', '% Voix/Ins_candi6', '% Voix/Exp_candi6', '% Voix/Ins_candi7', '% Voix/Exp_candi7', '% Voix/Ins_candi8', '% Voix/Exp_candi8', '% Voix/Ins_candi9', '% Voix/Exp_candi9', '% Voix/Ins_candi10', '% Voix/Exp_candi10', '% Voix/Ins_candi11', '% Voix/Exp_candi11'], inplace=True)

# Regroupper les données par communes
num_col = ['Inscrits','Abstentions','Votants','Blancs','Nuls','Exprimés','Voix_candi1','Voix_candi2','Voix_candi3','Voix_candi4','Voix_candi5','Voix_candi6','Voix_candi7','Voix_candi8','Voix_candi9','Voix_candi10','Voix_candi11']
group_col = [col for col in df_elec_2017.columns if col not in num_col]
df_elec_2017 = df_elec_2017.groupby(group_col, as_index=False).sum()

print(df_elec_2017.head(10))

"""
print(df_pop_2017)
print(df_pop_2018)
print(df_pop_2019)
print(df_pop_2020)
print(df_pop_2021)
print(df_pop_2022)
"""