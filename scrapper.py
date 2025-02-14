import pandas as pd

df_2021 = pd.read_csv('raw_data/donnees_communes_2021.csv', sep=';')
df_2021.rename(columns={'COM': 'Code Postale', 'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)

df_2020 = pd.read_csv('raw_data/donnees_communes_2020.csv', sep=';')
df_2020.rename(columns={'COM': 'Code Postale', 'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
