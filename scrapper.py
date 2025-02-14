import pandas as pd

df_2021 = pd.read_csv('raw_data/donnees_communes_2021.csv', sep=';')
df_2021.rename(columns={'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
print(df_2021.columns)

df_2020 = pd.read_csv('raw_data/donnees_communes_2020.csv', sep=';')
df_2020.rename(columns={'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
print(df_2020.columns)

df_2019 = pd.read_csv('raw_data/donnees_communes_2019.csv', sep=';')
df_2019.rename(columns={'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
print(df_2019.columns)

df_2018 = pd.read_csv('raw_data/donnees_communes_2018.csv', sep=';')
df_2018.rename(columns={'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
print(df_2018.columns)

df_2017 = pd.read_csv('raw_data/donnees_communes_2017.csv', sep=';')
df_2017.rename(columns={'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)
print(df_2019.columns)