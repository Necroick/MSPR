import pandas as pd

df = pd.read_csv('raw_data/donnees_communes_2021.csv', sep=';')
df.rename(columns={'COM': 'Code Postale', 'PMUN': 'Population municipale', 'PCAP': 'Population comptée à part', 'PTOT': 'Population totale'}, inplace=True)

print(df.columns)