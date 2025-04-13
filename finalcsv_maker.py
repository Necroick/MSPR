import pandas as pd

df_chomage = pd.read_csv('cleaned_data\\data_chomage.csv')
df_creat_entreprise = pd.read_csv('cleaned_data\\data_creations_entreprises.csv')
df_criminalite = pd.read_csv('cleaned_data\\data_criminalite.csv', sep=';')
df_election = pd.read_csv('cleaned_data\\data_election.csv')
df_population = pd.read_csv('cleaned_data\\data_population.csv')
df_final = pd.DataFrame()

print(df_chomage.head())
print(df_creat_entreprise.head())
print(df_criminalite.head())
print(df_election.head())
print(df_population.head())

df_final.to_csv('final_data.csv', index=False)