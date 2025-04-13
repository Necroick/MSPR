import pandas as pd

# Récupération des données de votes de 2022 trié par Awa
df_elec_2022 = pd.read_csv('awa_data\\election-2022.csv', sep=',')

# Suppression des colonnes inutiles
df_elec_2022.drop('Code de la circonscription', axis=1, inplace=True)
df_elec_2022.drop('Libellé de la circonscription', axis=1, inplace=True)
df_elec_2022.drop('lib_du_b_vote', axis=1, inplace=True)
df_elec_2022.drop('scrutin_code', axis=1, inplace=True)
df_elec_2022.drop('Code Officiel EPCI', axis=1, inplace=True)
df_elec_2022.drop('Nom Officiel EPCI', axis=1, inplace=True)
df_elec_2022.drop('Sexe', axis=1, inplace=True)
df_elec_2022.drop('Code_département', axis=1, inplace=True)
df_elec_2022.drop('Libellé du département', axis=1, inplace=True)
df_elec_2022.drop('Code Officiel Région', axis=1, inplace=True)
df_elec_2022.drop('Nom Officiel Région', axis=1, inplace=True)
df_elec_2022.drop('location', axis=1, inplace=True)
df_elec_2022.drop('Code du b.vote', axis=1, inplace=True)

# Suppression des données qui perderont en cohérence lors de la concaténation
df_elec_2022.drop('% Abs/Ins', axis=1, inplace=True)
df_elec_2022.drop('% Vot/Ins', axis=1, inplace=True)
df_elec_2022.drop('% Blancs/Ins', axis=1, inplace=True)
df_elec_2022.drop('% Blancs/Vot', axis=1, inplace=True)
df_elec_2022.drop('% Nuls/Ins', axis=1, inplace=True)
df_elec_2022.drop('% Nuls/Vot', axis=1, inplace=True)
df_elec_2022.drop('% Exp/Ins', axis=1, inplace=True)
df_elec_2022.drop('% Exp/Vot', axis=1, inplace=True)
df_elec_2022.drop('% Voix/Ins', axis=1, inplace=True)
df_elec_2022.drop('% Voix/Exp', axis=1, inplace=True)

# Concaténation des données par ville et par candidat
grp_col = ['Code de la commune', 'Libellé de la commune', 'N°Panneau', 'Nom', 'Prénom']
df_elec_2022 = df_elec_2022.groupby(grp_col, as_index=False).sum()

# Pour chaque ville, on garde le candidat qui a gagner
idx = df_elec_2022.groupby(['Code de la commune'])['Voix'].idxmax()
df_elec_2022 = df_elec_2022.loc[idx]

# Suppression des champs devenues inutiles (+ regroupement de Nom et Prénom)
df_elec_2022['NomPrénom'] = df_elec_2022['Nom'] + ' ' + df_elec_2022['Prénom']
df_elec_2022.drop('Nom', axis=1, inplace=True)
df_elec_2022.drop('Prénom', axis=1, inplace=True)
df_elec_2022.drop('Voix', axis=1, inplace=True)
df_elec_2022.drop('N°Panneau', axis=1, inplace=True)
df_elec_2022.drop('Blancs', axis=1, inplace=True)
df_elec_2022.drop('Nuls', axis=1, inplace=True)
df_elec_2022.drop('Exprimés', axis=1, inplace=True)

# Ajout de la colonne 'Année'
df_elec_2022['Année'] = 22

df_elec_2022.to_csv('test.csv', index=False)