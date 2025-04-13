import pandas as pd

def createDepcom(df):
    coddep = list(df['CODDEP'])
    codcom = list(df['CODCOM'])
    depcom = []
    for i in range(len(codcom)):
        if len(str(coddep[i])) == 2 :
            if len(str(codcom[i])) == 1 :
                depcom.append(str(coddep[i]) + '00' + str(codcom[i]))
            elif len(str(codcom[i])) == 2 :
                depcom.append(str(coddep[i]) + '0' + str(codcom[i]))
            elif len(str(codcom[i])) == 3 :
                depcom.append(str(coddep[i]) + str(codcom[i]))
        elif len(str(coddep[i])) == 3 :
            if len(str(codcom[i])) == 1 :
                depcom.append(str(coddep[i]) + '0' + str(codcom[i]))
            elif len(str(codcom[i])) == 2 :
                depcom.append(str(coddep[i]) + str(codcom[i]))
            elif len(str(codcom[i])) == 3 :
                depcom.append(str(coddep[i]) + str(codcom[i])[1:])
    df['DEPCOM'] = depcom
    return df

def createCodeCommune(df):
    coddep = list(df['Code du département'])
    codcom = list(df['Code de la commune'])
    depcom = []
    for i in range(len(codcom)):
        #print(i, df.iloc[i]['Libellé de la commune'], df.iloc[i]['Code du département'], df.iloc[i]['Code de la commune'])
        if len(str(coddep[i])) == 1 :
            if len(str(codcom[i])) == 1 :
                depcom.append('0' + str(coddep[i]) + '00' + str(codcom[i]))
            elif len(str(codcom[i])) == 2 :
                depcom.append('0' + str(coddep[i]) + '0' + str(codcom[i]))
            elif len(str(codcom[i])) == 3 :
                depcom.append('0' + str(coddep[i]) + str(codcom[i]))
        elif len(str(coddep[i])) == 2 :
            if len(str(codcom[i])) == 1 :
                depcom.append(str(coddep[i]) + '00' + str(codcom[i]))
            elif len(str(codcom[i])) == 2 :
                depcom.append(str(coddep[i]) + '0' + str(codcom[i]))
            elif len(str(codcom[i])) == 3 :
                depcom.append(str(coddep[i]) + str(codcom[i]))
        elif len(str(coddep[i])) == 3 :
            if len(str(codcom[i])) == 1 :
                depcom.append(str(coddep[i]) + '0' + str(codcom[i]))
            elif len(str(codcom[i])) == 2 :
                depcom.append(str(coddep[i]) + str(codcom[i]))
            elif len(str(codcom[i])) == 3 :
                depcom.append(str(coddep[i]) + str(codcom[i])[1:])
    df['Code Commune'] = depcom
    return df

def resElectionClean2017(df):
    # Création des entête manuellement
    base_columns = "Code du département;Libellé du département;Code de la circonscription;Libellé de la circonscription;Code de la commune;Libellé de la commune;Code du b.vote;Inscrits;Abstentions;% Abs/Ins;Votants;% Vot/Ins;Blancs;% Blancs/Ins;% Blancs/Vot;Nuls;% Nuls/Ins;% Nuls/Vot;Exprimés;% Exp/Ins;% Exp/Vot".split(';')
    candidat_columns = "N°Panneau;Sexe;Nom;Prénom;Voix;% Voix/Ins;% Voix/Exp".split(';')
    candidat_columns_renamed = []
    for i in range(11):
        candidat_columns_renamed.extend([f'{col}_candi{i+1}' for col in candidat_columns])
    df.columns = base_columns+candidat_columns_renamed

    # Suppression des entête inutiles
    df.drop(columns=['Code de la circonscription','Libellé de la circonscription','Code du b.vote','% Abs/Ins', '% Vot/Ins', '% Blancs/Ins', '% Blancs/Vot', '% Nuls/Ins', '% Nuls/Vot', '% Exp/Ins', '% Exp/Vot', '% Voix/Ins_candi1', '% Voix/Exp_candi1', '% Voix/Ins_candi2', '% Voix/Exp_candi2', '% Voix/Ins_candi3', '% Voix/Exp_candi3', '% Voix/Ins_candi4', '% Voix/Exp_candi4', '% Voix/Ins_candi5', '% Voix/Exp_candi5', '% Voix/Ins_candi6', '% Voix/Exp_candi6', '% Voix/Ins_candi7', '% Voix/Exp_candi7', '% Voix/Ins_candi8', '% Voix/Exp_candi8', '% Voix/Ins_candi9', '% Voix/Exp_candi9', '% Voix/Ins_candi10', '% Voix/Exp_candi10', '% Voix/Ins_candi11', '% Voix/Exp_candi11'], inplace=True)

    # Regroupper les données par communes
    num_col = ['Inscrits','Abstentions','Votants','Blancs','Nuls','Exprimés','Voix_candi1','Voix_candi2','Voix_candi3','Voix_candi4','Voix_candi5','Voix_candi6','Voix_candi7','Voix_candi8','Voix_candi9','Voix_candi10','Voix_candi11']
    group_col = [col for col in df.columns if col not in num_col]
    df = df.groupby(group_col, as_index=False).sum()

    # Récupération du gagnant de chaque ville
    vote_columns = [f'Voix_candi{i}' for i in range(1, 12)]
    winning_vote_col = df[vote_columns].idxmax(axis=1)
    winner_indices = winning_vote_col.str.extract(r'(\d+)$', expand=False)

    # Récupérer les infos du gagnant en utilisant le numéro extrait
    df['Nom_Gagnant'] = None
    df['Prenom_Gagnant'] = None
    df['Voix_Gagnant'] = None
    for index, row in df.iterrows():
        winner_idx = winner_indices.loc[index]
        df.loc[index, 'Nom_Gagnant'] = row[f'Nom_candi{winner_idx}']
        df.loc[index, 'Prenom_Gagnant'] = row[f'Prénom_candi{winner_idx}']
        df.loc[index, 'Voix_Gagnant'] = row[f'Voix_candi{winner_idx}']

    # Suppression des colonnes inutiles
    df.drop('Voix_candi1', axis=1, inplace=True)
    df.drop('N°Panneau_candi1', axis=1, inplace=True)
    df.drop('Sexe_candi1', axis=1, inplace=True)
    df.drop('Nom_candi1', axis=1, inplace=True)
    df.drop('Prénom_candi1', axis=1, inplace=True)
    df.drop('Voix_candi2', axis=1, inplace=True)
    df.drop('N°Panneau_candi2', axis=1, inplace=True)
    df.drop('Sexe_candi2', axis=1, inplace=True)
    df.drop('Nom_candi2', axis=1, inplace=True)
    df.drop('Prénom_candi2', axis=1, inplace=True)
    df.drop('Voix_candi3', axis=1, inplace=True)
    df.drop('N°Panneau_candi3', axis=1, inplace=True)
    df.drop('Sexe_candi3', axis=1, inplace=True)
    df.drop('Nom_candi3', axis=1, inplace=True)
    df.drop('Prénom_candi3', axis=1, inplace=True)
    df.drop('Voix_candi4', axis=1, inplace=True)
    df.drop('N°Panneau_candi4', axis=1, inplace=True)
    df.drop('Sexe_candi4', axis=1, inplace=True)
    df.drop('Nom_candi4', axis=1, inplace=True)
    df.drop('Prénom_candi4', axis=1, inplace=True)
    df.drop('Voix_candi5', axis=1, inplace=True)
    df.drop('N°Panneau_candi5', axis=1, inplace=True)
    df.drop('Sexe_candi5', axis=1, inplace=True)
    df.drop('Nom_candi5', axis=1, inplace=True)
    df.drop('Prénom_candi5', axis=1, inplace=True)
    df.drop('Voix_candi6', axis=1, inplace=True)
    df.drop('N°Panneau_candi6', axis=1, inplace=True)
    df.drop('Sexe_candi6', axis=1, inplace=True)
    df.drop('Nom_candi6', axis=1, inplace=True)
    df.drop('Prénom_candi6', axis=1, inplace=True)
    df.drop('Voix_candi7', axis=1, inplace=True)
    df.drop('N°Panneau_candi7', axis=1, inplace=True)
    df.drop('Sexe_candi7', axis=1, inplace=True)
    df.drop('Nom_candi7', axis=1, inplace=True)
    df.drop('Prénom_candi7', axis=1, inplace=True)
    df.drop('Voix_candi8', axis=1, inplace=True)
    df.drop('N°Panneau_candi8', axis=1, inplace=True)
    df.drop('Sexe_candi8', axis=1, inplace=True)
    df.drop('Nom_candi8', axis=1, inplace=True)
    df.drop('Prénom_candi8', axis=1, inplace=True)
    df.drop('Voix_candi9', axis=1, inplace=True)
    df.drop('N°Panneau_candi9', axis=1, inplace=True)
    df.drop('Sexe_candi9', axis=1, inplace=True)
    df.drop('Nom_candi9', axis=1, inplace=True)
    df.drop('Prénom_candi9', axis=1, inplace=True)
    df.drop('Voix_candi10', axis=1, inplace=True)
    df.drop('N°Panneau_candi10', axis=1, inplace=True)
    df.drop('Sexe_candi10', axis=1, inplace=True)
    df.drop('Nom_candi10', axis=1, inplace=True)
    df.drop('Prénom_candi10', axis=1, inplace=True)
    df.drop('Voix_candi11', axis=1, inplace=True)
    df.drop('N°Panneau_candi11', axis=1, inplace=True)
    df.drop('Sexe_candi11', axis=1, inplace=True)
    df.drop('Nom_candi11', axis=1, inplace=True)
    df.drop('Prénom_candi11', axis=1, inplace=True)

    df.drop('Abstentions', axis=1, inplace=True)
    df.drop('Votants', axis=1, inplace=True)
    df.drop('Blancs', axis=1, inplace=True)
    df.drop('Nuls', axis=1, inplace=True)

    df.drop('Voix_Gagnant', axis=1, inplace=True)
    df['Nom_Prenom_Gagnant'] = df['Prenom_Gagnant'] + ' ' + df['Nom_Gagnant']
    df.drop('Nom_Gagnant', axis=1, inplace=True)
    df.drop('Prenom_Gagnant', axis=1, inplace=True)

    # Suppression des données liés aux français de l'étranger
    df = df[df['Code du département'] != 'ZZ']
    df = df[df['Code du département'] != 'ZX']
    df = df[df['Code du département'] != 'ZW']
    df = df[df['Code du département'] != 'ZS']
    df = df[df['Code du département'] != 'ZP']
    df = df[df['Code du département'] != 'ZN']
    df = df[df['Code du département'] != 'ZM']

    # Modification du code departement des territoires français de l'étranger
    df['Code du département'].replace(['ZA', 'ZB', 'ZC', 'ZD'], '97', inplace=True)

    return df

def resElectionClean2022(df):
     # Création des entête manuellement
    base_columns = "Code du département;Libellé du département;Code de la circonscription;Libellé de la circonscription;Code de la commune;Libellé de la commune;Code du b.vote;Inscrits;Abstentions;% Abs/Ins;Votants;% Vot/Ins;Blancs;% Blancs/Ins;% Blancs/Vot;Nuls;% Nuls/Ins;% Nuls/Vot;Exprimés;% Exp/Ins;% Exp/Vot".split(';')
    candidat_columns = "N°Panneau;Sexe;Nom;Prénom;Voix;% Voix/Ins;% Voix/Exp".split(';')
    candidat_columns_renamed = []
    for i in range(12):
        candidat_columns_renamed.extend([f'{col}_candi{i+1}' for col in candidat_columns])
    df.columns = base_columns+candidat_columns_renamed

    # Suppression des entête inutiles
    df.drop(columns=['Code de la circonscription','Libellé de la circonscription','Code du b.vote','% Abs/Ins', '% Vot/Ins', '% Blancs/Ins', '% Blancs/Vot', '% Nuls/Ins', '% Nuls/Vot', '% Exp/Ins', '% Exp/Vot', '% Voix/Ins_candi1', '% Voix/Exp_candi1', '% Voix/Ins_candi2', '% Voix/Exp_candi2', '% Voix/Ins_candi3', '% Voix/Exp_candi3', '% Voix/Ins_candi4', '% Voix/Exp_candi4', '% Voix/Ins_candi5', '% Voix/Exp_candi5', '% Voix/Ins_candi6', '% Voix/Exp_candi6', '% Voix/Ins_candi7', '% Voix/Exp_candi7', '% Voix/Ins_candi8', '% Voix/Exp_candi8', '% Voix/Ins_candi9', '% Voix/Exp_candi9', '% Voix/Ins_candi10', '% Voix/Exp_candi10', '% Voix/Ins_candi11', '% Voix/Exp_candi11', '% Voix/Ins_candi12', '% Voix/Exp_candi12'], inplace=True)

    # Regroupper les données par communes
    num_col = ['Inscrits','Abstentions','Votants','Blancs','Nuls','Exprimés','Voix_candi1','Voix_candi2','Voix_candi3','Voix_candi4','Voix_candi5','Voix_candi6','Voix_candi7','Voix_candi8','Voix_candi9','Voix_candi10','Voix_candi11','Voix_candi12']
    group_col = [col for col in df.columns if col not in num_col]
    df = df.groupby(group_col, as_index=False).sum()

    # Récupération du gagnant de chaque ville
    vote_columns = [f'Voix_candi{i}' for i in range(1, 13)]
    winning_vote_col = df[vote_columns].idxmax(axis=1)
    winner_indices = winning_vote_col.str.extract(r'(\d+)$', expand=False)

    # Récupérer les infos du gagnant en utilisant le numéro extrait
    df['Nom_Gagnant'] = None
    df['Prenom_Gagnant'] = None
    df['Voix_Gagnant'] = None
    for index, row in df.iterrows():
        winner_idx = winner_indices.loc[index]
        df.loc[index, 'Nom_Gagnant'] = row[f'Nom_candi{winner_idx}']
        df.loc[index, 'Prenom_Gagnant'] = row[f'Prénom_candi{winner_idx}']
        df.loc[index, 'Voix_Gagnant'] = row[f'Voix_candi{winner_idx}']

    # Suppression des colonnes inutiles
    df.drop('Voix_candi1', axis=1, inplace=True)
    df.drop('N°Panneau_candi1', axis=1, inplace=True)
    df.drop('Sexe_candi1', axis=1, inplace=True)
    df.drop('Nom_candi1', axis=1, inplace=True)
    df.drop('Prénom_candi1', axis=1, inplace=True)
    df.drop('Voix_candi2', axis=1, inplace=True)
    df.drop('N°Panneau_candi2', axis=1, inplace=True)
    df.drop('Sexe_candi2', axis=1, inplace=True)
    df.drop('Nom_candi2', axis=1, inplace=True)
    df.drop('Prénom_candi2', axis=1, inplace=True)
    df.drop('Voix_candi3', axis=1, inplace=True)
    df.drop('N°Panneau_candi3', axis=1, inplace=True)
    df.drop('Sexe_candi3', axis=1, inplace=True)
    df.drop('Nom_candi3', axis=1, inplace=True)
    df.drop('Prénom_candi3', axis=1, inplace=True)
    df.drop('Voix_candi4', axis=1, inplace=True)
    df.drop('N°Panneau_candi4', axis=1, inplace=True)
    df.drop('Sexe_candi4', axis=1, inplace=True)
    df.drop('Nom_candi4', axis=1, inplace=True)
    df.drop('Prénom_candi4', axis=1, inplace=True)
    df.drop('Voix_candi5', axis=1, inplace=True)
    df.drop('N°Panneau_candi5', axis=1, inplace=True)
    df.drop('Sexe_candi5', axis=1, inplace=True)
    df.drop('Nom_candi5', axis=1, inplace=True)
    df.drop('Prénom_candi5', axis=1, inplace=True)
    df.drop('Voix_candi6', axis=1, inplace=True)
    df.drop('N°Panneau_candi6', axis=1, inplace=True)
    df.drop('Sexe_candi6', axis=1, inplace=True)
    df.drop('Nom_candi6', axis=1, inplace=True)
    df.drop('Prénom_candi6', axis=1, inplace=True)
    df.drop('Voix_candi7', axis=1, inplace=True)
    df.drop('N°Panneau_candi7', axis=1, inplace=True)
    df.drop('Sexe_candi7', axis=1, inplace=True)
    df.drop('Nom_candi7', axis=1, inplace=True)
    df.drop('Prénom_candi7', axis=1, inplace=True)
    df.drop('Voix_candi8', axis=1, inplace=True)
    df.drop('N°Panneau_candi8', axis=1, inplace=True)
    df.drop('Sexe_candi8', axis=1, inplace=True)
    df.drop('Nom_candi8', axis=1, inplace=True)
    df.drop('Prénom_candi8', axis=1, inplace=True)
    df.drop('Voix_candi9', axis=1, inplace=True)
    df.drop('N°Panneau_candi9', axis=1, inplace=True)
    df.drop('Sexe_candi9', axis=1, inplace=True)
    df.drop('Nom_candi9', axis=1, inplace=True)
    df.drop('Prénom_candi9', axis=1, inplace=True)
    df.drop('Voix_candi10', axis=1, inplace=True)
    df.drop('N°Panneau_candi10', axis=1, inplace=True)
    df.drop('Sexe_candi10', axis=1, inplace=True)
    df.drop('Nom_candi10', axis=1, inplace=True)
    df.drop('Prénom_candi10', axis=1, inplace=True)
    df.drop('Voix_candi11', axis=1, inplace=True)
    df.drop('N°Panneau_candi11', axis=1, inplace=True)
    df.drop('Sexe_candi11', axis=1, inplace=True)
    df.drop('Nom_candi11', axis=1, inplace=True)
    df.drop('Prénom_candi11', axis=1, inplace=True)
    df.drop('Voix_candi12', axis=1, inplace=True)
    df.drop('N°Panneau_candi12', axis=1, inplace=True)
    df.drop('Sexe_candi12', axis=1, inplace=True)
    df.drop('Nom_candi12', axis=1, inplace=True)
    df.drop('Prénom_candi12', axis=1, inplace=True)

    df.drop('Abstentions', axis=1, inplace=True)
    df.drop('Votants', axis=1, inplace=True)
    df.drop('Blancs', axis=1, inplace=True)
    df.drop('Nuls', axis=1, inplace=True)

    df.drop('Voix_Gagnant', axis=1, inplace=True)
    df['Nom_Prenom_Gagnant'] = df['Prenom_Gagnant'] + ' ' + df['Nom_Gagnant']
    df.drop('Nom_Gagnant', axis=1, inplace=True)
    df.drop('Prenom_Gagnant', axis=1, inplace=True)

    # Suppression des données liés aux français de l'étranger
    df = df[df['Code du département'] != 'ZZ']
    df = df[df['Code du département'] != 'ZX']
    df = df[df['Code du département'] != 'ZW']
    df = df[df['Code du département'] != 'ZS']
    df = df[df['Code du département'] != 'ZP']
    df = df[df['Code du département'] != 'ZN']
    df = df[df['Code du département'] != 'ZM']

    # Modification du code departement des territoires français de l'étranger
    df['Code du département'].replace(['ZA', 'ZB', 'ZC', 'ZD'], '97', inplace=True)

    return df