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
    return df.groupby(group_col, as_index=False).sum()