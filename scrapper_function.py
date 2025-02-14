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