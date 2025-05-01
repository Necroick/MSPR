# Choix de l'entrainement des modèles d'IA
## Traitement des donnée socio-économiques
Création de variable 'Diff' expirmant l'évolution des données, afin d'expirmer plus explicitement l'évolution des situations sociales (plutôt que les situations sociales elle-même)
Les données 'Taux de chômage' et 'Création' sont les moyenne pour le pays entier, alors on ne passe que la valeur 'Diff'

## Création des ensembles d'entrainement
On commence par définir la cible et les features (cible : donnée que l'on souhaite prédir, features : donnée qui devront influée sur la prédiction).
Ensuite, on sépare le dataset en ensemble d'entrainement

## Entrainement du modèle - Random Forest
Choix du modèle : réputé pour avoir de bonne performances pour la classification avec des données structurées complexes
Accuracy avec 100 arbres : 0.6507
Accuracy avec 125 arbres : 0.6521
Accuracy avec 150 arbres : 0.6497
Accuracy avec 200 arbres : 0.6487
Accuracy avec 250 arbres : 0.6518
Accuracy avec 300 arbres : 0.6517
On a donc choisi 125 arbres, qui semble être le plateau

## Estimation des données pour 2027 en Auvergne-Rhône-Alpes
Estimation des données socio-économiques pour 2027 par extrapolation

## Prédiction des résultats pour 2027 en Auvergne-Rhône-Alpes - Random Forest