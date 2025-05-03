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

## Entrainement du modèle - Gradient Boosting
Choix du modèle : 
Accuracy avec 75 estimateurs : 0.6601
Accuracy avec 100 estimateurs : 0.6631
Accuracy avec 125 estimateurs : 0.6663
Accuracy avec 150 estimateurs : 0.6654
On a donc choisi 100 estimateurs, augementer n'apportant pas beaucoup d'accuracy et augmentant le temps de calcul

## Entrainement du modèle - K-Means
Choix du modèle : 
Accuracy pour 15 cluster : 0.5861
Accuracy pour 20 cluster : 0.5839
Accuracy pour 50 cluster : 0.6109
Accuracy pour 100 cluster : 0.6439
Accuracy pour 250 cluster : 0.6541
Augmenter le nombre de cluster semble augmenter l'accuraccy de manière exponentielle, on a donc choisis 250 cluster qui parraissait un bon compromis entre précision et temps de calcul

## Estimation des données pour 2027 en Auvergne-Rhône-Alpes
Estimation des données socio-économiques pour 2027 par extrapolation

## Prédiction des résultats pour 2027 en Auvergne-Rhône-Alpes - Random Forest
Marine LE PEN         2158
Emmanuel MACRON       1464
Jean-Luc MELENCHON     394
Jean LASSALLE            3
Valerie PECRESSE         2
Eric ZEMMOUR             2
Nathalie ARTHAUD         2
Yannick JADOT            1

## Prédiction des résultats pour 2027 en Auvergne-Rhône-Alpes - Gradient Boosting
Marine LE PEN         2690
Emmanuel MACRON       1029
Jean-Luc MELENCHON     287
Eric ZEMMOUR             7
Valerie PECRESSE         7
Fabien ROUSSEL           2
Jean LASSALLE            2
Nathalie ARTHAUD         2

## Prédiction des résultats pour 2027 en Auvergne-Rhône-Alpes - K-Means
Marine LE PEN         3298
Emmanuel MACRON        383
Jean-Luc MELENCHON     345