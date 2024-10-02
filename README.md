# DSLR
Création d'un algorithme de régression logistique

## Table des Matières
1. [Installation](#installation)
2. [Utilisation](#utilisation)
3. [Fonctionnalités](#fonctionnalités)
4. [Contributeurs](#contributeurs)
5. [Licence](#licence)

## Installation
Instructions pour installer le projet : rien de speciale juste git clone et c'est ok ok.

```sh
git clone https://github.com/Np93/DSLR.git
cd votre_projet
```

## Utilisation

Utiliser les commande suivante pour tester chaque fonctionnaliter du projets, il faut bien sur entrainer le modele avant d'avoir une prediction.

```sh
# pour installer les dependance de poetry commencer par:
poetry install

# ensuite:
poetry run python3 dslr/scripts/describe.py data/dataset_train.csv
poetry run python3 dslr/scripts/histogram.py
poetry run python3 dslr/scripts/pair_plot.py
poetry run python3 dslr/scripts/scatter_plot.py

poetry run python3 dslr/models/logreg_train.py data/dataset_train.csv
poetry run python3 dslr/models/logreg_predict.py data/dataset_train.csv trained_weights.json
```

## Fonctionnalités

1. Entraînement d'un modèle de régression logistique.
2. Prédiction des maisons de Poudlard à partir des données de test.
3. ...

## Contributeurs

- Np93
- ...

## Licence

## Remerciements

merci a j pour son aide et sont travail au projet