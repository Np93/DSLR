import os
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
import json
from dslr.models.logreg_train import LogisticRegressionOVR_train, load_config
from dslr.models.logreg_predict import LogisticRegressionOVR_predict

# Fonction pour diviser les données en deux ensembles (entraînement et test)
def split_data(input_file: str, train_file: str, test_file: str, train_size: int = 1200, test_size: int = 400) -> None:
    # Lire les données d'origine
    data = pd.read_csv(input_file)

    # Mélanger les lignes de façon aléatoire
    # data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data = data.sample(frac=1).reset_index(drop=True)

    # Séparer en deux jeux de données
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:train_size + test_size]

    # Modifier la colonne 'Index' existante pour qu'elle commence à 0 et continue jusqu'à max index
    train_data.loc[:, 'Index'] = range(0, len(train_data))
    test_data.loc[:, 'Index'] = range(0, len(test_data))

    # Sauvegarder les données dans les fichiers spécifiés
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

# Pytest pour la division, l'entraînement, la prédiction et la vérification des résultats
@pytest.fixture
def setup_data():
    # Diviser les données d'origine en fichiers d'entraînement et de test
    split_data('data/dataset_train.csv', 'data/train_data.csv', 'data/test_data.csv')

@pytest.fixture
def train_model(setup_data):
    # Charger la configuration
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    config_path = os.path.join(project_root, "config.yaml")
    config = load_config(config_path)

    # Entraîner le modèle sur les données d'entraînement
    model = LogisticRegressionOVR_train(config)
    weights = model.fit('data/train_data.csv')

    # Sauvegarder les poids dans un fichier JSON
    with open('trained_weights.json', 'w') as f:
        json.dump(weights, f, indent=4)

@pytest.fixture
def predict(train_model):
    # Charger les données de test
    hptest = pd.read_csv('data/test_data.csv', index_col="Index")

    # Charger les poids du modèle
    with open('trained_weights.json', 'r') as f:
        weights_dict = json.load(f)

    # Convertir les poids en une liste de tuples (poids, classe)
    weights = [(np.array(v), k) for k, v in weights_dict.items()]

    # Initialiser le modèle de prédiction
    model = LogisticRegressionOVR_predict()
    
    # Faire les prédictions pour les données de test
    predicts = model.predict(hptest, weights)

    # Sauvegarder les prédictions dans un fichier CSV
    houses = pd.DataFrame({'Index': range(len(predicts)), 'Hogwarts House': predicts})
    houses.to_csv('houses.csv', index=False)

# Test principal pour vérifier la précision des prédictions
def test_accuracy(predict):
    # Charger les données de prédiction
    pred_df = pd.read_csv('houses.csv')

    # Charger les vraies données
    true_df = pd.read_csv('data/test_data.csv')

    # Fusionner les prédictions avec les vraies étiquettes en utilisant l'index comme clé
    merged_df = pd.merge(true_df, pred_df, on="Index", suffixes=('_true', '_pred'))

    # Calculer la précision
    accuracy = accuracy_score(merged_df['Hogwarts House_true'], merged_df['Hogwarts House_pred'])

    # Si la précision est inférieure à 99%, imprimer combien de prédictions sont correctes
    if accuracy < 0.99:
        correct_predictions = (merged_df['Hogwarts House_true'] == merged_df['Hogwarts House_pred']).sum()
        total = len(merged_df)
        print(f"Nombre de prédictions correctes : {correct_predictions} sur {total}")

    # Test si la précision est égale ou supérieure à 99%
    assert accuracy >= 0.99, f"La précision est de {accuracy:.2%}. Elle est inférieure à 99%."