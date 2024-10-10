import os
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
import json
from typing import List, Dict
from dslr.models.logreg_train import LogisticRegressionOVR_train, load_config
from dslr.models.logreg_predict import LogisticRegressionOVR_predict

def split_data(input_file: str, train_file: str, test_file: str, train_size: int = 1200, test_size: int = 400) -> None:
    data = pd.read_csv(input_file)
    # Mélanger les lignes de façon aléatoire
    # data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data = data.sample(frac=1).reset_index(drop=True)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:train_size + test_size]
    train_data.loc[:, 'Index'] = range(0, len(train_data))
    test_data.loc[:, 'Index'] = range(0, len(test_data))
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

@pytest.fixture
def setup_data():
    split_data('data/dataset_train.csv', 'data/train_data.csv', 'data/test_data.csv')

@pytest.fixture
def train_model(setup_data):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    config_path = os.path.join(project_root, "config.yaml")
    config = load_config(config_path)
    model = LogisticRegressionOVR_train(config)
    weights = model.fit('data/train_data.csv')
    with open('trained_weights.json', 'w') as f:
        json.dump(weights, f, indent=4)

@pytest.fixture
def predict(train_model):
    hptest = pd.read_csv('data/test_data.csv', index_col="Index")
    with open('trained_weights.json', 'r') as f:
        weights_dict = json.load(f)
    weights = [(np.array(v), k) for k, v in weights_dict.items()]
    model = LogisticRegressionOVR_predict()
    predicts = model.predict(hptest, weights)
    houses = pd.DataFrame({'Index': range(len(predicts)), 'Hogwarts House': predicts})
    houses.to_csv('houses.csv', index=False)

def analyze_weights(weights_dict: Dict[str, List[float]], feature_names: List[str]) -> None:
    for class_label, weights in weights_dict.items():
        print(f"\nHouse '\033[38;5;214m{class_label}\033[0m' :")
        # weights_with_features = list(zip(feature_names, weights))
        weights_with_features = list(zip(feature_names, weights[1:]))
        weights_with_features_sorted = sorted(weights_with_features, key=lambda x: abs(x[1]), reverse=True)
        print("Caractéristiques les plus influentes (en termes de poids) :")
        for feature, weight in weights_with_features_sorted[:5]:
            # print(f"{feature}: {weight:.4f}")
            print(f"{feature}: \033[32m{weight:.4f}\033[0m")
        print("\nCaractéristiques les moins influentes (poids proches de 0) :")
        for feature, weight in weights_with_features_sorted[-5:]:
            # print(f"{feature}: {weight:.4f}")
            print(f"{feature}: \033[31m{weight:.4f}\033[0m")

def colorize_dataframe(df):
    colored_df = df.copy()
    colored_df['Hogwarts House_true'] = df['Hogwarts House_true'].apply(lambda x: f"\033[32m{x}\033[0m")  # Vert
    colored_df['Hogwarts House_pred'] = df['Hogwarts House_pred'].apply(lambda x: f"\033[31m{x}\033[0m")  # Rouge
    return colored_df

def test_accuracy(predict):
    pred_df = pd.read_csv('houses.csv')
    true_df = pd.read_csv('data/test_data.csv')
    merged_df = pd.merge(true_df, pred_df, on="Index", suffixes=('_true', '_pred'))
    accuracy = accuracy_score(merged_df['Hogwarts House_true'], merged_df['Hogwarts House_pred'])
    if accuracy < 1.0:
        incorrect_predictions = merged_df[merged_df['Hogwarts House_true'] != merged_df['Hogwarts House_pred']]
        correct_predictions = (merged_df['Hogwarts House_true'] == merged_df['Hogwarts House_pred']).sum()
        total = len(merged_df)
        incorrect_predictions = incorrect_predictions.sort_values(by='Hogwarts House_true')
        colored_df = colorize_dataframe(incorrect_predictions)
        print(f"Nombre de prédictions correctes : {correct_predictions} sur {total}")
        print("\nErreurs de prédictions :")
        print(colored_df[['Index', 'Hogwarts House_true', 'Hogwarts House_pred']])
    with open('trained_weights.json', 'r') as f:
        weights_dict = json.load(f)
    feature_names = pd.read_csv('data/train_data.csv').columns[6:]  # À partir de la 6ème
    analyze_weights(weights_dict, feature_names)
    assert accuracy >= 1.0, f"La précision est de {accuracy:.2%}. Elle est inférieure à 100%."