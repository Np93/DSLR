import os
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
import json
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

def test_accuracy(predict):
    pred_df = pd.read_csv('houses.csv')
    true_df = pd.read_csv('data/test_data.csv')
    merged_df = pd.merge(true_df, pred_df, on="Index", suffixes=('_true', '_pred'))
    accuracy = accuracy_score(merged_df['Hogwarts House_true'], merged_df['Hogwarts House_pred'])
    if accuracy < 1.0:
        correct_predictions = (merged_df['Hogwarts House_true'] == merged_df['Hogwarts House_pred']).sum()
        total = len(merged_df)
        print(f"Nombre de prédictions correctes : {correct_predictions} sur {total}")
    assert accuracy >= 1.0, f"La précision est de {accuracy:.2%}. Elle est inférieure à 99%."