import os
import numpy as np
import pandas as pd
import sys
import json
import yaml
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder
import math
from dslr.utils import load_data

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def manual_median(data: List[float]) -> float:
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        median = (sorted_data[n//2] + sorted_data[n//2 - 1]) / 2
    else:
        median = sorted_data[n//2]
    return median

def manual_mean(data: List[float]) -> float:
    return sum(data) / len(data) if len(data) > 0 else 0

def manual_std(data: List[float], mean: float) -> float:
    variance = sum((x - mean) ** 2 for x in data) / len(data) if len(data) > 0 else 0
    return math.sqrt(variance)

def standardize_column(data: List[float], mean: float, std: float) -> List[float]:
    if std == 0:
        return data
    return [(x - mean) / std for x in data]

class LogisticRegressionOVR_train:
    def __init__(self, config: Dict) -> None:
        """
        Initialiser le modèle de régression logistique one-vs-all.

        Paramètres :
        config (Dict) : Dictionnaire de configuration avec les hyperparamètres.
        """
        self.eta = config['eta']
        self.n_iter = config['n_iter']
        self.lambda_ = config['lambda_']
        self.gradient_type = config['gradient_type']
        self.batch_size = config.get('batch_size', 32)  # Taille par défaut pour mini-batch GD
        self.early_stopping = config.get('early_stopping', False)
        self.patience = config.get('patience', 10)
        self.tolerance = config.get('tolerance', 1e-4)
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        data = load_data(filepath)
        X = data.iloc[:, 5:].select_dtypes(include=[np.number]).apply(
            lambda col: col.fillna(manual_median(col.dropna()))
        )
        y = self.label_encoder.fit_transform(data['Hogwarts House'])
        for col in X.columns:
            column_data = X[col].tolist()
            mean = manual_mean(column_data)
            std = manual_std(column_data, mean)
            X[col] = standardize_column(column_data, mean, std)
        return X.values, y, list(self.label_encoder.classes_)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[float, np.ndarray]:
        m = len(y)
        h = self.sigmoid(np.dot(X, weights))
        epsilon = 1e-5
        cost = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        cost += self.lambda_ / (2 * m) * np.sum(weights**2)  # Régularisation L2
        gradient = np.dot(X.T, h - y) / m + (self.lambda_ / m) * weights
        return cost, gradient

    def early_stopping_check(self, cost_history: List[float], patience: int, tolerance: float) -> bool:
        """
        Vérifier si Early Stopping doit être activé.
        Si le coût n'a pas diminué de manière significative pendant 'patience' itérations, arrêter l'entraînement.
        """
        if len(cost_history) > patience:
            recent_costs = cost_history[-patience:]
            tolerance = float(tolerance)
            if np.abs(recent_costs[-1] - recent_costs[0]) < tolerance:
                return True
        return False

    def train_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        weights = np.random.rand(X.shape[1]) * 0.01
        cost_history = []
        for _ in range(self.n_iter):
            cost, gradient = self.compute_cost(X, y, weights)
            weights -= self.eta * gradient
            cost_history.append(cost)
            if np.isnan(cost):
                break
            if self.early_stopping and self.early_stopping_check(cost_history, self.patience, self.tolerance):
                print(f"Early stopping activated at iteration {_}")
                break
        return weights

    def train_stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        weights = np.random.rand(X.shape[1]) * 0.01
        m = len(y)
        cost_history = []
        for _ in range(self.n_iter):
            for i in range(m):
                xi = X[i:i+1]
                yi = y[i:i+1]
                cost, gradient = self.compute_cost(xi, yi, weights)
                weights -= self.eta * gradient
                cost_history.append(cost)
                if np.isnan(cost):
                    break
                if self.early_stopping and self.early_stopping_check(cost_history, self.patience, self.tolerance):
                    print(f"Early stopping activated at iteration {_}")
                    break
        return weights

    def train_mini_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        weights = np.random.rand(X.shape[1]) * 0.01
        m = len(y)
        cost_history = []
        for _ in range(self.n_iter):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                cost, gradient = self.compute_cost(X_batch, y_batch, weights)
                weights -= self.eta * gradient
                cost_history.append(cost)
                if np.isnan(cost):
                    break
                if self.early_stopping and self.early_stopping_check(cost_history, self.patience, self.tolerance):
                    print(f"Early stopping activated at iteration {_}")
                    break
        return weights

    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.gradient_type == 'batch':
            return self.train_batch_gradient_descent(X, y)
        elif self.gradient_type == 'sgd':
            return self.train_stochastic_gradient_descent(X, y)
        elif self.gradient_type == 'mini-batch':
            return self.train_mini_batch_gradient_descent(X, y)
        else:
            raise ValueError("Invalid gradient type. Choose 'batch', 'sgd', or 'mini-batch'.")

    def fit(self, filepath: str) -> Dict[str, List[float]]:
        X, y, class_labels = self.preprocess_data(filepath)
        weights_dict = {}
        for i, cls_label in enumerate(class_labels):
            y_binary = (y == i).astype(int)
            weights = self.train_logistic_regression(X, y_binary)
            weights_dict[cls_label] = weights.tolist()
        return weights_dict

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py <path_to_data>")
    else:
        project_root = get_project_root()
        config_path = os.path.join(project_root, "config.yaml")
        config = load_config(config_path)
        model = LogisticRegressionOVR_train(config)
        weights = model.fit(sys.argv[1])

        with open(os.path.join(project_root, 'trained_weights.json'), 'w') as f:
            json.dump(weights, f, indent=4)

        print("Weights saved in 'trained_weights.json'.")
