import os
import numpy as np
import pandas as pd
import sys
import json
import yaml
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder

# Fonction pour charger la configuration YAML
def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Fonction pour obtenir le chemin vers la racine du projet
def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Calcul de la médiane manuelle
def manual_median(data: List[float]) -> float:
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        median = (sorted_data[n//2] + sorted_data[n//2 - 1]) / 2
    else:
        median = sorted_data[n//2]
    return median

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
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        data = pd.read_csv(filepath)
        X = data.iloc[:, 5:].select_dtypes(include=[np.number]).apply(lambda col: col.fillna(manual_median(col.dropna())))
        y = self.label_encoder.fit_transform(data['Hogwarts House'])
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

    def train_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.random.rand(X.shape[1]) * 0.01
        for _ in range(self.n_iter):
            cost, gradient = self.compute_cost(X, y, weights)
            weights -= self.eta * gradient
            if np.isnan(cost):
                break
        return weights

    def train_stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.random.rand(X.shape[1]) * 0.01
        m = len(y)
        for _ in range(self.n_iter):
            for i in range(m):
                xi = X[i:i+1]  # Un seul exemple de données
                yi = y[i:i+1]
                cost, gradient = self.compute_cost(xi, yi, weights)
                weights -= self.eta * gradient
                if np.isnan(cost):
                    break
        return weights

    def train_mini_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.random.rand(X.shape[1]) * 0.01
        m = len(y)
        for _ in range(self.n_iter):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                cost, gradient = self.compute_cost(X_batch, y_batch, weights)
                weights -= self.eta * gradient
                if np.isnan(cost):
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
        
        # Charger la configuration
        config = load_config(config_path)
        
        # Initialiser et entraîner le modèle
        model = LogisticRegressionOVR_train(config)
        weights = model.fit(sys.argv[1])
        
        # Sauvegarder les poids appris dans un fichier JSON
        with open(os.path.join(project_root, 'trained_weights.json'), 'w') as f:
            json.dump(weights, f, indent=4)
        
        print("Weights saved in 'trained_weights.json'.")




# import numpy as np
# import pandas as pd
# import sys
# import json
# from typing import List, Tuple, Dict
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# def manual_median(data: List[float]) -> float:
#     """
#     Calculer la médiane manuellement.

#     Paramètres :
#     data (list) : Liste des valeurs.

#     Retourne :
#     float : La médiane des valeurs.
#     """
#     sorted_data = sorted(data)  # Trier les données
#     n = len(sorted_data)
#     if n % 2 == 0:
#         # Si le nombre de valeurs est pair, la médiane est la moyenne des deux valeurs centrales
#         median1 = sorted_data[n//2]
#         median2 = sorted_data[n//2 - 1]
#         median = (median1 + median2) / 2
#     else:
#         # Si le nombre de valeurs est impair, la médiane est la valeur centrale
#         median = sorted_data[n//2]
#     return median

# class LogisticRegressionOVR_train:
#     def __init__(self, eta: float = 5e-5, n_iter: int = 10000, lambda_: float = 0.01) -> None:
#         """
#         Initialiser le modèle de régression logistique one-vs-all.

#         Paramètres :
#         eta (float) : Taux d'apprentissage.
#         n_iter (int) : Nombre d'itérations.
#         lambda_ (float) : Paramètre de régularisation L2.
#         """
#         self.eta = eta
#         self.n_iter = n_iter
#         self.lambda_ = lambda_
#         self.label_encoder = LabelEncoder()

#     def preprocess_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#         """
#         Prétraiter les données en chargeant le fichier CSV et en traitant les caractéristiques numériques.

#         Paramètres :
#         filepath (str) : Chemin vers le fichier CSV contenant les données.

#         Retourne :
#         tuple : Un tuple contenant les caractéristiques (X), les étiquettes (y) et les classes.
#         """
#         data = pd.read_csv(filepath)  # Charger les données à partir du fichier CSV
#         # Sélectionner les colonnes numériques à partir de la 6ème colonne et remplacer les valeurs manquantes par la médiane manuelle
#         X = data.iloc[:, 5:].select_dtypes(include=[np.number]).apply(lambda col: col.fillna(manual_median(col.dropna())))
#         y = self.label_encoder.fit_transform(data['Hogwarts House'])  # Encoder les étiquettes des classes
#         return X.values, y, self.label_encoder.classes_

#     def sigmoid(self, z: np.ndarray) -> np.ndarray:
#         """
#         Calculer la fonction sigmoïde.

#         Paramètres :
#         z (numpy.ndarray) : Valeurs d'entrée pour la fonction sigmoïde.

#         Retourne :
#         numpy.ndarray : Valeurs après application de la fonction sigmoïde.
#         """
#         return 1 / (1 + np.exp(-z))

#     def compute_cost(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[float, np.ndarray]:
#         """
#         Calculer le coût et le gradient pour la régression logistique.

#         Paramètres :
#         X (numpy.ndarray) : Caractéristiques des données.
#         y (numpy.ndarray) : Étiquettes des données.
#         weights (numpy.ndarray) : Poids du modèle.

#         Retourne :
#         tuple : Un tuple contenant le coût et le gradient.
#         """
#         m = len(y)
#         h = self.sigmoid(np.dot(X, weights))  # Prédictions de l'hypothèse
#         epsilon = 1e-5
#         # Calcul du coût avec une petite valeur epsilon pour éviter log(0)
#         cost = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
#         cost += self.lambda_ / (2 * m) * np.sum(weights**2)  # Terme de régularisation L2
#         gradient = np.dot(X.T, h - y) / m + (self.lambda_ / m) * weights  # Gradient avec régularisation L2
#         return cost, gradient

#     def train_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
#         """
#         Entraîner le modèle de régression logistique en utilisant la descente de gradient.

#         Paramètres :
#         X (numpy.ndarray) : Caractéristiques des données.
#         y (numpy.ndarray) : Étiquettes des données.

#         Retourne :
#         numpy.ndarray : Poids appris pour le modèle.
#         """
#         weights = np.random.rand(X.shape[1]) * 0.01  # Initialiser les poids aléatoirement
#         for i in range(self.n_iter):
#             cost, gradient = self.compute_cost(X, y, weights)  # Calculer le coût et le gradient
#             weights -= self.eta * gradient  # Mettre à jour les poids
#             # if i % 1000 == 0:
#             #     print(f"Iteration {i}: Cost = {cost}")  # Afficher le coût toutes les 1000 itérations
#             if np.isnan(cost):
#                 print("NaN detected in cost\nBreaking due to NaN in cost at iteration", i)
#                 break
#         return weights

#     def fit(self, filepath: str) -> Dict[str, List[float]]:
#         """
#         Entraîner le modèle pour chaque classe en utilisant l'approche one-vs-all.

#         Paramètres :
#         filepath (str) : Chemin vers le fichier CSV contenant les données d'entraînement.

#         Retourne :
#         dict : Dictionnaire contenant les poids pour chaque classe.
#         """
#         X, y, class_labels = self.preprocess_data(filepath)  # Prétraiter les données
#         weights_dict = {}
#         for i, cls_label in enumerate(class_labels):
#             # print(f"Training for class {cls_label}")
#             y_binary = (y == i).astype(int)  # Créer des étiquettes binaires pour la classe actuelle
#             weights = self.train_logistic_regression(X, y_binary)  # Entraîner le modèle pour cette classe
#             weights_dict[cls_label] = weights.tolist()  # Sauvegarder les poids pour cette classe
#         return weights_dict

# if __name__ == "__main__":
#     # Vérifier si le chemin vers le fichier de données est fourni en argument
#     if len(sys.argv) < 2:
#         print("Usage: python logreg_train.py <path_to_data>")
#     else:
#         # Initialiser et entraîner le modèle
#         model = LogisticRegressionOVR_train()
#         weights = model.fit(sys.argv[1])
#         # Sauvegarder les poids appris dans un fichier JSON
#         with open('trained_weights.json', 'w') as f:
#             json.dump(weights, f, indent=4)
#         print("Weights saved in 'trained_weights.json'.")