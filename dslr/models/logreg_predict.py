import numpy as np
import pandas as pd
import json
import sys
from typing import List, Tuple
from dslr.utils import load_data

class LogisticRegressionOVR_predict:
    def __init__(self, eta: float = 5e-5, n_iter: int = 30000) -> None:
        """
        Initialiser le modèle de régression logistique one-vs-all.

        Paramètres :
        eta (float) : Taux d'apprentissage.
        n_iter (int) : Nombre d'itérations.
        """
        self.eta = eta
        self.n_iter = n_iter

    def _scaling(self, X: np.ndarray) -> np.ndarray:
        """
        Standardiser les caractéristiques.

        Paramètres :
        X (numpy.ndarray) : Les données à standardiser.

        Retourne :
        numpy.ndarray : Les données standardisées.
        """
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def _processing(self, hptest: pd.DataFrame) -> np.ndarray:
        """
        Sélectionner et traiter les caractéristiques numériques à partir de la 6ème colonne.

        Paramètres :
        hptest (pandas.DataFrame) : Les données de test.

        Retourne :
        numpy.ndarray : Les caractéristiques traitées.
        """
        hp_features = hptest.iloc[:, 5:].select_dtypes(include=[np.number])
        hp_features = hp_features.ffill()
        hp_features = self._scaling(hp_features)
        return hp_features

    def _predict_one(self, x: np.ndarray, weights: List[Tuple[np.ndarray, int]]) -> int:
        """
        Calculer les prédictions pour une instance.

        Paramètres :
        x (numpy.ndarray) : Les caractéristiques de l'instance.
        weights (list) : Les poids des modèles.

        Retourne :
        int : La classe prédite.
        """
        return max((np.dot(x, w), c) for w, c in weights)[1]

    def predict(self, hptest: pd.DataFrame, weights: List[Tuple[np.ndarray, int]]) -> List[int]:
        """
        Prédire les classes pour les données de test.

        Paramètres :
        hptest (pandas.DataFrame) : Les données de test.
        weights (list) : Les poids des modèles.

        Retourne :
        list : Les classes prédites pour chaque instance.
        """
        X = self._processing(hptest)
        X = np.insert(X, 0, 1, axis=1)
        predictions = [self._predict_one(i, weights) for i in X]
        return predictions

if __name__ == "__main__":
    load_data(sys.argv[1])
    hptest = pd.read_csv(sys.argv[1], index_col="Index")
    with open(sys.argv[2], 'r') as f:
        weights_dict = json.load(f)
    weights = [(np.array(v), k) for k, v in weights_dict.items()]
    model = LogisticRegressionOVR_predict()
    predicts = model.predict(hptest, weights)
    print("Predictions saved to 'houses.csv':", predicts)
    houses = pd.DataFrame({'Index': range(len(predicts)), 'Hogwarts House': predicts})
    houses.to_csv('houses.csv', index=False)
