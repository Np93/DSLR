import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(filepath):
    """
    Charger les données à partir d'un fichier CSV.

    Paramètres :
    filepath (str) : Chemin vers le fichier CSV.

    Retourne :
    DataFrame : Données chargées sous forme de DataFrame Pandas.
    """
    return pd.read_csv(filepath)

def numeric_features(dataset):
    """
    Identifier les caractéristiques numériques dans le jeu de données.

    Paramètres :
    dataset (DataFrame) : Jeu de données contenant les caractéristiques.

    Retourne :
    list : Liste des noms des colonnes numériques.
    """
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols

def preprocess_data_for_correlation(dataset, cols, min_non_na=100):
    """
    Prétraiter les données en gardant uniquement les colonnes avec suffisamment de valeurs non-NA.

    Paramètres :
    dataset (DataFrame) : Jeu de données à prétraiter.
    cols (list) : Liste des colonnes à vérifier.
    min_non_na (int) : Nombre minimum de valeurs non-NA requis pour conserver une colonne.

    Retourne :
    DataFrame, list : Jeu de données prétraité et liste des colonnes valides.
    """
    valid_cols = [col for col in cols if dataset[col].count() >= min_non_na]
    return dataset.dropna(subset=valid_cols), valid_cols

def calculate_correlations(dataset, numeric_cols):
    """
    Calculer les corrélations entre les caractéristiques numériques en utilisant des données nettoyées.

    Paramètres :
    dataset (DataFrame) : Jeu de données à utiliser pour le calcul.
    numeric_cols (list) : Liste des colonnes numériques.

    Retourne :
    DataFrame : Matrice de corrélation.
    """
    dataset_clean, valid_numeric_cols = preprocess_data_for_correlation(dataset, numeric_cols)
    correlations = pd.DataFrame(index=valid_numeric_cols, columns=valid_numeric_cols)
    for col1 in valid_numeric_cols:
        for col2 in valid_numeric_cols:
            if col1 != col2:
                correlation = np.corrcoef(dataset_clean[col1], dataset_clean[col2])[0, 1]
                correlations.at[col1, col2] = correlation if np.isfinite(correlation) else np.nan
    return correlations

def plot_similar_features(dataset, feature1, feature2):
    """
    Tracer les deux caractéristiques les plus similaires.

    Paramètres :
    dataset (DataFrame) : Jeu de données contenant les caractéristiques à tracer.
    feature1 (str) : Nom de la première caractéristique.
    feature2 (str) : Nom de la deuxième caractéristique.

    Retourne :
    None
    """
    plt.figure(figsize=(10, 6))
    # Préparer les données en excluant les valeurs manquantes pour les deux caractéristiques
    valid_data = dataset.dropna(subset=[feature1, feature2])
    
    # Tracer feature1 en rouge et feature2 en bleu
    plt.scatter(valid_data[feature1], valid_data[feature2], c=np.where(valid_data[feature1] > valid_data[feature2], 'red', 'blue'), alpha=0.5)
    
    plt.title(f'Scatter Plot between {feature1} and {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.colorbar(label='Red: feature1 > feature2, Blue: feature2 >= feature1')
    plt.show()

def main():
    # Charger les données à partir du fichier CSV
    dataset = load_data("data/dataset_train.csv")
    # Identifier les colonnes numériques
    numeric_cols = numeric_features(dataset)
    # Calculer les corrélations entre les caractéristiques numériques
    correlations = calculate_correlations(dataset, numeric_cols)
    if not correlations.empty:
        # Identifier la paire de caractéristiques la plus corrélée
        most_correlated_pair = correlations.unstack().dropna().idxmax()
        # Tracer les caractéristiques les plus corrélées
        plot_similar_features(dataset, *most_correlated_pair)
    else:
        print("No sufficiently correlated features found to plot.")

if __name__ == "__main__":
    main()