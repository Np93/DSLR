import pandas as pd
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

def preprocess_data(dataset):
    """
    Prétraiter les données en supprimant les colonnes non numériques et en gérant les valeurs manquantes.

    Paramètres :
    dataset (DataFrame) : DataFrame contenant les données à prétraiter.

    Retourne :
    DataFrame : DataFrame prétraitée avec uniquement les colonnes numériques et sans valeurs manquantes.
    """
    # Filtrer les colonnes numériques
    numeric_dataset = dataset.select_dtypes(include=[float, int])
    # Supprimer les colonnes où toutes les valeurs sont NaN
    numeric_dataset = numeric_dataset.dropna(axis=1, how='all')
    # Supprimer les lignes avec des valeurs NaN restantes
    numeric_dataset.dropna(inplace=True)
    return numeric_dataset

def plot_pair_matrix(dataset, columns=None):
    """
    Tracer une matrice de graphiques de dispersion pour chaque paire de colonnes numériques.

    Paramètres :
    dataset (DataFrame) : DataFrame contenant les données à tracer.
    columns (list) : Liste des colonnes à inclure dans la matrice de graphiques. Si None, toutes les colonnes sont utilisées.

    Retourne :
    None
    """
    if columns is None:
        columns = dataset.columns
    # Créer une figure avec des sous-graphiques pour chaque paire de colonnes
    fig, axs = plt.subplots(len(columns), len(columns), figsize=(12, 12))
    for i, feature1 in enumerate(columns):
        for j, feature2 in enumerate(columns):
            if i == j:
                # Si les indices sont égaux, afficher le nom de la caractéristique
                axs[i, j].text(0.5, 0.5, feature1, ha='center')
            else:
                # Sinon, tracer un graphique de dispersion
                axs[i, j].scatter(dataset[feature1], dataset[feature2], alpha=0.5, marker='.')
            # Supprimer les ticks des axes pour une meilleure lisibilité
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.show()

def main():
    # Charger les données à partir du fichier CSV
    df = load_data("data/dataset_train.csv")
    # Prétraiter les données
    cleaned_data = preprocess_data(df)
    if not cleaned_data.empty:
        # Tracer la matrice de graphiques de dispersion
        plot_pair_matrix(cleaned_data)
    else:
        print("No valid data available for plotting after preprocessing.")

if __name__ == "__main__":
    main()