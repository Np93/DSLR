import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from dslr.utils import manual_count, manual_mean, manual_std
# from describe import manual_mean, manual_std, manual_count  # Import specific functions

def standardize_data(dataset, columns) -> any:
    """
    Standardise les colonnes numériques présentes dans le set de donnée dataset,
    de sorte que la moyenne µ = 0 et l'écart type (standard deviation) de 1

    Paramètres :
    dataset (DataFrame) : DataFrame des observations
    columns : la liste des colonnes à standardiser

    Retourne :
    standardized_data : l'ensemble dataset et les données standardisées

    """
    standardized_data = dataset.copy()

    for col in columns:
        mean = manual_mean(dataset[col])
        std = manual_std(dataset[col], mean)
        standardized_data[col] = (dataset[col] - mean) / std

    return standardized_data

def Pearson_correlation_coefficient(dataset_X, dataset_Y) -> any:
    """
    Calcule le coefficient de correlation entre X et Y comme le fait np.corrcoef
    

    Paramètres :
    dataset_X : les données X
    dataset_Y : les données Y

    Remarque: 
    Sxx, Sxy et Syy sont des fonctions standards en statistiques.
    Sxy est la somme des produits des différences entre x et sa moyenne et entre y et sa moyenne

    Retourne :
    corr, le coefficient de correlation 
    """

    X_mean = manual_mean(dataset_X)
    Y_mean = manual_mean(dataset_Y)
    n = manual_count(dataset_X) - 1
    
    if np.std(dataset_X) == 0 or np.std(dataset_Y) == 0:
        return np.nan  # Return NaN if variance is zero (no correlation can be computed)

    Sxx, Sxy, Syy = 0.0, 0.0, 0.0
    for x in dataset_X:
        Sxx += (x - X_mean) ** 2
    Sxx = Sxx / n
    for y in dataset_Y:
        Syy += (y - Y_mean) ** 2
    Syy = Syy / n
    for x, y in zip(dataset_X,dataset_Y):
        Sxy += (x - X_mean)*(y - Y_mean)
    Sxy = Sxy / n
    corr = Sxy/np.sqrt(Sxx*Syy)
    return corr

def plot_similar_features(dataset, feature1, feature2, max_corr) -> any:
    """
    Crée un scatter plot des deux cours les plus semblables d'après leur coefficient de correlations, ceci en choississnant
    le coefficient le plus proche de 1. De plus les données sont standardisées et affichées en comparaison avec la fonction identité. 
    

    Paramètres :
    dataset : l'ensemble des données dataset
    feature1, feature2 : les 2 cours les plus proches
    max_corr, le coefficient de correlation

    Retourne :
    Rien
    """
    # Standardize the two features before plotting
    standardized_dataset = standardize_data(dataset, [feature1, feature2])
    valid_data = standardized_dataset.dropna(subset=[feature1, feature2])

    plt.figure(figsize=(10, 10), num=f"Scatter plot between {feature1} and {feature2}")
    # Assign a color to each Hogwarts House
    house_colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow'
    }

    # Map the houses to colors for the scatter plot
    colors = valid_data['Hogwarts_House'].map(house_colors)

    # Plot the scatter plot with colors based on Hogwarts House
    plt.scatter(valid_data[feature1], valid_data[feature2], c=colors, alpha=0.6, edgecolor='k')

    # Add plot titles and labels
    plt.title(f'Scatter Plot between {feature1} and {feature2} (Standardized) with correlation {max_corr:.2f}')
    plt.xlabel(f'Standardized {feature1}')
    plt.ylabel(f'Standardized {feature2}')

    # Set x and y axis to have the same scale
    plt.axis('equal')  # This ensures the aspect ratio is 1:1

    # Add the identity line (y = x)
    min_value = min(valid_data[feature1].min(), valid_data[feature2].min())
    max_value = max(valid_data[feature1].max(), valid_data[feature2].max())
    plt.xlim(-3, 3)

    # Set the y-axis limits (optional, depending on your data)
    plt.ylim(-3, 3)
    plt.plot([min_value, max_value], [min_value, max_value], color='black', linestyle='--', label='y = x')

    # Add a legend for the houses
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=house_colors[house], markersize=10) for house in house_colors]
    labels = list(house_colors.keys())
    plt.legend(handles, labels, title="Hogwarts House")
    plt.show()

def load_data(filepath) -> any:
    return pd.read_csv(filepath)

def numeric_features(dataset:any) -> any:
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols

def preprocess_data_for_correlation(dataset, cols, min_non_na=100) -> tuple[any, list]:
    valid_cols = [col for col in cols if dataset[col].count() >= min_non_na]
    return dataset.dropna(subset=valid_cols), valid_cols


def preprocess_data(dataset) -> any:
    # Filtrer les colonnes numériques
    numeric_dataset = dataset.select_dtypes(include=[float, int])
    # Add the 'Hogwarts House' column back to the DataFrame
    categorical_column = dataset['Hogwarts House']
    # Supprimer les colonnes où toutes les valeurs sont NaN
    numeric_dataset = numeric_dataset.dropna(axis=1, how='all')
    # Supprimer les lignes avec des valeurs NaN restantes

    # Combine the numeric data with the categorical 'Hogwarts House'
    combined_data = numeric_dataset.assign(Hogwarts_House=categorical_column)
    combined_data.dropna(inplace=True)
    return combined_data




def calculate_correlations(dataset, numeric_cols) -> any:
    dataset_clean, valid_numeric_cols = preprocess_data_for_correlation(dataset, numeric_cols)
    correlations = pd.DataFrame(index=valid_numeric_cols, columns=valid_numeric_cols)
    for col1 in valid_numeric_cols:
        for col2 in valid_numeric_cols:
            if col1 != col2:
                correlation = Pearson_correlation_coefficient(dataset_clean[col1], dataset_clean[col2])
                correlations.at[col1, col2] = correlation if np.isfinite(correlation) else np.nan
    return correlations



def get_max_correlation(correlations) -> any:
    # Unstack the correlation matrix to turn it into a long format
    corr_unstacked = correlations.unstack()

    # Remove NaN values and self-correlations (where feature1 == feature2)
    corr_filtered = corr_unstacked.dropna().loc[lambda x: x != 1]

    # Get the maximum correlation value
    max_corr_value = corr_filtered.max()

    return max_corr_value

def main():
    dataset = load_data("data/dataset_train.csv")
    numeric_cols = numeric_features(dataset)
    cleaned_data = preprocess_data(dataset)
    # Calculer les corrélations entre les caractéristiques numériques
    correlations = calculate_correlations(dataset, numeric_cols)
    if not correlations.empty:
        # Identifier la paire de caractéristiques la plus corrélée
        most_correlated_pair = correlations.unstack().dropna().idxmax()
        # Tracer les caractéristiques les plus corrélées
        max_corr = get_max_correlation(correlations)
        plot_similar_features(cleaned_data, *most_correlated_pair, max_corr)

    else:
        print("No sufficiently correlated features found to plot.")
        


if __name__ == "__main__":
    main()