import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_data(filepath) -> any:
    """
    Charger les données à partir d'un fichier CSV.

    Paramètres :
    filepath (str) : Chemin vers le fichier CSV.

    Retourne :
    DataFrame : Données chargées sous forme de DataFrame Pandas.
    """
    return pd.read_csv(filepath)

def preprocess_data(dataset) -> any:
    """
    Modifié
    Prétraiter les données en supprimant les colonnes non numériques et en gérant les valeurs manquantes.

    Paramètres :
    dataset (DataFrame) : DataFrame contenant les données à prétraiter.

    Retourne :
    DataFrame : DataFrame prétraitée avec uniquement les colonnes numériques et sans valeurs manquantes ET les noms de house.
    """
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


def plot_pair_matrix(dataset, hue_column , label_fontsize=10, title_fontsize=12) -> None:
    """
    Tracer une matrice de graphiques de dispersion pour chaque paire de colonnes numériques, et en diagonal les histogrammes.

    Paramètres :
    dataset (DataFrame) : DataFrame contenant les données à tracer.
    hue_columns : La colonne target (Hogwart house)
    label_fontsize : la taille des caractères des labels
    title_fontsize : la taille des caractères des titres

    Retourne :
    None
    """

    house_colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow'
    }

    # find and get the numeric columns (the course)
    columns = dataset.select_dtypes(include=[float, int]).columns

    #creates subplot
    n = len(columns)
    fig, axs = plt.subplots(n,n, figsize=(12,12), num="Pair Plot feature1 vs feature2")

    #iterate through each pair of feature
    for i, feature1 in enumerate(columns):
        for j, feature2 in enumerate(columns):
            if i == j:
                #draw histogram
                axs[i,j].hist(dataset[feature1], color='lightgray', edgecolor='black', alpha=0.7)
            else:
                # create a bit mask for selecting specific houses (for coloring purpose)
                for house in dataset[hue_column].unique():
                    subset = dataset[dataset[hue_column] == house]
                    axs[i,j].scatter(subset[feature1], subset[feature2], label=house, color=house_colors[house], alpha=0.3)

            #hide ticks
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    # writing the outer line and column (and managing long strings):
    for i, feature1 in enumerate(columns):
        if feature1 == "Defense Against the Dark Arts":
            axs[i, 0].set_ylabel("Defense Against\nthe Dark Arts",  fontsize=label_fontsize)
        elif feature1 == "Care of Magical Creatures":
            axs[i, 0].set_ylabel("Care of \nMagical Creatures",  fontsize=label_fontsize)
        else:
            axs[i, 0].set_ylabel(feature1,  fontsize=label_fontsize)

    for j, feature2 in enumerate(columns):
        if feature2 == "Defense Against the Dark Arts":
            axs[0, j].set_title("Defense Against\nthe Dark Arts", fontsize=title_fontsize)
        elif feature2 == "Care of Magical Creatures":
            axs[0, j].set_title("Care of \nMagical Creatures",  fontsize=label_fontsize)
        else:
            axs[0, j].set_title(feature2,  fontsize=label_fontsize)

    # Create a legend for the scatter plots
    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1),title=hue_column, fontsize=label_fontsize)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9)
    plt.show()

def main():
    df = load_data("data/dataset_train.csv")
    cleaned_data = preprocess_data(df)
    if not cleaned_data.empty:
        plot_pair_matrix(cleaned_data, 'Hogwarts_House',label_fontsize=6, title_fontsize=6)
    else:
        print("No valid data available for plotting after preprocessing.")


if __name__ == "__main__":
    main()
