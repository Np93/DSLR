import pandas as pd
import sys
import matplotlib.pyplot as plt
from dslr.utils import manual_min, manual_max, calculate_bins, assign_data_to_bins

def load_data(filepath):
    """
    Charger les données à partir d'un fichier CSV.

    Paramètres :
    filepath (str) : Chemin vers le fichier CSV.

    Retourne :
    DataFrame : Données chargées sous forme de DataFrame Pandas.
    """
    return pd.read_csv(filepath)

def get_house_data(df, course):
    """
    Collecter les données pour chaque maison pour un cours spécifique.

    Paramètres :
    df (DataFrame) : DataFrame contenant les données.
    course (str) : Nom du cours pour lequel collecter les données.

    Retourne :
    dict : Dictionnaire contenant les données pour chaque maison.
    """
    houses = df['Hogwarts House'].dropna().unique()
    house_data = {house: df[(df['Hogwarts House'] == house) & df[course].notna()][course].tolist() for house in houses}
    return house_data

# def manual_sum(data):
#     """
#     Calculer la somme d'une liste de nombres.

#     Paramètres :
#     data (list) : Liste des nombres.

#     Retourne :
#     float : Somme des nombres dans la liste.
#     """
#     return sum(data) if data else 0

# def manual_min(data):
#     """
#     Trouver la valeur minimale dans les données manuellement.

#     Paramètres :
#     data (list) : Liste des nombres.

#     Retourne :
#     float : Valeur minimale dans la liste.
#     """
#     return min(data) if data else 0

# def manual_max(data):
#     """
#     Trouver la valeur maximale dans les données manuellement.

#     Paramètres :
#     data (list) : Liste des nombres.

#     Retourne :
#     float : Valeur maximale dans la liste.
#     """
#     return max(data) if data else 0

# def calculate_bins(data, n_bins=10):
#     """
#     Calculer les bins de l'histogramme manuellement.

#     Paramètres :
#     data (list) : Liste des nombres.
#     n_bins (int) : Nombre de bins à calculer.

#     Retourne :
#     list : Liste des bordures des bins.
#     """
#     min_val = manual_min(data)
#     max_val = manual_max(data)
#     bin_width = (max_val - min_val) / n_bins if min_val != max_val else 1
#     return [min_val + i * bin_width for i in range(n_bins + 1)]

# def assign_data_to_bins(data, bins):
#     """
#     Assigner les points de données aux bins pour l'histogramme.

#     Paramètres :
#     data (list) : Liste des nombres.
#     bins (list) : Liste des bordures des bins.

#     Retourne :
#     list : Liste des comptes de données dans chaque bin.
#     """
#     bin_counts = [0] * (len(bins) - 1)
#     for value in data:
#         for i, bin_edge in enumerate(bins):
#             if i > 0 and value <= bin_edge:
#                 bin_counts[i - 1] += 1
#                 break
#     return bin_counts

def plot_manual_histogram(house_data, course, bins, house_colors):
    """
    Tracer l'histogramme des scores du cours pour chaque maison de Poudlard.

    Paramètres :
    house_data (dict) : Dictionnaire contenant les scores pour chaque maison.
    course (str) : Nom du cours.
    bins (list) : Liste des bordures des bins.
    house_colors (dict) : Dictionnaire contenant les couleurs pour chaque maison.

    Retourne :
    None
    """
    plt.figure(figsize=(10, 6))
    for house, scores in house_data.items():
        bin_counts = assign_data_to_bins(scores, bins)
        plt.bar([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)], bin_counts, width=(bins[1] - bins[0]) * 0.8, alpha=0.5, label=f'{house} ({len(scores)} scores)', color=house_colors.get(house, 'gray'))
    plt.title(f'Score Distribution Across Houses for {course}')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Charger le fichier de données
    df = load_data("data/dataset_train.csv")
    
    # Liste des cours à analyser
    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    
    # Dictionnaire des couleurs pour chaque maison
    house_colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow'
    }

    # Analyser et tracer les histogrammes pour chaque cours
    for course in courses:
        house_data = get_house_data(df, course)
        if any(house_data.values()):
            scores = [score for scores_list in house_data.values() for score in scores_list]
            bins = calculate_bins(scores, n_bins=15)
            plot_manual_histogram(house_data, course, bins, house_colors)
        else:
            print(f"No scores available for the course {course}. Cannot plot histogram.")

if __name__ == "__main__":
    main()