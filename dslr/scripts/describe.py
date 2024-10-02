import pandas as pd
import sys
from dslr.utils import manual_count, manual_sum, manual_mean, manual_std, manual_min, manual_max, manual_quantile

def describe_data(data_path):
    """
    Calcule et affiche des statistiques descriptives pour les données numériques d'un fichier CSV.

    Paramètres :
    data_path (str) : Chemin vers le fichier CSV contenant les données.

    Retourne :
    None
    """
    # Charger les données à partir du fichier CSV
    df = pd.read_csv(data_path)

    # Vérifier les colonnes entièrement NaN
    if df.isna().all().any():
        print("Error: One or more columns are fully NaN.")
        return

    # Sélectionner uniquement les colonnes numériques
    num_df = df.select_dtypes(include=['float64', 'int64'])

    # Initialiser le dictionnaire pour les statistiques
    stats = {'Feature': [], 'Count': [], 'Mean': [], 'Std': [], 'Min': [], '25%': [], '50%': [], '75%': [], 'Max': []}

    # Calculer les statistiques pour chaque colonne numérique
    for column in num_df.columns:
        column_data = num_df[column].dropna().astype(float).tolist()
        mean = manual_mean(column_data)
        stats['Feature'].append(column)
        stats['Count'].append(manual_count(column_data))
        stats['Mean'].append(mean)
        stats['Std'].append(manual_std(column_data, mean))
        stats['Min'].append(manual_min(column_data))
        stats['25%'].append(manual_quantile(column_data, 0.25))
        stats['50%'].append(manual_quantile(column_data, 0.50))
        stats['75%'].append(manual_quantile(column_data, 0.75))
        stats['Max'].append(manual_max(column_data))

    # Convertir les résultats en DataFrame pour un affichage formaté
    result_df = pd.DataFrame(stats)
    print(result_df.T)  # Transposé pour correspondre au format requis

if __name__ == "__main__":
    # Vérifier si le chemin vers le fichier de données est fourni en argument
    if len(sys.argv) < 2:
        print("Usage: python script_name.py path_to_data.csv")
    else:
        # Décrire les données du fichier CSV fourni
        describe_data(sys.argv[1])