import pandas as pd
import sys
from dslr.utils import manual_count, manual_sum, manual_mean, manual_std, manual_min, manual_max, manual_quantile, load_data

def describe_data(data_path: str) -> None:
    """
    Calcule et affiche des statistiques descriptives pour les données numériques d'un fichier CSV.

    Paramètres :
    data_path (str) : Chemin vers le fichier CSV contenant les données.

    Retourne :
    None
    """
    df = load_data(data_path)

    if df.isna().all().any():
        print("Error: One or more columns are fully NaN.")
        return

    num_df = df.select_dtypes(include=['float64', 'int64'])

    stats = {'Feature': [], 'Count': [], 'Mean': [], 'Std': [], 'Min': [], '25%': [], '50%': [], '75%': [], 'Max': []}

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

    result_df = pd.DataFrame(stats)
    print(result_df.T)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py path_to_data.csv")
    else:
        describe_data(sys.argv[1])