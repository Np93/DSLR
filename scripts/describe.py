import pandas as pd
import sys

def manual_count(data):
    """
    Compte le nombre d'éléments dans une liste.

    Paramètres :
    data (list) : Liste des éléments à compter.

    Retourne :
    int : Nombre d'éléments dans la liste.
    """
    count = 0
    for _ in data:
        count += 1
    return count

def manual_sum(data):
    """
    Calcule la somme des éléments dans une liste.

    Paramètres :
    data (list) : Liste des éléments à sommer.

    Retourne :
    float : Somme des éléments dans la liste.
    """
    total = 0
    for num in data:
        total += num
    return total

def manual_mean(data):
    """
    Calcule la moyenne des éléments dans une liste.

    Paramètres :
    data (list) : Liste des éléments.

    Retourne :
    float : Moyenne des éléments dans la liste.
    """
    total = manual_sum(data)
    count = manual_count(data)
    return total / count if count != 0 else 0

def manual_std(data, mean):
    """
    Calcule l'écart type des éléments dans une liste.

    Paramètres :
    data (list) : Liste des éléments.
    mean (float) : Moyenne des éléments.

    Retourne :
    float : Écart type des éléments dans la liste.
    """
    n = manual_count(data)
    variance = 0
    for x in data:
        variance += (x - mean) ** 2
    return (variance / (n - 1)) ** 0.5 if n > 1 else 0

def manual_min(data):
    """
    Trouve le minimum des éléments dans une liste.

    Paramètres :
    data (list) : Liste des éléments.

    Retourne :
    float : Minimum des éléments dans la liste.
    """
    if not data:
        return None
    minimum = data[0]
    for num in data:
        if num < minimum:
            minimum = num
    return minimum

def manual_max(data):
    """
    Trouve le maximum des éléments dans une liste.

    Paramètres :
    data (list) : Liste des éléments.

    Retourne :
    float : Maximum des éléments dans la liste.
    """
    if not data:
        return None
    maximum = data[0]
    for num in data:
        if num > maximum:
            maximum = num
    return maximum

def manual_quantile(data, quantile):
    """
    Calcule le quantile spécifié des éléments dans une liste.

    Paramètres :
    data (list) : Liste des éléments.
    quantile (float) : Quantile à calculer (entre 0 et 1).

    Retourne :
    float : Quantile des éléments dans la liste.
    """
    if not data:
        return None
    sorted_data = sorted(data)
    index = int(len(sorted_data) * quantile)
    return sorted_data[index]

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