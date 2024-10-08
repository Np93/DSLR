import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dslr.utils import load_data, manual_count, manual_mean, manual_std

def standardize_data(dataset: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
	"""
	Standardise les colonnes numériques présentes dans le set de donnée dataset,
	de sorte que la moyenne µ = 0 et l'écart type (standard deviation) de 1

	Paramètres :
	pd.DataFrame : dataset des observations.
	list[str] :columns, la liste des colonnes à standardiser (contient 2 labels de colonnes).

	Retourne :
	pd.DataFrame:l'ensemble dataset et les données standardisées.
	"""
	standardized_data = dataset.copy()
	for col in columns:
		mean = manual_mean(dataset[col])
		std = manual_std(dataset[col], mean)
		standardized_data[col] = (dataset[col] - mean) / std
	return standardized_data

def Pearson_correlation_coefficient(series_x: pd.Series, series_y: pd.Series) -> float:
	"""
	Calcule le coefficient de correlation entre X et Y comme le fait np.corrcoef

	Paramètres :
	pd.Series: series_x, la série des données X.
	pd.Series: series_y, la série des données Y.

	Remarque:
	Sxx: Variance de x
	Syy: Variance de y
	Sxy: Covariance de x et z

	Retourne :
	float : le coefficient de corrélation
	"""
	X_mean = manual_mean(series_x)
	Y_mean = manual_mean(series_y)
	n = manual_count(series_x) - 1

	if np.std(series_x) == 0 or np.std(series_y) == 0:
		return np.nan

	Sxx, Sxy, Syy = 0.0, 0.0, 0.0
	for x in series_x:
		Sxx += (x - X_mean) ** 2
	Sxx = Sxx / n
	for y in series_y:
		Syy += (y - Y_mean) ** 2
	Syy = Syy / n
	for x, y in zip(series_x,series_y):
		Sxy += (x - X_mean) * (y - Y_mean)
	Sxy = Sxy / n
	correlation_coefficient = Sxy / np.sqrt(Sxx * Syy)
	return correlation_coefficient

def plot_similar_features(dataset: pd.DataFrame,  most_similar_feature1: str,  most_similar_feature2: str, max_corr: float) -> None:
	"""
	Crée un scatter plot des deux cours les plus semblables d'après leur coefficient de correlations, ceci en choississnant
	le coefficient le plus proche de 1. De plus les données sont standardisées et affichées en comparaison avec la fonction identité.

	Paramètres :
	pd.DataFrame: dataset, l'ensemble des données.
	str: most_similar_feature1,le premier des deux cours les plus proches.
	str: most_similar_feature2, le second des deux cours les plus proches.
	float: max_corr, le coefficient de corrélation maximal (obtenu de get_max_correlation()).

	Retourne :
	None
	"""
	standardized_dataset = standardize_data(dataset, [ most_similar_feature1, most_similar_feature2])
	valid_data = standardized_dataset.dropna(subset=[ most_similar_feature1, most_similar_feature2])

	plt.figure(figsize=(10, 10), num=f"Scatter plot between { most_similar_feature1} and {most_similar_feature2}")
	house_colors = {
		'Gryffindor': 'red',
		'Slytherin': 'green',
		'Ravenclaw': 'blue',
		'Hufflepuff': 'yellow'
	}

	colors = valid_data['Hogwarts_House'].map(house_colors)
	plt.scatter(valid_data[ most_similar_feature1], valid_data[most_similar_feature2], c=colors, alpha=0.6, edgecolor='k')
	plt.title(f'Scatter Plot between { most_similar_feature1} and {most_similar_feature2} (Standardized) with correlation {max_corr:.2f}')
	plt.xlabel(f'Standardized { most_similar_feature1}')
	plt.ylabel(f'Standardized {most_similar_feature2}')
	plt.axis('equal')

	min_value = min(valid_data[ most_similar_feature1].min(), valid_data[most_similar_feature2].min())
	max_value = max(valid_data[ most_similar_feature1].max(), valid_data[most_similar_feature2].max())
	plt.xlim(-3, 3)
	plt.ylim(-3, 3)
	plt.plot([min_value, max_value], [min_value, max_value], color='black', linestyle='--', label='y = x')

	handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=house_colors[house], markersize=10) for house in house_colors]
	labels = list(house_colors.keys())
	plt.legend(handles, labels, title="Hogwarts House")
	plt.show()

def numeric_features(dataset:pd.DataFrame) -> list[str]:
	"""
	Extrait les colonnes numériques du dataset donné.

	Cette fonction identifie toutes les colonnes du dataset qui contiennent des types de données numériques
	(comme des entiers ou des flottants) et les renvoie sous forme de liste.

	Paramètres :
	pd.DataFrame: dataset, l'ensemble des données.

	Retourne :
	list[str] : Une liste de noms de colonnes correspondant aux colonnes numériques du dataset.
	"""
	numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
	return numeric_columns

def preprocess_data_for_correlation(dataset: pd.DataFrame, column_names: list[str], min_non_na: int = 100) -> tuple[pd.DataFrame, list[str]]:
	"""
	Prépare les données pour l'analyse de corrélation en sélectionnant les colonnes valides et en supprimant les lignes avec des valeurs manquantes.

	Paramètres :
	pd.DataFrame: dataset, l'ensemble des données.
	list[str]: column_names, une liste des noms de colonnes à évaluer pour le nombre de valeurs non manquantes.
	int: min_non_na (optionnel): Le nombre minimum de valeurs non manquantes qu'une colonne doit avoir
	pour être considérée comme valide (par défaut, 100).

	Retourne :
	tuple[pd.DataFrame, list[str]] : Un tuple contenant :
		- pd.DataFrame: Le DataFrame prétraité avec les lignes supprimées si elles contiennent des valeurs manquantes dans les colonnes valides.
		- list[str]]: Une liste de noms de colonnes valides qui répondent au critère de seuil de valeurs non manquantes.
	"""
	valid_columns = [col for col in column_names if dataset[col].count() >= min_non_na]
	cleaned_dataset = dataset.dropna(subset=valid_columns)
	return cleaned_dataset, valid_columns

def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
	"""
	Prétraiter les données en supprimant les colonnes non numériques et en gérant les valeurs manquantes.

	Paramètres :
	pd.DataFrame: dataset, l'ensemble des données.

	Retourne :
	pd.DataFrame: l'ensemble des données prétraitées avec uniquement les colonnes numériques et sans valeurs manquantes ET les noms de house.
	"""
	get_numeric_dataset = dataset.select_dtypes(include=[float, int])
	categorical_column = dataset['Hogwarts House']
	get_numeric_dataset = get_numeric_dataset.dropna(axis=1, how='all')
	add_categorical_data = get_numeric_dataset.assign(Hogwarts_House=categorical_column)
	add_categorical_data.dropna(inplace=True)
	return add_categorical_data

def calculate_correlations(dataset: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
	"""
	Calcule les coefficients de corrélation de Pearson entre toutes les paires de colonnes numériques dans le dataset.

	Paramètres :
	pd.DataFrame: dataset, l'ensemble des données.
	list[str]: column_names, la liste des colonnes numériques pour lesquelles les corrélations doivent être calculées.

	Retourne :
	pd.DataFrame: Un DataFrame où chaque cellule [i, j] contient le coefficient de corrélation entre la ligne i et la colonne j.
	"""
	dataset_clean, valid_numeric_columns = preprocess_data_for_correlation(dataset, column_names)
	correlations_matrix = pd.DataFrame(index=valid_numeric_columns, columns=valid_numeric_columns)
	for column1 in valid_numeric_columns:
		for column2 in valid_numeric_columns:
			if column1 != column2:
				correlation= Pearson_correlation_coefficient(dataset_clean[column1], dataset_clean[column2])
				correlations_matrix.at[column1, column2] = correlation if np.isfinite(correlation) else np.nan
	return correlations_matrix

def get_max_correlation(correlations_matrix: pd.DataFrame) -> float:
	"""
	Récupère la valeur maximale de corrélation dans la matrice de corrélation.

	Paramètres :
	pd.DataFrame: correlations, un DataFrame où chaque cellule [i, j] contient le coefficient de corrélation entre la ligne i et la colonne j.

	Retourne :
	float : La valeur maximale de corrélation trouvée entre deux colonnes différentes.
	"""
	correlation_matrix_filtered = correlations_matrix.unstack().dropna().loc[lambda x: x != 1]
	correlation_coefficient_max = correlation_matrix_filtered.max()
	return correlation_coefficient_max

def main():
	dataset = load_data("data/dataset_train.csv")
	numeric_columns = numeric_features(dataset)
	cleaned_data = preprocess_data(dataset)
	correlations_matrix = calculate_correlations(dataset, numeric_columns)
	if not correlations_matrix.empty:
		most_correlated_pair = correlations_matrix.unstack().dropna().idxmax()
		correlation_coefficient_max = get_max_correlation(correlations_matrix)
		plot_similar_features(cleaned_data, *most_correlated_pair, correlation_coefficient_max)
	else:
		print("No sufficiently correlated features found to plot.")

if __name__ == "__main__":
	main()
