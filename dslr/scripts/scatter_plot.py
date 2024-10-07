import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dslr.utils import load_data, manual_count, manual_mean, manual_std

def standardize_data(dataset, columns) -> pd.DataFrame:
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

def Pearson_correlation_coefficient(dataset_X, dataset_Y) -> float:
	"""
	Calcule le coefficient de correlation entre X et Y comme le fait np.corrcoef

	Paramètres :
	dataset_X : les données X
	dataset_Y : les données Y

	Remarque:
	Sxx, Sxy et Syy sont des fonctions standards en statistiques.
	Sxy est la somme des produits des différences entre x et sa moyenne et entre y et sa moyenne

	Retourne :
	float : le coefficient de corrélation
	"""
	X_mean = manual_mean(dataset_X)
	Y_mean = manual_mean(dataset_Y)
	n = manual_count(dataset_X) - 1

	if np.std(dataset_X) == 0 or np.std(dataset_Y) == 0:
		return np.nan

	Sxx, Sxy, Syy = 0.0, 0.0, 0.0
	for x in dataset_X:
		Sxx += (x - X_mean) ** 2
	Sxx = Sxx / n
	for y in dataset_Y:
		Syy += (y - Y_mean) ** 2
	Syy = Syy / n
	for x, y in zip(dataset_X,dataset_Y):
		Sxy += (x - X_mean) * (y - Y_mean)
	Sxy = Sxy / n
	correlation_coefficient = Sxy / np.sqrt(Sxx * Syy)
	return correlation_coefficient

def plot_similar_features(dataset, feature1, feature2, max_corr) -> None:
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
	standardized_dataset = standardize_data(dataset, [feature1, feature2])
	valid_data = standardized_dataset.dropna(subset=[feature1, feature2])

	plt.figure(figsize=(10, 10), num=f"Scatter plot between {feature1} and {feature2}")
	house_colors = {
		'Gryffindor': 'red',
		'Slytherin': 'green',
		'Ravenclaw': 'blue',
		'Hufflepuff': 'yellow'
	}

	colors = valid_data['Hogwarts_House'].map(house_colors)
	plt.scatter(valid_data[feature1], valid_data[feature2], c=colors, alpha=0.6, edgecolor='k')
	plt.title(f'Scatter Plot between {feature1} and {feature2} (Standardized) with correlation {max_corr:.2f}')
	plt.xlabel(f'Standardized {feature1}')
	plt.ylabel(f'Standardized {feature2}')
	plt.axis('equal')

	min_value = min(valid_data[feature1].min(), valid_data[feature2].min())
	max_value = max(valid_data[feature1].max(), valid_data[feature2].max())
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
	dataset (pd.DataFrame) : Le dataset d'entrée contenant différentes colonnes avec différents types de données.

	Retourne :
	list[str] : Une liste de noms de colonnes correspondant aux colonnes numériques du dataset.
	"""
	numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
	return numeric_cols

def preprocess_data_for_correlation(dataset: pd.DataFrame, cols: list[str], min_non_na=100) -> tuple[pd.DataFrame, list[str]]:
	"""
	Prépare les données pour l'analyse de corrélation en sélectionnant les colonnes valides et en supprimant les lignes avec des valeurs manquantes.

	Paramètres :
	dataset (pd.DataFrame) : Le dataset d'entrée contenant les observations.
	cols (list[str]) : Une liste des noms de colonnes à évaluer pour le nombre de valeurs non manquantes.
	min_non_na (int, optionnel) : Le nombre minimum de valeurs non manquantes qu'une colonne doit avoir pour être considérée comme valide.
								  Par défaut, 100.

	Retourne :
	tuple[pd.DataFrame, list[str]] : Un tuple contenant :
		- Le DataFrame prétraité avec les lignes supprimées si elles contiennent des valeurs manquantes dans les colonnes valides.
		- Une liste de noms de colonnes valides qui répondent au critère de seuil de valeurs non manquantes.
	"""
	valid_cols = [col for col in cols if dataset[col].count() >= min_non_na]
	return dataset.dropna(subset=valid_cols), valid_cols

def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
	"""
	Prétraiter les données en supprimant les colonnes non numériques et en gérant les valeurs manquantes.

	Paramètres :
	dataset (DataFrame) : DataFrame contenant les données à prétraiter.

	Retourne :
	DataFrame : DataFrame prétraitée avec uniquement les colonnes numériques et sans valeurs manquantes ET les noms de house.
	"""
	numeric_dataset = dataset.select_dtypes(include=[float, int])
	categorical_column = dataset['Hogwarts House']
	numeric_dataset = numeric_dataset.dropna(axis=1, how='all')
	combined_data = numeric_dataset.assign(Hogwarts_House=categorical_column)
	combined_data.dropna(inplace=True)
	return combined_data

def calculate_correlations(dataset, numeric_cols) -> float:
	"""
	Calcule les coefficients de corrélation de Pearson entre toutes les paires de colonnes numériques dans le dataset.

	Paramètres :
	dataset (pd.DataFrame) : Le dataset d'entrée contenant les observations.
	numeric_cols (list[str]) : La liste des colonnes numériques pour lesquelles les corrélations doivent être calculées.

	Retourne :
	pd.DataFrame : Une matrice DataFrame où chaque cellule [i, j] contient le coefficient de corrélation entre
				   les colonnes numériques i et j.
	"""
	dataset_clean, valid_numeric_cols = preprocess_data_for_correlation(dataset, numeric_cols)
	correlations = pd.DataFrame(index=valid_numeric_cols, columns=valid_numeric_cols)
	for col1 in valid_numeric_cols:
		for col2 in valid_numeric_cols:
			if col1 != col2:
				correlation = Pearson_correlation_coefficient(dataset_clean[col1], dataset_clean[col2])
				correlations.at[col1, col2] = correlation if np.isfinite(correlation) else np.nan
	return correlations

def get_max_correlation(correlations) -> float:
	"""
	Récupère la valeur maximale de corrélation entre deux colonnes dans la matrice de corrélation.

	Paramètres :
	correlations (pd.DataFrame) : Une matrice de corrélation des colonnes numériques.

	Retourne :
	float : La valeur maximale de corrélation trouvée entre deux colonnes différentes.
	"""
	corr_unstacked = correlations.unstack()
	corr_filtered = corr_unstacked.dropna().loc[lambda x: x != 1]
	max_corr_value = corr_filtered.max()
	return max_corr_value

def main():
	dataset = load_data("data/dataset_train.csv")
	numeric_cols = numeric_features(dataset)
	cleaned_data = preprocess_data(dataset)
	correlations = calculate_correlations(dataset, numeric_cols)
	if not correlations.empty:
		most_correlated_pair = correlations.unstack().dropna().idxmax()
		max_corr = get_max_correlation(correlations)
		plot_similar_features(cleaned_data, *most_correlated_pair, max_corr)
	else:
		print("No sufficiently correlated features found to plot.")

if __name__ == "__main__":
	main()
