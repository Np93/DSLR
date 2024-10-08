import pandas as pd
import matplotlib.pyplot as plt
import sys
from dslr.utils import load_data

def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
	"""
	Filtrer les données du fichier (dataset) en supprimant les colonnes non numériques et en gérant les valeurs manquantes, 
	en vue de la régression logistique.

	Paramètres :
	pd.DataFrame: dataset contenant les données à prétraiter.

	Retourne :
	pd.DataFrame: l'ensemble des données filtrées avec en plus les noms de maison.
	"""
	
	numeric_dataset = dataset.select_dtypes(include=[float, int])
	numeric_dataset = numeric_dataset.iloc[:, 1:]
	categorical_column = dataset['Hogwarts House']
	numeric_dataset = numeric_dataset.dropna(axis=1, how='all')
	combined_data = numeric_dataset.assign(Hogwarts_House=categorical_column)
	combined_data.dropna(inplace=True)
	return combined_data

def plot_pair_matrix(dataset: pd.DataFrame, target_label: str , label_fontsize: int = 10, title_fontsize: int = 12) -> None:
	"""
	Tracer une matrice de graphiques de dispersion pour chaque paire de colonnes numériques, et en diagonal les histogrammes,
	le tout en fonction des maisons.

	Paramètres :
	pd.DataFrame: dataset contenant les données à tracer.
	str: target_label : Le nom de la colonne target (Hogwart house).
	int: label_fontsize,la taille des caractères des labels.
	int: title_fontsize,la taille des caractères des titres.

	Retourne :
	None
	"""
	house_colors = {
		'Gryffindor': 'red',
		'Slytherin': 'green',
		'Ravenclaw': 'blue',
		'Hufflepuff': 'yellow'
	}
	numeric_columns = dataset.select_dtypes(include=[float, int]).columns
	num_columns = len(numeric_columns)
	fig, axs = plt.subplots(num_columns,num_columns, figsize=(12, 12), num="Pair Plot feature1 vs feature2")

	for i, feature1 in enumerate(numeric_columns):
		for j, feature2 in enumerate(numeric_columns):
			if i == j:
				for house in dataset[target_label].unique():
					subset = dataset[dataset[target_label] == house]
					axs[i,j].hist(subset[feature1], bins = 50, label=house, color=house_colors[house], alpha=0.7)
			else:
				for house in dataset[target_label].unique():
					subset = dataset[dataset[target_label] == house]
					axs[i,j].scatter(subset[feature1], subset[feature2], label=house, color=house_colors[house], alpha=0.3)
			axs[i, j].set_xticks([])
			axs[i, j].set_yticks([])

	for i, feature in enumerate(numeric_columns):
		ylabel = feature.replace("Defense Against the Dark Arts", "Defense Against\nthe Dark Arts").replace(
			"Care of Magical Creatures", "Care of\nMagical Creatures")
		axs[i, 0].set_ylabel(ylabel, fontsize=label_fontsize)
		title = feature.replace("Defense Against the Dark Arts", "Defense Against\nthe Dark Arts").replace(
				"Care of Magical Creatures", "Care of\nMagical Creatures")
		axs[0, i].set_title(title, fontsize=title_fontsize)

	handles, labels = axs[0, 1].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), title=target_label, fontsize=label_fontsize)
	plt.subplots_adjust(left=0.1, right=0.9, top=0.9)
	plt.show()

def main() -> None:
	df = load_data("data/dataset_train.csv")
	cleaned_data = preprocess_data(df)
	if not cleaned_data.empty:
		plot_pair_matrix(cleaned_data, 'Hogwarts_House',label_fontsize=6, title_fontsize=6)
	else:
		print("No valid data available for plotting after preprocessing.")

if __name__ == "__main__":
	main()
