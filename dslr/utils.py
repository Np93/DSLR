import pandas as pd
from typing import List
from dslr.csvchecker.CSV_Checker import CSV_Checker

def manual_count(data: list[float]) -> int:
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

def manual_sum(data: List[float]) -> float:
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

def manual_mean(data: list[float]) -> float:
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

def manual_std(data: list[float], mean: float) -> float:
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

def manual_min(data: list[float]) -> float:
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

def manual_max(data: list[float]) -> float:
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

def manual_quantile(data: list[float], quantile: float) -> float:
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

def calculate_bins(data: List[float], n_bins: int = 10) -> List[float]:
	"""
	Calculer les bins de l'histogramme manuellement.

	Paramètres :
	data (list) : Liste des nombres.
	n_bins (int) : Nombre de bins à calculer.

	Retourne :
	list : Liste des bordures des bins.
	"""
	min_val = manual_min(data)
	max_val = manual_max(data)
	bin_width = (max_val - min_val) / n_bins if min_val != max_val else 1
	return [min_val + i * bin_width for i in range(n_bins + 1)]

def assign_data_to_bins(data: List[float], bins: List[float]) -> List[int]:
	"""
	Assigner les points de données aux bins pour l'histogramme.

	Paramètres :
	data (list) : Liste des nombres.
	bins (list) : Liste des bordures des bins.

	Retourne :
	list : Liste des comptes de données dans chaque bin.
	"""
	bin_counts = [0] * (len(bins) - 1)
	for value in data:
		for i, bin_edge in enumerate(bins):
			if i > 0 and value <= bin_edge:
				bin_counts[i - 1] += 1
				break
	return bin_counts

def load_data(filepath) -> pd.DataFrame:
	"""
	Charger les données à partir d'un fichier CSV.

	Paramètres :
	str: filepath, le hemin vers le fichier CSV.

	Retourne :
	pd.DataFrame: DataFrame, les données chargées sous forme de DataFrame Pandas.
	"""
	checker = CSV_Checker(file_path=filepath, delimiter=',')
	
	checker.check_delimiter_inconsistency()
	checker.check_unnamed_variables()
	checker.analyze_data_type()
	print(f"check error found {checker.error_found}")
	checker.load_csv()
	
	df = checker.df
	return df
	#return pd.read_csv(filepath)