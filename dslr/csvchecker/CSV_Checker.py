import pandas as pd
import numpy as np
import json
import sys
import os
from typing import List, Tuple
def is_integer(value):
	try:
		int(value)
		return True
	except ValueError:
		return False

def is_float(value):
	if value == "":
		return True
	try:
		float(value)
		return True
	except ValueError:
		return False
	
class CSV_Checker:
	def __init__(self, file_path: str, delimiter: str =',', encoding: str ='utf-8') -> None:
		"""
		Initialise la classe CSV_Checker, qui s'occupe de tester la conformité du fichier à analyser.

		Paramètres :
		str: file_path, le fichier .csv à analyser.
		str: delimiter, le delimiteur entre les données
		str: encoding, le format de l'alphabet
		pd.DataFrame: df, la variable contenatn la conversion du fichier en dataframe par panda
		int: expected_num_column, le nombre de colonnes attendue
		int: num_columns et num_rows, les nombres de colonnes et de lignes du fichier
		list[str]: columns, les intitulés des paramètres
		#dict{}:column_types (not used anymore), conteneur des types pour chaque colonnes
		dict[int]{str} delimiter_inconsistency, log de l'analyse durant le check sur délimiteur
		dict[int]{list[int]} None_value_dict, la liste de toute les valeurs vides trouvées
		dict[int]{list[int]} string_in_numeric_dict, la liste de toutes les erreurs de types string dans un champ numériques
		bool : error_found, le drapeau déterminant si oui ou noon on charge le fichier
		"""
		self.file_path = file_path
		self.delimiter = delimiter
		self.encoding = encoding
		self.df = None
		self.expected_num_column = 0
		self.num_columns = 0
		self.num_rows = 0
		self.columns = []
		#self.column_types = {}
		self.delimiter_inconsistency = {}
		self.None_value_dict = {}
		self.string_in_numeric_dict = {}
		self.error_found = False

	def load_csv(self):
		"""Load le fichier dans le dataframe"""
		try:
			if not self.error_found:
				self.df = pd.read_csv(self.file_path, delimiter=self.delimiter, encoding=self.encoding)
				self.num_columns = len(self.df.columns)
				self.num_rows = self.df.shape[0]
				self.columns = list(self.df.columns)
				print(f"CSV file loaded successfully with {self.num_columns} columns and {self.num_rows} rows")
			else:
				print(f"CSV file is corrupted, therefore not loaded")
				sys.exit(1)
				
		except Exception as e:
			print("Error loading CSV: {e}")
			sys.exit(1)
	
	def check_delimiter_inconsistency(self):
		"""
		Compte pour chaque ligne du fichier le nombre de délimiteurs, en corollaire il détecte les lignes vides
		set(int): delimiter_counts, l'ensemble des nombres de délimiteurs par ligne trouvé
		int: expected_num_delimiter, la valeur attendue
		"""
		delimiter_counts = set()
		self.delimiter_inconsistency = {} #dictionary with key the line number and value the difference between found and expected number of delimiter
		expected_num_delimiter = None
		try:
			with open(self.file_path, 'r', encoding=self.encoding) as f:
				for i, line in enumerate(f):
					line = line.strip()
					if not line:
						self.delimiter_inconsistency[i+1] = "Empty row"
						continue

					count_delimiter = line.count(self.delimiter)
					delimiter_counts.add(count_delimiter)
					if i == 0:
						expected_num_delimiter = count_delimiter
					if len(delimiter_counts) > 1:
						self.delimiter_inconsistency[i+1] = "Number of delimiter not the one expected"
						delimiter_counts.clear()
						break
				
			if self.delimiter_inconsistency:
				print(f"delimiter inconsistency found !")
				for line_number, error in self.delimiter_inconsistency.items():
					print(f"Line {line_number}: {error} ")
				self.error_found = True
			else:
				self.expected_num_column = expected_num_delimiter + 1
				print(f"No delimiter inconsistency detected, {self.expected_num_column} columns")
			
		except Exception as e:
			print(f"Error checking for delimiter inconsistencies: {e}")
			sys.exit(1)
			
	
	def check_unnamed_variables(self):
		"""
		Vérifie que tout les paramètres sont décrits par leurs noms
		"""
		try:
			with open(self.file_path, 'r', encoding=self.encoding) as f:
				header = f.readline().strip().split(self.delimiter)
				unnamed_variables = [i for i, col_name in enumerate(header, start=1) if not col_name]

				if unnamed_variables:
					print(f"Empty column headers found at positions: {unnamed_variables}")
					self.error_found = True
				else:
					print("No empty column headers detected.")
					
		except Exception as e:
			print(f"Error while checking column headers: {e}")
			sys.exit(1)
	

	def filter_cell_nonnumeric(self, column_types: list[set]):
		"""
		Vérifie que sur les champs numériques, il n'y a que des nombres, nan ou ''. Et crée une liste de toutes les erreurs,
		si des fois on voudrait les corriger !
		list[set] de data type: column_types, liste pour chaque colonne tout les types de valeurs trouvées
		list[int]: cell_columns, l'index des colonnes contenant des string et des nombres.
		"""
		cell_columns = []
		try:
			with open(self.file_path, 'r', encoding=self.encoding) as f:
				for index, types in enumerate(column_types):
					# Check if both str and numeric types are present in this column
					if str in types and (int in types or float in types):
						print(f"Column {index}: {types} : str-num collision")
						cell_columns.append(index)
						self.error_found = True

			with open(self.file_path, 'r', encoding=self.encoding) as f:
				f.readline()
				for i, line in enumerate(f):
					data_row = line.strip().split(self.delimiter)
					row_str_list = []
					for col_index in cell_columns:
						value = data_row[col_index].strip()
						if not (is_integer(value) or is_float(value)):
							row_str_list.append(col_index)
					if row_str_list:        
						self.string_in_numeric_dict[i]=row_str_list
				
			for row_idx, col_list in self.string_in_numeric_dict.items():
				print(f"Row {row_idx} has strings in numeric columns at positions: {col_list}")

		except Exception as e:
			print(f"Error checking for delimiter inconsistencies: {e}")
			sys.exit(1)	

	def analyze_data_type(self):
		"""
		Détecte tout les types des valeurs du fichier et pour chaque colonne leur assigne leurs types dans une liste de set.
		None: correspond pour les NaN et ''.
		Puis envoye cette liste pour checker des incohérences entre str-int ou str-float ()
		str: header, la ligne contenant les paramètres (pour connaitre le nb de colonnes)
		list[set] de data type: column_types, liste pour chaque colonne tout les types de valeurs trouvées
		set{str}, exception_value, la liste des types "None"
		list[int], la liste de position des Nones dans le fichier
		"""
		try:
			with open(self.file_path, 'r', encoding=self.encoding) as f:
				header = f.readline().strip().split(self.delimiter)
				column_types = [set() for _ in range(len(header))]
			
				exception_value = {'nan', ''}

				for i, line in enumerate(f):
					data_row = line.strip().split(self.delimiter)
					row_None_list = []

					for index, value in enumerate(data_row):
						value = value.strip()
						if value.lower() in exception_value:
							row_None_list.append(index)
							column_types[index].add(None)
							continue
						
						if is_integer(value):
							column_types[index].add(int)
						elif is_float(value):
							column_types[index].add(float)
						else:
							column_types[index].add(str)
				
					if row_None_list:
						self.None_value_dict[i] = row_None_list
			self.filter_cell_nonnumeric(column_types)
		except Exception as e:
			print(f"Error checking for delimiter inconsistencies: {e}")
			sys.exit(1)
			
		


