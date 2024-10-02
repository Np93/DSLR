import pytest
import dslr.scripts.describe  # Remplace par le chemin réel de ton fichier
import os
import pandas as pd

def test_describe_data_train_output(capsys):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    # Chemin du fichier de données et de la sortie attendue
    data_path = "data/dataset_train.csv"
    expected_output_path = "tests/scripts/tool/expected_output.txt"

    # Appeler la fonction principale qui génère la sortie
    dslr.scripts.describe.describe_data(data_path)

    # Capturer la sortie générée par le script
    captured = capsys.readouterr()

    # Charger la sortie attendue depuis un fichier
    with open(expected_output_path, 'r') as f:
        expected_output = f.read()

    # Comparer la sortie générée avec la sortie attendue
    assert captured.out.strip() == expected_output.strip(), "La sortie générée ne correspond pas à la sortie attendue."