import pytest
import dslr.scripts.describe
import os
import pandas as pd

def test_describe_data_train_output(capsys):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    data_path = "data/dataset_train.csv"
    expected_output_path = "tests/scripts/tool/expected_output.txt"

    dslr.scripts.describe.describe_data(data_path)

    captured = capsys.readouterr()

    with open(expected_output_path, 'r') as f:
        expected_output = f.read()

    assert captured.out.strip() == expected_output.strip(), "La sortie générée ne correspond pas à la sortie attendue."