import pytest
from unittest import mock
import matplotlib.pyplot as plt
import dslr.scripts.histogram  # Remplace par le nom réel de ton fichier Python à tester

def test_histogram(mocker):
    # Mock plt.show pour compter combien de fois il est appelé
    mock_show = mocker.patch('matplotlib.pyplot.show')
    dslr.scripts.histogram.main()
    expected_graph_count = 13
    assert mock_show.call_count == expected_graph_count, f"Expected {expected_graph_count} plots, but got {mock_show.call_count}"