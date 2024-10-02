import pytest
from unittest import mock
import matplotlib.pyplot as plt
import dslr.scripts.pair_plot

def test_pair_plot_matrix(mocker):
    mock_show = mocker.patch('matplotlib.pyplot.show')
    dslr.scripts.pair_plot.main()
    expected_graph_count = 1
    assert mock_show.call_count == expected_graph_count, f"Expected {expected_graph_count} plots, but got {mock_show.call_count}"