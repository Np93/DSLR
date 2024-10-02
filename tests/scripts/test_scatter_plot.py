import pytest
from unittest import mock
import matplotlib.pyplot as plt
import dslr.scripts.scatter_plot

def test_scatter_plot_matrix(mocker):
    mock_show = mocker.patch('matplotlib.pyplot.show')
    dslr.scripts.scatter_plot.main()
    expected_graph_count = 1  
    assert mock_show.call_count == expected_graph_count, f"Expected {expected_graph_count} plots, but got {mock_show.call_count}"