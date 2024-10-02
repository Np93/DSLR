import pytest
from unittest import mock
import matplotlib.pyplot as plt
import dslr.scripts.histogram

def test_histogram(mocker):
    mock_show = mocker.patch('matplotlib.pyplot.show')
    dslr.scripts.histogram.main()
    expected_graph_count = 13
    assert mock_show.call_count == expected_graph_count, f"Expected {expected_graph_count} plots, but got {mock_show.call_count}"