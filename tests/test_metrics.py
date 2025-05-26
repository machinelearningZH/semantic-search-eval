import numpy as np
import pytest

from semsearcheval.data import Result
from semsearcheval.metrics import Accuracy, Latency


@pytest.mark.parametrize(
    "similarity,gold_indices,k,expected",
    [
        (np.array([[1, 0, 0], [0, 1, 0]]), np.array([0, 1]), 1, 100.0),  # Both correct at top 1
        (np.array([[0, 1, 0], [0, 0, 1]]), np.array([0, 1]), 1, 0.0),  # None correct at top 1
        (np.array([[0, 1, 0], [0, 0, 1]]), np.array([0, 1]), 2, 50.0),  # One correct in top 2
    ],
)
def test_accuracy_at_k(similarity, gold_indices, k, expected):
    metric = Accuracy(name=f"accuracy@{k}")
    result = Result(similarity=similarity, time=0.1, gold_indices=gold_indices)
    accuracy, unit = metric.compute(result)
    assert accuracy == expected
    assert unit == "%"


def test_latency():
    metric = Latency(name="latency")
    result = Result(similarity=[[0.0]], time=345.678, gold_indices=[0])
    latency, unit = metric.compute(result)

    # Consider using pytest.approx for floating point comparisons
    assert latency == pytest.approx(345.678)
    assert unit == "s"
