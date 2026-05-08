import numpy as np
import pytest

from semsearcheval.data import Result
from semsearcheval.metrics import NDCG, Accuracy, Latency


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


@pytest.mark.parametrize(
    "similarity,gold_indices,k,expected",
    [
        # Gold doc at rank 1 for both queries: NDCG = 1/log2(2) = 1.0 each -> 100%
        (np.array([[1, 0, 0], [0, 1, 0]]), np.array([0, 1]), 10, 100.0),
        # Gold doc at rank 2 for query 0, not in top 1 for query 1
        # Query 0: gold=0, scores=[0,1,0] -> ranked [1,0,2] -> gold at rank 2 -> 1/log2(3)
        # Query 1: gold=1, scores=[0,0,1] -> ranked [2,0,1] -> gold at rank 3 -> 1/log2(4)
        (
            np.array([[0, 1, 0], [0, 0, 1]]),
            np.array([0, 1]),
            10,
            (1.0 / np.log2(3) + 1.0 / np.log2(4)) / 2 * 100,
        ),
        # Gold doc outside top 1 for both -> 0%
        (np.array([[0, 1, 0], [0, 0, 1]]), np.array([0, 1]), 1, 0.0),
    ],
)
def test_ndcg_at_k(similarity, gold_indices, k, expected):
    metric = NDCG(name=f"ndcg@{k}")
    result = Result(similarity=similarity, time=0.1, gold_indices=gold_indices)
    ndcg, unit = metric.compute(result)
    assert ndcg == pytest.approx(expected)
    assert unit == "%"


def test_latency():
    metric = Latency(name="latency")
    result = Result(similarity=[[0.0]], time=345.678, gold_indices=[0])
    latency, unit = metric.compute(result)

    # Consider using pytest.approx for floating point comparisons
    assert latency == pytest.approx(345.678)
    assert unit == "s"
