from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from semsearcheval.data import Result


class Metric(ABC):
    """
    Abstract base class for all metrics.

    Subclasses must implement compute to calculate the metric
    score based on the provided Result object.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def compute(self, result: Result) -> Tuple[float, str]:
        pass


class Accuracy(Metric):
    """
    Computes top-k accuracy over all queries.

    For each query, checks if the gold index is in the top-k most similar results.
    The metric name should be in the form: accuracy@k.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.k = self._parse_k(name)

    def _parse_k(self, name: str) -> None:
        """Extracts the top-k cutoff from the metric name."""
        if "@" not in name:
            raise ValueError(f"Invalid metric name: {name}. Expected format: accuracy@k")
        k = int(name.split("@")[1])
        if k <= 0:
            raise ValueError(f"Invalid k value: {k}. Must be a positive integer.")
        return k

    def compute(self, result: Result) -> Tuple[float, str]:
        """
        Compute top-k accuracy: proportion of queries where the gold document index
        appears in the top-k predicted documents (by similarity score).
        """
        correct_retrieved = 0
        for q_sims, gold_index in zip(result.similarity, result.gold_indices):
            # Get indices of top-k most similar documents (descending order as higher similarity is better)
            top_k = np.argsort(q_sims)[::-1][: self.k]
            if gold_index in top_k:
                correct_retrieved += 1

        # Return fraction of queries that were correct
        return correct_retrieved / result.similarity.shape[0] * 100, "%"


class Latency(Metric):
    """
    Reports the time taken to compute the similarity matrix.

    The Result object contains a `.time` attribute in seconds.
    """

    def compute(self, result: Result) -> Tuple[float, str]:
        return result.time, "s"
