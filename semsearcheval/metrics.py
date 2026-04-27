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

    def _parse_k(self, name: str) -> None:
        """Extracts the top-k cutoff from the metric name."""
        if "@" not in name:
            raise ValueError(f"Invalid metric name: {name}. Expected format: metric@k")
        k = int(name.split("@")[1])
        if k <= 0:
            raise ValueError(f"Invalid k value: {k}. Must be a positive integer.")
        return k

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

    def compute(self, result: Result) -> Tuple[float, str]:
        """
        Compute top-k accuracy: proportion of queries where the gold document index
        appears in the top-k predicted documents (by similarity score).
        """
        correct_retrieved = 0
        for q_sims, gold_index in zip(result.similarity, result.gold_indices):
            # Sort similarity scores in descending order (highest similarity first)
            # and select the indices of the top-k documents
            top_k = np.argsort(q_sims)[::-1][: self.k]
            
            # Check if the gold (correct) document is among the top-k predictions
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


class NDCG(Metric):
    """
    Computes NDCG@k (Normalized Discounted Cumulative Gain) over all queries.

    With a single relevant document per query the ideal DCG is always 1.0,
    so NDCG simplifies to 1/log2(rank+1) if the gold doc is within the top k,
    and 0 otherwise. The result is averaged across all queries.
    The metric name should be in the form: ndcg@k.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.k = self._parse_k(name)

    def compute(self, result: Result) -> Tuple[float, str]:
        """
        Compute NDCG@k. For each query, find the rank of the gold document
        within the top k. Score is 1/log2(rank+1) if found, 0 otherwise.
        """
        total_ndcg = 0.0
        for q_sims, gold_index in zip(result.similarity, result.gold_indices):
            # Sort similarity scores in descending order and select top-k document indices
            ranked = np.argsort(q_sims)[::-1][: self.k]
            
            # Find if and where the gold document appears in the top-k ranked results
            positions = np.where(ranked == gold_index)[0]
            
            if len(positions) > 0:
                # Convert 0-based position to 1-based rank
                rank = positions[0] + 1
                
                # Apply logarithmic discount: 1/log2(rank+1)
                # This rewards higher-ranked results more than lower-ranked ones
                total_ndcg += 1.0 / np.log2(rank + 1)
            # If gold doc not in top-k, contributes 0 to the score
        
        # Calculate average NDCG across all queries and convert to percentage
        avg_ndcg = total_ndcg / result.similarity.shape[0] * 100
        return avg_ndcg, "%"
