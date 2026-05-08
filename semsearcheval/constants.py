"""
Central registry for supported model and metric classes used in config.
"""

from typing import Dict, Type

from semsearcheval.metrics import NDCG, Accuracy, Latency, Metric
from semsearcheval.models import (
    BM25Model,
    HuggingFaceModel,
    Model,
    OpenAIModel,
)


model_registry: Dict[str, Type[Model]] = {
    "huggingface": HuggingFaceModel,
    "lexical": BM25Model,
    "open-ai": OpenAIModel,
}

metric_registry: Dict[str, Type[Metric]] = {
    "accuracy": Accuracy,
    "latency": Latency,
    "ndcg": NDCG,
}
