"""
Central registry for supported model and metric classes used in config.
"""

from typing import Dict, Type

from semsearcheval.metrics import Accuracy, Latency, Metric, NDCG
from semsearcheval.models import (
    BM25Model,
    HuggingFaceModel,
    IntFloatInstructModel,
    IntFloatModel,
    Model,
    OpenAIModel,
)


model_registry: Dict[str, Type[Model]] = {
    "huggingface": HuggingFaceModel,
    "intfloat": IntFloatModel,
    "intfloat-instruct": IntFloatInstructModel,
    "lexical": BM25Model,
    "open-ai": OpenAIModel,
}

metric_registry: Dict[str, Type[Metric]] = {
    "accuracy": Accuracy,
    "latency": Latency,
    "ndcg": NDCG,
}
