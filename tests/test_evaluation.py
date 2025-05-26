from pathlib import Path

from semsearcheval.evaluate import load_dataset, load_metrics, load_models
from semsearcheval.query_creator import RandomKeywordQueryCreator
from semsearcheval.utils import truncate_to_max_len


def test_config_parsing(test_config):
    assert test_config["folder"] == "test_results"
    assert test_config["name"] == "test_example"
    assert test_config["metrics"] == ["accuracy@1", "accuracy@5", "accuracy@10", "latency"]


def test_load_dataset(test_config):
    folder = Path(test_config["folder"])
    dataset = load_dataset(folder, test_config)
    assert dataset.prefix == "test_example"
    assert len(dataset.docs) == 5
    assert len(dataset.queries) == 5
    assert dataset.max_len == 256
    assert isinstance(dataset.query_creator, RandomKeywordQueryCreator)


def test_load_metrics(test_config):
    metrics = load_metrics(test_config["metrics"])
    assert len(metrics) == 4
    assert all([metric.name in test_config["metrics"] for metric in metrics])


def test_load_model(test_config):
    model = list(load_models(test_config["models"]))
    assert len(model) == 2
    assert model[0].name == "bm25"
    assert model[1].name == "jina-v2"


def test_truncate_to_max_len():
    texts = ["This is a test sentence that is too long.", "Short sentence."]
    max_len = 5
    truncated_texts = truncate_to_max_len(texts, max_len, "docs")
    assert len(truncated_texts) == 2
    assert truncated_texts[0] == "This is a test sentence"
    assert truncated_texts[1] == "Short sentence."
