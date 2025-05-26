"""Contains global fixtures for unit tests."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def test_config():
    """Load the YAML test configuration."""
    config_path = Path(__file__).parent / "test_config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


@pytest.fixture
def test_docs():
    return [
        "Eiffel Tower is in Paris",
        "Statue of Liberty is in New York",
        "Statue of Liberty came from Paris",
    ]


@pytest.fixture
def test_queries():
    return ["Tower Paris", "New York Statue", "came from Paris"]
