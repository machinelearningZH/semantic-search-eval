import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import polars as pl
from dotenv import load_dotenv

from semsearcheval.constants import metric_registry, model_registry
from semsearcheval.data import Dataset
from semsearcheval.logger import logger
from semsearcheval.metrics import Metric, Result
from semsearcheval.models import Model
from semsearcheval.utils import read_yaml
from semsearcheval.visualize import visualize_results


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tool to evaluate different embedding models on different datasets for semantic search."
    )

    parser.add_argument(
        "-c", "--config", type=Path, required=True, help="Path to configuration file."
    )
    return parser.parse_args()


def load_dataset(
    folder: Path,
    config: Dict[str, Any],
) -> List[Dataset]:
    """Loads the dataset as specified in the config attributes."""
    return Dataset(folder, config)


def load_models(models: Dict[str, Dict[str, str]]) -> Generator[Model, None, None]:
    """Loads and yields model instances from the configuration."""
    for model_type, instances in models.items():
        for instance, info in instances.items():
            # If info is a string, only model path was provided
            if type(info) is str:
                yield model_registry[model_type](instance, info)
            # Otherwise additional arguments were provided
            else:
                model = info.pop("model")
                yield model_registry[model_type](instance, model, **info)


def load_metrics(metrics: Dict[str, Dict[str, str]]) -> List[Metric]:
    """Loads metrics from the configuration."""
    return [metric_registry[re.sub(r"@\d+$", "", name)](name) for name in metrics]


def load_precomputed_results(
    results_path: Path, time_path: Path, dataset: Dataset, model: Model
) -> Result:
    """Loads precomputed results from disk if they exist."""
    if results_path.is_file():
        logger.info(
            f"Loading similarities for {dataset.prefix} with {model.identifier} from results folder."
        )
        similarity = np.load(results_path)
        time = json.loads(time_path.read_text(encoding="UTF-8"))
        return Result(similarity, time, dataset.gold_indices)
    return None


def compute_and_save_results(
    results_folder: Path, results_path: Path, time_path: Path, dataset: Dataset, model: Model
) -> None:
    """Computes and saves the results to disk."""
    results_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Computing similarities for {dataset.prefix} with {model.identifier}.")
    result = model.run(dataset.queries, dataset.docs)
    result.gold_indices = dataset.gold_indices
    np.save(results_path, result.similarity)
    time_path.write_text(json.dumps(result.time))
    return result


def evaluate_experiments(
    folder: Path,
    dataset: Dataset,
    models: List[Model],
    metrics: List[Metric],
) -> pl.DataFrame:
    """Evaluates the models on the dataset using the provided metrics."""
    schema = {
        "dataset": pl.String,
        "model": pl.String,
        **{
            k: v
            for metric in metrics
            for k, v in [
                (metric.name, pl.Float64),
                (f"{metric.name}_unit", pl.String),
            ]
        },
    }
    df = pl.DataFrame(schema=schema)

    for model in models:
        results_folder = folder / "embeddings" / model.name
        results_path = results_folder / f"{dataset.prefix}.npy"
        time_path = results_folder / f"{dataset.prefix}.json"

        # If results were computed before, load them instead of recomputing
        result = load_precomputed_results(results_path, time_path, dataset, model)
        if result is None:
            model.load_model()
            result = compute_and_save_results(
                results_folder, results_path, time_path, dataset, model
            )

        # Record results for each metric
        row = {"dataset": dataset.prefix, "model": model.name}
        for metric in metrics:
            score, unit = metric.compute(result)
            row[metric.name] = round(score, 4)
            row[f"{metric.name}_unit"] = unit
        df = pl.concat([df, pl.DataFrame(row)])
    return df


def main(args: argparse.Namespace) -> None:
    config = read_yaml(args.config)
    folder = Path(config["folder"])
    load_dotenv()

    logger.info(f"Starting evaluation based on {args.config}.")

    # Load queries and doc pool
    dataset = load_dataset(folder, config)

    # Initialize models
    models = load_models(config["models"])

    # Initialize metrics
    metrics = load_metrics(config["metrics"])

    # Evaluate different experiments
    df = evaluate_experiments(folder, dataset, models, metrics)

    # Save and visualize the results
    path = folder / "stats.csv"
    df.write_csv(path)
    logger.info(f"Saved results to {path}.")
    visualize_results(folder, df, config["metrics"])


if __name__ == "__main__":
    args = parse_args()
    main(args)
