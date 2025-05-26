from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl

from semsearcheval.logger import logger
from semsearcheval.query_creator import OpenAIQueryCreator, RandomKeywordQueryCreator
from semsearcheval.utils import truncate_to_max_len


@dataclass
class Result:
    """Class for keeping track of result values"""

    similarity: List[List[float]]
    time: float
    gold_indices: List[int]


class Dataset:
    """Loads documents and search queries, creating them if necessary."""

    def __init__(
        self,
        folder: Path,
        config: Dict[str, Any],
    ) -> None:
        self.prefix = config["name"]
        self.query_creator = (
            OpenAIQueryCreator(config["queries-per-doc"], config["openai-model"])
            if config["is-public-data"]
            else RandomKeywordQueryCreator(config["queries-per-doc"], config["spacy-model"])
        )
        self.path = folder / "datasets" / self.prefix
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_len = config["max-len"]

        self.docs = self._prepare_docs(Path(config["docs"]))
        self.queries, self.gold_indices = self._prepare_queries(
            Path(config["queries"]) if config["queries"] else None
        )

    @staticmethod
    def read_dataframe(file_path: Path) -> pl.DataFrame:
        if file_path.suffix.lower() in [".parquet", ".parq"]:
            return pl.read_parquet(file_path)
        elif file_path.suffix.lower() == ".csv":
            return pl.read_csv(file_path)
        raise ValueError(
            f"File {file_path.suffix} is not a supported file type. Please provide a CSV or a parquet file."
        )

    @staticmethod
    def validate_file_exists(file_path: Path) -> None:
        if not file_path.is_file():
            raise ValueError(f"File {file_path} does not exist. Please check the path.")

    def _load_file(
        self, file_path: Path, save_path: Path, required_columns: List[str]
    ) -> pl.DataFrame:
        # Check the users file exists
        self.validate_file_exists(file_path)

        # Save in testset folder
        df = self.read_dataframe(file_path)
        if any(column not in df.columns for column in required_columns):
            raise ValueError(
                f"Input file must contain the following columns: {', '.join(required_columns)}."
            )
        df.write_parquet(save_path)
        return df

    def _prepare_docs(self, file_path: Path) -> List[str]:
        doc_path = self.path / f"{self.prefix}.docs.parquet"
        df = self._load_file(file_path, doc_path, ["text"])
        docs = df["text"].to_list()
        return truncate_to_max_len(docs, self.max_len, "docs")

    def _prepare_queries(self, file_path: Path) -> Tuple[List[str], List[int]]:
        query_path = self.path / f"{self.prefix}.queries.parquet"
        # If user provided a file, check it exists and write it to the testset folder
        if file_path is not None:
            df = self._load_file(file_path, query_path, ["search_query", "idx"])

        # If no file provided, check if it was saved in previous iteration
        elif query_path.is_file():
            df = self.read_dataframe(query_path)

        # Create new queries
        else:
            logger.info(f"Creating queries using a {type(self.query_creator).__name__}.")
            df = self.query_creator.get_queries_with_indices(self.docs)
            df.write_parquet(query_path)

        queries, gold_indices = df["search_query"].to_list(), df["idx"].to_list()
        return truncate_to_max_len(queries, self.max_len, "queries"), gold_indices
