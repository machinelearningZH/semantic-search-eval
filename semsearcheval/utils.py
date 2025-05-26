from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import tiktoken
from ruamel.yaml import YAML
from tqdm import tqdm

from semsearcheval.logger import logger


MAX_THREADPOOL_WORKERS = 10
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def read_yaml(path: Path) -> Dict[str, Any]:
    """Reads a YAML file and returns its contents as a dictionary."""
    yaml = YAML(typ="safe")
    return yaml.load(path)


def run_funct_in_parallel(function: Callable[[Any], Any], items: Iterable[Any]) -> List[Any]:
    """Executes a function in parallel on a list of inputs and returns the results."""
    with ThreadPoolExecutor(max_workers=MAX_THREADPOOL_WORKERS) as executor:
        futures = {executor.submit(function, item): i for i, item in enumerate(items)}

        results = [None] * len(futures)
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing...",
        ):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = f"Error: {str(e)}"
                logger.exception(f"Error processing input {index}: {str(e)}")
        return results


def truncate_to_max_len(texts: List[str], max_len: int, source: str) -> List[str]:
    """Truncates a list of texts to a specified maximum length in tokens."""
    truncated = []
    shorter = 0
    for text in texts:
        tokens = TOKENIZER.encode(text)
        # If the text is already within the limit, keep it as is
        if len(tokens) <= max_len:
            truncated.append(text)
            continue
        # If the text exceeds the limit, truncate it
        tokens = tokens[:max_len]
        truncated.append(TOKENIZER.decode(tokens))
        shorter += 1
    logger.info(
        f"Shortened {round((shorter / len(texts)) * 100, 2)}% of {source} "
        f"({shorter} out of {len(texts)}). All {source} now ≤ {max_len} tokens."
    )
    return truncated
