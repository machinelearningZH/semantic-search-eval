import os
import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import List

import numpy as np
import polars as pl
import spacy
from openai import OpenAI
from ordered_set import OrderedSet
from pydantic import BaseModel

from semsearcheval.logger import logger
from semsearcheval.prompts import BASE_PROMPT
from semsearcheval.utils import run_funct_in_parallel


class SearchQuery(BaseModel):
    """Schema for a single search query returned by the model."""

    query: str


class SearchQueriesOutput(BaseModel):
    """Expected structured response format from the OpenAI API."""

    search_queries: List[SearchQuery]


class QueryCreator(ABC):
    """
    Abstract base class for all query creators.

    Subclasses must implement get_queries_with_indices to generate
    queries from a list of documents and return them as a DataFrame.
    """

    def __init__(self, max_queries: int = 10) -> None:
        self.max_queries = max_queries

    @abstractmethod
    def get_queries_with_indices(self, docs: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError


class RandomKeywordQueryCreator(QueryCreator):
    """Generates keyword-based queries using capitalized tokens as pseudo-nouns."""

    def __init__(self, max_queries: int = 10, spacy_model_name: str = "de_core_news_sm") -> None:
        super().__init__(max_queries)
        self.nlp = spacy.load(spacy_model_name, disable=["ner", "textcat"])

    def generate_random_keywords(self, texts: List[str]) -> List[str]:
        """
        Extracts random pseudo-keywords from capitalized tokens in the input text.
        Returns up to 10 randomly formed queries.
        """
        queries_per_doc = []
        for doc in self.nlp.pipe(texts):
            # Find nouns and proper nouns and consider their frequency
            nouns = [
                token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop
            ]
            noun_counts = Counter(nouns).most_common()
            unique_nouns = [noun for noun, _ in noun_counts]
            probabilities = [count / len(nouns) for _, count in noun_counts]

            queries = []
            np.random.seed(42)
            random.seed(42)
            for _ in range(self.max_queries):
                # Randomly select between 1 and 5 unique nouns to form a query
                query_words = np.random.choice(
                    unique_nouns, size=random.randint(1, 5), p=probabilities
                )
                query = " ".join(list(OrderedSet(query_words)))
                queries.append(query)
            queries_per_doc.append(queries)
        return queries_per_doc

    def get_queries_with_indices(self, docs: List[str]) -> pl.DataFrame:
        """
        Generate queries for each document in parallel, then return a DataFrame
        with the query and the document index.
        """
        queries_per_doc = self.generate_random_keywords(docs)

        # Prepare the data for DataFrame
        data = {"search_query": [], "idx": []}
        for i, queries in enumerate(queries_per_doc):
            for query in queries[: self.max_queries]:
                data["search_query"].append(query)
                data["idx"].append(i)

        return pl.DataFrame(data)


class OpenAIQueryCreator(QueryCreator):
    """
    Uses the OpenAI Chat Completions API with structured response parsing
    to generate realistic search queries from documents.
    """

    def __init__(self, max_queries: int = 10, openai_model_name: str = "gpt-4.1-mini") -> None:
        super().__init__(max_queries)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = openai_model_name
        self.temperature = 0.8
        self.max_tokens = 4096
        self.response_format = SearchQueriesOutput

    def generate_queries(
        self,
        input: str,
    ) -> List[str]:
        """
        Send a document prompt to the OpenAI API, expecting a structured list
        of search queries in response. Falls back to empty list on error.
        """
        try:
            # We use the beta API for chat completions to allow for structured output.
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": input},
                ],
                response_format=self.response_format,
            )
            answer = completion.choices[0].message.parsed
            return [query.query for query in answer.search_queries]

        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            return []

    def create_final_prompt(self, text: str, base_prompt: str = BASE_PROMPT) -> str:
        """
        Insert the document text into the base prompt template.
        """
        return base_prompt.format(
            self.max_queries, int(0.8 * self.max_queries), int(0.3 * self.max_queries), text
        )

    def get_queries_with_indices(self, docs: List[str]) -> pl.DataFrame:
        """
        Prepares prompts from the documents, sends them to OpenAI in parallel,
        and returns a DataFrame of search queries and their corresponding doc indices.
        """
        # Wrap in Polars DataFrame to map transformation
        df = pl.DataFrame({"text": docs})

        # Create input prompts by formatting each document into the base prompt
        df = df.with_columns(
            pl.col("text")
            .map_elements(self.create_final_prompt, return_dtype=pl.String)
            .alias("prompt")
        )
        prompts = df["prompt"].to_list()

        # Generate queries for each prompt using parallel execution
        queries_per_doc = run_funct_in_parallel(self.generate_queries, prompts)

        data = {"search_query": [], "idx": []}
        for i, queries in enumerate(queries_per_doc):
            for query in queries[: self.max_queries]:
                data["search_query"].append(query)
                data["idx"].append(i)

        return pl.DataFrame(data)
