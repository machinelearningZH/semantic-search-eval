import os
import time
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import spacy
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from semsearcheval.data import Result


class Model(ABC):
    """
    Abstract base class for all embedding models.

    Subclasses must implement encode to generate
    embeddings and compute_similarity to compute
    similarity scores between queries and documents.
    """

    def __init__(self, name: str, model_path: str) -> None:
        self.name = name
        self.identifier = f"model {model_path}"
        self.model_path = model_path

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def encode(self, segments: List[str], prompt_name: str = None) -> List[List[float]]:
        pass

    @abstractmethod
    def compute_similarity(self, queries: List[str], docs: List[str]) -> np.array:
        pass

    @staticmethod
    def _prepend_prefix(texts: List[str], prefix: str) -> List[str]:
        """Prepends a custom prefix string to each text."""
        return [f"{prefix}{text}" for text in texts]

    def run(self, queries: List[str], docs: List[str]) -> Result:
        """
        A convenience method that runs the `compute_similarity` function and
        measures the time it takes. Returns a Result object.
        """
        start = time.time()
        similarity = self.compute_similarity(queries, docs)
        end = time.time()
        result = Result(similarity, end - start, gold_indices=[])
        return result


class HuggingFaceModel(Model):
    """
    A model that uses Hugging Face's SentenceTransformers to generate embeddings.

    Supports optional built-in prompts and custom prefixes for queries and passages.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        set_builtin_query_prompt: str = None,
        set_builtin_passage_prompt: str = None,
        set_custom_query_prefix: str = None,
        set_custom_passage_prefix: str = None,
        set_query_task_prompt: str = None,
        set_passage_task_prompt: str = None,
    ) -> None:
        super().__init__(name, model_path)
        self.identifier = f"open-source model {model_path}"

        # Validate: prompt_name and custom prefix are mutually exclusive.
        both_specified_query = (
            set_builtin_query_prompt is not None and set_custom_query_prefix is not None
        )
        both_specified_passage = (
            set_builtin_passage_prompt is not None and set_custom_passage_prefix is not None
        )
        if both_specified_query or both_specified_passage:
            raise ValueError(
                f"Model '{name}': cannot use both prompt_name (set_builtin_query_prompt / set_builtin_passage_prompt) "
                "and custom prefix (set_custom_query_prefix / set_custom_passage_prefix) at the same time."
            )

        self.builtin_query_name = set_builtin_query_prompt
        self.builtin_passage_name = set_builtin_passage_prompt
        self.custom_query_prefix = set_custom_query_prefix
        self.custom_passage_prefix = set_custom_passage_prefix
        self.query_task_prompt = set_query_task_prompt
        self.passage_task_prompt = set_passage_task_prompt

    def load_model(self) -> None:
        """Load the SentenceTransformer model."""
        self.model = SentenceTransformer(self.model_path, trust_remote_code=True)

    def encode(
        self, segments: List[str], prompt_name: str = None, task: str = None
    ) -> List[List[float]]:
        """Encodes the input segments into embeddings using the SentenceTransformer model."""
        kwargs = dict(
            normalize_embeddings=True,
            prompt_name=prompt_name,
            show_progress_bar=True,
            task=task,
        )
        return self.model.encode(segments, **kwargs)

    def encode_with_prompt(
        self, input: List[str], prompt_name: str, task: str = None
    ) -> List[List[float]]:
        """Helper function to encode input with a specific prompt_name if needed."""
        if prompt_name:
            return self.encode(input, prompt_name=prompt_name, task=task)
        return self.encode(input)

    def compute_similarity(self, queries: List[str], docs: List[str]) -> np.array:
        """Computes the cosine similarity between query and document embeddings."""
        # Prepend custom prefixes if specified. This modifies the input texts before encoding.
        if self.custom_query_prefix:
            queries = self._prepend_prefix(queries, self.custom_query_prefix)

        if self.custom_passage_prefix:
            docs = self._prepend_prefix(docs, self.custom_passage_prefix)

        # Encode queries and documents with the appropriate prompts if specified.
        query_emb = self.encode_with_prompt(
            queries, prompt_name=self.builtin_query_name, task=self.query_task_prompt
        )
        doc_emb = self.encode_with_prompt(
            docs, prompt_name=self.builtin_passage_name, task=self.passage_task_prompt
        )
        return self.model.similarity(query_emb, doc_emb).numpy()


class OpenAIModel(Model):
    """
    A model that uses the OpenAI API to generate embeddings and compute similarity.
    """

    def __init__(self, name: str, model_id: str) -> None:
        super().__init__(name, model_id)
        self.model = model_id
        self.identifier = f"OpenAI model {model_id}"

    def load_model(self) -> None:
        """Connect to the OpenAI client."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def encode(self, segments: List[str]) -> np.ndarray:
        """
        Encodes a list of text segments using the OpenAI API.
        The `batch_size` defines how many segments are processed in one API call.
        """
        batch_size = 100
        results = []
        for pos in range(0, len(segments), batch_size):
            batch = segments[pos : pos + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            results += [x.embedding for x in response.data]
        return np.array(results)

    def compute_similarity(self, queries: List[str], docs: List[str]) -> np.ndarray:
        """Encodes queries and documents, then computes cosine similarity."""
        queries = self.encode(queries)
        docs = self.encode(docs)
        return cosine_similarity(queries, docs)


class BM25Model(Model):
    """
    A model that uses the BM25 algorithm for information retrieval.
    """

    def __init__(self, name: str, model_id: str):
        super().__init__(name, model_id)
        self.identifier = "lemmatized BM25"

    def load_model(self) -> None:
        self.nlp = spacy.load(self.model_path, disable=["ner", "textcat"])

    def preprocess(self, texts: List[str]) -> List[str]:
        """Tokenizes and lemmatizes the input texts, removing stop words."""
        tokenized_corpus = []
        for doc in self.nlp.pipe(texts):
            tokens = [text.lemma_ for text in doc if not text.is_stop]
            tokenized_corpus.append(tokens)
        return tokenized_corpus

    def build_index(self, docs: List[str]) -> BM25Okapi:
        """Tokenizes documents and builds a BM25 index (inverted index)."""
        tokenized_corpus = self.preprocess(docs)
        return BM25Okapi(tokenized_corpus)

    def encode(self, segments: List[str]) -> BM25Okapi:
        return self.build_index(segments)

    def compute_similarity(self, queries: List[str], docs: List[str]) -> np.array:
        """Computes BM25 scores for each query-document pair. BM25 returns relevance scores, which are then used to calculate the similarity."""
        bm25 = self.build_index(docs)
        similarities = []
        for query in queries:
            tokenized_query = self.preprocess([query])[0]
            scores = bm25.get_scores(tokenized_query)
            similarities.append(scores.tolist())

        return np.array(similarities)
