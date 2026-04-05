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
from semsearcheval.logger import logger


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

    Supports several encoding mechanisms (checked in priority order):

    1. jina_adapter ("v3" or "v5"): uses hardcoded task/prompt_name pairs for
       Jina embedding models. Overrides all other prompt/prefix settings.
    2. custom prefix (set_custom_query_prefix / set_custom_doc_prefix): raw
       strings prepended to each text at inference time. Mutually exclusive
       with prompt_name booleans.
    3. prompt_name (use_query_prompt / use_passage_prompt): uses
       SentenceTransformer's built-in prompt_name feature.
    4. Default: plain encode with no prompt or task.
    """

    # Hardcoded Jina adapter configurations.
    # Each maps to (query_task, query_prompt, doc_task, doc_prompt).
    JINA_CONFIGS = {
        "v3": ("retrieval.query", "retrieval.query", "retrieval.passage", "retrieval.passage"),
        "v5": ("retrieval", "query", "retrieval", "document"),
    }

    def __init__(
        self,
        name: str,
        model_path: str,
        use_query_prompt: bool = False,
        use_passage_prompt: bool = False,
        set_custom_query_prefix: str = None,
        set_custom_doc_prefix: str = None,
        jina_adapter: str = None,
    ) -> None:
        super().__init__(name, model_path)
        self.identifier = f"open-source model {model_path}"

        # Jina adapter overrides all other prompt/prefix settings.
        if jina_adapter is not None:
            if jina_adapter not in self.JINA_CONFIGS:
                raise ValueError(
                    f"Model '{name}': jina_adapter must be one of "
                    f"{list(self.JINA_CONFIGS.keys())}, got '{jina_adapter}'."
                )
            has_other = (
                use_query_prompt or use_passage_prompt
                or set_custom_query_prefix is not None
                or set_custom_doc_prefix is not None
            )
            if has_other:
                logger.info(
                    f"Model '{name}': jina_adapter='{jina_adapter}' is set. "
                    "Other prompt/prefix parameters will be ignored."
                )
            self.jina_adapter = jina_adapter
            self.query_name = None
            self.passage_name = None
            self.custom_query_prefix = None
            self.custom_doc_prefix = None
            return

        self.jina_adapter = None

        # Validate: prompt_name and custom prefix are mutually exclusive.
        has_prompt_name = use_query_prompt or use_passage_prompt
        has_custom_prefix = set_custom_query_prefix is not None or set_custom_doc_prefix is not None
        if has_prompt_name and has_custom_prefix:
            raise ValueError(
                f"Model '{name}': cannot use both prompt_name (use_query_prompt / use_passage_prompt) "
                "and custom prefix (set_custom_query_prefix / set_custom_doc_prefix) at the same time."
            )

        self.query_name = "query" if use_query_prompt else None
        self.passage_name = "passage" if use_passage_prompt else None
        self.custom_query_prefix = set_custom_query_prefix
        self.custom_doc_prefix = set_custom_doc_prefix

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
        )
        if task is not None:
            kwargs["task"] = task
        return self.model.encode(segments, **kwargs)

    def encode_with_prompt(self, input: List[str], prompt: str):
        """Helper function to encode input with a specific prompt_name if needed."""
        if prompt:
            return self.encode(input, prompt_name=prompt)
        return self.encode(input)

    def _prepend_prefix(self, texts: List[str], prefix: str) -> List[str]:
        """Prepends a custom prefix string to each text."""
        if prefix is None:
            return texts
        return [f"{prefix}{text}" for text in texts]

    def compute_similarity(self, queries: List[str], docs: List[str]) -> np.array:
        """Computes the cosine similarity between query and document embeddings."""
        # Jina adapter path: use hardcoded task + prompt_name per role.
        if self.jina_adapter is not None:
            q_task, q_prompt, d_task, d_prompt = self.JINA_CONFIGS[self.jina_adapter]
            query_emb = self.encode(queries, prompt_name=q_prompt, task=q_task)
            doc_emb = self.encode(docs, prompt_name=d_prompt, task=d_task)
            return self.model.similarity(query_emb, doc_emb).numpy()

        # Custom prefix path: prepend strings and encode without prompt_name.
        if self.custom_query_prefix is not None or self.custom_doc_prefix is not None:
            queries = self._prepend_prefix(queries, self.custom_query_prefix)
            docs = self._prepend_prefix(docs, self.custom_doc_prefix)
            query_emb = self.encode(queries)
            doc_emb = self.encode(docs)
            return self.model.similarity(query_emb, doc_emb).numpy()

        # Default path: use prompt_name mechanism (or plain encode if no prompts).
        query_emb = self.encode_with_prompt(queries, self.query_name)
        doc_emb = self.encode_with_prompt(docs, self.passage_name)
        return self.model.similarity(query_emb, doc_emb).numpy()


class IntFloatModel(HuggingFaceModel):
    """
    A variation of HuggingFaceModel that prepends 'query:' to queries and 'passage:' to documents.
    Computes similarity between these enriched queries and documents.
    """

    def compute_similarity(self, queries: List[str], docs: List[str]) -> np.array:
        # Prepend the string 'query: ' and 'passage: ' to each query and document respectively
        queries = [f"query: {query}" for query in queries]
        docs = [f"passage: {doc}" for doc in docs]

        # Encode the modified queries and documents
        input_texts = queries + docs
        embeddings = self.encode(input_texts)

        # Compute similarity as the dot product of the query and document embeddings (identical to cosine similarity as vectors are normalized)
        scores = (embeddings[: len(queries)] @ embeddings[len(queries) :].T).tolist()
        return np.array(scores)


class IntFloatInstructModel(HuggingFaceModel):
    """
    Similar to IntFloatModel, but generates an instruction-based prompt for each query.
    This is useful for retrieval tasks that require structured input (e.g., "search for a document").
    """

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def compute_similarity(self, queries: List[str], docs: List[str]) -> np.array:
        # Add instruction-based prompt to each query
        task = "Given a web search query, retrieve relevant passages that answer the query"
        queries = [self.get_detailed_instruct(task, query) for query in queries]

        # Encode the modified queries and documents
        input_texts = queries + docs
        embeddings = self.encode(input_texts)

        scores = (embeddings[: len(queries)] @ embeddings[len(queries) :].T).tolist()
        return np.array(scores)


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
