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

    Supports optional query and passage prompts for specialized encoding.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        use_query_prompt: bool = False,
        use_passage_prompt: bool = False,
    ) -> None:
        super().__init__(name, model_path)
        self.identifier = f"open-source model {model_path}"
        self.query_name = "query" if use_query_prompt else None
        self.passage_name = "passage" if use_passage_prompt else None

    def load_model(self) -> None:
        """Load the SentenceTransformer model."""
        self.model = SentenceTransformer(self.model_path, trust_remote_code=True)

    def encode(self, segments: List[str], prompt_name: str = None) -> List[List[float]]:
        """Encodes the input segments into embeddings using the SentenceTransformer model."""
        return self.model.encode(
            segments,
            normalize_embeddings=True,
            prompt_name=prompt_name,
            show_progress_bar=True,
        )

    def encode_with_prompt(self, input: List[str], prompt: str):
        """Helper function to encode input with a specific prompt if needed."""
        if prompt:
            return self.encode(input, prompt_name=prompt)
        return self.encode(input)

    def compute_similarity(self, queries: List[str], docs: List[str]) -> np.array:
        """Computes the cosine similarity between query and document embeddings."""
        queries = self.encode_with_prompt(queries, self.query_name)
        docs = self.encode_with_prompt(docs, self.passage_name)
        return self.model.similarity(queries, docs).numpy()


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
