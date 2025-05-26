import typing as t

import numpy as np
from pydantic import BaseModel
from pytest_mock import MockerFixture

from semsearcheval.models import BM25Model, HuggingFaceModel, OpenAIModel


def test_bm25_model(test_queries, test_docs):
    bm25 = BM25Model("bm25", "de_core_news_sm")
    bm25.load_model()
    similarity = bm25.compute_similarity(test_queries, test_docs)

    assert similarity.shape == (3, 3)
    assert np.argmax(similarity[0]) == 0
    assert np.argmax(similarity[1]) == 1
    assert np.argmax(similarity[2]) == 2


def test_huggingface_model(test_queries, test_docs):
    hf_model = HuggingFaceModel("hf_model", "sentence-transformers/all-MiniLM-L6-v2")
    hf_model.load_model()
    similarity = hf_model.compute_similarity(test_queries, test_docs)

    assert similarity.shape == (3, 3)
    assert np.argmax(similarity[0]) == 0
    assert np.argmax(similarity[1]) == 1
    assert np.argmax(similarity[2]) == 2


def test_openai_model(mocker: MockerFixture, test_queries, test_docs):
    class EmbeddingResult(BaseModel):
        embedding: t.List[float]

    class EmbeddingResponse(BaseModel):
        data: t.List[EmbeddingResult]

    output = EmbeddingResponse(
        data=[
            EmbeddingResult(embedding=[0.1, 0.2, 0.3]),
            EmbeddingResult(embedding=[0.4, 0.5, 0.6]),
            EmbeddingResult(embedding=[0.7, 0.8, 0.9]),
        ]
    )

    mock_parse = mocker.Mock(return_value=output)

    mock_client = mocker.Mock()
    mock_client.embeddings.create = mock_parse

    mocker.patch("semsearcheval.models.OpenAI", return_value=mock_client)
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    model = OpenAIModel("open-ai-3-small", "text-embedding-3-small")
    model.load_model()

    similarity = model.compute_similarity(test_queries, test_docs)
    assert similarity.shape == (3, 3)
    assert np.argmax(similarity[0]) == 0
    assert np.argmax(similarity[1]) == 1
    assert np.argmax(similarity[2]) == 2
