from pytest_mock import MockerFixture

from semsearcheval.query_creator import (
    OpenAIQueryCreator,
    RandomKeywordQueryCreator,
    SearchQueriesOutput,
)


def test_generate_random_queries():
    docs = ["The Eiffel Tower is located in Paris, France."]
    query_creator = RandomKeywordQueryCreator()
    queries = query_creator.get_queries_with_indices(docs)
    assert len(queries) == 10
    assert all([isinstance(query, str) for query in queries["search_query"]])
    assert queries["search_query"][4] == "France Paris"


def test_openai_query_creator(mocker: MockerFixture):
    docs = ["The Eiffel Tower is located in Paris, France."]
    output = SearchQueriesOutput(
        search_queries=[
            {"query": "Where is the Eiffel Tower?"},
            {"query": "What is the height of the Eiffel Tower?"},
            {"query": "What is the history of the Eiffel Tower?"},
            {"query": "What is the best time to visit the Eiffel Tower?"},
            {"query": "What are the opening hours of the Eiffel Tower?"},
            {"query": "How much does it cost to go up the Eiffel Tower?"},
            {"query": "Is there a restaurant in the Eiffel Tower?"},
            {"query": "What is the view like from the Eiffel Tower?"},
            {"query": "How do I get to the Eiffel Tower?"},
            {"query": "Are there any events at the Eiffel Tower?"},
        ]
    )

    mock_parse = mocker.Mock(
        return_value=mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(parsed=output))])
    )

    mock_client = mocker.Mock()
    mock_client.beta.chat.completions.parse = mock_parse

    mocker.patch("semsearcheval.query_creator.OpenAI", return_value=mock_client)
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    query_creator = OpenAIQueryCreator()

    queries = query_creator.get_queries_with_indices(docs)
    assert len(queries) == 10
    assert queries["search_query"][0] == "Where is the Eiffel Tower?"
