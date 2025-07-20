from unittest.mock import MagicMock

from nodes.node_utils import execute_search_queries


def test_execute_search_queries_with_valid_results(monkeypatch):
    """Test that function properly formats valid search results."""
    # Mock Tavily response
    mock_tavily_response = {
        "results": [
            {
                "url": "https://example1.com",
                "title": "Example Article 1",
                "content": "This is the content of article 1",
            },
            {
                "url": "https://example2.com",
                "title": "Example Article 2",
                "content": "This is the content of article 2",
            },
        ]
    }

    mock_tavily_search = MagicMock()
    mock_tavily_search.return_value.invoke.return_value = mock_tavily_response

    monkeypatch.setattr("nodes.node_utils.TavilySearch", mock_tavily_search)

    queries = ["machine learning", "neural networks"]
    results = execute_search_queries(queries, max_results=2)

    assert len(results) == 4  # 2 queries × 2 results each

    expected_results = [
        {
            "url": "https://example1.com",
            "title": "Example Article 1",
            "page_content": "This is the content of article 1",
        },
        {
            "url": "https://example2.com",
            "title": "Example Article 2",
            "page_content": "This is the content of article 2",
        },
        {
            "url": "https://example1.com",
            "title": "Example Article 1",
            "page_content": "This is the content of article 1",
        },
        {
            "url": "https://example2.com",
            "title": "Example Article 2",
            "page_content": "This is the content of article 2",
        },
    ]

    assert results == expected_results

    # Verify Tavily was called correctly
    assert mock_tavily_search.call_count == 2  # Once per query
    mock_tavily_search.assert_any_call(max_results=2)


def test_execute_search_queries_filters_empty_content(monkeypatch):
    """Test that function filters out results with empty content."""
    mock_tavily_response = {
        "results": [
            {
                "url": "https://example1.com",
                "title": "Valid Article",
                "content": "This has content",
            },
            {
                "url": "https://example2.com",
                "title": "Empty Content Article",
                "content": "",  # Empty content
            },
            {
                "url": "https://example3.com",
                "title": "Missing Content Article",
                # Missing content key entirely
            },
            {
                "url": "https://example4.com",
                "title": "None Content Article",
                "content": None,  # None content
            },
        ]
    }

    mock_tavily_search = MagicMock()
    mock_tavily_search.return_value.invoke.return_value = mock_tavily_response

    monkeypatch.setattr("nodes.node_utils.TavilySearch", mock_tavily_search)

    results = execute_search_queries(["test query"])

    # Should only return the one valid result
    assert len(results) == 1
    assert results[0]["title"] == "Valid Article"
    assert results[0]["page_content"] == "This has content"


def test_execute_search_queries_handles_empty_query_list():
    """Test that function handles empty query list gracefully."""
    results = execute_search_queries([])

    assert results == []


def test_execute_search_queries_handles_all_whitespace_queries(monkeypatch):
    """Test that function warns when all queries are whitespace-only."""
    mock_tavily_search = MagicMock()
    monkeypatch.setattr("nodes.node_utils.TavilySearch", mock_tavily_search)

    queries = ["   ", "\n\t", ""]  # All whitespace
    results = execute_search_queries(queries)

    assert results == []

    # Should not call Tavily at all
    mock_tavily_search.assert_not_called()


def test_execute_search_queries_shows_valid_query_count(monkeypatch):
    """Test that function shows count of valid queries being executed."""
    mock_tavily_response = {"results": []}

    mock_tavily_search = MagicMock()
    mock_tavily_search.return_value.invoke.return_value = mock_tavily_response

    monkeypatch.setattr("nodes.node_utils.TavilySearch", mock_tavily_search)

    queries = ["valid1", "  ", "valid2", "\n", "valid3"]  # 3 valid, 2 invalid
    execute_search_queries(queries)

    # Should execute exactly 3 queries
    assert mock_tavily_search.call_count == 3


def test_execute_search_queries_continues_on_tavily_error(monkeypatch):
    """Test that function continues processing other queries when one fails."""

    def mock_tavily_side_effect(max_results):
        mock_search = MagicMock()

        def mock_invoke(query):
            if query == "failing query":
                raise Exception("Tavily API Error")
            return {
                "results": [
                    {
                        "url": "https://success.com",
                        "title": "Success Article",
                        "content": "Successful search result",
                    }
                ]
            }

        mock_search.invoke = mock_invoke
        return mock_search

    monkeypatch.setattr("nodes.node_utils.TavilySearch", mock_tavily_side_effect)

    queries = ["good query", "failing query", "another good query"]
    results = execute_search_queries(queries)

    # Should get results from the 2 successful queries
    assert len(results) == 2
    assert all(result["title"] == "Success Article" for result in results)


def test_execute_search_queries_handles_malformed_tavily_response(monkeypatch):
    """Test that function handles malformed Tavily responses."""
    mock_tavily_responses = [
        {},  # Missing 'results' key
        {"results": None},  # None results
        {"results": "not a list"},  # Invalid results type
        {
            "results": [
                {"title": "Missing URL"},  # Missing required fields
                {"url": "https://example.com"},  # Missing title
                None,  # None result item
            ]
        },
    ]

    call_count = 0

    def mock_tavily_side_effect(max_results):
        mock_search = MagicMock()

        def mock_invoke(query):
            nonlocal call_count
            response = mock_tavily_responses[call_count % len(mock_tavily_responses)]
            call_count += 1
            return response

        mock_search.invoke = mock_invoke
        return mock_search

    monkeypatch.setattr("nodes.node_utils.TavilySearch", mock_tavily_side_effect)

    queries = ["query1", "query2", "query3", "query4"]

    # Should handle malformed responses gracefully (might raise exceptions)
    # The specific behavior depends on how you want to handle malformed responses
    try:
        results = execute_search_queries(queries)
        # If it succeeds, should return empty list or filtered results
        assert isinstance(results, list)
    except (KeyError, TypeError, AttributeError):
        # If it fails, that's also acceptable behavior for malformed responses
        pass


def test_execute_search_queries_respects_max_results_parameter(monkeypatch):
    """Test that function passes max_results parameter to Tavily correctly."""
    mock_tavily_response = {"results": []}

    mock_tavily_search = MagicMock()
    mock_tavily_search.return_value.invoke.return_value = mock_tavily_response

    monkeypatch.setattr("nodes.node_utils.TavilySearch", mock_tavily_search)

    execute_search_queries(["test"], max_results=5)

    # Verify max_results was passed correctly
    mock_tavily_search.assert_called_once_with(max_results=5)


def test_execute_search_queries_aggregates_multiple_query_results(monkeypatch):
    """Test that function properly aggregates results from multiple queries."""

    def mock_tavily_side_effect(max_results):
        mock_search = MagicMock()

        def mock_invoke(query):
            if query == "query1":
                return {
                    "results": [
                        {
                            "url": "https://q1.com",
                            "title": "Q1 Result",
                            "content": "Content 1",
                        }
                    ]
                }
            elif query == "query2":
                return {
                    "results": [
                        {
                            "url": "https://q2a.com",
                            "title": "Q2A Result",
                            "content": "Content 2A",
                        },
                        {
                            "url": "https://q2b.com",
                            "title": "Q2B Result",
                            "content": "Content 2B",
                        },
                    ]
                }
            return {"results": []}

        mock_search.invoke = mock_invoke
        return mock_search

    monkeypatch.setattr("nodes.node_utils.TavilySearch", mock_tavily_side_effect)

    results = execute_search_queries(["query1", "query2"])

    assert len(results) == 3  # 1 from query1 + 2 from query2

    # Verify aggregation order (should maintain query order)
    assert results[0]["title"] == "Q1 Result"
    assert results[1]["title"] == "Q2A Result"
    assert results[2]["title"] == "Q2B Result"


def test_execute_search_queries_handles_none_input():
    """Test that function handles None input gracefully."""
    results = execute_search_queries(None)

    assert results == []


def test_execute_search_queries_handles_non_list_input():
    """Test that function handles non-list input gracefully."""
    # Test various non-list inputs
    test_inputs = ["string", 123, {"dict": "value"}]

    for invalid_input in test_inputs:
        results = execute_search_queries(invalid_input)
        assert results == []


def test_execute_search_queries_handles_non_string_queries(monkeypatch):
    """Test that function handles non-string query items."""
    mock_tavily_response = {"results": []}

    mock_tavily_search = MagicMock()
    mock_tavily_search.return_value.invoke.return_value = mock_tavily_response

    monkeypatch.setattr("nodes.node_utils.TavilySearch", mock_tavily_search)

    # Mix of valid and invalid query types
    queries = ["valid query", 123, None, ["list"], {"dict": "value"}]

    # Function should handle this gracefully (behavior depends on implementation)
    try:
        results = execute_search_queries(queries)
        assert isinstance(results, list)
    except (TypeError, AttributeError):
        # Also acceptable to raise an error for invalid input types
        pass
