import pytest
from unittest.mock import MagicMock, PropertyMock, patch
from nodes.a3_nodes import make_references_generator_node
from consts import (
    REFERENCES_GEN_MESSAGES,
    INPUT_TEXT,
    MANAGER_BRIEF,
    REFERENCES_FEEDBACK,
    REFERENCES_APPROVED,
    REFERENCE_SEARCH_QUERIES,
    CANDIDATE_REFERENCES,
)


@pytest.fixture
def mock_queries():
    return ["machine learning algorithms", "neural networks", "deep learning"]


@pytest.fixture
def mock_search_results():
    return [
        {
            "url": "https://example1.com/ml-paper",
            "title": "Machine Learning Fundamentals",
            "page_content": "This paper discusses the fundamentals of machine learning algorithms...",
        },
        {
            "url": "https://example2.com/neural-nets",
            "title": "Neural Networks in Practice",
            "page_content": "Neural networks have revolutionized the field of artificial intelligence...",
        },
        {
            "url": "https://example3.com/deep-learning",
            "title": "Deep Learning Architectures",
            "page_content": "Deep learning models have shown remarkable success in various domains...",
        },
    ]


@pytest.fixture
def empty_search_results():
    return []


@pytest.fixture
def references_state():
    return {
        INPUT_TEXT: "Sample research paper about machine learning algorithms",
        MANAGER_BRIEF: "This paper discusses ML algorithms",
        REFERENCES_GEN_MESSAGES: ["system_message"],
        REFERENCES_FEEDBACK: "",
        REFERENCES_APPROVED: False,
    }


def test_references_generator_node_skips_when_already_approved(
    monkeypatch, references_state
):
    """Test that node skips processing when references are already approved."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    # Set references as already approved - need to modify the dict, not reassign
    references_state[REFERENCES_APPROVED] = True

    node = make_references_generator_node(llm_model="mock-model")
    result = node(references_state)

    # Should return empty dict and not call LLM
    assert result == {}
    mock_llm_obj.with_structured_output.assert_not_called()


def test_references_generator_node_generates_queries_and_searches(
    monkeypatch, references_state, mock_queries, mock_search_results
):
    """Test successful query generation and search execution."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.queries = (
        mock_queries
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes.execute_search_queries", lambda queries: mock_search_results
    )

    node = make_references_generator_node(llm_model="mock-model")
    result = node(references_state)

    # Verify results
    assert REFERENCE_SEARCH_QUERIES in result
    assert CANDIDATE_REFERENCES in result
    assert result[REFERENCE_SEARCH_QUERIES] == mock_queries
    assert result[CANDIDATE_REFERENCES] == mock_search_results


def test_references_generator_node_constructs_correct_messages(
    monkeypatch, references_state, mock_queries, empty_search_results
):
    """Test that node constructs the correct message sequence for LLM."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.queries = (
        mock_queries
    )

    # Mock helper functions with identifiable returns
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes._get_manager_brief_message", lambda state: "manager_message"
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_reviewer_message",
        lambda state, feedback_key: "reviewer_message",
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_input_text_message", lambda state: "input_text_message"
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_begin_task_message", lambda: "begin_task_message"
    )
    monkeypatch.setattr(
        "nodes.a3_nodes.execute_search_queries", lambda queries: empty_search_results
    )

    # Add additional system messages to test spreading
    references_state[REFERENCES_GEN_MESSAGES] = ["system_message_1", "system_message_2"]

    node = make_references_generator_node(llm_model="mock-model")
    result = node(references_state)

    # Verify the LLM was called with the correct message sequence
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]

    assert isinstance(call_args, list)
    assert len(call_args) == 6  # 2 system + manager + reviewer + input + begin_task

    # Verify message order
    expected_messages = [
        "system_message_1",
        "system_message_2",
        "manager_message",
        "reviewer_message",
        "input_text_message",
        "begin_task_message",
    ]
    assert call_args == expected_messages


def test_references_generator_node_handles_llm_exceptions(
    monkeypatch, references_state
):
    """Test that node properly wraps and propagates LLM exceptions."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.side_effect = Exception(
        "LLM API Error"
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_references_generator_node(llm_model="mock-model")

    # Should raise RuntimeError wrapping the original exception
    with pytest.raises(RuntimeError, match="References extraction failed"):
        node(references_state)


def test_references_generator_node_handles_search_exceptions(
    monkeypatch, references_state, mock_queries
):
    """Test that node handles search execution exceptions."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.queries = (
        mock_queries
    )

    # Mock search function to raise exception
    def mock_search_with_error(queries):
        raise Exception("Search API Error")

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr("nodes.a3_nodes.execute_search_queries", mock_search_with_error)

    node = make_references_generator_node(llm_model="mock-model")

    # Should raise RuntimeError wrapping the search exception
    with pytest.raises(RuntimeError, match="References extraction failed"):
        node(references_state)


def test_references_generator_node_handles_empty_queries(
    monkeypatch, references_state, empty_search_results
):
    """Test that node handles empty query list from LLM."""
    empty_queries = []  # Empty queries

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.queries = (
        empty_queries
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes.execute_search_queries", lambda queries: empty_search_results
    )

    node = make_references_generator_node(llm_model="mock-model")
    result = node(references_state)

    # Should handle gracefully
    assert result[REFERENCE_SEARCH_QUERIES] == []
    assert result[CANDIDATE_REFERENCES] == []


def test_references_generator_node_verifies_search_function_called_correctly(
    monkeypatch, references_state, mock_queries, mock_search_results
):
    """Test that the search function is called with the correct queries."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.queries = (
        mock_queries
    )

    # Create a mock search function that we can verify was called correctly
    mock_search_function = MagicMock(return_value=mock_search_results)

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr("nodes.a3_nodes.execute_search_queries", mock_search_function)

    node = make_references_generator_node(llm_model="mock-model")
    result = node(references_state)

    # Verify the search function was called with the correct queries
    mock_search_function.assert_called_once_with(mock_queries)

    # Verify results are returned correctly
    assert result[REFERENCE_SEARCH_QUERIES] == mock_queries
    assert result[CANDIDATE_REFERENCES] == mock_search_results


def test_references_generator_node_handles_missing_queries_attribute(
    monkeypatch, references_state
):
    """Test that node handles LLM response without queries attribute."""
    # Create a mock response object that raises AttributeError when accessing .queries
    mock_llm_response = MagicMock()
    del (
        mock_llm_response.queries
    )  # Remove the queries attribute to force AttributeError

    # Configure the mock to raise AttributeError when .queries is accessed
    type(mock_llm_response).queries = PropertyMock(
        side_effect=AttributeError("'MockResponse' object has no attribute 'queries'")
    )

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value = (
        mock_llm_response
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_references_generator_node(llm_model="mock-model")

    # Should raise RuntimeError wrapping the AttributeError
    with pytest.raises(RuntimeError, match="References extraction failed"):
        node(references_state)
