import pytest
from unittest.mock import MagicMock

from nodes.a3_nodes import make_manager_node
from consts import MANAGER_BRIEF, MANAGER_MESSAGES, INPUT_TEXT


def test_manager_node_returns_stripped_output(monkeypatch):
    """Test that manager node returns properly cleaned brief content."""
    # Mock LLM response with extra whitespace
    mock_ai_response = MagicMock()
    mock_ai_response.content = "  This is the manager's brief for the task.  \n\n  "

    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_manager_node(llm_model="mock-model")

    state = {
        INPUT_TEXT: "Sample article content about machine learning.",
        MANAGER_MESSAGES: ["system message"],
    }

    result = node(state)

    assert MANAGER_BRIEF in result
    assert (
        result[MANAGER_BRIEF] == "This is the manager's brief for the task."
    )  # Stripped

    # Verify LLM was called
    mock_llm_obj.invoke.assert_called_once()


def test_manager_node_handles_empty_brief(monkeypatch):
    """Test that manager node handles empty or whitespace-only responses."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = "   \n\n   "  # Only whitespace

    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_manager_node(llm_model="mock-model")

    state = {
        INPUT_TEXT: "Sample content",
        MANAGER_MESSAGES: ["system message"],
    }

    result = node(state)

    assert MANAGER_BRIEF in result
    assert result[MANAGER_BRIEF] == ""  # Should be empty string after strip


def test_manager_node_raises_error_on_empty_input_text(monkeypatch):
    """Test that manager node raises ValueError for empty input text."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_manager_node(llm_model="mock-model")

    # Test with None input
    state = {
        INPUT_TEXT: None,
        MANAGER_MESSAGES: ["system message"],
    }

    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(state)

    # Test with empty string
    state[INPUT_TEXT] = ""
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(state)

    # Test with whitespace only
    state[INPUT_TEXT] = "   \n\n   "
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(state)

    # Verify LLM was never called
    mock_llm_obj.invoke.assert_not_called()


def test_manager_node_passes_correct_messages_to_llm(monkeypatch):
    """Test that manager node constructs and passes correct messages to LLM."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = "Manager brief response"

    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response

    # Mock the helper functions
    mock_input_text_message = {"role": "human", "content": "Input text message"}
    mock_begin_task_message = {"role": "human", "content": "Begin task message"}

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes._get_input_text_message", lambda _: mock_input_text_message
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_begin_task_message", lambda: mock_begin_task_message
    )

    node = make_manager_node(llm_model="mock-model")

    initial_messages = [
        {"role": "system", "content": "System message 1"},
        {"role": "system", "content": "System message 2"},
    ]
    state = {
        INPUT_TEXT: "Valid input text content",
        MANAGER_MESSAGES: initial_messages,
    }

    result = node(state)

    # Verify the correct messages were passed to LLM
    expected_messages = [
        {"role": "system", "content": "System message 1"},
        {"role": "system", "content": "System message 2"},
        {"role": "human", "content": "Input text message"},
        {"role": "human", "content": "Begin task message"},
    ]

    mock_llm_obj.invoke.assert_called_once_with(expected_messages)
    assert result[MANAGER_BRIEF] == "Manager brief response"


def test_manager_node_with_various_input_text_lengths(monkeypatch):
    """Test that manager node handles different input text lengths."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = "Brief content"

    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_manager_node(llm_model="mock-model")

    # Test short input
    state = {
        INPUT_TEXT: "Short",
        MANAGER_MESSAGES: [],
    }
    result = node(state)
    assert MANAGER_BRIEF in result

    # Test long input
    long_text = "A" * 10000  # Very long text
    state[INPUT_TEXT] = long_text
    result = node(state)
    assert MANAGER_BRIEF in result

    # Test input with special characters
    special_text = "Text with émojis 🚀 and spëcial chars & symbols!"
    state[INPUT_TEXT] = special_text
    result = node(state)
    assert MANAGER_BRIEF in result


def test_manager_node_propagates_llm_exceptions(monkeypatch):
    """Test that manager node properly propagates LLM exceptions."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.side_effect = Exception("LLM API Error")

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_manager_node(llm_model="mock-model")

    state = {
        INPUT_TEXT: "Valid input text",
        MANAGER_MESSAGES: ["system message"],
    }

    with pytest.raises(Exception, match="LLM API Error"):
        node(state)


def test_manager_node_handles_none_content_response(monkeypatch):
    """Test that manager node gracefully handles LLM response with None content."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = None

    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_manager_node(llm_model="mock-model")

    state = {
        INPUT_TEXT: "Valid input text",
        MANAGER_MESSAGES: ["system message"],
    }

    result = node(state)

    # Should handle None gracefully and return empty string
    assert MANAGER_BRIEF in result
    assert result[MANAGER_BRIEF] == ""

    # Verify LLM was still called
    mock_llm_obj.invoke.assert_called_once()
