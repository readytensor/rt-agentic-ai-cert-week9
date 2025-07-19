import pytest
from unittest.mock import MagicMock
from nodes.a3_nodes import make_title_generator_node
from consts import TITLE, TITLE_APPROVED, TITLE_FEEDBACK, TITLE_GEN_MESSAGES


@pytest.fixture
def title_generator_state():
    """Basic state for title generator testing."""
    return {
        TITLE_APPROVED: False,
        TITLE_GEN_MESSAGES: ["system message"],
        "input_text": "Sample article content about machine learning",
        "manager_brief": "Create an engaging title",
        TITLE_FEEDBACK: "Make it more specific",
    }


def test_title_generator_node_returns_cleaned_title(monkeypatch, title_generator_state):
    """Test that title generator returns properly cleaned title content."""
    # Mock LLM response with extra whitespace
    mock_ai_response = MagicMock()
    mock_ai_response.content = "  Machine Learning in Healthcare: A Comprehensive Review  \n\n  "
    
    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response
    
    # Mock helper functions
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr("nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"])
    monkeypatch.setattr("nodes.a3_nodes._get_reviewer_message", lambda state, feedback_type: ["reviewer message"])
    monkeypatch.setattr("nodes.a3_nodes._get_input_text_message", lambda state: ["input message"])
    monkeypatch.setattr("nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"])
    
    node = make_title_generator_node(llm_model="mock-model")
    result = node(title_generator_state)
    
    assert TITLE in result
    assert result[TITLE] == "Machine Learning in Healthcare: A Comprehensive Review"  # Stripped
    assert result[TITLE_FEEDBACK] == ""  # Reset feedback
    
    # Verify LLM was called
    mock_llm_obj.invoke.assert_called_once()


def test_title_generator_node_skips_when_already_approved(monkeypatch, title_generator_state):
    """Test that title generator skips processing when already approved."""
    title_generator_state[TITLE_APPROVED] = True
    
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    
    node = make_title_generator_node(llm_model="mock-model")
    result = node(title_generator_state)
    
    # Should return empty dict when skipping
    assert result == {}
    
    # LLM should not be called
    mock_llm_obj.invoke.assert_not_called()


def test_title_generator_node_handles_empty_title_response(monkeypatch, title_generator_state):
    """Test that title generator handles empty or whitespace-only responses."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = "   \n\n   "  # Only whitespace
    
    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response
    
    # Mock helper functions
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr("nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"])
    monkeypatch.setattr("nodes.a3_nodes._get_reviewer_message", lambda state, feedback_type: ["reviewer message"])
    monkeypatch.setattr("nodes.a3_nodes._get_input_text_message", lambda state: ["input message"])
    monkeypatch.setattr("nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"])
    
    node = make_title_generator_node(llm_model="mock-model")
    result = node(title_generator_state)
    
    assert TITLE in result
    assert result[TITLE] == ""  # Should be empty string after strip
    assert result[TITLE_FEEDBACK] == ""


def test_title_generator_node_handles_none_content_response(monkeypatch, title_generator_state):
    """Test that title generator handles LLM response with None content."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = None
    
    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response
    
    # Mock helper functions
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr("nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"])
    monkeypatch.setattr("nodes.a3_nodes._get_reviewer_message", lambda state, feedback_type: ["reviewer message"])
    monkeypatch.setattr("nodes.a3_nodes._get_input_text_message", lambda state: ["input message"])
    monkeypatch.setattr("nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"])
    
    node = make_title_generator_node(llm_model="mock-model")
    result = node(title_generator_state)
    
    # Should handle None gracefully and return empty string
    assert TITLE in result
    assert result[TITLE] == ""
    assert result[TITLE_FEEDBACK] == ""
    
    # Verify LLM was still called
    mock_llm_obj.invoke.assert_called_once()


def test_title_generator_node_constructs_correct_message_sequence(monkeypatch, title_generator_state):
    """Test that title generator constructs messages in correct order."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = "Generated Title"
    
    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response
    
    # Mock helper functions with identifiable returns
    mock_manager_msg = "manager_brief_message"
    mock_reviewer_msg = "reviewer_feedback_message"
    mock_input_msg = "input_text_message"
    mock_begin_msg = "begin_task_message"
    
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr("nodes.a3_nodes._get_manager_brief_message", lambda state: mock_manager_msg)
    monkeypatch.setattr("nodes.a3_nodes._get_reviewer_message", lambda state, feedback_type: mock_reviewer_msg)
    monkeypatch.setattr("nodes.a3_nodes._get_input_text_message", lambda state: mock_input_msg)
    monkeypatch.setattr("nodes.a3_nodes._get_begin_task_message", lambda: mock_begin_msg)
    
    node = make_title_generator_node(llm_model="mock-model")
    result = node(title_generator_state)
    
    # Verify the correct message sequence was passed to LLM
    expected_messages = [
        "system message",           # from TITLE_GEN_MESSAGES
        "manager_brief_message",    # from _get_manager_brief_message
        "reviewer_feedback_message", # from _get_reviewer_message
        "input_text_message",       # from _get_input_text_message
        "begin_task_message",       # from _get_begin_task_message
    ]
    
    mock_llm_obj.invoke.assert_called_once_with(expected_messages)
    assert result[TITLE] == "Generated Title"


def test_title_generator_node_propagates_llm_exceptions(monkeypatch, title_generator_state):
    """Test that title generator properly propagates LLM exceptions."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.side_effect = Exception("LLM API Error")
    
    # Mock helper functions
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr("nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"])
    monkeypatch.setattr("nodes.a3_nodes._get_reviewer_message", lambda state, feedback_type: ["reviewer message"])
    monkeypatch.setattr("nodes.a3_nodes._get_input_text_message", lambda state: ["input message"])
    monkeypatch.setattr("nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"])
    
    node = make_title_generator_node(llm_model="mock-model")
    
    with pytest.raises(Exception, match="LLM API Error"):
        node(title_generator_state)


def test_title_generator_node_passes_correct_feedback_type(monkeypatch, title_generator_state):
    """Test that title generator passes the correct feedback type to reviewer message helper."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = "Generated Title"
    
    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response
    
    # Track calls to _get_reviewer_message
    mock__get_reviewer_message = MagicMock(return_value=["reviewer message"])
    
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr("nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"])
    monkeypatch.setattr("nodes.a3_nodes._get_reviewer_message", mock__get_reviewer_message)
    monkeypatch.setattr("nodes.a3_nodes._get_input_text_message", lambda state: ["input message"])
    monkeypatch.setattr("nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"])
    
    node = make_title_generator_node(llm_model="mock-model")
    result = node(title_generator_state)
    
    # Verify _get_reviewer_message was called with correct feedback type
    mock__get_reviewer_message.assert_called_once_with(title_generator_state, TITLE_FEEDBACK)
    assert result[TITLE] == "Generated Title"


def test_title_generator_node_resets_title_feedback(monkeypatch, title_generator_state):
    """Test that title generator resets TITLE_FEEDBACK to empty string."""
    # Set some existing feedback
    title_generator_state[TITLE_FEEDBACK] = "Previous feedback about the title"
    
    mock_ai_response = MagicMock()
    mock_ai_response.content = "New Generated Title"
    
    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response
    
    # Mock helper functions
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr("nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"])
    monkeypatch.setattr("nodes.a3_nodes._get_reviewer_message", lambda state, feedback_type: ["reviewer message"])
    monkeypatch.setattr("nodes.a3_nodes._get_input_text_message", lambda state: ["input message"])
    monkeypatch.setattr("nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"])
    
    node = make_title_generator_node(llm_model="mock-model")
    result = node(title_generator_state)
    
    assert result[TITLE] == "New Generated Title"
    assert result[TITLE_FEEDBACK] == ""  # Should be reset to empty string