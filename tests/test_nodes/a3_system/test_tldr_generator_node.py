import pytest
from unittest.mock import MagicMock
from nodes.a3_nodes import make_tldr_generator_node
from consts import TLDR, TLDR_APPROVED, TLDR_FEEDBACK, TLDR_GEN_MESSAGES


@pytest.fixture
def tldr_generator_state():
    """Basic state for TL;DR generator testing."""
    return {
        TLDR_APPROVED: False,
        TLDR_GEN_MESSAGES: ["system message"],
        "input_text": "Sample article content about machine learning applications in healthcare",
        "manager_brief": "Create a concise summary",
        TLDR_FEEDBACK: "Make it more technical",
    }


def test_tldr_generator_node_returns_cleaned_tldr(monkeypatch, tldr_generator_state):
    """Test that TL;DR generator returns properly cleaned summary content."""
    # Mock LLM response with extra whitespace
    mock_ai_response = MagicMock()
    mock_ai_response.content = "  This study demonstrates ML applications in healthcare, improving diagnosis accuracy by 25%.  \n\n  "

    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response

    # Mock helper functions
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"]
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_reviewer_message",
        lambda state, feedback_type: ["reviewer message"],
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_input_text_message", lambda state: ["input message"]
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"]
    )

    node = make_tldr_generator_node(llm_model="mock-model")
    result = node(tldr_generator_state)

    assert TLDR in result
    assert (
        result[TLDR]
        == "This study demonstrates ML applications in healthcare, improving diagnosis accuracy by 25%."
    )  # Stripped
    assert result[TLDR_FEEDBACK] == ""  # Reset feedback

    # Verify LLM was called
    mock_llm_obj.invoke.assert_called_once()


def test_tldr_generator_node_skips_when_already_approved(
    monkeypatch, tldr_generator_state
):
    """Test that TL;DR generator skips processing when already approved."""
    tldr_generator_state[TLDR_APPROVED] = True

    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tldr_generator_node(llm_model="mock-model")
    result = node(tldr_generator_state)

    # Should return empty dict when skipping
    assert result == {}

    # LLM should not be called
    mock_llm_obj.invoke.assert_not_called()


def test_tldr_generator_node_handles_none_content_response(
    monkeypatch, tldr_generator_state
):
    """Test that TL;DR generator handles LLM response with None content gracefully."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = None

    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response

    # Mock helper functions
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"]
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_reviewer_message",
        lambda state, feedback_type: ["reviewer message"],
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_input_text_message", lambda state: ["input message"]
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"]
    )

    node = make_tldr_generator_node(llm_model="mock-model")
    result = node(tldr_generator_state)

    # Should handle None gracefully and return empty string
    assert TLDR in result
    assert result[TLDR] == ""
    assert result[TLDR_FEEDBACK] == ""

    # Verify LLM was still called
    mock_llm_obj.invoke.assert_called_once()


def test_tldr_generator_node_constructs_correct_message_sequence(
    monkeypatch, tldr_generator_state
):
    """Test that TL;DR generator constructs messages in correct order."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = "Generated TL;DR Summary"

    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response

    # Mock helper functions with identifiable returns
    mock_manager_msg = "manager_brief_message"
    mock_reviewer_msg = "reviewer_feedback_message"
    mock_input_msg = "input_text_message"
    mock_begin_msg = "begin_task_message"

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes._get_manager_brief_message", lambda state: mock_manager_msg
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_reviewer_message",
        lambda state, feedback_type: mock_reviewer_msg,
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_input_text_message", lambda state: mock_input_msg
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_begin_task_message", lambda: mock_begin_msg
    )

    node = make_tldr_generator_node(llm_model="mock-model")
    result = node(tldr_generator_state)

    # Verify the correct message sequence was passed to LLM
    expected_messages = [
        "system message",  # from TLDR_GEN_MESSAGES
        "manager_brief_message",  # from _get_manager_brief_message
        "reviewer_feedback_message",  # from _get_reviewer_message
        "input_text_message",  # from _get_input_text_message
        "begin_task_message",  # from _get_begin_task_message
    ]

    mock_llm_obj.invoke.assert_called_once_with(expected_messages)
    assert result[TLDR] == "Generated TL;DR Summary"


def test_tldr_generator_node_passes_correct_feedback_type(
    monkeypatch, tldr_generator_state
):
    """Test that TL;DR generator passes the correct feedback type to reviewer message helper."""
    mock_ai_response = MagicMock()
    mock_ai_response.content = "Generated Summary"

    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.return_value = mock_ai_response

    # Track calls to _get_reviewer_message
    mock__get_reviewer_message = MagicMock(return_value=["reviewer message"])

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"]
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_reviewer_message", mock__get_reviewer_message
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_input_text_message", lambda state: ["input message"]
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"]
    )

    node = make_tldr_generator_node(llm_model="mock-model")
    result = node(tldr_generator_state)

    # Verify _get_reviewer_message was called with correct feedback type
    mock__get_reviewer_message.assert_called_once_with(
        tldr_generator_state, TLDR_FEEDBACK
    )
    assert result[TLDR] == "Generated Summary"


def test_tldr_generator_node_propagates_llm_exceptions(
    monkeypatch, tldr_generator_state
):
    """Test that TL;DR generator properly propagates LLM exceptions."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.invoke.side_effect = Exception("LLM API Error")

    # Mock helper functions
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes._get_manager_brief_message", lambda state: ["manager message"]
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_reviewer_message",
        lambda state, feedback_type: ["reviewer message"],
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_input_text_message", lambda state: ["input message"]
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_begin_task_message", lambda: ["begin message"]
    )

    node = make_tldr_generator_node(llm_model="mock-model")

    with pytest.raises(Exception, match="LLM API Error"):
        node(tldr_generator_state)
