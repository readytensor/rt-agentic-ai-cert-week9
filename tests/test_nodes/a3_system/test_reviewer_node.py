import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from nodes.a3_nodes import make_reviewer_node
from consts import (
    REVIEWER_MESSAGES,
    INPUT_TEXT,
    MANAGER_BRIEF,
    TITLE,
    TLDR,
    SELECTED_REFERENCES,
    REVISION_ROUND,
    MAX_REVISIONS,
    NEEDS_REVISION,
    TITLE_APPROVED,
    TLDR_APPROVED,
    REFERENCES_APPROVED,
    TITLE_FEEDBACK,
    TLDR_FEEDBACK,
    REFERENCES_FEEDBACK,
)


@pytest.fixture
def reviewer_state():
    return {
        INPUT_TEXT: "Sample research paper about machine learning algorithms",
        MANAGER_BRIEF: "This paper discusses ML algorithms",
        REVIEWER_MESSAGES: ["system_message"],
        TITLE: "Machine Learning in Practice",
        TLDR: "This paper explores various ML algorithms and their applications",
        SELECTED_REFERENCES: [
            {
                "url": "https://example1.com/ml-paper",
                "title": "ML Fundamentals",
                "page_content": "ML content...",
            }
        ],
        REVISION_ROUND: 0,
        MAX_REVISIONS: 3,
    }


@pytest.fixture
def mock_review_response_all_approved():
    """Mock ReviewOutput with all components approved."""
    response = MagicMock()
    response.title_approved = True
    response.tldr_approved = True
    response.references_approved = True
    response.title_feedback = "Great title!"
    response.tldr_feedback = "Clear summary!"
    response.references_feedback = "Good sources!"
    return response


@pytest.fixture
def mock_review_response_needs_revision():
    """Mock ReviewOutput with some components needing revision."""
    response = MagicMock()
    response.title_approved = False
    response.tldr_approved = True
    response.references_approved = False
    response.title_feedback = "Make title more specific"
    response.tldr_feedback = "TLDR looks good"
    response.references_feedback = "Add more recent sources"
    return response


def test_reviewer_node_validates_input_text(monkeypatch, reviewer_state):
    """Test that node validates input text and raises appropriate errors."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_reviewer_node(llm_model="mock-model")

    # Test with None input text
    reviewer_state[INPUT_TEXT] = None
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(reviewer_state)

    # Test with empty string input text
    reviewer_state[INPUT_TEXT] = ""
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(reviewer_state)

    # Test with whitespace-only input text
    reviewer_state[INPUT_TEXT] = "   \n\t  "
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(reviewer_state)


def test_reviewer_node_forces_approval_at_max_revisions(monkeypatch, reviewer_state):
    """Test that node forces approval when max revisions is reached."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    # Set revision round to max revisions
    reviewer_state[REVISION_ROUND] = 3
    reviewer_state[MAX_REVISIONS] = 3

    node = make_reviewer_node(llm_model="mock-model")
    result = node(reviewer_state)

    # Should force approval without calling LLM
    assert result[NEEDS_REVISION] is False
    assert result[TITLE_APPROVED] is True
    assert result[TLDR_APPROVED] is True
    assert result[REFERENCES_APPROVED] is True
    mock_llm_obj.with_structured_output.assert_not_called()


def test_reviewer_node_all_components_approved(
    monkeypatch, reviewer_state, mock_review_response_all_approved
):
    """Test successful review with all components approved."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value = (
        mock_review_response_all_approved
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_reviewer_node(llm_model="mock-model")
    result = node(reviewer_state)

    # Verify all approved
    assert result[NEEDS_REVISION] is False
    assert result[TITLE_APPROVED] is True
    assert result[TLDR_APPROVED] is True
    assert result[REFERENCES_APPROVED] is True
    assert result[REVISION_ROUND] == 1  # Should increment

    # Verify feedback is included
    assert result[TITLE_FEEDBACK] == "Great title!"
    assert result[TLDR_FEEDBACK] == "Clear summary!"
    assert result[REFERENCES_FEEDBACK] == "Good sources!"


def test_reviewer_node_some_components_need_revision(
    monkeypatch, reviewer_state, mock_review_response_needs_revision
):
    """Test review with some components needing revision."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value = (
        mock_review_response_needs_revision
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_reviewer_node(llm_model="mock-model")
    result = node(reviewer_state)

    # Verify needs revision
    assert result[NEEDS_REVISION] is True
    assert result[TITLE_APPROVED] is False
    assert result[TLDR_APPROVED] is True
    assert result[REFERENCES_APPROVED] is False
    assert result[REVISION_ROUND] == 1  # Should increment

    # Verify feedback is included
    assert result[TITLE_FEEDBACK] == "Make title more specific"
    assert result[TLDR_FEEDBACK] == "TLDR looks good"
    assert result[REFERENCES_FEEDBACK] == "Add more recent sources"


def test_reviewer_node_constructs_correct_messages(
    monkeypatch, reviewer_state, mock_review_response_all_approved
):
    """Test that node constructs the correct message sequence for LLM."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value = (
        mock_review_response_all_approved
    )

    # Mock helper functions with identifiable returns
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes._get_manager_brief_message", lambda state: "manager_message"
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_input_text_message", lambda state: "input_text_message"
    )
    monkeypatch.setattr(
        "nodes.a3_nodes._get_begin_task_message", lambda: "begin_task_message"
    )
    monkeypatch.setattr(
        "nodes.a3_nodes.format_references_for_prompt",
        lambda refs: "formatted_references",
    )

    # Add additional system messages to test spreading
    reviewer_state[REVIEWER_MESSAGES] = ["system_message_1", "system_message_2"]

    node = make_reviewer_node(llm_model="mock-model")
    result = node(reviewer_state)

    # Verify the LLM was called with the correct message sequence
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]

    assert isinstance(call_args, list)
    assert len(call_args) == 6  # 2 system + manager + input + review + begin_task

    # Verify message order and types
    assert call_args[0] == "system_message_1"
    assert call_args[1] == "system_message_2"
    assert call_args[2] == "manager_message"
    assert call_args[3] == "input_text_message"
    assert isinstance(call_args[4], HumanMessage)  # review message
    assert call_args[5] == "begin_task_message"

    # Verify review message content
    review_message = call_args[4]
    content = review_message.content
    assert "Please review the following content and provide feedback:" in content
    assert "Machine Learning in Practice" in content  # title
    assert "This paper explores various ML algorithms" in content  # tldr
    assert "formatted_references" in content  # references


def test_reviewer_node_handles_missing_components(
    monkeypatch, reviewer_state, mock_review_response_all_approved
):
    """Test that node handles missing title, tldr, or references gracefully."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value = (
        mock_review_response_all_approved
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes.format_references_for_prompt", lambda refs: "no_references"
    )

    # Remove components to test default handling
    reviewer_state[TITLE] = None  # Will become "Not generated"
    reviewer_state[TLDR] = None  # Will become "Not generated"
    reviewer_state[SELECTED_REFERENCES] = []  # Empty list

    node = make_reviewer_node(llm_model="mock-model")
    result = node(reviewer_state)

    # Should complete without errors and include defaults in review message
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]
    review_message = call_args[3]
    content = review_message.content

    assert "No title provided" in content
    assert "No TLDR provided" in content
    assert "No references provided" in content


def test_reviewer_node_handles_llm_exceptions(monkeypatch, reviewer_state):
    """Test that node propagates LLM exceptions."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.side_effect = Exception(
        "LLM API Error"
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_reviewer_node(llm_model="mock-model")

    # Should propagate the LLM exception
    with pytest.raises(Exception, match="LLM API Error"):
        node(reviewer_state)


def test_reviewer_node_increments_revision_round(
    monkeypatch, reviewer_state, mock_review_response_needs_revision
):
    """Test that revision round is properly incremented."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value = (
        mock_review_response_needs_revision
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    # Start at revision round 1
    reviewer_state[REVISION_ROUND] = 1

    node = make_reviewer_node(llm_model="mock-model")
    result = node(reviewer_state)

    # Should increment to round 2
    assert result[REVISION_ROUND] == 2


def test_reviewer_node_handles_missing_revision_round(
    monkeypatch, reviewer_state, mock_review_response_all_approved
):
    """Test that node handles missing REVISION_ROUND gracefully."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value = (
        mock_review_response_all_approved
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    # Remove revision round key
    del reviewer_state[REVISION_ROUND]

    node = make_reviewer_node(llm_model="mock-model")
    result = node(reviewer_state)

    # Should default to 0 and increment to 1
    assert result[REVISION_ROUND] == 1


def test_reviewer_node_formats_review_input_correctly(
    monkeypatch, reviewer_state, mock_review_response_all_approved
):
    """Test that the review input is formatted correctly with proper sections."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value = (
        mock_review_response_all_approved
    )

    # Track calls to format_references_for_prompt
    mock_format_function = MagicMock(return_value="formatted_ref_output")

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes.format_references_for_prompt", mock_format_function
    )

    node = make_reviewer_node(llm_model="mock-model")
    result = node(reviewer_state)

    # Verify format function was called with selected references
    mock_format_function.assert_called_once_with(reviewer_state[SELECTED_REFERENCES])

    # Verify the review message contains properly formatted sections
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]
    review_message = call_args[3]
    content = review_message.content

    # Check for section headers and content
    assert "# Title(s):" in content
    assert "# TLDR(s):" in content
    assert "# References:" in content
    assert "-------------" in content  # Section dividers
    assert "formatted_ref_output" in content
