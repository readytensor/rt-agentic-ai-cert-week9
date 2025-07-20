import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from nodes.output_types import Reference  # Import your Reference model

from nodes.a3_nodes import make_references_selector_node
from consts import (
    REFERENCES_SELECTOR_MESSAGES,
    INPUT_TEXT,
    MANAGER_BRIEF,
    REFERENCES_FEEDBACK,
    REFERENCES_APPROVED,
    CANDIDATE_REFERENCES,
    SELECTED_REFERENCES,
)


@pytest.fixture
def references_selector_state():
    return {
        INPUT_TEXT: "Sample research paper about machine learning algorithms",
        MANAGER_BRIEF: "This paper discusses ML algorithms",
        REFERENCES_SELECTOR_MESSAGES: ["system_message"],
        REFERENCES_FEEDBACK: "",
        REFERENCES_APPROVED: False,
        CANDIDATE_REFERENCES: [
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
                "url": "https://example3.com/transformers",
                "title": "Transformers Explained",
                "page_content": "Transformers are a type of model architecture that has shown great success...",
            },
        ],
    }


@pytest.fixture
def mock_selected_references():
    return [
        Reference(
            url="https://example.com/article1",
            title="Article 1",
            page_content="This is the first article content."
        ),
        Reference(
            url="https://example.com/article2",
            title="Article 2",
            page_content="Second article content."
        ),
    ]


@pytest.fixture
def expected_selected_references():
    """Expected output format after processing the mock references."""
    return [
        {
            "url": "https://example.com/article1",
            "title": "Article 1", 
            "page_content": "This is the first article content.",
        },
        {
            "url": "https://example.com/article2",
            "title": "Article 2",
            "page_content": "Second article content.",
        },
    ]



def test_references_selector_node_skips_when_already_approved(
    monkeypatch, references_selector_state
):
    """Test that node skips processing when references are already approved."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    # Set references as already approved
    references_selector_state[REFERENCES_APPROVED] = True

    node = make_references_selector_node(llm_model="mock-model")
    result = node(references_selector_state)

    # Should return empty dict and not call LLM
    assert result == {}
    mock_llm_obj.with_structured_output.assert_not_called()


def test_references_selector_node_handles_empty_candidate_references(
    monkeypatch, references_selector_state
):
    """Test that node handles empty candidate references gracefully."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    # Set empty candidate references
    references_selector_state[CANDIDATE_REFERENCES] = []

    node = make_references_selector_node(llm_model="mock-model")
    result = node(references_selector_state)

    # Should return empty dict and not call LLM
    assert result == {}
    mock_llm_obj.with_structured_output.assert_not_called()


def test_references_selector_node_handles_missing_candidate_references(
    monkeypatch, references_selector_state
):
    """Test that node handles missing CANDIDATE_REFERENCES key gracefully."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    # Remove candidate references key
    del references_selector_state[CANDIDATE_REFERENCES]

    node = make_references_selector_node(llm_model="mock-model")
    result = node(references_selector_state)

    # Should return empty dict and not call LLM
    assert result == {}
    mock_llm_obj.with_structured_output.assert_not_called()


def test_references_selector_node_selects_references_successfully(
    monkeypatch,
    references_selector_state,
    mock_selected_references,
    expected_selected_references,
):
    """Test successful reference selection."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.references = (
        mock_selected_references
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_references_selector_node(llm_model="mock-model")
    result = node(references_selector_state)

    # Verify results
    assert SELECTED_REFERENCES in result
    print(f"Selected references: {result[SELECTED_REFERENCES]}")
    print(f"Expected references: {expected_selected_references}")
    assert result[SELECTED_REFERENCES] == expected_selected_references


def test_references_selector_node_constructs_correct_messages(
    monkeypatch, references_selector_state, mock_selected_references
):
    """Test that node constructs the correct message sequence for LLM."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.references = (
        mock_selected_references
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
        "nodes.a3_nodes.format_references_for_prompt",
        lambda refs: "formatted_references",
    )

    # Add additional system messages to test spreading
    references_selector_state[REFERENCES_SELECTOR_MESSAGES] = [
        "system_message_1",
        "system_message_2",
    ]

    node = make_references_selector_node(llm_model="mock-model")
    result = node(references_selector_state)

    # Verify the LLM was called with the correct message sequence
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]

    assert isinstance(call_args, list)
    assert (
        len(call_args) == 7
    )  # 2 system + manager + reviewer + input + candidate_refs + begin_task

    # Verify message order and types
    assert call_args[0] == "system_message_1"
    assert call_args[1] == "system_message_2"
    assert call_args[2] == "manager_message"
    assert call_args[3] == "reviewer_message"
    assert call_args[4] == "input_text_message"
    assert isinstance(call_args[5], HumanMessage)  # candidate references message
    assert call_args[6] == "begin_task_message"

    # Verify candidate references message content
    candidate_refs_message = call_args[5]
    assert (
        "Here are your candidate references to select from:"
        in candidate_refs_message.content
    )
    assert "formatted_references" in candidate_refs_message.content


def test_references_selector_node_formats_candidate_references_correctly(
    monkeypatch, references_selector_state, mock_selected_references
):
    """Test that candidate references are properly formatted in the message."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.references = (
        mock_selected_references
    )

    # Track calls to format_references_for_prompt
    mock_format_function = MagicMock(return_value="formatted_candidate_references")

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.a3_nodes.format_references_for_prompt", mock_format_function
    )

    node = make_references_selector_node(llm_model="mock-model")
    result = node(references_selector_state)

    # Verify format function was called with candidate references
    mock_format_function.assert_called_once_with(
        references_selector_state[CANDIDATE_REFERENCES]
    )

    # Verify the formatted references appear in the LLM message
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]
    candidate_refs_message = call_args[
        4
    ]  # Should be the HumanMessage with candidate refs

    assert isinstance(candidate_refs_message, HumanMessage)
    assert "formatted_candidate_references" in candidate_refs_message.content


def test_references_selector_node_handles_llm_exceptions(
    monkeypatch, references_selector_state
):
    """Test that node propagates LLM exceptions."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.side_effect = Exception(
        "LLM API Error"
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_references_selector_node(llm_model="mock-model")

    # Should propagate the LLM exception
    with pytest.raises(Exception, match="LLM API Error"):
        node(references_selector_state)


def test_references_selector_node_handles_missing_references_attribute(
    monkeypatch, references_selector_state
):
    """Test that node handles LLM response without references attribute."""
    mock_llm_response = MagicMock()
    # Remove the references attribute to force AttributeError
    del mock_llm_response.references

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value = (
        mock_llm_response
    )

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_references_selector_node(llm_model="mock-model")

    # Should raise AttributeError when trying to access .references
    with pytest.raises(AttributeError):
        node(references_selector_state)

def test_references_selector_node_handles_malformed_llm_response_objects(
    monkeypatch, references_selector_state
):
    """Test that node handles reference objects missing required attributes."""
    # Create objects that will actually cause AttributeError when accessing missing attributes
    class MalformedRef1:
        def __init__(self):
            self.url = "https://example.com"
            # Missing title and page_content attributes
    
    class MalformedRef2:
        def __init__(self):
            self.title = "Some Title"
            # Missing url and page_content attributes
    
    # Create a valid reference for comparison
    class ValidRef:
        def __init__(self):
            self.url = "https://valid-example.com"
            self.title = "Valid Reference"
            self.page_content = "This is valid content"

    malformed_refs = [MalformedRef1(), MalformedRef2(), ValidRef()]

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.references = malformed_refs

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_references_selector_node(llm_model="mock-model")
    result = node(references_selector_state)

    # Should skip malformed references and only return the valid one
    assert SELECTED_REFERENCES in result
    assert len(result[SELECTED_REFERENCES]) == 1
    assert result[SELECTED_REFERENCES][0] == {
        "url": "https://valid-example.com",
        "title": "Valid Reference",
        "page_content": "This is valid content",
    }


def test_references_selector_node_handles_empty_references_response(
    monkeypatch, references_selector_state
):
    """Test that node handles empty references list from LLM."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.references = []

    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_references_selector_node(llm_model="mock-model")
    result = node(references_selector_state)

    # Should handle gracefully and return empty list
    assert SELECTED_REFERENCES in result
    assert result[SELECTED_REFERENCES] == []


def test_references_selector_node_validates_input_text(
    monkeypatch, references_selector_state
):
    """Test that node validates input text and raises appropriate errors."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.a3_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_references_selector_node(llm_model="mock-model")

    # Test with None input text
    references_selector_state[INPUT_TEXT] = None
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(references_selector_state)

    # Test with empty string input text
    references_selector_state[INPUT_TEXT] = ""
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(references_selector_state)

    # Test with whitespace-only input text
    references_selector_state[INPUT_TEXT] = "   \n\t  "
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(references_selector_state)
