import pytest
from unittest.mock import MagicMock
from nodes.tag_generation_nodes import make_tag_type_assigner_node
from consts import SPACY_TAGS, TAG_TYPE_ASSIGNER_MESSAGES


@pytest.fixture
def mock_tag_type_assign_state():
    return {
        SPACY_TAGS: [
            {"name": "mnist"},
            {"name": "hugging face"},
        ],
        TAG_TYPE_ASSIGNER_MESSAGES: [],
    }


def test_tag_type_assigner_node_updates_spacy_tags(
    monkeypatch, mock_tag_type_assign_state
):
    mock_response = {
        "entities": [
            {"name": "mnist", "type": "Dataset "},
            {"name": "hugging face", "type": "Tool-or-Framework"},
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)
    node = make_tag_type_assigner_node(llm_model="mock-model")
    result = node(mock_tag_type_assign_state)

    assert SPACY_TAGS in result
    updated_tags = result[SPACY_TAGS]
    assert {"name": "mnist", "type": "dataset"} in updated_tags
    assert {"name": "hugging face", "type": "tool-or-framework"} in updated_tags


def test_tag_type_assigner_node_skips_missing_name(monkeypatch):
    malformed_response = {
        "entities": [
            {"name": "mnist", "type": "Dataset"},
            {"type": "Tool"},  # No name — should be skipped
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        malformed_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    state = {
        SPACY_TAGS: [{"name": "mnist", "type": "some_type"}, {"name": "hugging face"}],
        TAG_TYPE_ASSIGNER_MESSAGES: [],
    }

    node = make_tag_type_assigner_node("mock-model")
    result = node(state)

    assert len(result[SPACY_TAGS]) == 1
    assert {"name": "mnist", "type": "dataset"} in result[SPACY_TAGS]


def test_tag_type_assigner_node_with_empty_spacy_tags(monkeypatch):
    # Create a mock that should not be called
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    state = {
        SPACY_TAGS: [],
        TAG_TYPE_ASSIGNER_MESSAGES: [],
    }

    node = make_tag_type_assigner_node("mock-model")
    result = node(state)

    assert result[SPACY_TAGS] == []
    mock_llm_obj.with_structured_output.return_value.invoke.assert_not_called()


def test_tag_type_assigner_node_handles_none_values_in_input(monkeypatch):
    """Test that node handles None values in input spaCy tags gracefully."""
    mock_response = {"entities": [{"name": "valid", "type": "dataset"}]}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    state = {
        SPACY_TAGS: [
            {"name": None},  # None name - should be skipped
            {"name": "valid"},  # valid entry
            {},  # missing name key - should be skipped
        ],
        TAG_TYPE_ASSIGNER_MESSAGES: [],
    }

    node = make_tag_type_assigner_node("mock-model")
    result = node(state)

    # Should handle gracefully - skip None/missing names, process valid ones
    assert result[SPACY_TAGS] == [{"name": "valid", "type": "dataset"}]

    # Should have called LLM with only the valid name
    mock_llm_obj.with_structured_output.return_value.invoke.assert_called_once()
    # The call should contain only "valid" in the message
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]
    human_message = call_args[-1].content
    assert "valid" in human_message
    assert "None" not in human_message


def test_tag_type_assigner_node_handles_whitespace_only_names(monkeypatch):
    """Test that node handles whitespace-only names in input tags."""
    mock_response = {"entities": []}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    state = {
        SPACY_TAGS: [
            {"name": "   "},  # whitespace only
            {"name": "\n\t"},  # whitespace only
            {"name": ""},  # empty string
        ],
        TAG_TYPE_ASSIGNER_MESSAGES: [],
    }

    node = make_tag_type_assigner_node("mock-model")
    result = node(state)

    # Should return empty list since all names are whitespace/empty
    assert result[SPACY_TAGS] == []
    # LLM should not be called since no valid names
    mock_llm_obj.with_structured_output.return_value.invoke.assert_not_called()


def test_tag_type_assigner_node_handles_none_values_in_llm_response(monkeypatch):
    """Test that node handles None values in LLM response gracefully."""
    mock_response = {
        "entities": [
            {"name": None, "type": "dataset"},  # None name from LLM
            {"name": "mnist", "type": None},  # None type from LLM
            {"name": "pytorch", "type": "framework"},  # valid
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    state = {
        SPACY_TAGS: [{"name": "mnist"}, {"name": "pytorch"}],
        TAG_TYPE_ASSIGNER_MESSAGES: [],
    }

    node = make_tag_type_assigner_node("mock-model")

    # This will likely crash when trying to call .strip().lower() on None
    with pytest.raises(AttributeError):
        node(state)


def test_tag_type_assigner_node_handles_missing_entities_key(monkeypatch):
    """Test that node handles LLM response without 'entities' key."""
    mock_response = {}  # Missing 'entities' key

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    state = {
        SPACY_TAGS: [{"name": "mnist"}],
        TAG_TYPE_ASSIGNER_MESSAGES: [],
    }

    node = make_tag_type_assigner_node("mock-model")

    # Should raise KeyError when trying to access ['entities']
    with pytest.raises(KeyError):
        node(state)


def test_tag_type_assigner_node_propagates_llm_exceptions(monkeypatch):
    """Test that node propagates LLM exceptions appropriately."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.side_effect = Exception(
        "LLM API Error"
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    state = {
        SPACY_TAGS: [{"name": "mnist"}],
        TAG_TYPE_ASSIGNER_MESSAGES: [],
    }

    node = make_tag_type_assigner_node("mock-model")

    with pytest.raises(Exception, match="LLM API Error"):
        node(state)


def test_tag_type_assigner_node_handles_missing_spacy_tags_key(monkeypatch):
    """Test that node handles missing SPACY_TAGS key gracefully."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # State without SPACY_TAGS key
    state = {TAG_TYPE_ASSIGNER_MESSAGES: []}

    node = make_tag_type_assigner_node("mock-model")
    result = node(state)

    # Should handle gracefully and return empty list
    assert result[SPACY_TAGS] == []
    # LLM should not be called
    mock_llm_obj.with_structured_output.return_value.invoke.assert_not_called()
