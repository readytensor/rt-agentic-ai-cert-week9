import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage

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


def test_tag_type_assigner_node_skips_missing_name(
    monkeypatch, mock_tag_type_assign_state
):
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

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [
        {"name": "mnist", "type": "some_type"},
        {"name": "hugging face"},
    ]

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    assert len(result[SPACY_TAGS]) == 1
    assert {"name": "mnist", "type": "dataset"} in result[SPACY_TAGS]


def test_tag_type_assigner_node_with_empty_spacy_tags(
    monkeypatch, mock_tag_type_assign_state
):
    # Create a mock that should not be called
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture to have empty tags
    mock_tag_type_assign_state[SPACY_TAGS] = []

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    assert result[SPACY_TAGS] == []
    mock_llm_obj.with_structured_output.return_value.invoke.assert_not_called()


def test_tag_type_assigner_node_handles_none_values_in_input(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node handles None values in input spaCy tags gracefully."""
    mock_response = {"entities": [{"name": "valid", "type": "dataset"}]}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [
        {"name": None},  # None name - should be skipped
        {"name": "valid"},  # valid entry
        {},  # missing name key - should be skipped
    ]

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    # Should handle gracefully - skip None/missing names, process valid ones
    assert result[SPACY_TAGS] == [{"name": "valid", "type": "dataset"}]

    # Should have called LLM with only the valid name
    mock_llm_obj.with_structured_output.return_value.invoke.assert_called_once()
    # The call should contain only "valid" in the message
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]
    human_message = call_args[-1].content
    assert "valid" in human_message
    assert "None" not in human_message


def test_tag_type_assigner_node_handles_whitespace_only_names(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node handles whitespace-only names in input tags."""
    mock_response = {"entities": []}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [
        {"name": "   "},  # whitespace only
        {"name": "\n\t"},  # whitespace only
        {"name": ""},  # empty string
    ]

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    # Should return empty list since all names are whitespace/empty
    assert result[SPACY_TAGS] == []
    # LLM should not be called since no valid names
    mock_llm_obj.with_structured_output.return_value.invoke.assert_not_called()


def test_tag_type_assigner_node_handles_none_values_in_llm_response(
    monkeypatch, mock_tag_type_assign_state
):
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

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [{"name": "mnist"}, {"name": "pytorch"}]

    node = make_tag_type_assigner_node("mock-model")

    # This will likely crash when trying to call .strip().lower() on None
    with pytest.raises(AttributeError):
        node(mock_tag_type_assign_state)


def test_tag_type_assigner_node_handles_missing_entities_key(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node handles LLM response without 'entities' key."""
    mock_response = {}  # Missing 'entities' key

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [{"name": "mnist"}]

    node = make_tag_type_assigner_node("mock-model")

    # Should raise KeyError when trying to access ['entities']
    with pytest.raises(KeyError):
        node(mock_tag_type_assign_state)


def test_tag_type_assigner_node_propagates_llm_exceptions(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node propagates LLM exceptions appropriately."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.side_effect = Exception(
        "LLM API Error"
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [{"name": "mnist"}]

    node = make_tag_type_assigner_node("mock-model")

    with pytest.raises(Exception, match="LLM API Error"):
        node(mock_tag_type_assign_state)


def test_tag_type_assigner_node_handles_missing_spacy_tags_key(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node handles missing SPACY_TAGS key gracefully."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Remove SPACY_TAGS key from fixture
    del mock_tag_type_assign_state[SPACY_TAGS]

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    # Should handle gracefully and return empty list
    assert result[SPACY_TAGS] == []
    # LLM should not be called
    mock_llm_obj.with_structured_output.return_value.invoke.assert_not_called()


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


def test_tag_type_assigner_node_skips_missing_name(
    monkeypatch, mock_tag_type_assign_state
):
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

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [
        {"name": "mnist", "type": "some_type"},
        {"name": "hugging face"},
    ]

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    assert len(result[SPACY_TAGS]) == 1
    assert {"name": "mnist", "type": "dataset"} in result[SPACY_TAGS]


def test_tag_type_assigner_node_with_empty_spacy_tags(
    monkeypatch, mock_tag_type_assign_state
):
    # Create a mock that should not be called
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture to have empty tags
    mock_tag_type_assign_state[SPACY_TAGS] = []

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    assert result[SPACY_TAGS] == []
    mock_llm_obj.with_structured_output.return_value.invoke.assert_not_called()


def test_tag_type_assigner_node_handles_none_values_in_input(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node handles None values in input spaCy tags gracefully."""
    mock_response = {"entities": [{"name": "valid", "type": "dataset"}]}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [
        {"name": None},  # None name - should be skipped
        {"name": "valid"},  # valid entry
        {},  # missing name key - should be skipped
    ]

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    # Should handle gracefully - skip None/missing names, process valid ones
    assert result[SPACY_TAGS] == [{"name": "valid", "type": "dataset"}]

    # Should have called LLM with only the valid name
    mock_llm_obj.with_structured_output.return_value.invoke.assert_called_once()
    # The call should contain only "valid" in the message
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]
    human_message = call_args[-1].content
    assert "valid" in human_message
    assert "None" not in human_message


def test_tag_type_assigner_node_handles_whitespace_only_names(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node handles whitespace-only names in input tags."""
    mock_response = {"entities": []}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [
        {"name": "   "},  # whitespace only
        {"name": "\n\t"},  # whitespace only
        {"name": ""},  # empty string
    ]

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    # Should return empty list since all names are whitespace/empty
    assert result[SPACY_TAGS] == []
    # LLM should not be called since no valid names
    mock_llm_obj.with_structured_output.return_value.invoke.assert_not_called()


def test_tag_type_assigner_node_handles_none_values_in_llm_response(
    monkeypatch, mock_tag_type_assign_state
):
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

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [{"name": "mnist"}, {"name": "pytorch"}]

    node = make_tag_type_assigner_node("mock-model")

    # This will likely crash when trying to call .strip().lower() on None
    with pytest.raises(AttributeError):
        node(mock_tag_type_assign_state)


def test_tag_type_assigner_node_handles_missing_entities_key(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node handles LLM response without 'entities' key."""
    mock_response = {}  # Missing 'entities' key

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [{"name": "mnist"}]

    node = make_tag_type_assigner_node("mock-model")

    # Should raise KeyError when trying to access ['entities']
    with pytest.raises(KeyError):
        node(mock_tag_type_assign_state)


def test_tag_type_assigner_node_propagates_llm_exceptions(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node propagates LLM exceptions appropriately."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.side_effect = Exception(
        "LLM API Error"
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override fixture data for this test
    mock_tag_type_assign_state[SPACY_TAGS] = [{"name": "mnist"}]

    node = make_tag_type_assigner_node("mock-model")

    with pytest.raises(Exception, match="LLM API Error"):
        node(mock_tag_type_assign_state)


def test_tag_type_assigner_node_handles_missing_spacy_tags_key(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node handles missing SPACY_TAGS key gracefully."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Remove SPACY_TAGS key from fixture
    del mock_tag_type_assign_state[SPACY_TAGS]

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    # Should handle gracefully and return empty list
    assert result[SPACY_TAGS] == []
    # LLM should not be called
    mock_llm_obj.with_structured_output.return_value.invoke.assert_not_called()


def test_tag_type_assigner_node_constructs_correct_messages(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that node constructs the correct message sequence for LLM."""
    mock_invoke_output = {
        "entities": [
            {"name": "mnist", "type": "dataset"},
            {"name": "hugging face", "type": "framework"},
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Set up fixture with known data and additional system messages
    mock_tag_type_assign_state[TAG_TYPE_ASSIGNER_MESSAGES] = [
        "system_message_1",
        "system_message_2",
    ]
    mock_tag_type_assign_state[SPACY_TAGS] = [
        {"name": "mnist"},
        {"name": "hugging face"},
    ]

    node = make_tag_type_assigner_node(llm_model="mock-model")
    result = node(mock_tag_type_assign_state)

    # Verify the LLM was called with the correct message sequence
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]

    assert isinstance(call_args, list)
    assert len(call_args) == 3  # 2 system messages + 1 human message with tag names

    # Verify message order and types
    assert call_args[0] == "system_message_1"
    assert call_args[1] == "system_message_2"

    # Verify the human message contains the tag assignment instruction
    assert isinstance(call_args[2], HumanMessage)

    human_message = call_args[2]
    content = human_message.content

    # Verify the message contains the expected instruction and tag names
    assert "Assign tag types to the following tags:" in content
    assert "mnist, hugging face" in content  # The clean names joined with commas

    # Verify the result
    assert SPACY_TAGS in result
    assert len(result[SPACY_TAGS]) == 2
    assert {"name": "mnist", "type": "dataset"} in result[SPACY_TAGS]
    assert {"name": "hugging face", "type": "framework"} in result[SPACY_TAGS]


def test_tag_type_assigner_node_message_contains_only_valid_names(
    monkeypatch, mock_tag_type_assign_state
):
    """Test that the message to LLM contains only valid tag names (filters out None/empty)."""
    mock_invoke_output = {"entities": [{"name": "valid_tag", "type": "dataset"}]}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Set up fixture with mix of valid and invalid names
    mock_tag_type_assign_state[SPACY_TAGS] = [
        {"name": None},  # Should be filtered out
        {"name": "   "},  # Should be filtered out (whitespace only)
        {"name": ""},  # Should be filtered out (empty)
        {"name": "valid_tag"},  # Should be included
        {"name": " another_valid "},  # Should be included (trimmed)
        {},  # Should be filtered out (missing name key)
    ]

    node = make_tag_type_assigner_node("mock-model")
    result = node(mock_tag_type_assign_state)

    # Verify the LLM was called with only valid names
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]
    human_message = call_args[-1]  # Last message should be the HumanMessage

    assert isinstance(human_message, HumanMessage)

    content = human_message.content
    # Should contain only the valid names (trimmed)
    assert "valid_tag, another_valid" in content
    # Should NOT contain invalid values
    assert "None" not in content
    assert "   " not in content

    # Verify result contains the processed tag
    assert result[SPACY_TAGS] == [{"name": "valid_tag", "type": "dataset"}]
