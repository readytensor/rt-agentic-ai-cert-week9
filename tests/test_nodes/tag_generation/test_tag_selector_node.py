import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from nodes.tag_generation_nodes import make_tag_selector_node
from consts import SELECTED_TAGS, CANDIDATE_TAGS, TAGS_SELECTOR_MESSAGES, INPUT_TEXT


@pytest.fixture
def selector_state():
    return {
        INPUT_TEXT: "Sample input text for testing",
        CANDIDATE_TAGS: [
            {"name": "MNIST", "type": "dataset"},
            {"name": "transformer", "type": "algorithm"},
            {"name": "irrelevant-tag", "type": "algorithm"},
        ],
        TAGS_SELECTOR_MESSAGES: [],
    }


def test_tag_selector_node_returns_cleaned_tags(monkeypatch, selector_state):
    mock_response = {
        "entities": [
            {"name": " MNIST", "type": "Dataset "},
            {"name": "transformer", "type": "Algorithm"},
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)
    node = make_tag_selector_node(llm_model="mock", max_tags=5)
    result = node(selector_state)

    assert SELECTED_TAGS in result
    tags = result[SELECTED_TAGS]
    assert len(tags) == 2
    print(tags)
    assert {"name": "mnist", "type": "dataset"} in tags
    assert {"name": "transformer", "type": "algorithm"} in tags


def test_tag_selector_node_ignores_malformed_tags(monkeypatch, selector_state):
    mock_llm_response = {
        "entities": [
            {"name": "MNIST", "type": ""},  # valid
            {"type": "dataset"},  # missing name — ignored
            {},  # completely malformed — ignored
        ]
    }

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_llm_response
    )
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm)

    # Override candidate tags for this specific test
    selector_state[CANDIDATE_TAGS] = [{"name": "MNIST", "type": ""}]

    node = make_tag_selector_node("mock-model", max_tags=3)
    result = node(selector_state)

    assert result[SELECTED_TAGS] == [{"name": "mnist", "type": ""}]


def test_tag_selector_node_handles_empty_input(monkeypatch, selector_state):
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value.model_dump.return_value = {
        "entities": []
    }

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm)

    # Override candidate tags to be empty for this test
    selector_state[CANDIDATE_TAGS] = []

    node = make_tag_selector_node("mock", max_tags=3)
    result = node(selector_state)

    assert result[SELECTED_TAGS] == []


def test_tag_selector_node_respects_max_tags_limit(monkeypatch, selector_state):
    # Simulate 5 tags returned, but max_tags is set to 3
    mock_response = {
        "entities": [{"name": f"Tag{i}", "type": "Type"} for i in range(5)]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    # Override candidate tags for this test
    selector_state[CANDIDATE_TAGS] = [
        {"name": f"Tag{i}", "type": "Type"} for i in range(5)
    ]

    node = make_tag_selector_node("mock-model", max_tags=3)
    result = node(selector_state)

    assert SELECTED_TAGS in result
    tags = result[SELECTED_TAGS]

    assert len(tags) == 3
    for i in range(3):
        assert tags[i]["name"] == f"tag{i}"
        assert tags[i]["type"] == "type"


def test_tag_selector_node_handles_none_name_values(monkeypatch, selector_state):
    """Test that node handles None values in name field gracefully."""
    mock_response = {
        "entities": [
            {"name": None, "type": "algorithm"},  # None name - should be skipped
            {"name": "mnist", "type": "dataset"},  # valid
            {
                "name": "pytorch",
                "type": None,
            },  # None type - should be kept with empty type
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tag_selector_node("mock-model", max_tags=5)
    result = node(selector_state)

    # Should skip None name, keep valid entries, handle None type gracefully
    expected = [
        {"name": "mnist", "type": "dataset"},
        {"name": "pytorch", "type": ""},  # None type becomes empty string
    ]

    assert result[SELECTED_TAGS] == expected


def test_tag_selector_node_handles_missing_entities_key(monkeypatch, selector_state):
    """Test that node handles LLM response without 'entities' key."""
    mock_response = {}  # Missing 'entities' key

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tag_selector_node("mock-model", max_tags=5)
    result = node(selector_state)

    # Should handle gracefully and return empty list
    assert result[SELECTED_TAGS] == []


def test_tag_selector_node_handles_missing_candidate_tags(monkeypatch, selector_state):
    """Test that node handles missing CANDIDATE_TAGS key gracefully."""
    mock_response = {"entities": []}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tag_selector_node("mock-model", max_tags=5)

    # Remove CANDIDATE_TAGS key from fixture
    del selector_state[CANDIDATE_TAGS]

    result = node(selector_state)
    assert result[SELECTED_TAGS] == []


def test_tag_selector_node_propagates_llm_exceptions(monkeypatch, selector_state):
    """Test that node propagates LLM exceptions appropriately."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.side_effect = Exception(
        "LLM API Error"
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tag_selector_node("mock-model", max_tags=5)

    with pytest.raises(Exception, match="LLM API Error"):
        node(selector_state)


def test_tag_selector_node_handles_max_tags_zero(monkeypatch, selector_state):
    """Test that node handles max_tags=0 correctly."""
    mock_response = {
        "entities": [
            {"name": "mnist", "type": "dataset"},
            {"name": "pytorch", "type": "framework"},
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tag_selector_node("mock-model", max_tags=0)
    result = node(selector_state)

    # Should return empty list when max_tags=0
    assert result[SELECTED_TAGS] == []


def test_tag_selector_node_validates_input_text(monkeypatch, selector_state):
    """Test that node validates input text and raises appropriate errors."""
    mock_llm_obj = MagicMock()
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tag_selector_node(llm_model="mock-model", max_tags=5)

    # Test with None input text
    selector_state[INPUT_TEXT] = None
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(selector_state)

    # Test with empty string input text
    selector_state[INPUT_TEXT] = ""
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(selector_state)

    # Test with whitespace-only input text
    selector_state[INPUT_TEXT] = "   \n\t  "
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        node(selector_state)


def test_tag_selector_node_constructs_correct_messages(monkeypatch, selector_state):
    """Test that node constructs the correct message sequence for LLM."""
    mock_invoke_output = {"entities": [{"name": "test", "type": "algorithm"}]}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    # Mock helper functions with identifiable returns
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.tag_generation_nodes._get_manager_brief_message",
        lambda state: "manager_message",
    )
    monkeypatch.setattr(
        "nodes.tag_generation_nodes._get_input_text_message",
        lambda state: "input_text_message",
    )
    monkeypatch.setattr(
        "nodes.tag_generation_nodes._get_begin_task_message",
        lambda: "begin_task_message",
    )

    # Add additional messages to test the spreading of TAGS_SELECTOR_MESSAGES
    selector_state[TAGS_SELECTOR_MESSAGES] = ["system_message_1", "system_message_2"]

    node = make_tag_selector_node(llm_model="mock-model", max_tags=5)
    result = node(selector_state)

    # Verify the LLM was called with the correct message sequence
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]

    assert isinstance(call_args, list)
    assert (
        len(call_args) == 6
    )  # 2 system + manager + input_text + selection + begin_task

    # Verify message order and types based on actual node implementation:
    # [system_message_1, system_message_2, manager_brief, input_text, selection_instruction, begin_task]
    assert call_args[0] == "system_message_1"
    assert call_args[1] == "system_message_2"
    assert call_args[2] == "manager_message"
    assert call_args[3] == "input_text_message"
    assert isinstance(call_args[4], HumanMessage)  # selection instruction
    assert call_args[5] == "begin_task_message"

    # Verify selection instruction content
    selection_msg = call_args[4]  # selection instruction is at index 4
    assert "candidate tags" in selection_msg.content.lower()
    assert str(selector_state[CANDIDATE_TAGS]) in selection_msg.content
    assert "maximum 5" in selection_msg.content

    # Verify the result
    assert SELECTED_TAGS in result
    assert result[SELECTED_TAGS] == [{"name": "test", "type": "algorithm"}]


def test_tag_selector_node_formats_candidate_tags_correctly_in_message(
    monkeypatch, selector_state
):
    """Test that candidate tags are properly formatted in the selection instruction."""
    mock_invoke_output = {"entities": []}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)
    monkeypatch.setattr(
        "nodes.tag_generation_nodes._get_manager_brief_message",
        lambda state: "manager_message",
    )
    monkeypatch.setattr(
        "nodes.tag_generation_nodes._get_input_text_message",
        lambda state: "input_text_message",
    )
    monkeypatch.setattr(
        "nodes.tag_generation_nodes._get_begin_task_message",
        lambda: "begin_task_message",
    )

    # Use fixture data but override TAGS_SELECTOR_MESSAGES for this test
    selector_state[TAGS_SELECTOR_MESSAGES] = ["system_message"]

    node = make_tag_selector_node(llm_model="mock-model", max_tags=2)
    node(selector_state)

    # Extract the selection instruction message
    call_args = mock_llm_obj.with_structured_output.return_value.invoke.call_args[0][0]
    selection_msg = call_args[
        3
    ]  # selection instruction is at index 3 (system + manager + input + selection)

    assert isinstance(selection_msg, HumanMessage)

    # Verify the message contains the candidate tags and max limit
    content = selection_msg.content
    assert "candidate tags" in content.lower()
    assert str(selector_state[CANDIDATE_TAGS]) in content
    assert "maximum 2" in content

    # Verify the exact format expected
    expected_content_parts = [
        "Here is the list of candidate tags (name and type):",
        str(selector_state[CANDIDATE_TAGS]),
        "Please return a refined list of the most important tags (maximum 2).",
    ]
    for part in expected_content_parts:
        assert part in content
