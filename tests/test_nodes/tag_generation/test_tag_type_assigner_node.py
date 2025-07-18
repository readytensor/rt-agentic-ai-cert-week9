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


def test_tag_type_assigner_node_updates_spacy_tags(monkeypatch, mock_tag_type_assign_state):
    mock_response = {
        "entities": [
            {"name": "mnist", "type": "Dataset "},
            {"name": "hugging face", "type": "Tool-or-Framework"},
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj\
        .with_structured_output.return_value\
        .invoke.return_value\
        .model_dump.return_value = mock_response

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
    mock_llm_obj\
        .with_structured_output.return_value\
        .invoke.return_value\
        .model_dump.return_value = malformed_response

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