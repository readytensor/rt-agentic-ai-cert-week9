import pytest
from unittest.mock import MagicMock
from nodes.tag_generation_nodes import make_llm_tag_generator_node
from consts import LLM_TAGS, LLM_TAGS_GEN_MESSAGES


def test_llm_tag_generator_node_returns_cleaned_tags(monkeypatch):
    mock_invoke_output = {
        "entities": [
            {"name": "MNIST ", "type": "Dataset "},
            {"name": "transformer", "type": "Algorithm"},
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj\
        .with_structured_output.return_value\
            .invoke.return_value\
                .model_dump.return_value = mock_invoke_output

    # This mocks:
    # llm.with_structured_output(...).invoke(...).model_dump()

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")
    result = node({
        LLM_TAGS_GEN_MESSAGES: ["some message"],  # Can be anything; not used in mock
    })

    assert LLM_TAGS in result
    tags = result[LLM_TAGS]
    assert len(tags) == 2

    assert {"name": "mnist", "type": "dataset"} in tags
    assert {"name": "transformer", "type": "algorithm"} in tags


def test_llm_tag_generator_node_ignores_malformed_tags(monkeypatch):
    # Simulate a mix of valid and invalid tags
    malformed_response = {
        "entities": [
            {"name": "transformer", "type": "Algorithm"},  # valid
            {"name": "mnist"},  # missing type (should be kept with empty type)
            {"type": "dataset"},  # missing name (should be ignored)
            {},  # completely empty
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj\
        .with_structured_output.return_value\
        .invoke.return_value\
        .model_dump.return_value = malformed_response

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node("mock-model")
    result = node({
        LLM_TAGS_GEN_MESSAGES: ["some message"],  # Can be anything; not used in mock
    })

    assert LLM_TAGS in result

    # Should only keep entries with a non-empty 'name'
    # If 'type' is missing, it's set to empty string
    assert result[LLM_TAGS] == [
        {"name": "transformer", "type": "algorithm"},
        {"name": "mnist", "type": ""},
    ]
