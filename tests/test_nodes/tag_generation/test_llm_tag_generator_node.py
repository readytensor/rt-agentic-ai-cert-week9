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
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    # This mocks:
    # llm.with_structured_output(...).invoke(...).model_dump()

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")
    result = node(
        {
            LLM_TAGS_GEN_MESSAGES: [
                "some message"
            ],  # Can be anything; not used in mock
        }
    )

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
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        malformed_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node("mock-model")
    result = node(
        {
            LLM_TAGS_GEN_MESSAGES: [
                "some message"
            ],  # Can be anything; not used in mock
        }
    )

    assert LLM_TAGS in result

    # Should only keep entries with a non-empty 'name'
    # If 'type' is missing, it's set to empty string
    assert result[LLM_TAGS] == [
        {"name": "transformer", "type": "algorithm"},
        {"name": "mnist", "type": ""},
    ]


def test_llm_tag_generator_node_handles_empty_entities_list(monkeypatch):
    """Test that node handles empty entities list gracefully."""
    mock_invoke_output = {"entities": []}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")
    result = node({LLM_TAGS_GEN_MESSAGES: ["some message"]})

    assert LLM_TAGS in result
    assert result[LLM_TAGS] == []


def test_llm_tag_generator_node_handles_missing_entities_key(monkeypatch):
    """Test that node handles response without 'entities' key."""
    mock_invoke_output = {}  # Missing 'entities' key

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")

    # Should raise KeyError when trying to access ['entities']
    with pytest.raises(KeyError):
        node({LLM_TAGS_GEN_MESSAGES: ["some message"]})


def test_llm_tag_generator_node_handles_whitespace_only_names(monkeypatch):
    """Test that node properly filters out whitespace-only names."""
    mock_invoke_output = {
        "entities": [
            {"name": "   ", "type": "Algorithm"},  # whitespace only
            {"name": "\n\t", "type": "Dataset"},  # whitespace only
            {"name": " valid ", "type": "Method"},  # valid with whitespace
            {"name": "", "type": "Tool"},  # empty string
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")
    result = node({LLM_TAGS_GEN_MESSAGES: ["some message"]})

    assert LLM_TAGS in result
    # Only the valid name should remain
    assert result[LLM_TAGS] == [{"name": "valid", "type": "method"}]


def test_llm_tag_generator_node_handles_special_characters(monkeypatch):
    """Test that node handles special characters in tag names and types."""
    mock_invoke_output = {
        "entities": [
            {"name": "C++", "type": "Programming Language"},
            {"name": "GPT-4", "type": "AI Model"},
            {"name": "sci-fi", "type": "Genre"},
            {"name": "résumé", "type": "Document"},  # Unicode
            {"name": "ML/AI", "type": "Field"},  # Special chars
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")
    result = node({LLM_TAGS_GEN_MESSAGES: ["some message"]})

    assert LLM_TAGS in result
    expected_tags = [
        {"name": "c++", "type": "programming language"},
        {"name": "gpt-4", "type": "ai model"},
        {"name": "sci-fi", "type": "genre"},
        {"name": "résumé", "type": "document"},
        {"name": "ml/ai", "type": "field"},
    ]
    assert result[LLM_TAGS] == expected_tags


def test_llm_tag_generator_node_handles_very_long_names(monkeypatch):
    """Test that node handles very long tag names and types."""
    long_name = "a" * 1000  # Very long name
    long_type = "b" * 500  # Very long type

    mock_invoke_output = {
        "entities": [
            {"name": long_name, "type": long_type},
            {"name": "normal", "type": "standard"},
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")
    result = node({LLM_TAGS_GEN_MESSAGES: ["some message"]})

    assert LLM_TAGS in result
    assert len(result[LLM_TAGS]) == 2
    assert result[LLM_TAGS][0]["name"] == long_name.lower()
    assert result[LLM_TAGS][0]["type"] == long_type.lower()


def test_llm_tag_generator_node_handles_none_type_values(monkeypatch):
    """Test that node handles None values in name and type fields."""
    mock_invoke_output = {
        "entities": [
            {"name": None, "type": "Algorithm"},  # None name
            {"name": "transformer", "type": None},  # None type
            {"name": "valid", "type": "Method"},  # Valid entry
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")

    # This might raise AttributeError when trying to call .lower() on None
    # Depending on your implementation preference, you might want to handle this
    with pytest.raises(AttributeError):
        node({LLM_TAGS_GEN_MESSAGES: ["some message"]})


def test_llm_tag_generator_node_propagates_llm_exceptions(monkeypatch):
    """Test that node propagates LLM-related exceptions."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.side_effect = Exception(
        "LLM API Error"
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")

    with pytest.raises(Exception, match="LLM API Error"):
        node({LLM_TAGS_GEN_MESSAGES: ["some message"]})


def test_llm_tag_generator_node_handles_non_string_name_types(monkeypatch):
    """Test that node handles non-string values in name and type fields."""
    mock_invoke_output = {
        "entities": [
            {"name": 123, "type": "Number"},  # Integer name
            {"name": "valid", "type": 456},  # Integer type
            {"name": True, "type": "Boolean"},  # Boolean name
            {"name": "test", "type": False},  # Boolean type
            {"name": [], "type": "List"},  # List name
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")

    # This will likely raise AttributeError when trying to call .lower() on non-strings
    with pytest.raises(AttributeError):
        node({LLM_TAGS_GEN_MESSAGES: ["some message"]})


def test_llm_tag_generator_node_handles_large_entity_list(monkeypatch):
    """Test that node handles a large number of entities efficiently."""
    # Create 100 tags
    large_entities = [{"name": f"tag_{i}", "type": f"type_{i % 5}"} for i in range(100)]

    mock_invoke_output = {"entities": large_entities}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_invoke_output
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_llm_tag_generator_node(llm_model="mock-model")
    result = node({LLM_TAGS_GEN_MESSAGES: ["some message"]})

    assert LLM_TAGS in result
    assert len(result[LLM_TAGS]) == 100

    # Verify first and last entries are properly cleaned
    assert result[LLM_TAGS][0] == {"name": "tag_0", "type": "type_0"}
    assert result[LLM_TAGS][-1] == {"name": "tag_99", "type": "type_4"}
