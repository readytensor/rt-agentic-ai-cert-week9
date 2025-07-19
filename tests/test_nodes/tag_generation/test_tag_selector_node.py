import pytest
from unittest.mock import MagicMock
from nodes.tag_generation_nodes import make_tag_selector_node
from consts import SELECTED_TAGS, CANDIDATE_TAGS, TAGS_SELECTOR_MESSAGES


@pytest.fixture
def selector_state():
    return {
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


def test_tag_selector_node_ignores_malformed_tags(monkeypatch):
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

    state = {
        CANDIDATE_TAGS: [{"name": "MNIST", "type": ""}],
        TAGS_SELECTOR_MESSAGES: [],
    }

    node = make_tag_selector_node("mock-model", max_tags=3)
    result = node(state)

    assert result[SELECTED_TAGS] == [{"name": "mnist", "type": ""}]


def test_tag_selector_node_handles_empty_input(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value.model_dump.return_value = {
        "entities": []
    }

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm)

    node = make_tag_selector_node("mock", max_tags=3)
    result = node(
        {
            CANDIDATE_TAGS: [],
            TAGS_SELECTOR_MESSAGES: [],
        }
    )

    assert result[SELECTED_TAGS] == []


def test_tag_selector_node_respects_max_tags_limit(monkeypatch):
    # Simulate 5 tags returned, but max_tags is set to 3
    mock_response = {
        "entities": [{"name": f"Tag{i}", "type": "Type"} for i in range(5)]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )
    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    state = {
        CANDIDATE_TAGS: [{"name": f"Tag{i}", "type": "Type"} for i in range(5)],
        TAGS_SELECTOR_MESSAGES: [],
    }

    node = make_tag_selector_node("mock-model", max_tags=3)
    result = node(state)

    assert SELECTED_TAGS in result
    tags = result[SELECTED_TAGS]

    assert len(tags) == 3
    for i in range(3):
        assert tags[i]["name"] == f"tag{i}"
        assert tags[i]["type"] == "type"


def test_tag_selector_node_handles_none_name_values(monkeypatch):
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

    state = {
        CANDIDATE_TAGS: [{"name": "test", "type": "test"}],
        TAGS_SELECTOR_MESSAGES: [],
    }

    result = node(state)

    # Should skip None name, keep valid entries, handle None type gracefully
    expected = [
        {"name": "mnist", "type": "dataset"},
        {"name": "pytorch", "type": ""},  # None type becomes empty string
    ]

    assert result[SELECTED_TAGS] == expected


def test_tag_selector_node_handles_missing_entities_key(monkeypatch):
    """Test that node handles LLM response without 'entities' key."""
    mock_response = {}  # Missing 'entities' key

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tag_selector_node("mock-model", max_tags=5)

    state = {
        CANDIDATE_TAGS: [{"name": "test", "type": "test"}],
        TAGS_SELECTOR_MESSAGES: [],
    }

    result = node(state)

    # Should handle gracefully and return empty list
    assert result[SELECTED_TAGS] == []


def test_tag_selector_node_handles_missing_candidate_tags(monkeypatch):
    """Test that node handles missing CANDIDATE_TAGS key gracefully."""
    mock_response = {"entities": []}

    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.return_value.model_dump.return_value = (
        mock_response
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tag_selector_node("mock-model", max_tags=5)

    # State without CANDIDATE_TAGS key
    state = {TAGS_SELECTOR_MESSAGES: []}

    result = node(state)

    assert result[SELECTED_TAGS] == []


def test_tag_selector_node_propagates_llm_exceptions(monkeypatch):
    """Test that node propagates LLM exceptions appropriately."""
    mock_llm_obj = MagicMock()
    mock_llm_obj.with_structured_output.return_value.invoke.side_effect = Exception(
        "LLM API Error"
    )

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm_obj)

    node = make_tag_selector_node("mock-model", max_tags=5)

    state = {
        CANDIDATE_TAGS: [{"name": "test", "type": "test"}],
        TAGS_SELECTOR_MESSAGES: [],
    }

    with pytest.raises(Exception, match="LLM API Error"):
        node(state)


def test_tag_selector_node_handles_max_tags_zero(monkeypatch):
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

    state = {
        CANDIDATE_TAGS: [{"name": "test", "type": "test"}],
        TAGS_SELECTOR_MESSAGES: [],
    }

    result = node(state)

    # Should return empty list when max_tags=0
    assert result[SELECTED_TAGS] == []
