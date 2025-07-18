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
    mock_llm_obj\
        .with_structured_output.return_value\
        .invoke.return_value\
        .model_dump.return_value = mock_response

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
            {"type": "dataset"},            # missing name — ignored
            {},                              # completely malformed — ignored
        ]
    }

    mock_llm = MagicMock()
    mock_llm\
        .with_structured_output.return_value\
            .invoke.return_value.\
                model_dump.return_value = mock_llm_response
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
    mock_llm\
        .with_structured_output.return_value\
        .invoke.return_value\
        .model_dump.return_value = {"entities": []}

    monkeypatch.setattr("nodes.tag_generation_nodes.get_llm", lambda _: mock_llm)

    node = make_tag_selector_node("mock", max_tags=3)
    result = node({
        CANDIDATE_TAGS: [],
        TAGS_SELECTOR_MESSAGES: [],
    })

    assert result[SELECTED_TAGS] == []

def test_tag_selector_node_respects_max_tags_limit(monkeypatch):
    # Simulate 5 tags returned, but max_tags is set to 3
    mock_response = {
        "entities": [
            {"name": f"Tag{i}", "type": "Type"} for i in range(5)
        ]
    }

    mock_llm_obj = MagicMock()
    mock_llm_obj\
        .with_structured_output.return_value\
            .invoke.return_value\
                .model_dump.return_value = mock_response
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