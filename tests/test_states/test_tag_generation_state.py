from langchain_core.messages import SystemMessage

from states.tag_generation_state import initialize_tag_generation_state

from langchain_core.messages import SystemMessage
from states.tag_generation_state import initialize_tag_generation_state


def test_initialize_tag_generation_state(mock_tag_config, sample_tag_types):
    """Test tag generation state initialization from config."""
    # Create a config structure that matches what the function expects
    config = {
        "agents": {
            "llm_tags_generator": {
                "prompt_config": mock_tag_config["llm_tags_generator"]
            },
            "tag_type_assigner": {
                "prompt_config": mock_tag_config["tag_type_assigner"]
            },
            "tags_selector": {"prompt_config": mock_tag_config["tags_selector"]},
        },
        "tag_types": sample_tag_types,
        "max_tags": 7,
    }

    state = initialize_tag_generation_state(
        config=config,
        input_text="Example input text",
    )

    # Basic keys exist and are populated properly
    assert state["input_text"] == "Example input text"
    assert isinstance(state["llm_tags_gen_messages"], list)
    assert isinstance(state["tag_type_assigner_messages"], list)
    assert isinstance(state["tags_selector_messages"], list)

    # Ensure each message list has the right system prompts
    for messages in [
        state["llm_tags_gen_messages"],
        state["tag_type_assigner_messages"],
        state["tags_selector_messages"],
    ]:
        assert len(messages) == 2
        assert all(isinstance(m, SystemMessage) for m in messages)

    # Validate presence of tag types prompt
    assert "tag types you can assign" in state["llm_tags_gen_messages"][1].content
    assert (
        "**task**: ML objective like classification"
        in state["llm_tags_gen_messages"][1].content
    )

    # Selector message should mention max tags
    assert "select at most 7 tags" in state["tags_selector_messages"][1].content

    # Lists should be initialized empty
    assert state["llm_tags"] == []
    assert state["spacy_tags"] == []
    assert state["gazetteer_tags"] == []
    assert state["candidate_tags"] == []
    assert state["selected_tags"] == []

    # Other config fields
    assert state["max_tags"] == 7
    assert state["tag_types"] == sample_tag_types


def test_initialize_tag_generation_state_without_input_text(
    mock_tag_config, sample_tag_types
):
    """Test tag generation state template creation (no input text)."""
    config = {
        "agents": {
            "llm_tags_generator": {
                "prompt_config": mock_tag_config["llm_tags_generator"]
            },
            "tag_type_assigner": {
                "prompt_config": mock_tag_config["tag_type_assigner"]
            },
            "tags_selector": {"prompt_config": mock_tag_config["tags_selector"]},
        },
        "tag_types": sample_tag_types,
        "max_tags": 5,
    }

    state = initialize_tag_generation_state(config)

    # Should handle None input text gracefully
    assert state["input_text"] is None
    assert state["max_tags"] == 5
    assert len(state["llm_tags_gen_messages"]) == 2
    assert len(state["tag_type_assigner_messages"]) == 2
    assert len(state["tags_selector_messages"]) == 2


def test_initialize_tag_generation_state_extracts_config_correctly(
    mock_tag_config, sample_tag_types
):
    """Test that function correctly extracts values from config structure."""
    config = {
        "agents": {
            "llm_tags_generator": {
                "prompt_config": mock_tag_config["llm_tags_generator"]
            },
            "tag_type_assigner": {
                "prompt_config": mock_tag_config["tag_type_assigner"]
            },
            "tags_selector": {"prompt_config": mock_tag_config["tags_selector"]},
        },
        "tag_types": sample_tag_types,
        "max_tags": 10,
    }

    state = initialize_tag_generation_state(config, input_text="test")

    # Verify that max_tags from config is used
    assert state["max_tags"] == 10
    assert "select at most 10 tags" in state["tags_selector_messages"][1].content

    # Verify that tag_types from config is used
    assert state["tag_types"] == sample_tag_types
