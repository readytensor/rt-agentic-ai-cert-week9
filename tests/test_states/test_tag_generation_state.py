from states.tag_generation_state import initialize_tag_generation_state


from states.tag_generation_state import initialize_tag_generation_state
from langchain_core.messages import SystemMessage


def test_initialize_tag_generation_state(
    tag_generation_prompt_configs, sample_tag_types
):
    state = initialize_tag_generation_state(
        llm_tags_generator_prompt_cfg=tag_generation_prompt_configs[
            "llm_tags_generator"
        ],
        tag_type_assigner_prompt_cfg=tag_generation_prompt_configs["tag_type_assigner"],
        tags_selector_prompt_cfg=tag_generation_prompt_configs["tags_selector"],
        tag_types=sample_tag_types,
        max_tags=7,
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
