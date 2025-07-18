from states.a3_state import initialize_a3_state
from langchain_core.messages import SystemMessage

def test_initialize_a3_state(minimal_prompt_configs, sample_tag_types):
    state = initialize_a3_state(
        input_text="Example article",
        manager_prompt_cfg=minimal_prompt_configs["manager"],
        llm_tags_generator_prompt_cfg=minimal_prompt_configs["llm_tags_generator"],
        tag_type_assigner_prompt_cfg=minimal_prompt_configs["tag_type_assigner"],
        tags_selector_prompt_cfg=minimal_prompt_configs["tags_selector"],
        title_gen_prompt_cfg=minimal_prompt_configs["title_generator"],
        tldr_gen_prompt_cfg=minimal_prompt_configs["tldr_generator"],
        references_gen_prompt_cfg=minimal_prompt_configs["references_generator"],
        references_selector_prompt_cfg=minimal_prompt_configs["references_selector"],
        reviewer_prompt_cfg=minimal_prompt_configs["reviewer"],
        max_tags=8,
        tag_types=sample_tag_types,
        max_search_queries=4,
        max_references=5,
        max_revisions=2,
    )

    # General fields
    assert state["input_text"] == "Example article"
    assert state["manager_brief"] is None
    assert state["revision_round"] == 0
    assert state["needs_revision"] is False
    assert state["tldr_approved"] is False
    assert state["title_approved"] is False
    assert state["references_approved"] is False
    assert state["max_tags"] == 8
    assert state["max_references"] == 5
    assert state["max_revisions"] == 2
    assert state["max_search_queries"] == 4

    # All optional outputs should be initialized to empty or None
    assert state["tldr"] is None
    assert state["title"] is None
    assert state["reference_search_queries"] is None
    assert state["candidate_references"] == []
    assert state["selected_references"] == []
    assert state["tldr_feedback"] is None
    assert state["title_feedback"] is None
    assert state["references_feedback"] is None

    # Tagging-related lists
    assert state["llm_tags"] == []
    assert state["spacy_tags"] == []
    assert state["gazetteer_tags"] == []
    assert state["selected_tags"] == []
    assert state["candidate_tags"] == []

    # Check that every agent message list exists and is correctly formatted
    for field in [
        "manager_messages",
        "title_gen_messages",
        "tldr_gen_messages",
        "references_gen_messages",
        "references_selector_messages",
        "reviewer_messages",
        "llm_tags_gen_messages",
        "tag_type_assigner_messages",
        "tags_selector_messages",
    ]:
        assert field in state
        assert isinstance(state[field], list)
        assert all(isinstance(m, SystemMessage) for m in state[field])
        assert len(state[field]) >= 1  # Some have 1, others 2

    # Check presence of tag type instructions in LLM tag generator messages
    llm_tag_msg = state["llm_tags_gen_messages"][1].content
    assert "tag types you can assign" in llm_tag_msg
    assert "**task**: ML objective like classification" in state["llm_tags_gen_messages"][1].content

    # Check selector constraint
    assert "select at most 8 tags" in state["tags_selector_messages"][1].content

    # Check query constraint
    assert "generate at most 4 search queries" in state["references_gen_messages"][1].content

    # Check reference selector constraint
    assert "select at most 5 references" in state["references_selector_messages"][1].content
