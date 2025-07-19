from nodes.tag_generation_nodes import aggregate_tags_node
from consts import GAZETTEER_TAGS, SPACY_TAGS, CANDIDATE_TAGS, LLM_TAGS


def test_aggregator_combines_all_tags(simple_tag_state):
    simple_tag_state["llm_tags"] = [{"name": "gan", "type": "algorithm"}]
    simple_tag_state["spacy_tags"] = [{"name": "mnist", "type": "dataset"}]
    simple_tag_state["gazetteer_tags"] = [
        {"name": "pytorch", "type": "tool-or-framework"}
    ]

    result = aggregate_tags_node(simple_tag_state)

    assert result[CANDIDATE_TAGS] == [
        {"name": "gan", "type": "algorithm"},
        {"name": "mnist", "type": "dataset"},
        {"name": "pytorch", "type": "tool-or-framework"},
    ]


def test_aggregator_deduplicates_tags(simple_tag_state):
    simple_tag_state["llm_tags"] = [{"name": "gan", "type": "algorithm"}]
    simple_tag_state["spacy_tags"] = [{"name": "GAN", "type": "algorithm"}]
    simple_tag_state["gazetteer_tags"] = [{"name": "Gan", "type": "algorithm"}]

    result = aggregate_tags_node(simple_tag_state)

    assert result[CANDIDATE_TAGS] == [{"name": "gan", "type": "algorithm"}]


def test_aggregator_with_no_tags(simple_tag_state):
    simple_tag_state["llm_tags"] = []
    simple_tag_state["spacy_tags"] = []
    simple_tag_state["gazetteer_tags"] = []

    result = aggregate_tags_node(simple_tag_state)

    assert result[CANDIDATE_TAGS] == []


def test_aggregate_tags_node_handles_missing_sources():
    state = {
        "llm_tags": [{"name": "gan", "type": "algorithm"}],
        # spacy_tags and gazetteer_tags are missing
    }

    result = aggregate_tags_node(state)

    assert CANDIDATE_TAGS in result
    assert result[CANDIDATE_TAGS] == [{"name": "gan", "type": "algorithm"}]


def test_aggregator_with_partial_sources(simple_tag_state):
    simple_tag_state["llm_tags"] = [{"name": "mnist", "type": "dataset"}]
    simple_tag_state["spacy_tags"] = []
    simple_tag_state["gazetteer_tags"] = [{"name": "gan", "type": "algorithm"}]

    result = aggregate_tags_node(simple_tag_state)

    assert result[CANDIDATE_TAGS] == [
        {"name": "mnist", "type": "dataset"},
        {"name": "gan", "type": "algorithm"},
    ]


def test_aggregate_retains_same_name_different_type():
    state = {
        LLM_TAGS: [{"name": "transformer", "type": "algorithm"}],
        SPACY_TAGS: [{"name": "transformer", "type": "tool-or-framework"}],
        GAZETTEER_TAGS: [],
    }

    result = aggregate_tags_node(state)

    tags = result[CANDIDATE_TAGS]
    assert len(tags) == 2
    assert {"name": "transformer", "type": "algorithm"} in tags
    assert {"name": "transformer", "type": "tool-or-framework"} in tags


def test_aggregate_tags_node_ignores_malformed_tags():
    state = {
        "llm_tags": [{"name": "gan"}],  # missing type
        "spacy_tags": [{"type": "algorithm"}],  # missing name
        "gazetteer_tags": [{"name": "mnist", "type": "dataset"}],  # valid tag
    }

    result = aggregate_tags_node(state)

    assert CANDIDATE_TAGS in result
    assert result[CANDIDATE_TAGS] == [{"name": "mnist", "type": "dataset"}]


def test_aggregator_handles_whitespace_in_names_and_types():
    """Test that aggregator properly strips whitespace and handles empty strings."""
    state = {
        LLM_TAGS: [
            {"name": "  gan  ", "type": "  algorithm  "},  # whitespace
            {"name": "mnist", "type": "dataset"},  # normal
            {"name": "   ", "type": "algorithm"},  # whitespace-only name
            {"name": "pytorch", "type": "   "},  # whitespace-only type
        ],
        SPACY_TAGS: [],
        GAZETTEER_TAGS: [],
    }

    result = aggregate_tags_node(state)

    # Should strip whitespace and exclude entries with empty name/type after stripping
    expected = [
        {"name": "gan", "type": "algorithm"},  # stripped
        {"name": "mnist", "type": "dataset"},  # unchanged
        # whitespace-only entries should be excluded
    ]

    assert result[CANDIDATE_TAGS] == expected


def test_aggregator_handles_none_values():
    """Test that aggregator handles None values gracefully."""
    state = {
        LLM_TAGS: [
            {"name": None, "type": "algorithm"},  # None name
            {"name": "mnist", "type": None},  # None type
            {"name": "pytorch", "type": "framework"},  # valid
        ],
        SPACY_TAGS: [],
        GAZETTEER_TAGS: [],
    }

    result = aggregate_tags_node(state)

    # Should exclude entries with None values, keep valid ones
    assert result[CANDIDATE_TAGS] == [{"name": "pytorch", "type": "framework"}]


def test_aggregator_handles_missing_name_type_keys():
    """Test that aggregator handles tags missing name or type keys."""
    state = {
        LLM_TAGS: [
            {},  # completely empty
            {"name": "gan"},  # missing type key
            {"type": "algorithm"},  # missing name key
            {"name": "mnist", "type": "dataset"},  # valid
        ],
        SPACY_TAGS: [],
        GAZETTEER_TAGS: [],
    }

    result = aggregate_tags_node(state)

    # Should only keep the valid tag
    assert result[CANDIDATE_TAGS] == [{"name": "mnist", "type": "dataset"}]


def test_aggregator_normalizes_case_consistently():
    """Test that aggregator consistently normalizes case across all sources."""
    state = {
        LLM_TAGS: [{"name": "PyTorch", "type": "FRAMEWORK"}],
        SPACY_TAGS: [{"name": "MNIST", "type": "dataset"}],
        GAZETTEER_TAGS: [{"name": "gan", "type": "Algorithm"}],
    }

    result = aggregate_tags_node(state)

    expected = [
        {"name": "pytorch", "type": "framework"},
        {"name": "mnist", "type": "dataset"},
        {"name": "gan", "type": "algorithm"},
    ]

    assert result[CANDIDATE_TAGS] == expected


def test_aggregator_deduplication_across_different_sources():
    """Test that aggregator deduplicates same entities from different sources."""
    state = {
        LLM_TAGS: [{"name": "transformer", "type": "algorithm"}],
        SPACY_TAGS: [
            {"name": "TRANSFORMER", "type": "ALGORITHM"}
        ],  # same but different case
        GAZETTEER_TAGS: [
            {"name": "Transformer", "type": "Algorithm"},  # same but different case
            {"name": "mnist", "type": "dataset"},  # different entity
        ],
    }

    result = aggregate_tags_node(state)

    # Should deduplicate the transformer entries, keep mnist
    expected = [
        {"name": "transformer", "type": "algorithm"},
        {"name": "mnist", "type": "dataset"},
    ]

    assert result[CANDIDATE_TAGS] == expected
