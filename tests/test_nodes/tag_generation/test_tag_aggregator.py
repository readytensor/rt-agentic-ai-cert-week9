from nodes.tag_generation_nodes import aggregate_tags_node
from consts import GAZETTEER_TAGS, SPACY_TAGS, CANDIDATE_TAGS, LLM_TAGS


def test_aggregator_combines_all_tags(simple_tag_state):
    simple_tag_state["llm_tags"] = [{"name": "gan", "type": "algorithm"}]
    simple_tag_state["spacy_tags"] = [{"name": "mnist", "type": "dataset"}]
    simple_tag_state["gazetteer_tags"] = [{"name": "pytorch", "type": "tool-or-framework"}]

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
        "gazetteer_tags": [{"name": "mnist", "type": "dataset"}], # valid tag
    }

    result = aggregate_tags_node(state)

    assert CANDIDATE_TAGS in result
    assert result[CANDIDATE_TAGS] == [{"name": "mnist", "type": "dataset"}]
