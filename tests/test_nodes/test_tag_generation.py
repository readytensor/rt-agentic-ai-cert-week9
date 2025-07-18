import pytest
from nodes.tag_generation_nodes import make_gazetteer_tag_generator_node, aggregate_tags_node
from consts import GAZETTEER_TAGS, SPACY_TAGS, CANDIDATE_TAGS, LLM_TAGS



def test_gazetteer_detects_multiple_known_terms(simple_tag_state):
    simple_tag_state["input_text"] = (
        "We used a transformer model trained on the MNIST dataset "
        "with PyTorch to perform image generation and anomaly detection."
    )
    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    assert GAZETTEER_TAGS in result

    expected = {
        "transformer": "algorithm",
        "mnist": "dataset",
        "pytorch": "tool-or-framework",
        "image generation": "task",
        "anomaly detection": "task",
    }

    for name, type_ in expected.items():
        assert {"name": name, "type": type_} in result[GAZETTEER_TAGS]


def test_gazetteer_is_case_insensitive(simple_tag_state):
    simple_tag_state["input_text"] = "We used PyTorch and MNIST and Hugging Face."
    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    expected = {"pytorch", "mnist", "hugging face"}
    found = {tag["name"] for tag in result[GAZETTEER_TAGS]}
    assert expected.issubset(found)


def test_gazetteer_handles_duplicates(simple_tag_state):
    simple_tag_state["input_text"] = "gan gan GAN"
    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    assert result[GAZETTEER_TAGS] == [{"name": "gan", "type": "algorithm"}]


def test_gazetteer_no_match_returns_empty(simple_tag_state):
    simple_tag_state["input_text"] = "No match here."
    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)
    assert result[GAZETTEER_TAGS] == []


def test_spacy_detects_named_entities(spacy_node, simple_tag_state):
    simple_tag_state["input_text"] = "Hugging Face released a new model using PyTorch and trained on CIFAR-10."
    result = spacy_node(simple_tag_state)
    tags = result[SPACY_TAGS]

    assert isinstance(tags, list)
    assert any("hugging face" in tag["name"] for tag in tags)


def test_spacy_excludes_dates_and_cardinals(spacy_node, simple_tag_state):
    simple_tag_state["input_text"] = "The model was trained for 10 days and tested on January 1, 2024."
    result = spacy_node(simple_tag_state)
    tags = result[SPACY_TAGS]

    for tag in tags:
        assert tag["type"] not in {"DATE", "CARDINAL"}


def test_spacy_deduplicates_entities(spacy_node, simple_tag_state):
    simple_tag_state["input_text"] = "We used PyTorch with PyTorch in our PyTorch experiments."
    result = spacy_node(simple_tag_state)
    tags = result[SPACY_TAGS]

    # Ensure only one unique entry for "pytorch"
    names = [tag["name"] for tag in tags]
    assert names.count("pytorch") == 1


def test_spacy_returns_empty_for_no_entities(spacy_node, simple_tag_state):
    simple_tag_state["input_text"] = "abcd efgh ijkl"
    result = spacy_node(simple_tag_state)

    assert result[SPACY_TAGS] == []


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
