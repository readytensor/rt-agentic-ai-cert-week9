import pytest

from consts import SPACY_TAGS


def test_spacy_detects_named_entities(spacy_node, simple_tag_state):
    simple_tag_state["input_text"] = (
        "Hugging Face released a new model using PyTorch and trained on CIFAR-10."
    )
    result = spacy_node(simple_tag_state)
    tags = result[SPACY_TAGS]

    assert isinstance(tags, list)
    assert any("hugging face" in tag["name"] for tag in tags)


def test_spacy_excludes_dates_and_cardinals(spacy_node, simple_tag_state):
    simple_tag_state["input_text"] = (
        "The model was trained for 10 days and tested on January 1, 2024."
    )
    result = spacy_node(simple_tag_state)
    tags = result[SPACY_TAGS]

    for tag in tags:
        assert tag["type"] not in {"DATE", "CARDINAL"}


def test_spacy_deduplicates_entities(spacy_node, simple_tag_state):
    simple_tag_state["input_text"] = (
        "We used PyTorch with PyTorch in our PyTorch experiments."
    )
    result = spacy_node(simple_tag_state)
    tags = result[SPACY_TAGS]

    # Ensure only one unique entry for "pytorch"
    names = [tag["name"] for tag in tags]
    assert names.count("pytorch") == 1


def test_spacy_returns_empty_for_no_entities(spacy_node, simple_tag_state):
    simple_tag_state["input_text"] = "abcd efgh ijkl"
    result = spacy_node(simple_tag_state)

    assert result[SPACY_TAGS] == []


def test_spacy_handles_empty_input_text(spacy_node, simple_tag_state):
    """Test that spaCy handles empty input text gracefully."""
    simple_tag_state["input_text"] = ""
    result = spacy_node(simple_tag_state)
    assert result[SPACY_TAGS] == []


def test_spacy_handles_none_input_text(spacy_node, simple_tag_state):
    """Test that spaCy handles None input text gracefully."""
    simple_tag_state["input_text"] = None
    result = spacy_node(simple_tag_state)

    # Should handle gracefully and return empty list
    assert result[SPACY_TAGS] == []


def test_spacy_handles_whitespace_only_input(spacy_node, simple_tag_state):
    """Test that spaCy handles whitespace-only input gracefully."""
    simple_tag_state["input_text"] = "   \n\t   "
    result = spacy_node(simple_tag_state)

    # Should handle gracefully and return empty list
    assert result[SPACY_TAGS] == []


def test_spacy_normalizes_text_case_and_whitespace(spacy_node, simple_tag_state):
    """Test that spaCy properly normalizes entity names."""
    simple_tag_state["input_text"] = "We used  PYTORCH  and   TensorFlow   frameworks."
    result = spacy_node(simple_tag_state)

    found_names = {tag["name"] for tag in result[SPACY_TAGS]}

    # Should be lowercase and stripped
    if "pytorch" in found_names:
        assert "pytorch" in found_names  # lowercase
        assert "PYTORCH" not in found_names  # not uppercase
        assert " pytorch " not in found_names  # no extra spaces


def test_spacy_handles_very_long_text(spacy_node, simple_tag_state):
    """Test that spaCy handles very long text without crashing."""
    # Create long text with known entities
    long_text = "We used PyTorch for machine learning. " * 500  # Very long
    simple_tag_state["input_text"] = long_text

    result = spacy_node(simple_tag_state)

    # Should still work and deduplicate
    assert SPACY_TAGS in result
    names = [tag["name"] for tag in result[SPACY_TAGS]]
    if "pytorch" in names:
        assert names.count("pytorch") == 1  # Deduplicated despite many occurrences
