from consts import SPACY_TAGS

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

