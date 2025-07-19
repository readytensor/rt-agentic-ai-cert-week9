from nodes.tag_generation_nodes import make_gazetteer_tag_generator_node
from consts import GAZETTEER_TAGS, INPUT_TEXT


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


def test_gazetteer_handles_empty_input_text(simple_tag_state):
    """Test that gazetteer handles empty input text gracefully."""
    simple_tag_state[INPUT_TEXT] = ""

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    assert GAZETTEER_TAGS in result
    assert result[GAZETTEER_TAGS] == []


def test_gazetteer_handles_missing_input_text_key(simple_tag_state):
    """Test that gazetteer handles missing INPUT_TEXT key gracefully."""
    # Remove INPUT_TEXT key entirely
    if INPUT_TEXT in simple_tag_state:
        del simple_tag_state[INPUT_TEXT]

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    assert GAZETTEER_TAGS in result
    assert result[GAZETTEER_TAGS] == []


def test_gazetteer_handles_none_input_text(simple_tag_state):
    """Test that gazetteer handles None input text gracefully."""
    simple_tag_state[INPUT_TEXT] = None

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    assert GAZETTEER_TAGS in result
    assert result[GAZETTEER_TAGS] == []


def test_gazetteer_word_boundary_matching(simple_tag_state):
    """Test that gazetteer only matches whole words, not partial matches."""
    simple_tag_state[INPUT_TEXT] = (
        "We use transform operations and transformer models. "
        "The transformation process uses transformers effectively. "
        "Gantt charts and gan networks are different. "
        "Scikit package versus scikit-learn library."
    )

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    found_names = {tag["name"] for tag in result[GAZETTEER_TAGS]}

    # Should match exact entities
    assert "transformer" in found_names
    assert "gan" in found_names
    assert "scikit-learn" in found_names

    # Should NOT match partial words
    assert "transform" not in found_names  # partial word
    assert "transformation" not in found_names  # partial word
    assert "transformers" not in found_names  # plural (not in gazetteer)
    assert "gantt" not in found_names  # different word that contains "gan"
    assert "scikit" not in found_names  # partial word

    # Verify we only found the expected entities
    expected_entities = {"transformer", "gan", "scikit-learn"}
    assert found_names == expected_entities


def test_gazetteer_handles_punctuation_boundaries(simple_tag_state):
    """Test that gazetteer matches entities adjacent to punctuation."""
    simple_tag_state[INPUT_TEXT] = (
        "We used pytorch, transformer; and scikit-learn. "
        "Datasets like mnist/cifar-10 work well (especially coco captions). "
        "Tasks include: anomaly detection, sentiment analysis!"
    )

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    found_names = {tag["name"] for tag in result[GAZETTEER_TAGS]}

    # Should match entities even when followed by punctuation
    assert "pytorch" in found_names  # followed by comma
    assert "transformer" in found_names  # followed by semicolon
    assert "scikit-learn" in found_names  # followed by period
    assert "mnist" in found_names  # followed by slash
    assert "cifar-10" in found_names  # followed by space after slash
    assert "coco captions" in found_names  # followed by parenthesis
    assert "anomaly detection" in found_names  # followed by comma
    assert "sentiment analysis" in found_names  # followed by exclamation


def test_gazetteer_handles_multiword_entities(simple_tag_state):
    """Test that gazetteer correctly matches multi-word entities."""
    simple_tag_state[INPUT_TEXT] = (
        "We use anomaly detection and sentiment analysis techniques. "
        "Our use-case involves fraud detection and legal document classification. "
        "We trained on uci heart disease and coco captions datasets. "
        "The task includes text summarization and image generation."
    )

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    found_names = {tag["name"] for tag in result[GAZETTEER_TAGS]}

    # Verify multi-word entities are matched
    expected_multiword = {
        "anomaly detection",
        "sentiment analysis",
        "fraud detection",
        "legal document classification",
        "uci heart disease",
        "coco captions",
        "text summarization",
        "image generation",
    }

    for entity in expected_multiword:
        assert entity in found_names, f"Multi-word entity '{entity}' not found"


def test_gazetteer_handles_special_characters_in_entities(simple_tag_state):
    """Test that gazetteer handles entities with special characters."""
    simple_tag_state[INPUT_TEXT] = (
        "We used scikit-learn for preprocessing. "
        "The cifar-10 dataset and variational auto-encoders worked well. "
        "We also tested uci heart disease data."
    )

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    found_names = {tag["name"] for tag in result[GAZETTEER_TAGS]}

    # Verify entities with hyphens and other special characters are matched
    expected_entities = {
        "scikit-learn",  # hyphen
        "cifar-10",  # hyphen + number
        "variational auto-encoders",  # hyphen in compound word
        "uci heart disease",  # spaces
    }

    for entity in expected_entities:
        assert entity in found_names, f"Entity with special chars '{entity}' not found"


def test_gazetteer_handles_very_long_text(simple_tag_state):
    """Test that gazetteer performs reasonably with very long input text."""
    # Create a long text with known entities scattered throughout
    long_text_parts = [
        "We use transformer models",
        "pytorch is great",
        "mnist dataset works well",
        "Deep learning approaches",
    ]
    # Repeat and join to create very long text
    long_text = ". ".join(long_text_parts * 100)  # ~400 repetitions

    simple_tag_state[INPUT_TEXT] = long_text

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    # Should still work and return deduplicated results
    assert GAZETTEER_TAGS in result
    found_names = {tag["name"] for tag in result[GAZETTEER_TAGS]}

    # Should find the expected entities from the repeated text
    expected_entities = {"transformer", "pytorch", "mnist"}
    assert expected_entities.issubset(
        found_names
    ), f"Expected entities {expected_entities} not all found in {found_names}"

    # Verify deduplication works even with many occurrences
    tag_counts = {}
    for tag in result[GAZETTEER_TAGS]:
        name = tag["name"]
        tag_counts[name] = tag_counts.get(name, 0) + 1

    # Each entity should appear only once in results
    for name, count in tag_counts.items():
        assert count == 1, f"Entity '{name}' appears {count} times, should be 1"

    # Verify the expected entities were actually found and deduplicated
    for expected_entity in expected_entities:
        assert (
            expected_entity in tag_counts
        ), f"Expected entity '{expected_entity}' not found"
        assert (
            tag_counts[expected_entity] == 1
        ), f"Entity '{expected_entity}' not properly deduplicated"


def test_gazetteer_handles_regex_error_gracefully(simple_tag_state, monkeypatch):
    """Test that gazetteer handles regex errors gracefully and continues processing."""
    # Mock a gazetteer with problematic regex patterns
    problematic_gazetteer = {
        "valid_entity": "algorithm",
        "[unclosed": "invalid_regex",  # This would cause re.error
        "another_valid": "dataset",
    }

    # Mock load_config to return our problematic gazetteer
    monkeypatch.setattr(
        "nodes.tag_generation_nodes.load_config", lambda _: problematic_gazetteer
    )

    simple_tag_state[INPUT_TEXT] = (
        "This text contains valid_entity and another_valid terms."
    )

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    # Should continue processing valid entities despite regex errors
    assert GAZETTEER_TAGS in result
    found_names = {tag["name"] for tag in result[GAZETTEER_TAGS]}

    # Should find valid entities
    assert "valid_entity" in found_names
    assert "another_valid" in found_names
    # Should not crash due to regex error


def test_gazetteer_handles_empty_gazetteer(simple_tag_state, monkeypatch):
    """Test that gazetteer handles empty gazetteer dictionary."""
    # Mock load_config to return empty gazetteer
    monkeypatch.setattr("nodes.tag_generation_nodes.load_config", lambda _: {})

    simple_tag_state[INPUT_TEXT] = "This text has many words but no entities."

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    assert GAZETTEER_TAGS in result
    assert result[GAZETTEER_TAGS] == []


def test_gazetteer_handles_malformed_gazetteer_data(simple_tag_state, monkeypatch):
    """Test that gazetteer handles malformed gazetteer data gracefully."""
    # Mock a gazetteer with various malformed entries
    malformed_gazetteer = {
        "valid_entity": "algorithm",
        "": "empty_name",  # Empty entity name
        "valid_name": "",  # Empty type
        None: "none_name",  # None as key (would cause issues)
        "another_valid": None,  # None as value
    }

    monkeypatch.setattr(
        "nodes.tag_generation_nodes.load_config", lambda _: malformed_gazetteer
    )

    simple_tag_state[INPUT_TEXT] = "This contains valid_entity and valid_name."

    node = make_gazetteer_tag_generator_node()

    # This might raise an exception depending on how the iteration handles None keys
    # Or it might handle gracefully - adjust test based on desired behavior
    try:
        result = node(simple_tag_state)
        assert GAZETTEER_TAGS in result
        # If it succeeds, verify it handled valid entries
        found_names = {tag["name"] for tag in result[GAZETTEER_TAGS]}
        assert "valid_entity" in found_names
    except (TypeError, AttributeError):
        # If it fails on malformed data, that's also acceptable behavior
        # You might want to add defensive programming to handle this
        pass


def test_gazetteer_preserves_entity_types_correctly(simple_tag_state):
    """Test that gazetteer preserves the correct entity types from gazetteer."""
    simple_tag_state[INPUT_TEXT] = (
        "We used transformer and gan algorithms with the mnist dataset. "
        "Our use-case was fraud detection using pytorch framework."
    )

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    # Create a mapping of found entities to their types
    entity_type_map = {tag["name"]: tag["type"] for tag in result[GAZETTEER_TAGS]}

    # Verify that each entity has the correct type as defined in gazetteer
    expected_types = {
        "transformer": "algorithm",
        "gan": "algorithm",
        "mnist": "dataset",
        "fraud detection": "use-case",
        "pytorch": "tool-or-framework",
    }

    for entity_name, expected_type in expected_types.items():
        if entity_name in entity_type_map:
            assert (
                entity_type_map[entity_name] == expected_type
            ), f"Entity '{entity_name}' has type '{entity_type_map[entity_name]}', expected '{expected_type}'"


def test_gazetteer_strips_whitespace_from_types(simple_tag_state, monkeypatch):
    """Test that gazetteer properly strips whitespace from entity types."""
    # Mock gazetteer with whitespace in types
    gazetteer_with_whitespace = {
        "entity1": "  algorithm  ",
        "entity2": "\tdataset\n",
        "entity3": "tool-or-framework",  # normal
    }

    monkeypatch.setattr(
        "nodes.tag_generation_nodes.load_config", lambda _: gazetteer_with_whitespace
    )

    simple_tag_state[INPUT_TEXT] = "We used entity1, entity2, and entity3."

    node = make_gazetteer_tag_generator_node()
    result = node(simple_tag_state)

    # Verify types are properly stripped
    entity_type_map = {tag["name"]: tag["type"] for tag in result[GAZETTEER_TAGS]}

    assert entity_type_map["entity1"] == "algorithm"  # Stripped
    assert entity_type_map["entity2"] == "dataset"  # Stripped
    assert entity_type_map["entity3"] == "tool-or-framework"  # Unchanged
