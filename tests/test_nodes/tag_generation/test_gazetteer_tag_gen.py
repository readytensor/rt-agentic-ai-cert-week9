from nodes.tag_generation_nodes import make_gazetteer_tag_generator_node
from consts import GAZETTEER_TAGS


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

