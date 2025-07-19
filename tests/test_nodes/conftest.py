import pytest

from states.tag_generation_state import TagGenerationState
from nodes.tag_generation_nodes import make_spacy_tag_generator_node
from utils import load_config
from paths import GAZETTEER_ENTITIES_FILE_PATH


@pytest.fixture
def gazetteer_config():
    return load_config(GAZETTEER_ENTITIES_FILE_PATH)


@pytest.fixture
def simple_tag_state():
    return TagGenerationState(
        input_text="OpenAI is based in San Francisco. It was founded in 2015.",
        llm_tags_gen_messages=[],
        tag_type_assigner_messages=[],
        tags_selector_messages=[],
        llm_tags=[],
        spacy_tags=[],
        gazetteer_tags=[],
        candidate_tags=[],
        selected_tags=[],
        max_tags=5,
        tag_types=[],
    )


@pytest.fixture(scope="module")
def spacy_node():
    """Use a fixture because the node is expensive to create - involves model loading."""
    return make_spacy_tag_generator_node()
