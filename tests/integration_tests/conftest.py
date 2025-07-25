import pytest
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from pathlib import Path


@pytest.fixture(scope="session")
def hf_embedder():
    """
    Session-scoped fixture to load HuggingFace embedder once for all integration tests.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def a3_config():
    """
    Session-scoped fixture to load A3 system configuration.
    """
    from utils import load_config

    config = load_config("config/config.yaml")
    return config["a3_system"]


@pytest.fixture
def example1_data():
    input_path = Path("tests/data/input_text/publication_example1.md")
    expected_path = Path("tests/data/expected/publication_example1.json")

    input_text = input_path.read_text(encoding="utf-8")

    from utils import read_json

    expected = read_json(expected_path)

    return input_text, expected


EXAMPLE_IDS = [
    "publication_example1",
    "publication_example2",
    "publication_example3",
    "publication_example4",
]


@pytest.fixture(params=EXAMPLE_IDS)
def example_data(request):
    example_id = request.param
    input_path = Path(f"tests/data/input_text/{example_id}.md")
    expected_path = Path(f"tests/data/expected/{example_id}.json")

    input_text = input_path.read_text(encoding="utf-8")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    return example_id, input_text, expected
