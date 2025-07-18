import pytest


@pytest.fixture
def sample_input_text():
    return "This article explores how Transformers are applied to fraud detection in finance."


@pytest.fixture
def sample_tag_types():
    return [
        {"name": "task", "description": "ML objective like classification"},
        {"name": "algorithm", "description": "Named algorithm like Transformer"},
        {"name": "industry", "description": "Application domain like finance"},
    ]


@pytest.fixture
def sample_max_tags():
    return 5

@pytest.fixture
def sample_max_revisions():
    return 2

@pytest.fixture
def sample_max_search_queries():
    return 3

@pytest.fixture
def sample_max_references():
    return 4

@pytest.fixture
def sample_prompt_config():
    return {
        "role": "an assistant",
        "instruction": "Summarize the input clearly.",
        "output_constraints": ["Keep it short."],
        "style_or_tone": ["Clear and helpful"],
        "goal": "Generate a high-quality summary.",
    }

@pytest.fixture
def tag_generation_prompt_configs():
    return {
        "llm_tags_generator": {
            "role": "an analyst",
            "instruction": "Extract tags from the text.",
        },
        "tag_type_assigner": {
            "role": "a classifier",
            "instruction": "Assign types to the tags.",
        },
        "tags_selector": {
            "role": "a selector",
            "instruction": "Pick the most relevant tags.",
        }
    }

@pytest.fixture
def minimal_prompt_configs():
    return {
        "manager": {"role": "manager", "instruction": "Do manager task"},
        "llm_tags_generator": {"role": "tagger", "instruction": "Extract tags"},
        "tag_type_assigner": {"role": "classifier", "instruction": "Assign tag types"},
        "tags_selector": {"role": "selector", "instruction": "Select tags"},
        "title_generator": {"role": "title bot", "instruction": "Generate a title"},
        "tldr_generator": {"role": "summarizer", "instruction": "Summarize content"},
        "references_generator": {"role": "query bot", "instruction": "Generate queries"},
        "references_selector": {"role": "ref bot", "instruction": "Select references"},
        "reviewer": {"role": "reviewer", "instruction": "Review content"},
    }
