import pytest

from code.consts import (
    MANAGER_BRIEF,
    TITLE,
    TLDR,
    REFERENCES_FEEDBACK,
    REFERENCE_SEARCH_QUERIES,
    CANDIDATE_REFERENCES,
    SELECTED_REFERENCES,
    MANAGER_BRIEF,
    TITLE,
    TLDR,
    SELECTED_REFERENCES,
    REVISION_ROUND,
    NEEDS_REVISION,
    TITLE_APPROVED,
    TLDR_APPROVED,
    REFERENCES_APPROVED,
    TITLE_FEEDBACK,
    TLDR_FEEDBACK,
    REFERENCES_FEEDBACK,
)
from code.nodes.a3_nodes import (
    make_manager_node,
    make_title_generator_node,
    make_tldr_generator_node,
    make_references_generator_node,
    make_references_generator_node,
    make_references_selector_node,
    make_reviewer_node,
)
from code.states.a3_state import initialize_a3_state
from tests.integration_tests.utils.similarity import embed_and_score_similarity


@pytest.mark.integration
def test_manager_brief_generation(example1_data, hf_embedder, a3_config):
    """
    Test that the manager brief is semantically similar to the expected brief.

    Args:
        example1_data: Tuple containing input markdown and expected JSON.
        hf_embedder: HuggingFace embedder for semantic similarity scoring.
        a3_config: System configuration for agent selection.

    Asserts:
        - Non-empty manager brief is generated.
        - Semantic similarity > 0.5.
        - At least 50% of expected keywords appear.
        - Word count is between 100 and 350.
    """
    input_text, expected = example1_data
    expected_brief = expected.get("manager_brief", "")
    expected_keywords = expected.get("manager_brief_expected_keywords", [])

    state = initialize_a3_state(config=a3_config, input_text=input_text)

    node = make_manager_node(llm_model=a3_config["agents"]["manager"]["llm"])
    result_state = node(state)
    generated_brief = result_state.get(MANAGER_BRIEF, "")

    assert generated_brief, "Generated manager_brief is empty."

    similarity = embed_and_score_similarity(
        generated_brief, expected_brief, hf_embedder
    )
    assert similarity > 0.5, f"Low semantic similarity: {similarity:.2f}"

    # At least 50% of expected keywords should be present
    lowered_brief = generated_brief.lower()
    missing_keywords = [
        kw for kw in expected_keywords if kw.lower() not in lowered_brief
    ]
    coverage_ratio = (
        1 - (len(missing_keywords) / len(expected_keywords))
        if expected_keywords
        else 1.0
    )

    assert coverage_ratio >= 0.5, (
        f"Keyword coverage too low: {coverage_ratio:.2f}. "
        f"Missing: {missing_keywords}"
    )

    # Brief length sanity check
    word_count = len(generated_brief.split())
    assert 100 <= word_count <= 350, f"Brief word count out of range: {word_count}"


@pytest.mark.integration
def test_title_generator_node_output(example1_data, hf_embedder, a3_config):
    """
    Test that the title generator creates a title that is concise and relevant.

    Asserts:
        - Title is non-empty.
        - Semantic similarity with expected > 0.5.
        - Title has <= 12 words.
    """
    input_text, expected = example1_data
    expected_title = expected.get("title", "")
    manager_brief = expected.get("manager_brief", "")

    state = initialize_a3_state(config=a3_config, input_text=input_text)
    state[MANAGER_BRIEF] = manager_brief

    node = make_title_generator_node(
        llm_model=a3_config["agents"]["title_generator"]["llm"]
    )
    result = node(state)

    generated_title = result.get(TITLE, "")
    assert generated_title, "Generated title is empty."

    similarity = embed_and_score_similarity(
        generated_title, expected_title, hf_embedder
    )
    assert similarity > 0.5, f"Low title similarity: {similarity:.2f}"

    word_count = len(generated_title.split())
    assert word_count <= 12, f"Title too long: {word_count} words"


@pytest.mark.integration
def test_tldr_generator_node_output(example1_data, hf_embedder, a3_config):
    """
    Test that the TL;DR generator produces a semantically aligned summary.

    Asserts:
        - TL;DR is non-empty.
        - Semantic similarity with expected > 0.5.
        - Length is between 1 and 4 sentences.
    """
    input_text, expected = example1_data
    expected_tldr = expected.get("tldr", "")
    manager_brief = expected.get("manager_brief", "")

    state = initialize_a3_state(config=a3_config, input_text=input_text)
    state[MANAGER_BRIEF] = manager_brief

    node = make_tldr_generator_node(
        llm_model=a3_config["agents"]["tldr_generator"]["llm"]
    )
    result = node(state)

    generated_tldr = result.get(TLDR, "")
    assert generated_tldr, "Generated TLDR is empty."

    # Similarity check
    similarity = embed_and_score_similarity(generated_tldr, expected_tldr, hf_embedder)
    assert similarity > 0.5, f"Low TLDR similarity: {similarity:.2f}"

    # Length check — 2–3 sentences expected
    sentence_count = (
        generated_tldr.count(".")
        + generated_tldr.count("!")
        + generated_tldr.count("?")
    )
    assert 1 <= sentence_count <= 4, f"TLDR sentence count seems off: {sentence_count}"


@pytest.mark.integration
def test_references_generator_node_output_semantic(
    example1_data, hf_embedder, a3_config
):
    """
    Test that the references generator outputs relevant queries and reference titles.

    Asserts:
        - At least one search query and reference is returned.
        - Each query is semantically relevant to the input text.
        - Each reference title is semantically relevant to the input text.
    """
    input_text, expected = example1_data
    manager_brief = expected.get("manager_brief", "")

    state = initialize_a3_state(config=a3_config, input_text=input_text)
    state[MANAGER_BRIEF] = manager_brief

    node = make_references_generator_node(
        llm_model=a3_config["agents"]["references_generator"]["llm"]
    )
    result = node(state)

    queries = result.get(REFERENCE_SEARCH_QUERIES, [])
    references = result.get(CANDIDATE_REFERENCES, [])

    assert queries, "No search queries generated."
    assert isinstance(queries, list)

    # checking for semantic relevance of queries
    for q in queries:
        score = embed_and_score_similarity(q, input_text, hf_embedder)
        assert (
            score > 0.3
        ), f"Query not semantically aligned: '{q}' (similarity: {score:.2f})"

    assert references, "No references generated."
    # checking for semantic relevance of references
    for ref in references:
        title = ref.get("title", "")
        assert title, "Reference missing title"
        score = embed_and_score_similarity(title, input_text, hf_embedder)
        assert (
            score > 0.3
        ), f"Reference title not relevant: '{title}' (similarity: {score:.2f})"


@pytest.mark.integration
def test_references_selector_node_subset(example1_data, a3_config):
    """
    Test that the references selector only selects references from the given candidates.

    Asserts:
        - Selected references are not empty.
        - Selected references are a subset of the input candidate references.
    """
    input_text, expected = example1_data
    manager_brief = expected.get("manager_brief", "")
    expected_selected_refs = expected.get("selected_references", [])

    assert expected_selected_refs, "No selected references in expected data"

    # Use selected references as dummy candidates for testing
    candidate_refs = expected_selected_refs + [
        {
            "title": "Extra Reference",
            "url": "https://example.com",
            "page_content": "Irrelevant page content",
        }
    ]

    state = initialize_a3_state(config=a3_config, input_text=input_text)
    state[MANAGER_BRIEF] = manager_brief
    state[CANDIDATE_REFERENCES] = candidate_refs

    node = make_references_selector_node(
        llm_model=a3_config["agents"]["references_selector"]["llm"]
    )
    result = node(state)

    print("-" * 20)
    print(state.keys())
    print(state[CANDIDATE_REFERENCES])
    print(state[SELECTED_REFERENCES])
    print("-" * 20)

    selected_refs = result.get(SELECTED_REFERENCES, [])
    assert selected_refs, "No references selected"

    candidate_urls = {ref["url"] for ref in candidate_refs}
    selected_urls = {ref["url"] for ref in selected_refs}

    assert selected_urls.issubset(candidate_urls), (
        f"Selected references must be a subset of candidate references.\n"
        f"Unexpected selections: {selected_urls - candidate_urls}"
    )


@pytest.mark.integration
def test_reviewer_node_integration_with_bad_inputs(example1_data, a3_config):
    """
    Test that the reviewer node flags bad content and requests revision.

    Asserts:
        - Reviewer returns NEEDS_REVISION = True.
        - All components (title, TL;DR, references) are disapproved.
        - REVISION_ROUND is incremented correctly.
    """
    input_text, _ = example1_data
    state = initialize_a3_state(config=a3_config, input_text=input_text)

    # Simulate weak inputs
    state[TITLE] = "Weak Title"
    state[TLDR] = "Weak TLDR content that does not summarize the article well."
    state[SELECTED_REFERENCES] = [
        {
            "title": "How to make pizza",
            "url": "https://example.com/pizza",
            "page_content": "Pepperoni pizza is a popular dish.",
        }
    ]

    node = make_reviewer_node(a3_config["agents"]["reviewer"]["llm"])
    result = node(state)

    assert result[NEEDS_REVISION] is True, "Expected reviewer to request revision"
    assert result[REVISION_ROUND] == 1
    assert result[TITLE_APPROVED] is False
    assert result[TLDR_APPROVED] is False
    assert result[REFERENCES_APPROVED] is False

    # Optional: print feedback for debugging
    print("🔍 Reviewer Feedback:")
    print("Title Approved:", result[TITLE_APPROVED])
    print("TLDR Approved:", result[TLDR_APPROVED])
    print("References Approved:", result[REFERENCES_APPROVED])
    print("Title Feedback:", result[TITLE_FEEDBACK])
    print("TLDR Feedback:", result[TLDR_FEEDBACK])
    print("References Feedback:", result[REFERENCES_FEEDBACK])
