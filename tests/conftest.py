import pytest


@pytest.fixture
def minimal_prompt_config():
    return {
        "instruction": "Summarize the text accurately.",
    }


@pytest.fixture
def full_prompt_config():
    return {
        "role": "an assistant",
        "instruction": "Summarize the text accurately.",
        "context": "The text is an excerpt from a research paper.",
        "output_constraints": ["Keep it under 100 words."],
        "style_or_tone": ["Clear and academic"],
        "output_format": "- bullet 1\n- bullet 2",
        "examples": ["Example summary 1", "Example summary 2"],
        "goal": "Produce a high-quality summary for busy professionals.",
        "reasoning_strategy": "cot",
    }


@pytest.fixture
def dummy_input_text():
    return (
        "This project explores how diffusion models can be applied to protein folding."
    )


@pytest.fixture
def dummy_state():
    return {"input_text": "Mock publication text."}


@pytest.fixture
def mock_llm(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            class FakeResponse:
                content = "Mocked LLM response"

                def model_dump(self):
                    return {"queries": ["mock query"], "references": []}

            return FakeResponse()

        def with_structured_output(self, schema=None):
            return self

    monkeypatch.setattr("llm.get_llm", lambda *_: FakeLLM())
