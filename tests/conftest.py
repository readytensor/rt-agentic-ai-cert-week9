import pytest


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
