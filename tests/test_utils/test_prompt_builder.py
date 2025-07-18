import pytest

from prompt_builder import (
    build_prompt_body,
    build_one_shot_prompt,
    build_system_prompt_message,
)


def test_build_prompt_body_raises_if_instruction_missing():
    config = {"role": "test role"}  # no instruction
    with pytest.raises(ValueError, match="Missing required field: 'instruction'"):
        build_prompt_body(config)


def test_build_prompt_body_with_minimal_config(minimal_prompt_config):
    result = build_prompt_body(minimal_prompt_config)
    assert "Your task is as follows:" in result
    assert "Summarize the text accurately." in result
    assert result.strip().endswith("Now perform the task as instructed above.")


def test_build_prompt_body_with_all_fields(full_prompt_config):
    result = build_prompt_body(full_prompt_config, input_data="Some input text")

    # Check role is incorporated
    assert "You are an assistant." in result

    # Check instruction is rendered
    assert "Your task is as follows:" in result
    assert "Summarize the text accurately." in result

    # Check output constraints are present
    assert "Keep it under 100 words." in result

    # Check tone
    assert "Clear and academic" in result

    # Check examples
    assert "Example summary 1" in result
    assert "Example summary 2" in result

    # Check goal
    assert "Produce a high-quality summary" in result

    # Check input data formatting
    assert "<<<BEGIN CONTENT>>>" in result
    assert "Some input text" in result
    assert "<<<END CONTENT>>>" in result

    # Check reasoning strategy included
    assert (
        "Break down the problem into smaller steps" in result
        or "reason" in result.lower()
    )

    # Check final instruction
    assert "Now perform the task as instructed above." in result


def test_build_prompt_body_without_final_instruction(minimal_prompt_config):
    result = build_prompt_body(minimal_prompt_config, finalize=False)
    assert not result.strip().endswith("Now perform the task as instructed above.")


def test_build_prompt_body_raises_on_invalid_reasoning_strategy():
    # Given
    prompt_config = {
        "role": "a test role",
        "instruction": "Do something",
        "reasoning_strategy": "chain-of-thought",  # ❌ invalid key
    }

    # When / Then
    with pytest.raises(ValueError) as exc_info:
        build_prompt_body(prompt_config)

    # Validate the error message is helpful
    assert "Unknown reasoning strategy" in str(exc_info.value)
    assert "Expected one of:" in str(exc_info.value)
    assert "cot" in str(exc_info.value)  # assuming lowercase is correct key


def test_build_prompt_body_skips_input_data_if_empty(full_prompt_config):
    result = build_prompt_body(full_prompt_config, input_data="")
    assert "<<<BEGIN CONTENT>>>" not in result


def test_build_prompt_body_handles_string_instead_of_list():
    config = {
        "role": "a summarizer",
        "instruction": "Do the thing.",
        "output_constraints": "Use no more than 3 sentences.",
    }
    result = build_prompt_body(config)
    assert "Use no more than 3 sentences." in result


def test_build_prompt_body_raises_without_instruction():
    config = {
        "role": "something",
        # Missing "instruction"
    }
    with pytest.raises(ValueError, match="Missing required field: 'instruction'"):
        build_prompt_body(config)


def test_build_one_shot_prompt_appends_final_instruction(minimal_prompt_config):
    result = build_one_shot_prompt(minimal_prompt_config, input_data="Test input")

    # Core assertions
    assert "Your task is as follows:" in result
    assert "Summarize the text accurately." in result
    assert "Test input" in result
    assert "Now perform the task as instructed above." in result


def test_build_system_prompt_message_excludes_final_instruction(minimal_prompt_config):
    result = build_system_prompt_message(minimal_prompt_config)

    # Basic content check
    assert "Your task is as follows:" in result
    assert "Summarize the text accurately." in result

    # Final instruction should NOT be included
    assert "Now perform the task as instructed above." not in result


def test_build_prompt_body_without_input_data(full_prompt_config):
    result = build_prompt_body(full_prompt_config, input_data="")
    assert "<<<BEGIN CONTENT>>>" not in result


def test_build_prompt_body_with_unknown_field_ignored(minimal_prompt_config):
    minimal_prompt_config["unknown_field"] = "some junk"
    result = build_prompt_body(minimal_prompt_config)
    assert "some junk" not in result  # ignored
    assert "Summarize the text accurately." in result


def test_list_sections_render_with_bullets():
    config = {
        "instruction": "Do something",
        "output_constraints": ["Limit to 2 paragraphs", "No repetition"],
    }
    result = build_prompt_body(config)
    assert "- Limit to 2 paragraphs" in result
    assert "- No repetition" in result
