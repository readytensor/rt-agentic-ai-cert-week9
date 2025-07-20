from typing import Dict, List, TypedDict, Optional, Union
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import SystemMessage
from typing_extensions import Annotated

from prompt_builder import build_system_prompt_message
from consts import (
    LLM_TAGS_GENERATOR,
    TAG_TYPE_ASSIGNER,
    TAGS_SELECTOR,
)


class TagGenerationState(TypedDict):
    """State class for the tag extraction graph."""

    input_text: Optional[Union[str, None]]

    llm_tags_gen_messages: Annotated[list[AnyMessage], add_messages]
    tag_type_assigner_messages: Annotated[list[AnyMessage], add_messages]
    tags_selector_messages: Annotated[list[AnyMessage], add_messages]

    llm_tags: List[Dict[str, str]]
    spacy_tags: List[Dict[str, str]]
    gazetteer_tags: List[Dict[str, str]]
    candidate_tags: List[Dict[str, str]]
    selected_tags: List[Dict[str, str]]
    max_tags: int


def generate_tag_types_prompt(tag_types: List[Dict[str, str]]) -> str:
    """
    Generates a readable string version of tag types and their descriptions for LLM input.

    Args:
        tag_types: A list of dictionaries, each with 'name' and 'description' keys.

    Returns:
        A formatted string listing tag types with descriptions, suitable for use in prompts.
    """
    if not tag_types:
        return "No tag types provided."

    lines = []
    for tag in tag_types:
        name = tag.get("name", "").strip()
        description = tag.get("description", "").strip()
        if name and description:
            lines.append(f"- **{name}**: {description}")
    return "\n".join(lines)


def initialize_tag_generation_state(
    config: dict,
    input_text: str = None,
) -> TagGenerationState:
    """
    Initializes the state for the tag generation graph.

    Args:
        config: Tag generation configuration dictionary
        input_text: Optional input text for tag generation

    Returns:
        TagGenerationState with all values initialized
    """
    # Extract configuration values
    agents = config["agents"]
    tag_types = config["tag_types"]
    max_tags = config["max_tags"]

    # Build tag types prompt
    tag_types_prompt = generate_tag_types_prompt(tag_types)

    # Build system messages
    llm_tags_gen_messages = [
        SystemMessage(
            build_system_prompt_message(agents[LLM_TAGS_GENERATOR]["prompt_config"])
        ),
        SystemMessage(f"Here are the tag types you can assign:\n\n{tag_types_prompt}"),
    ]
    tag_type_assigner_messages = [
        SystemMessage(
            build_system_prompt_message(agents[TAG_TYPE_ASSIGNER]["prompt_config"])
        ),
        SystemMessage(f"Here are the tag types you can assign:\n\n{tag_types_prompt}"),
    ]
    tags_selector_messages = [
        SystemMessage(
            build_system_prompt_message(agents[TAGS_SELECTOR]["prompt_config"])
        ),
        SystemMessage(
            f"Please select at most {max_tags} tags from the generated list."
        ),
    ]

    return TagGenerationState(
        input_text=input_text,
        llm_tags_gen_messages=llm_tags_gen_messages,
        tag_type_assigner_messages=tag_type_assigner_messages,
        tags_selector_messages=tags_selector_messages,
        llm_tags=[],
        spacy_tags=[],
        gazetteer_tags=[],
        candidate_tags=[],
        selected_tags=[],
        max_tags=max_tags,
        tag_types=tag_types,
    )
