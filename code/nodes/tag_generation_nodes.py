from typing import Any, Callable, Dict
import spacy
import re
from typing import Dict
from langchain_core.messages import HumanMessage

from states.tag_generation_state import TagGenerationState
from llm import get_llm

from consts import (
    LLM_TAGS_GEN_MESSAGES,
    TAG_TYPE_ASSIGNER_MESSAGES,
    TAGS_SELECTOR_MESSAGES,
    INPUT_TEXT,
    LLM_TAGS,
    SPACY_TAGS,
    GAZETTEER_TAGS,
    CANDIDATE_TAGS,
    SELECTED_TAGS,
)
from paths import GAZETTEER_ENTITIES_FILE_PATH
from utils import load_config
from .output_types import Entities
from .node_utils import (
    _get_input_text_message,
    _get_manager_brief_message,
    _get_begin_task_message,
)

EXCLUDED_SPACY_ENTITY_TYPES = {"DATE", "CARDINAL"}


def make_llm_tag_generator_node(
    llm_model: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that extracts tags from the input text.
    """
    llm = get_llm(llm_model)

    def llm_tag_generator_node(state: TagGenerationState) -> Dict[str, Any]:
        """
        Extracts tags from the input text using the LLM.
        """
        # Prepare the input for the LLM
        input_text = state[INPUT_TEXT]
        if input_text is None or input_text.strip() == "":
            raise ValueError("Input text cannot be empty or None.")
        input_messages = [
            *state[LLM_TAGS_GEN_MESSAGES],
            _get_manager_brief_message(state),
            _get_input_text_message(state),
            _get_begin_task_message(),
        ]
        tags = (
            llm.with_structured_output(Entities)
            .invoke(input_messages)
            .model_dump()["entities"]
        )

        cleaned_tags = []
        for tag in tags:
            if not tag.get("name"):  # name is required
                continue
            name = tag["name"].lower().strip()
            if not name:
                continue
            tag_type = tag.get("type", "").lower().strip()
            cleaned_tags.append({"name": name, "type": tag_type})

        return {LLM_TAGS: cleaned_tags}

    return llm_tag_generator_node


def make_spacy_tag_generator_node() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that extracts tags using a pre-loaded spaCy model.
    """
    model = spacy.load("en_core_web_trf")

    def spacy_tag_generator_node(state: TagGenerationState) -> Dict[str, Any]:
        """
        Extracts unique named entities from the input text using spaCy.
        """
        input_text = state.get(INPUT_TEXT, "")

        # Handle edge cases gracefully
        if not input_text or not input_text.strip():
            return {SPACY_TAGS: []}

        doc = model(input_text)
        seen = set()
        entities = []
        for ent in doc.ents:
            if ent.label_ in EXCLUDED_SPACY_ENTITY_TYPES:
                continue
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append(
                    {
                        "name": ent.text.lower().strip(),
                        "type": ent.label_.strip(),
                    }
                )
        return {SPACY_TAGS: entities}

    return spacy_tag_generator_node


def make_gazetteer_tag_generator_node() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that extracts tags using a predefined gazetteer.
    """
    gazetteer = load_config(GAZETTEER_ENTITIES_FILE_PATH)

    def gazetteer_tag_generator_node(state: TagGenerationState) -> Dict[str, Any]:
        """
        Extracts unique entities from the input text using a regex-based gazetteer.
        """
        text = state.get(INPUT_TEXT, "")
        if not text:
            return {GAZETTEER_TAGS: []}

        seen = set()
        entities = []

        for entity_name, entity_type in gazetteer.items():
            pattern = r"\b" + re.escape(entity_name) + r"\b"
            try:
                for _ in re.finditer(pattern, text, re.IGNORECASE):
                    key = (entity_name.lower(), entity_type)
                    if key not in seen:
                        seen.add(key)
                        entities.append(
                            {
                                "name": entity_name.lower().strip(),
                                "type": entity_type.strip(),
                            }
                        )
            except re.error as e:
                print(f"Regex error for entity '{entity_name}': {e}")
                continue

        return {GAZETTEER_TAGS: entities}

    return gazetteer_tag_generator_node


def make_tag_type_assigner_node(
    llm_model: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that assigns tag types to extracted tags.
    """
    llm = get_llm(llm_model)

    def tag_type_assigner_node(state: TagGenerationState) -> Dict[str, Any]:
        """
        Assigns tag types to extracted tags using the LLM.
        - Skips tags with missing names.
        - Returns early if there are no valid tags to process.
        """
        raw_tags = state.get(SPACY_TAGS, [])
        clean_names = [
            tag["name"].strip()
            for tag in raw_tags
            if tag.get("name")
            and tag["name"].strip()  # Also check if non-empty after strip
        ]

        # No valid input? Return early
        if not clean_names:
            return {SPACY_TAGS: []}

        messages = list(state[TAG_TYPE_ASSIGNER_MESSAGES])  # Avoid mutating original
        messages.append(
            HumanMessage(
                content=f"Assign tag types to the following tags:\n {', '.join(clean_names)}"
            )
        )

        updated_tags = (
            llm.with_structured_output(Entities)
            .invoke(messages)
            .model_dump()["entities"]
        )

        cleaned_output = []
        for tag in updated_tags:
            if not tag.get("name"):
                continue  # Skip tags without a name
            cleaned_output.append(
                {
                    "name": tag["name"].strip().lower(),
                    "type": tag.get("type", "").strip().lower(),
                }
            )

        return {SPACY_TAGS: cleaned_output}

    return tag_type_assigner_node


def aggregate_tags_node(state: TagGenerationState) -> Dict[str, Any]:
    """
    Aggregates tags from LLM, spaCy, and Gazetteer into a single deduplicated list.

    Tags are considered duplicates if they have the same lowercase name and type.
    """
    all_tags = (
        state.get(LLM_TAGS, [])
        + state.get(SPACY_TAGS, [])
        + state.get(GAZETTEER_TAGS, [])
    )

    seen = set()
    deduped = []
    for tag in all_tags:
        # Handle None values by converting to string first
        raw_name = tag.get("name")
        raw_type = tag.get("type")

        # Skip if either name or type is None/empty
        if not raw_name or not raw_type:
            continue

        name = str(raw_name).lower().strip()
        tag_type = str(raw_type).lower().strip()

        # Skip if empty after conversion and stripping
        if not name or not tag_type:
            continue

        key = (name, tag_type)
        if key not in seen:
            seen.add(key)
            deduped.append({"name": name, "type": tag_type})

    return {CANDIDATE_TAGS: deduped}


def make_tag_selector_node(
    llm_model: str, max_tags: int
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that selects the most relevant tags using an LLM.

    Args:
        llm_model: The LLM to use for tag selection.
        max_tags: Maximum number of tags allowed in the final selection.

    Returns:
        A function that selects tags and updates the SELECTED_TAGS key in the state.
    """
    llm = get_llm(llm_model)

    def tag_selector_node(state: TagGenerationState) -> Dict[str, Any]:
        """
        Uses the LLM to select the most important tags from the candidate list.
        """
        candidate_tags = state.get(CANDIDATE_TAGS, [])
        base_messages = state.get(TAGS_SELECTOR_MESSAGES, [])

        selection_instruction = HumanMessage(
            content=(
                f"Here is the list of candidate tags (name and type):\n{candidate_tags}\n\n"
                f"Please return a refined list of the most important tags (maximum {max_tags})."
            )
        )
        full_prompt = base_messages + [selection_instruction]

        response = llm.with_structured_output(Entities).invoke(full_prompt).model_dump()

        tags = response.get("entities", [])
        cleaned_tags = []
        for tag in tags:
            name = tag.get("name")
            if not name:
                continue  # skip tag if name is missing or empty

            # Fix: Handle None type values
            tag_type = tag.get("type", "") or ""  # Convert None to empty string

            cleaned_tags.append(
                {
                    "name": str(name)
                    .lower()
                    .strip(),  # Also defensive: convert to string
                    "type": str(tag_type).lower().strip(),
                }
            )

        # Truncate if LLM returned too many
        cleaned_tags = cleaned_tags[:max_tags]
        return {SELECTED_TAGS: cleaned_tags}

    return tag_selector_node
