from typing import List, Optional, TypedDict, Dict, Union
from langchain_core.messages import SystemMessage
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import Annotated


from prompt_builder import build_system_prompt_message
from states.tag_generation_state import TagGenerationState, generate_tag_types_prompt

from consts import (
    MANAGER,
    LLM_TAGS_GENERATOR,
    TAG_TYPE_ASSIGNER,
    TAGS_SELECTOR,
    TLDR_GENERATOR,
    TITLE_GENERATOR,
    REFERENCES_GENERATOR,
    REFERENCES_SELECTOR,
    REVIEWER,
)


class A3SystemState(TypedDict, TagGenerationState):
    """State class for the A3 system."""

    input_text: Optional[Union[str, None]]

    manager_messages: Annotated[list[AnyMessage], add_messages]
    manager_brief: Optional[str]

    title_gen_messages: Annotated[list[AnyMessage], add_messages]
    tldr_gen_messages: Annotated[list[AnyMessage], add_messages]
    references_gen_messages: Annotated[list[AnyMessage], add_messages]
    references_selector_messages: Annotated[list[AnyMessage], add_messages]
    reviewer_messages: Annotated[list[AnyMessage], add_messages]

    tldr: Optional[str]
    title: Optional[str]
    reference_search_queries: Optional[str]
    candidate_references: Optional[List[Dict[str, str]]]
    selected_references: Optional[List[str]]

    # Revision information
    revision_round: Optional[int]
    needs_revision: Optional[bool]
    tldr_feedback: Optional[str]
    title_feedback: Optional[str]
    references_feedback: Optional[str]
    # Individual approval status for each component
    tldr_approved: Optional[bool]
    title_approved: Optional[bool]
    references_approved: Optional[bool]
    max_revisions: Optional[int]
    max_search_queries: Optional[int]
    max_references: Optional[int]


# def initialize_a3_state(a3_config: dict, input_text: str = None) -> A3SystemState:
#     """Initialize the A3 system state from config."""
#     # Extract agent configs
#     agents = a3_config["agents"]
#     # Build system messages
#     manager_messages = [
#         SystemMessage(build_system_prompt_message(agents[MANAGER]["prompt_config"])),
#     ]
#     title_gen_messages = [
#         SystemMessage(build_system_prompt_message(agents[TITLE_GENERATOR]["prompt_config"])),
#     ]
#     tldr_gen_messages = [
#         SystemMessage(build_system_prompt_message(agents[TLDR_GENERATOR]["prompt_config"])),
#     ]
#     # Tag-related messages
#     tag_types_prompt = generate_tag_types_prompt(a3_config["tag_types"])

#     llm_tags_gen_messages = [
#         SystemMessage(build_system_prompt_message(agents[LLM_TAGS_GENERATOR]["prompt_config"])),
#         SystemMessage(f"Here are the tag types you can assign:\n\n{tag_types_prompt}"),
#     ]
#     tag_type_assigner_messages = [
#         SystemMessage(build_system_prompt_message(agents[TAG_TYPE_ASSIGNER]["prompt_config"])),
#         SystemMessage(f"Here are the tag types you can assign:\n\n{tag_types_prompt}"),
#     ]
#     tags_selector_messages = [
#         SystemMessage(build_system_prompt_message(agents[TAGS_SELECTOR]["prompt_config"])),
#         SystemMessage(f"Please select at most {a3_config['max_tags']} tags from the generated list.\n"),
#     ]
#     references_gen_messages = [
#         SystemMessage(build_system_prompt_message(agents[REFERENCES_GENERATOR]["prompt_config"])),
#         SystemMessage(f"Please generate at most {a3_config['max_search_queries']} search queries from the generated list.\n"),
#     ]
#     references_selector_messages = [
#         SystemMessage(build_system_prompt_message(agents[REFERENCES_SELECTOR]["prompt_config"])),
#         SystemMessage(f"Please select at most {a3_config['max_references']} references from the given list of references.\n"),
#     ]
#     reviewer_messages = [
#         SystemMessage(build_system_prompt_message(agents[REVIEWER]["prompt_config"])),
#     ]
#     return A3SystemState(
#         input_text=input_text,
#         manager_brief=None,
#         manager_messages=manager_messages,
#         title_gen_messages=title_gen_messages,
#         llm_tags_gen_messages=llm_tags_gen_messages,
#         tag_type_assigner_messages=tag_type_assigner_messages,
#         tags_selector_messages=tags_selector_messages,
#         tldr_gen_messages=tldr_gen_messages,
#         references_gen_messages=references_gen_messages,
#         references_selector_messages=references_selector_messages,
#         reviewer_messages=reviewer_messages,
#         tldr=None,
#         title=None,
#         llm_tags=[],
#         spacy_tags=[],
#         gazetteer_tags=[],
#         candidate_tags=[],
#         selected_tags=[],
#         reference_search_queries=None,
#         candidate_references=[],
#         selected_references=[],
#         revision_round=0,
#         needs_revision=False,
#         tldr_feedback=None,
#         title_feedback=None,
#         references_feedback=None,
#         tldr_approved=False,
#         title_approved=False,
#         references_approved=False,
#         max_revisions=a3_config["max_revisions"],
#         max_tags=a3_config["max_tags"],
#         tag_types=a3_config["tag_types"],
#         max_search_queries=a3_config["max_search_queries"],
#         max_references=a3_config["max_references"],
#     )
