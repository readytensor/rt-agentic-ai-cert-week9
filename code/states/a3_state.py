from typing import List, Optional, TypedDict, Dict, Union
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import Annotated


from states.tag_generation_state import TagGenerationState


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
