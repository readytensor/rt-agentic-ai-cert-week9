from typing import Any, Callable, Dict, Literal
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

from states.a3_state import A3SystemState
from llm import get_llm
from .node_utils import execute_search_queries

from consts import (
    INPUT_TEXT,
    MANAGER_MESSAGES,
    MANAGER_BRIEF,
    SELECTED_REFERENCES,
    TITLE_GEN_MESSAGES,
    TITLE,
    TLDR_GEN_MESSAGES,
    TLDR,
    TITLE_APPROVED,
    TLDR_APPROVED,
    REFERENCES_APPROVED,
    REFERENCE_SEARCH_QUERIES,
    REFERENCES_GEN_MESSAGES,
    REFERENCES_SELECTOR_MESSAGES,
    CANDIDATE_REFERENCES,
    SELECTED_REFERENCES,
    REVIEWER_MESSAGES,
    MAX_REVISIONS,
    REVISION_ROUND,
    NEEDS_REVISION,
    TLDR_APPROVED,
    TITLE_APPROVED,
    REFERENCES_APPROVED,
    TITLE_FEEDBACK,
    TLDR_FEEDBACK,
    REFERENCES_FEEDBACK,
    TITLE_GENERATOR,
    TLDR_GENERATOR,
    REFERENCES_GENERATOR,
)
from .output_types import SearchQueries, References, ReviewOutput
from .node_utils import (
    _get_input_text_message,
    _get_manager_brief_message,
    _get_reviewer_message,
    _get_begin_task_message,
    format_references_for_prompt,
)


def make_manager_node(llm_model: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that wraps a manager node.
    """
    llm = get_llm(llm_model)

    def manager_node(state: A3SystemState) -> Dict[str, Any]:
        """
        Manager node that processes the input text and generates messages.
        """
        # Prepare the input for the LLM
        input_text = state[INPUT_TEXT]
        if input_text is None or input_text.strip() == "":
            raise ValueError("Input text cannot be empty or None.")
        input_messages = [
            *state[MANAGER_MESSAGES],
            _get_input_text_message(state),
            _get_begin_task_message(),
        ]

        ai_response = llm.invoke(input_messages)
        if ai_response.content is None:
            print("⚠️ Manager: LLM returned None content, using empty string")
            return {MANAGER_BRIEF: ""}

        return {MANAGER_BRIEF: ai_response.content.strip()}

    return manager_node


def make_title_generator_node(
    llm_model: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that wraps a title generator node.
    """
    llm = get_llm(llm_model)

    def title_generator_node(state: A3SystemState) -> Dict[str, Any]:
        """
        Title generator node that uses prompt configuration, manager brief, reviewer feedback,
        and input text to generate a proposed publication title.
        """
        # Check if this component needs revision (skip if already approved)
        if state[TITLE_APPROVED] is True:
            print("🎯 Title Generator: Already approved, skipping...")
            return {}

        input_text = state[INPUT_TEXT]
        if input_text is None or input_text.strip() == "":
            raise ValueError("Input text cannot be empty or None.")

        print("🎯 Title Generator: Creating title...")
        input_messages = [
            *state[TITLE_GEN_MESSAGES],
            _get_manager_brief_message(state),
            _get_reviewer_message(state, TITLE_FEEDBACK),
            _get_input_text_message(state),
            _get_begin_task_message(),
        ]
        ai_response = llm.invoke(input_messages)

        content = ai_response.content or ""

        return {
            TITLE: content.strip(),
            TITLE_FEEDBACK: "",
        }

    return title_generator_node


def make_tldr_generator_node(
    llm_model: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that wraps a TL;DR generator.
    """
    llm = get_llm(llm_model)

    def tldr_generator_node(state: A3SystemState) -> Dict[str, Any]:
        """
        TL;DR generator node that processes the input text and generates a summary.
        """
        # Check if this component needs revision (skip if already approved)
        if state[TLDR_APPROVED] is True:
            print("📝 TL;DR Generator: Already approved, skipping...")
            return {}

        input_text = state[INPUT_TEXT]
        if input_text is None or input_text.strip() == "":
            raise ValueError("Input text cannot be empty or None.")

        print("🎯 TL;DR Generator: Creating TL;DR...")
        input_messages = [
            *state[TLDR_GEN_MESSAGES],
            _get_manager_brief_message(state),
            _get_reviewer_message(state, TLDR_FEEDBACK),
            _get_input_text_message(state),
            _get_begin_task_message(),
        ]
        ai_response = llm.invoke(input_messages)
        content = ai_response.content or ""

        return {
            TLDR: content.strip(),
            TLDR_FEEDBACK: "",
        }

    return tldr_generator_node


def make_references_generator_node(
    llm_model: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that wraps a references generator.
    """
    llm = get_llm(llm_model)

    def references_generator_node(state: A3SystemState) -> Dict[str, Any]:
        """
        References generator node that processes the input text and generates references.
        """
        # Check if this component needs revision (skip if already approved)
        if state.get(REFERENCES_APPROVED, False):
            print("📚 References Generator: Already approved, skipping...")
            return {}

        input_text = state[INPUT_TEXT]
        if input_text is None or input_text.strip() == "":
            raise ValueError("Input text cannot be empty or None.")

        print("📚 References Generator: Extracting references...")
        input_messages = [
            *state[REFERENCES_GEN_MESSAGES],
            _get_manager_brief_message(state),
            _get_reviewer_message(state, REFERENCES_FEEDBACK),
            _get_input_text_message(state),
            _get_begin_task_message(),
        ]
        try:
            queries = (
                llm.with_structured_output(SearchQueries).invoke(input_messages).queries
            )
            print(f"✅ Queries to be executed: {queries}")

            search_results = execute_search_queries(queries)
            print(f"✅ # Search results obtained: {len(search_results)}")
            return {
                REFERENCE_SEARCH_QUERIES: queries,
                CANDIDATE_REFERENCES: search_results,
            }
        except Exception as e:
            print(f"❌ References extraction failed: {e}")
            raise RuntimeError(
                "References extraction failed. Please check your LLM configuration or input text."
            ) from e

    return references_generator_node


def make_references_selector_node(
    llm_model: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that wraps a references selector.
    """
    llm = get_llm(llm_model)

    def references_selector_node(state: A3SystemState) -> Dict[str, Any]:
        """
        References selector node that processes the input text and selects references.
        """
        if state.get(REFERENCES_APPROVED, False):
            print("📚 References Selector: Already approved, skipping...")
            return {}

        input_text = state[INPUT_TEXT]
        if input_text is None or input_text.strip() == "":
            raise ValueError("Input text cannot be empty or None.")

        print("📚 References Selector: Selecting references...")
        candidate_references = state.get(CANDIDATE_REFERENCES, [])
        if not candidate_references:
            print("❌ No candidate references available to select from.")
            return {}
        formatted_references = format_references_for_prompt(candidate_references)
        candidate_refs_message = HumanMessage(
            f"Here are your candidate references to select from:\n\n{formatted_references}"
        )

        input_messages = [
            *state[REFERENCES_SELECTOR_MESSAGES],
            _get_manager_brief_message(state),
            _get_reviewer_message(state, REFERENCES_FEEDBACK),
            _get_input_text_message(state),
            candidate_refs_message,
            _get_begin_task_message(),
        ]
        returned_references = (
            llm.with_structured_output(References).invoke(input_messages).references
        )
        cleaned_references = []
        for ref in returned_references:
            try:
                # Check if all required attributes exist and are truthy
                if not ref.url or not ref.title or not ref.page_content:
                    print(f"⚠️ Skipping reference with missing/empty attributes")
                    continue

                cleaned_references.append(
                    {
                        "url": ref.url,
                        "title": ref.title,
                        "page_content": ref.page_content,
                    }
                )
            except AttributeError as e:
                print(f"⚠️ Skipping malformed reference due to missing attribute: {e}")
                continue
        return {
            SELECTED_REFERENCES: cleaned_references,
        }

    return references_selector_node


def make_reviewer_node(
    llm_model: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a LangGraph-compatible node that wraps a reviewer node.
    """
    llm = get_llm(llm_model)

    def reviewer_node(state: A3SystemState) -> Dict[str, Any]:
        """
        Reviewer node that processes the input text and generates feedback.
        """

        # Force approval if we've reached max revisions to prevent infinite loops
        revision_round = state.get(REVISION_ROUND, 0)
        max_revisions = state[MAX_REVISIONS]
        if revision_round >= max_revisions:
            overall_approved = True  # Force approve all remaining components
            print(
                "🔒 Reviewer: Maximum revisions reached, forcing approval for all components."
            )
            return {
                NEEDS_REVISION: False,
                TITLE_APPROVED: True,
                TLDR_APPROVED: True,
                REFERENCES_APPROVED: True,
            }

        input_text = state[INPUT_TEXT]
        if input_text is None or input_text.strip() == "":
            raise ValueError("Input text cannot be empty or None.")

        print("📝 Reviewer: Generating feedback...")
        title = state.get(TITLE, None)
        if title is None or title.strip() == "":
            title = "No title provided"
        tldr = state.get(TLDR, None)
        if tldr is None or tldr.strip() == "":
            tldr = "No TLDR provided"
        selected_references = state.get(SELECTED_REFERENCES, [])
        if not selected_references:
            selected_references = []
            formatted_references = "No references provided."
        else:
            formatted_references = format_references_for_prompt(selected_references)

        # Build comprehensive input data for review
        review_input = f"""
        # Title(s):\n {title} \n ------------- \n
        # TLDR(s):\n {tldr} \n ------------- \n
        # References:\n {formatted_references} \n ------------- \n
        """
        review_message = HumanMessage(
            f"Please review the following content and provide feedback:\n\n{review_input}\n\n"
            "If you have any specific feedback for the TL;DR, title, or references, please "
            "include it."
        )
        input_messages = [
            *state[REVIEWER_MESSAGES],
            _get_manager_brief_message(state),
            _get_input_text_message(state),
            review_message,
            _get_begin_task_message(),
        ]
        response = llm.with_structured_output(ReviewOutput).invoke(input_messages)
        revision_round += 1

        # Handle individual component approvals
        overall_approved = (
            response.title_approved
            and response.tldr_approved
            and response.references_approved
        )

        print(f"✅ Review completed: approved = {overall_approved}")
        print(f"📋 Feedback: {response.model_dump()}")

        # Show individual component status
        components_status = [
            f"TLDR: {'✅' if response.tldr_approved else '❌'}",
            f"Title: {'✅' if response.title_approved else '❌'}",
            f"References: {'✅' if response.references_approved else '❌'}",
        ]
        print(f"📊 Component Status: {' | '.join(components_status)}")

        if not overall_approved:
            needs_revision_list = []
            if not response.tldr_approved:
                needs_revision_list.append("TLDR")
            if not response.title_approved:
                needs_revision_list.append("Title")
            if not response.references_approved:
                needs_revision_list.append("References")

            print(f"🔄 Components needing revision: {', '.join(needs_revision_list)}")

            return {
                NEEDS_REVISION: True,
                REVISION_ROUND: revision_round,
                TLDR_FEEDBACK: response.tldr_feedback,
                TITLE_FEEDBACK: response.title_feedback,
                REFERENCES_FEEDBACK: response.references_feedback,
                TLDR_APPROVED: response.tldr_approved,
                TITLE_APPROVED: response.title_approved,
                REFERENCES_APPROVED: response.references_approved,
            }
        else:
            print("✅ All components approved - proceeding to final output")

            return {
                NEEDS_REVISION: False,
                REVISION_ROUND: revision_round,
                TLDR_FEEDBACK: response.tldr_feedback,
                TITLE_FEEDBACK: response.title_feedback,
                REFERENCES_FEEDBACK: response.references_feedback,
                TLDR_APPROVED: response.tldr_approved,
                TITLE_APPROVED: response.title_approved,
                REFERENCES_APPROVED: response.references_approved,
            }

    return reviewer_node


def route_from_reviewer(
    state: A3SystemState,
) -> Literal["revision_dispatcher", "end"]:
    """
    Determines whether any component requires revision.
    If so, returns the list of components to revise.
    Otherwise, routes to END.
    """
    needs_revision = state.get(NEEDS_REVISION, False)

    if not needs_revision:
        print("✅ All components approved - routing to END")
        return "end"
    else:
        print("🔄 Some components need revision - routing to revision dispatcher")
        return [TLDR_GENERATOR, TITLE_GENERATOR, REFERENCES_GENERATOR]
