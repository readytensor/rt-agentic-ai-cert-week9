from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from typing import List, Dict

from states.a3_state import A3SystemState


from consts import (
    INPUT_TEXT,
    MANAGER_BRIEF,
)


def _get_input_text_message(state: A3SystemState) -> HumanMessage:
    """
    Returns the input text as a message from the state.
    """
    input_text = state.get(INPUT_TEXT, "No input text provided.")
    if input_text is None or input_text.strip() == "":
        raise ValueError("Input text cannot be empty or None.")
    return HumanMessage(f"Here's your input text:\n\n{input_text}\n\n")


def _get_manager_brief_message(state: A3SystemState) -> HumanMessage:
    """
    Returns the manager message from the state.
    """
    manager_brief = state.get(MANAGER_BRIEF, None)
    if manager_brief is None or manager_brief.strip() == "":
        return HumanMessage("No manager brief available.\n\n")
    return HumanMessage(
        f"This is your manager's brief for your review:\n\n {manager_brief}\n\n"
    )


def _get_reviewer_message(state: A3SystemState, recipient_key) -> HumanMessage:
    """
    Returns the reviewer message from the state.
    """
    reviewer_feedback = state.get(recipient_key, None)
    if reviewer_feedback is None or reviewer_feedback.strip() == "":
        return HumanMessage("No reviewer feedback available.\n\n")
    return HumanMessage(
        f"Following is the review from your reviewer:\n\n {reviewer_feedback}\n\n"
    )


def _get_begin_task_message() -> HumanMessage:
    """
    Returns a generic instructional cue message for the agent to begin its task.

    This message is intended to be appended at the end of a prompt sequence to signal
    that the setup is complete and the agent should proceed with generating a response.
    """
    return HumanMessage("Now perform your task.\n\n")


def format_references_for_prompt(references: list[dict[str, str]]) -> str:
    """Formats a list of references for inclusion in a prompt.

    Args:
        references (list[dict[str, str]]): A list of reference dictionaries, each containing
            'title', 'url', and optional 'page_content'.

    Returns:
        str: A formatted string representation of the references.
    """
    return "\n\n".join(
        f"- Title: {ref['title']}\n  URL: {ref['url']}\n  Content:\n{ref.get('page_content', '')[:5000]}"
        for ref in references
    )


def execute_search_queries(
    queries: List[str], max_results: int = 3
) -> List[Dict[str, str]]:
    """
    Execute a list of search queries and return aggregated results.

    Args:
        queries: List of search query strings
        max_results: Maximum results per query

    Returns:
        List of search results with url, title, and page_content
    """
    # Handle None input gracefully
    if queries is None:
        print("⚠️ Warning: No search queries provided (None input)")
        return []

    # Handle non-list input gracefully
    if not isinstance(queries, (list, tuple)):
        print(f"⚠️ Warning: Expected list of queries, got {type(queries).__name__}")
        return []

    # Filter out empty/whitespace queries upfront
    valid_queries = [query.strip() for query in queries if query and query.strip()]

    if not valid_queries:
        print("⚠️ Warning: No valid search queries provided (empty or whitespace-only)")
        return []

    print(f"🔍 Executing {len(valid_queries)} search queries...")
    search_results = []

    for query in valid_queries:
        print(f"🔍 Executing query: {query}")
        try:
            result = TavilySearch(max_results=max_results).invoke(query)["results"]
            search_results.extend(result)
            print(f"✅ Successfully executed query: {query}")
        except Exception as e:
            print(f"❌ Error executing query '{query}': {e}")
            continue

    # Filter and format results
    formatted_results = [
        {
            "url": search_result["url"],
            "title": search_result["title"],
            "page_content": search_result["content"],
        }
        for search_result in search_results
        if search_result.get("content")  # Ensure content is not empty
    ]

    return formatted_results
