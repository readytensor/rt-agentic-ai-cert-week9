from langchain_core.messages import HumanMessage

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
    manager_brief = state.get(MANAGER_BRIEF, "No manager brief available.\n\n")
    if manager_brief is None or manager_brief.strip() == "":
        return HumanMessage("No manager brief available.")
    return HumanMessage(
        f"This is your manager's brief for your review:\n\n {manager_brief}\n\n"
    )


def _get_reviewer_message(state: A3SystemState, recipient_key) -> HumanMessage:
    """
    Returns the reviewer message from the state.
    """
    reviewer_feedback = state.get(recipient_key, "No feedback provided")
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