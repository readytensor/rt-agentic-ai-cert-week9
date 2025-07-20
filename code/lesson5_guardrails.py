import os
import warnings

warnings.filterwarnings("ignore")

os.environ["OTEL_SDK_DISABLED"] = "true"

from typing import Any, Dict
from pprint import pprint

from graphs.a3_graph import A3System
from utils import (
    load_publication_example,
    load_config,
    load_toxic_example,
    load_unusual_prompt_example,
)
from langgraph_utils import save_graph_visualization
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, UnusualPrompt


def run_a3_graph(text: str) -> Dict[str, Any]:
    """
    Runs the A3 agentic authoring graph with the provided LLM and configurations.
    """

    # Load configurations
    a3_config = load_config()["a3_system"]
    graph = A3System(a3_config)
    save_graph_visualization(graph.graph, graph_name="a3_system")
    # Run the graph
    final_state = graph.process_article(text)
    return final_state


def validate_input(input_text: str) -> bool:
    guard = Guard().use_many(
        ToxicLanguage(threshold=0.5, on_fail=OnFailAction.EXCEPTION),
        UnusualPrompt(on_fail=OnFailAction.EXCEPTION),
    )

    guard.validate(input_text)


if __name__ == "__main__":

    example = "toxic"

    if example == "toxic":
        sample_text = load_toxic_example()
    elif example == "unusual":
        sample_text = load_unusual_prompt_example()
    elif example == "normal":
        sample_text = load_publication_example(1)
    else:
        raise ValueError(
            f"Invalid example: {example} - must be 'toxic', 'unusual', or 'normal'"
        )

    validate_input(sample_text)

    response = run_a3_graph(sample_text)

    print("=" * 80)
    print("🔍 A3-SYSTEM DEMO")
    print("=" * 80)
    print("Manager brief:")
    print(response["manager_brief"])
    print("=" * 80)
    print("Title:")
    print(response["title"])
    print("=" * 80)
    print("TL;DR:")
    print(response["tldr"])
    print("=" * 80)
    print("Tags:")
    pprint(response["selected_tags"])
    print("=" * 80)
    print("Search queries:")
    pprint(response["reference_search_queries"])
    print("=" * 80)
    print("References:")
    print("Selected # of references:", len(response["selected_references"]))
    for ref in response["selected_references"]:
        print(f"Title: {ref['title']}")
        print(f"URL: {ref['url']}")
        print("-" * 40)
    print("=" * 80)
