from typing import Any, Dict
from pprint import pprint
from langgraph.graph import StateGraph

from graphs.a3_graph import A3System
# from states.a3_state import initialize_a3_state
from utils import load_publication_example, load_config
from langgraph_utils import save_graph_visualization
from display_utils import print_a3_response


# def get_initialized_a3_graph():
#     # Load configurations
#     a3_config = load_config()["a3_system"]

#     # Initialize state
#     initial_state = initialize_a3_state(a3_config=a3_config)

#     # Build the graph
#     graph = build_a3_graph(a3_config)
#     # Add initial state to the graph    
#     graph = graph.with_config(initial_state)
#     save_graph_visualization(graph, graph_name="a3_system")
#     return graph


# def run_a3_graph(graph: StateGraph, text: str) -> Dict[str, Any]:
#     """
#     Runs the A3 agentic authoring graph with the provided LLM and configurations.
#     """
#     response = graph.invoke({"input_text": text})

#     print("=" * 80)
#     print("🔍 A3-SYSTEM DEMO")
#     print("=" * 80)
#     print("Manager brief:")
#     print(response["manager_brief"])
#     print("=" * 80)
#     print("Title:")
#     print(response["title"])
#     print("=" * 80)
#     print("TL;DR:")
#     print(response["tldr"])
#     print("=" * 80)
#     print("Tags:")
#     pprint(response["selected_tags"])
#     print("=" * 80)
#     print("Search queries:")
#     pprint(response["reference_search_queries"])
#     print("=" * 80)
#     print("References:")
#     print("Selected # of references:", len(response["selected_references"]))
#     print("-" * 40)
#     for ref in response["selected_references"]:
#         print(f"Title: {ref['title']}")
#         print(f"URL: {ref['url']}")
#         print("-" * 40)
#     print("=" * 80)
#     return response


if __name__ == "__main__":

    # ⚠️⚠️⚠️ CAUTION: LONG + EXPENSIVE INPUTS ⚠️⚠️⚠️
    # -------------------------------------------------------------------------------
    # 🚨 Publication examples 2 and 3 are large and may take several minutes to process.
    # 💸 They will consume a lot of tokens — which could cost you a few cents.
    #
    # 👉 For faster and cheaper runs:
    #    - Use example 1
    #    - Or create shorter versions of examples 2 and 3
    #
    # ✅ This project is for learning — don’t burn through tokens unnecessarily.
    # -------------------------------------------------------------------------------

    # Example usage
    sample_text = load_publication_example(1)  # ⚠️ CAUTION: SEE NOTE ABOVE

    # a3_graph = get_initialized_a3_graph()
    # save_graph_visualization(a3_graph, graph_name="a3_system")

    # a3_response = run_a3_graph(a3_graph, sample_text)

    a3_config = load_config()["a3_system"]

    a3_system = A3System(a3_config)
    save_graph_visualization(a3_system.graph, graph_name="a3_system")

    response = a3_system.process_article(sample_text)
    print_a3_response(response)

    sample_text2 = """
    In this project, we built a churn prediction model using a gradient boosting classifier 
    trained on customer behavior and demographic data. The dataset included features such as 
    contract type, service usage, and support interactions. After preprocessing and feature 
    engineering, we evaluated model performance using precision, recall, and ROC-AUC, achieving 
    an AUC score of 0.89. The model helps identify at-risk customers early, enabling targeted 
    retention strategies. Code and datasets are included for reproducibility, and the pipeline 
    can be adapted for similar business problems across telecom and subscription-based services.
"""
    response2 = a3_system.process_article(sample_text2)
    print_a3_response(response2)
