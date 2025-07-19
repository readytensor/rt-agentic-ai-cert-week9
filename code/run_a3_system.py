
from graphs.a3_graph import A3System

from utils import load_publication_example, load_config
from langgraph_utils import save_graph_visualization
from display_utils import print_a3_response



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

    a3_config = load_config()["a3_system"]

    a3_system = A3System(a3_config)
    save_graph_visualization(a3_system.graph, graph_name="a3_system")

    response = a3_system.process_article(sample_text)
    print_a3_response(response)
