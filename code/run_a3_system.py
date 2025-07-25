from graphs.a3_graph import A3System
from utils import load_publication_example, load_config
from langgraph_utils import save_graph_visualization
from display_utils import print_a3_response
from save_utils import save_a3_response_to_markdown
from paths import OUTPUTS_DIR

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
    example_id = 4  # Change to 2 or 3 for larger examples
    sample_text = load_publication_example(example_id)  # ⚠️ CAUTION: SEE NOTE ABOVE

    a3_config = load_config()["a3_system"]

    a3_system = A3System(a3_config)
    save_graph_visualization(a3_system.graph, graph_name="a3_system")

    response = a3_system.process_article(sample_text)
    print_a3_response(response)

    # Save to files (new functionality)
    markdown_path = save_a3_response_to_markdown(
        response,
        output_dir=OUTPUTS_DIR,
        filename=f"a3_response_{example_id}.md",
        include_debug=True,  # Include debug info
        include_metadata=True,  # Include metadata
    )

    print(f"\n📁 Files saved:")
    print(f"  📝 Markdown: {markdown_path}")
