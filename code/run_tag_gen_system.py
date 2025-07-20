from utils import load_publication_example, load_config
from graphs.tag_generation_graph import TagGenerationSystem
from langgraph_utils import save_graph_visualization
from display_utils import print_tag_generation_response


if __name__ == "__main__":

    # ⚠️⚠️⚠️ CAUTION: MODERATE TOKEN USAGE ⚠️⚠️⚠️
    # -------------------------------------------------------------------------------
    # 🚨 Tag generation uses fewer tokens than full A3 system but still costs money.
    # 💸 LLM calls for tag generation and selection will consume tokens.
    #
    # 👉 For cheaper runs:
    #    - Use example 1 or 4 (shorter text)
    #    - Or test with your own short text samples
    #
    # ✅ This is for learning and testing tag extraction capabilities.
    # -------------------------------------------------------------------------------

    # Example usage
    sample_text = load_publication_example(1)  # ⚠️ CAUTION: SEE NOTE ABOVE

    # Load tag generation specific config
    config = load_config()
    tag_config = config[
        "a3_system"
    ]  # Tag generation uses same agents from a3_system config

    tag_system = TagGenerationSystem(tag_config)
    save_graph_visualization(tag_system.graph, graph_name="tag_generation_system")

    response = tag_system.extract_tags(sample_text)
    print_tag_generation_response(response)
