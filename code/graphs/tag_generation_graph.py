from langgraph.graph import StateGraph, START, END
from typing import Any, Dict, List

from consts import (
    LLM_TAGS_GENERATOR,
    SPACY_TAGS_GENERATOR,
    GAZETTEER_TAGS_GENERATOR,
    TAG_TYPE_ASSIGNER,
    TAGS_AGGREGATOR,
    TAGS_SELECTOR,
)
from nodes.tag_generation_nodes import (
    make_llm_tag_generator_node,
    make_spacy_tag_generator_node,
    make_gazetteer_tag_generator_node,
    make_tag_type_assigner_node,
    aggregate_tags_node,
    make_tag_selector_node,
)
from states.tag_generation_state import (
    TagGenerationState,
    initialize_tag_generation_state,
)


def build_tag_generation_graph(tag_generation_config: Dict[str, Any]) -> StateGraph:
    graph = StateGraph(TagGenerationState)

    # Insert tag generation flow
    final_node = add_tag_generation_flow(
        graph=graph,
        entry_node=START,
        tag_generation_config=tag_generation_config,
    )

    graph.add_edge(final_node, END)
    return graph.compile()


def add_tag_generation_flow(
    graph: StateGraph,
    entry_node: str,
    tag_generation_config: Dict[str, Any],
) -> str:
    """
    Adds the tag generation flow into an existing LangGraph.

    Args:
        graph: The existing StateGraph to inject into.
        entry_node: The node in the existing graph after which the tag flow begins.
        tag_generation_config: Configuration dictionary for tag generation.

    Returns:
        str: The name of the final node in this subgraph (typically TAGS_SELECTOR).
    """

    agents = tag_generation_config["agents"]
    max_tags = tag_generation_config["max_tags"]

    # Create nodes
    llm_tags_generator_node = make_llm_tag_generator_node(
        llm_model=agents[LLM_TAGS_GENERATOR]["llm"]
    )
    graph.add_node(LLM_TAGS_GENERATOR, llm_tags_generator_node)

    spacy_tag_generator_node = make_spacy_tag_generator_node()
    graph.add_node(SPACY_TAGS_GENERATOR, spacy_tag_generator_node)

    tag_type_assigner_node = make_tag_type_assigner_node(
        llm_model=agents[TAG_TYPE_ASSIGNER]["llm"]
    )
    graph.add_node(TAG_TYPE_ASSIGNER, tag_type_assigner_node)

    gazetteer_tag_generator_node = make_gazetteer_tag_generator_node()
    graph.add_node(GAZETTEER_TAGS_GENERATOR, gazetteer_tag_generator_node)

    graph.add_node(TAGS_AGGREGATOR, aggregate_tags_node)

    tags_selector_node = make_tag_selector_node(
        llm_model=agents[TAGS_SELECTOR]["llm"],
        max_tags=max_tags,
    )
    graph.add_node(TAGS_SELECTOR, tags_selector_node)

    # Wire the subgraph
    graph.add_edge(entry_node, LLM_TAGS_GENERATOR)
    graph.add_edge(entry_node, SPACY_TAGS_GENERATOR)
    graph.add_edge(SPACY_TAGS_GENERATOR, TAG_TYPE_ASSIGNER)
    graph.add_edge(entry_node, GAZETTEER_TAGS_GENERATOR)

    graph.add_edge(
        [LLM_TAGS_GENERATOR, TAG_TYPE_ASSIGNER, GAZETTEER_TAGS_GENERATOR],
        TAGS_AGGREGATOR,
    )

    graph.add_edge(TAGS_AGGREGATOR, TAGS_SELECTOR)

    return TAGS_SELECTOR


class TagGenerationSystem:
    def __init__(self, config):
        self.config = config
        self.state_template = initialize_tag_generation_state(
            config=self.config,
            input_text=None,  # Will be overridden later
        )
        self.graph = build_tag_generation_graph(self.config)

    def extract_tags(self, text: str):
        # Merge template with new input
        runtime_state = {**self.state_template, "input_text": text}
        return self.graph.invoke(runtime_state)
