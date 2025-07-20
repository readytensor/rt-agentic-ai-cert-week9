from langgraph.graph import StateGraph, START, END

from consts import (
    MANAGER,
    TLDR_GENERATOR,
    TITLE_GENERATOR,
    REFERENCES_GENERATOR,
    REFERENCES_SELECTOR,
    REVIEWER,
)

from consts import (
    MANAGER,
    TLDR_GENERATOR,
    TITLE_GENERATOR,
    REFERENCES_GENERATOR,
    REFERENCES_SELECTOR,
    REVIEWER,
)
from states.a3_state import A3SystemState, initialize_a3_state
from graphs.tag_generation_graph import add_tag_generation_flow
from nodes.a3_nodes import (
    make_manager_node,
    make_title_generator_node,
    make_tldr_generator_node,
    make_references_generator_node,
    make_references_selector_node,
    make_reviewer_node,
    route_from_reviewer,
)


def build_a3_graph(config: dict) -> StateGraph:
    """
    Creates and returns the agentic authoring graph with hierarchical structure and feedback loop.

    Args:
        config: Configuration dictionary containing agent settings and other parameters.

    Returns:
        StateGraph: The compiled state graph for the A3 system.
    """
    agents = config["agents"]
    graph = StateGraph(A3SystemState)

    # Add nodes
    manager_node = make_manager_node(llm_model=agents[MANAGER]["llm"])
    graph.add_node(MANAGER, manager_node)

    title_gen_node = make_title_generator_node(llm_model=agents[TITLE_GENERATOR]["llm"])
    graph.add_node(TITLE_GENERATOR, title_gen_node)

    tldr_gen_node = make_tldr_generator_node(llm_model=agents[TLDR_GENERATOR]["llm"])
    graph.add_node(TLDR_GENERATOR, tldr_gen_node)

    references_generator_node = make_references_generator_node(
        llm_model=agents[REFERENCES_GENERATOR]["llm"]
    )
    graph.add_node(REFERENCES_GENERATOR, references_generator_node)

    references_selector_node = make_references_selector_node(
        llm_model=agents[REFERENCES_SELECTOR]["llm"]
    )
    graph.add_node(REFERENCES_SELECTOR, references_selector_node)

    reviewer_node = make_reviewer_node(llm_model=agents[REVIEWER]["llm"])
    graph.add_node(REVIEWER, reviewer_node)

    # Add edges and flows
    graph.add_edge(START, MANAGER)
    graph.add_edge(MANAGER, TITLE_GENERATOR)
    graph.add_edge(MANAGER, TLDR_GENERATOR)
    graph.add_edge(MANAGER, REFERENCES_GENERATOR)
    graph.add_edge(REFERENCES_GENERATOR, REFERENCES_SELECTOR)

    # Add tag generation flow
    tag_gen_exit_node = add_tag_generation_flow(
        graph=graph,
        entry_node=MANAGER,
        tag_generation_config=config,
    )

    graph.add_edge(tag_gen_exit_node, END)
    graph.add_edge([TITLE_GENERATOR, TLDR_GENERATOR, REFERENCES_SELECTOR], REVIEWER)

    graph.add_conditional_edges(
        REVIEWER,
        route_from_reviewer,
        {
            TLDR_GENERATOR: TLDR_GENERATOR,
            TITLE_GENERATOR: TITLE_GENERATOR,
            REFERENCES_GENERATOR: REFERENCES_GENERATOR,
            "end": END,
        },
    )
    return graph


class A3System:
    def __init__(self, config):
        self.config = config
        self.state_template = initialize_a3_state(
            config=self.config,
            input_text=None,  # Will be overridden later
        )
        self.graph = build_a3_graph(self.config).compile()

    def process_article(self, text: str):
        # Merge template with new input
        runtime_state = {**self.state_template, "input_text": text}
        return self.graph.invoke(runtime_state)
