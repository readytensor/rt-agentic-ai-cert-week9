from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage

from consts import (
    MANAGER,
    LLM_TAGS_GENERATOR,
    TAG_TYPE_ASSIGNER,
    TAGS_SELECTOR,
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
from states.a3_state import A3SystemState, generate_tag_types_prompt
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
from prompt_builder import build_system_prompt_message


class A3System:
    def __init__(self, config):
        self.config = config
        # Create a "template" state with all config/static values
        self.state_template = self._build_state_template()
        self.graph = self._build_graph()

    def _build_state_template(self):
        """Build state with all config values, but no input_text"""
        # Extract agent configs
        agents = self.config["agents"]
        # Build system messages
        manager_messages = [
            SystemMessage(
                build_system_prompt_message(agents[MANAGER]["prompt_config"])
            ),
        ]
        title_gen_messages = [
            SystemMessage(
                build_system_prompt_message(agents[TITLE_GENERATOR]["prompt_config"])
            ),
        ]
        tldr_gen_messages = [
            SystemMessage(
                build_system_prompt_message(agents[TLDR_GENERATOR]["prompt_config"])
            ),
        ]
        # Tag-related messages
        tag_types_prompt = generate_tag_types_prompt(self.config["tag_types"])

        llm_tags_gen_messages = [
            SystemMessage(
                build_system_prompt_message(agents[LLM_TAGS_GENERATOR]["prompt_config"])
            ),
            SystemMessage(
                f"Here are the tag types you can assign:\n\n{tag_types_prompt}"
            ),
        ]
        tag_type_assigner_messages = [
            SystemMessage(
                build_system_prompt_message(agents[TAG_TYPE_ASSIGNER]["prompt_config"])
            ),
            SystemMessage(
                f"Here are the tag types you can assign:\n\n{tag_types_prompt}"
            ),
        ]
        tags_selector_messages = [
            SystemMessage(
                build_system_prompt_message(agents[TAGS_SELECTOR]["prompt_config"])
            ),
            SystemMessage(
                f"Please select at most {self.config['max_tags']} tags from the generated list.\n"
            ),
        ]
        references_gen_messages = [
            SystemMessage(
                build_system_prompt_message(
                    agents[REFERENCES_GENERATOR]["prompt_config"]
                )
            ),
            SystemMessage(
                f"Please generate at most {self.config['max_search_queries']} search queries from the generated list.\n"
            ),
        ]
        references_selector_messages = [
            SystemMessage(
                build_system_prompt_message(
                    agents[REFERENCES_SELECTOR]["prompt_config"]
                )
            ),
            SystemMessage(
                f"Please select at most {self.config['max_references']} references from the given list of references.\n"
            ),
        ]
        reviewer_messages = [
            SystemMessage(
                build_system_prompt_message(agents[REVIEWER]["prompt_config"])
            ),
        ]
        return A3SystemState(
            input_text=None,  # Will be overridden
            manager_brief=None,
            manager_messages=manager_messages,
            title_gen_messages=title_gen_messages,
            llm_tags_gen_messages=llm_tags_gen_messages,
            tag_type_assigner_messages=tag_type_assigner_messages,
            tags_selector_messages=tags_selector_messages,
            tldr_gen_messages=tldr_gen_messages,
            references_gen_messages=references_gen_messages,
            references_selector_messages=references_selector_messages,
            reviewer_messages=reviewer_messages,
            tldr=None,
            title=None,
            llm_tags=[],
            spacy_tags=[],
            gazetteer_tags=[],
            candidate_tags=[],
            selected_tags=[],
            reference_search_queries=None,
            candidate_references=[],
            selected_references=[],
            revision_round=0,
            needs_revision=False,
            tldr_feedback=None,
            title_feedback=None,
            references_feedback=None,
            tldr_approved=False,
            title_approved=False,
            references_approved=False,
            max_revisions=self.config["max_revisions"],
            max_tags=self.config["max_tags"],
            tag_types=self.config["tag_types"],
            max_search_queries=self.config["max_search_queries"],
            max_references=self.config["max_references"],
        )

    def _build_graph(self) -> StateGraph:
        """
        Creates and returns the agentic authoring graph with hierarchical structure and feedback loop.
        """
        graph = StateGraph(A3SystemState)

        # Add nodes
        manager_node = make_manager_node(
            llm_model=self.config["agents"][MANAGER]["llm"]
        )
        graph.add_node(MANAGER, manager_node)

        title_gen_node = make_title_generator_node(
            llm_model=self.config["agents"][TITLE_GENERATOR]["llm"]
        )
        graph.add_node(TITLE_GENERATOR, title_gen_node)

        tldr_gen_node = make_tldr_generator_node(
            llm_model=self.config["agents"][TLDR_GENERATOR]["llm"]
        )
        graph.add_node(TLDR_GENERATOR, tldr_gen_node)

        references_generator_node = make_references_generator_node(
            llm_model=self.config["agents"][REFERENCES_GENERATOR]["llm"]
        )
        graph.add_node(REFERENCES_GENERATOR, references_generator_node)

        references_selector_node = make_references_selector_node(
            llm_model=self.config["agents"][REFERENCES_SELECTOR]["llm"]
        )
        graph.add_node(REFERENCES_SELECTOR, references_selector_node)

        reviewer_node = make_reviewer_node(
            llm_model=self.config["agents"][REVIEWER]["llm"]
        )
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
            tag_generation_config=self.config,
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

        return graph.compile()

    def process_article(self, text: str):
        # Merge template with new input
        runtime_state = {**self.state_template, "input_text": text}
        return self.graph.invoke(runtime_state)
