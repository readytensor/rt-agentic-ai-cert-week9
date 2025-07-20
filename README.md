# Ready Tensor Agentic AI Certification – Week 9

This repository contains lesson materials, code examples, and evaluation scripts for **Week 9** of the [Agentic AI Developer Certification Program](https://app.readytensor.ai/publications/HrJ0xWtLzLNt) by Ready Tensor. This week is about testing, safety and security in agentic AI systems.

**This repo is work in progress and will be updated with more content.**

---

## What You'll Learn

- Why traditional testing approaches fall short for dynamic, agentic AI systems
- Essential testing methodologies specifically designed for AI agents and multi-agent systems
- How to implement comprehensive unit testing for agentic components using **pytest**
- Critical security vulnerabilities in LLM applications and how to mitigate them using **OWASP Top 10 for LLMs**
- Safety and alignment testing techniques to identify and prevent ethical risks
- How to implement **guardrails** for runtime safety and output validation
- Automated bias and security scanning using **Giskard** for production-ready systems
- Real-world application of testing principles through a comprehensive case study of your Module 2 multi-agent system

---

## Lessons in This Repository

### 0. Agentic System Testing Preview

Get an overview of the unique testing challenges that agentic AI systems present and why traditional software testing methods need to be adapted.

### 1. Introduction to Testing

Foundation concepts for testing AI systems, including the differences between testing deterministic software and non-deterministic AI agents.

### 2a. Intro to Pytest

Master the pytest framework with hands-on examples tailored for AI applications, setting up the testing foundation for your agentic systems.

### 2b. Testing Agentic Systems

Learn how to test AI applications where outputs aren't always the same — especially those built with LLMs and agentic workflows.

### 3. App Security for AI: OWASP Top 10 for LLMs

Comprehensive walkthrough of the most critical security vulnerabilities in LLM applications, with practical mitigation strategies and code examples.

### 4. Safety & Alignment Testing – Mitigating Ethical Risk

Implement testing frameworks to identify potential ethical issues, bias, and harmful outputs in your agentic AI systems before deployment.

### 5. Guardrails: Add Runtime Safety & Output Validation

Build robust guardrail systems that monitor and control your AI agents in real-time, ensuring safe and appropriate behavior in production.

### 6. Giskard: Automated Scanning for Bias & Security

Hands-on tutorial with **Giskard** for automated detection of bias, security vulnerabilities, and performance issues in your AI systems.

### 7. Comprehensive Testing Case Study: Your Multi-Agent System

Apply everything you've learned by implementing a complete testing suite for your multi-agent system from Module 2, including unit tests, security assessments, safety validations, and guardrails integration.

---

## Repository Structure

```txt
rt-agentic-ai-cert-week9/
├── code/
│   ├── consts.py                             # Shared constant definitions
│   ├── display_utils.py                      # Formatting helpers for output visualization
│   ├── langgraph_utils.py                    # LangGraph graph visualizer
│   ├── lesson5_guardrails.py                 # Lesson 5: Runtime safety and validation via Guardrails
│   ├── lesson6_giskard.py                    # Lesson 6: Bias & security scanning with Giskard
│   ├── llm.py                                # LLM wrapper for OpenAI, Groq, etc.
│   ├── paths.py                              # Standardized file path management
│   ├── prompt_builder.py                     # Modular prompt construction helpers
│   ├── run_a3_system.py                      # Run the A3 multi-agent system (Module 2 case study)
│   ├── run_tag_gen_system.py                 # Run the tag generation system (single-agent pipeline)
│   ├── utils.py                              # General-purpose utilities
│   ├── graphs/
│   │   ├── a3_graph.py                       # LangGraph workflow for A3 multi-agent system
│   │   └── tag_generation_graph.py           # LangGraph workflow for the tag generation agent
│   ├── nodes/
│   │   ├── a3_nodes.py                       # All node definitions for A3 system (LLM + routing)
│   │   ├── node_utils.py                     # Reusable node prompt/message helpers
│   │   ├── output_types.py                   # LangChain structured output schemas
│   │   └── tag_generation_nodes.py           # All nodes for the tag generation system
│   ├── states/
│   │   ├── a3_state.py                       # State class + initializer for A3 system
│   │   └── tag_generation_state.py           # State class + initializer for tag generation system
├── config/                                   # Prompt configs, gazetteers, reasoning strategies
│   ├── config.yaml
│   ├── gazetteer_entities.yaml
│   └── reasoning.yaml
├── data/                                      # Input markdown files for publication examples
│   ├── publication_example1.md
│   └── ...
├── lessons/                                   # Lesson notebooks and supplementary content
├── outputs/                                   # Output logs, results, or final artifacts
├── tests/
│   ├── test_graphs/                           # Tests for LangGraph wiring and flows
│   ├── test_nodes/
│   │   ├── a3_system/                         # Unit tests for A3 system nodes
│   │   ├── tag_generation/                    # Unit tests for tag generation nodes
│   │   └── conftest.py                        # Shared fixtures for node tests
│   ├── test_states/                           # Tests for state initializers and logic
│   ├── test_utils/                            # Tests for prompt builder, node utils, etc.
│   └── conftest.py                            # Global pytest fixtures
├── .env.example                               # Example environment file (e.g., API keys)
├── .gitignore
├── LICENSE
├── README.md                                  # You are here
├── pytest.ini                                 # Pytest configuration (e.g., warning suppression)
└── requirements.txt                           # Python dependencies

```

---

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/readytensor/rt-agentic-ai-cert-week9.git
   cd rt-agentic-ai-cert-week9
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment variables:**

   Copy the `.env.example` to `.env` and fill in required values (e.g., OpenAI or Groq API keys):

   ```bash
   cp .env.example .env
   ```

---

## Running the Evaluation Examples

Each code example is runnable as a standalone script:

---

## Tests

This repo includes comprehensive unit tests for all major components, including nodes, state management, and utility functions.

```bash
pytest
```

Test organization:

```txt
tests/
├── test_graphs/                            # Tests for LangGraph-based workflows
├── test_nodes/
│   ├── a3_system/                          # Unit tests for A3 system nodes (e.g., manager, reviewer)
│   ├── tag_generation/                     # Unit tests for tag generation nodes (gazetteer, spaCy, LLM, etc.)
│   └── conftest.py                         # Node-level shared fixtures
├── test_states/                            # Tests for state initialization
├── test_utils/                             # Tests for utilities like prompt builders and tavily search function
└── conftest.py                             # Global fixtures
```

- To run a specific group of tests:

  ```bash
  pytest tests/test_nodes/tag_generation/
  ```

- To see detailed output with logs and print statements:

  ```bash
  pytest -s -v
  ```

All tests use `pytest` and follow best practices for mocking LLM calls when needed.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact

**Ready Tensor, Inc.**

- Email: contact at readytensor dot com
- Issues & Contributions: Open an issue or PR on this repo
- Website: [https://readytensor.ai](https://readytensor.ai)
