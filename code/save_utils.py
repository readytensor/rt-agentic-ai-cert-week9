import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


def save_a3_response_to_markdown(
    response: Dict[str, Any],
    output_dir: str = "outputs",
    filename: Optional[str] = None,
    include_debug: bool = False,
    include_metadata: bool = True,
) -> str:
    """
    Save A3 system response to a markdown file.

    Args:
        response: The response dictionary from A3 system
        output_dir: Directory to save the markdown file (default: "outputs")
        filename: Optional custom filename (default: auto-generated with timestamp)
        include_debug: Whether to include debug information
        include_metadata: Whether to include metadata section

    Returns:
        Path to the saved markdown file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"a3_output_{timestamp}.md"

    # Ensure .md extension
    if not filename.endswith(".md"):
        filename += ".md"

    filepath = os.path.join(output_dir, filename)

    # Generate markdown content
    markdown_content = _generate_a3_markdown(response, include_debug, include_metadata)

    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return filepath


def _generate_a3_markdown(
    response: Dict[str, Any], include_debug: bool, include_metadata: bool
) -> str:
    """Generate markdown content from A3 response."""
    lines = []

    # Header
    lines.append("# A3 System Output")
    lines.append("")

    if include_metadata:
        lines.extend(_generate_metadata_section(response))

    # Main content sections
    lines.extend(_generate_title_section(response))
    lines.extend(_generate_tldr_section(response))
    lines.extend(_generate_tags_section(response))
    lines.extend(_generate_references_section(response))

    if include_debug:
        lines.extend(_generate_debug_section(response))

    # Input text section (at the end)
    lines.extend(_generate_input_section(response))

    return "\n".join(lines)


def _generate_metadata_section(response: Dict[str, Any]) -> list[str]:
    """Generate metadata section."""
    lines = [
        "## 📋 Metadata",
        "",
        f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Revision Round**: {response.get('revision_round', 0)}",
        f"- **Manager Brief**: {len(response.get('manager_brief', '')) > 0}",
    ]

    # Component approval status
    title_approved = response.get("title_approved", False)
    tldr_approved = response.get("tldr_approved", False)
    references_approved = response.get("references_approved", False)

    lines.extend(
        [
            f"- **Title Approved**: {'✅' if title_approved else '❌'}",
            f"- **TL;DR Approved**: {'✅' if tldr_approved else '❌'}",
            f"- **References Approved**: {'✅' if references_approved else '❌'}",
            "",
        ]
    )

    return lines


def _generate_title_section(response: Dict[str, Any]) -> list[str]:
    """Generate title section."""
    title = response.get("title", "")
    if not title:
        return ["## 📝 Title", "", "*No title generated*", "", ""]

    return [
        "## 📝 Title",
        "",
        f"**{title}**",
        "",
    ]


def _generate_tldr_section(response: Dict[str, Any]) -> list[str]:
    """Generate TL;DR section."""
    tldr = response.get("tldr", "")
    if not tldr:
        return ["## 📄 TL;DR", "", "*No TL;DR generated*", "", ""]

    return [
        "## 📄 TL;DR",
        "",
        tldr,
        "",
    ]


def _generate_tags_section(response: Dict[str, Any]) -> list[str]:
    """Generate tags section."""
    selected_tags = response.get("selected_tags", [])

    lines = ["## 🏷️ Tags", ""]

    if not selected_tags:
        lines.extend(["*No tags generated*", ""])
        return lines

    # Group tags by type for better organization
    tags_by_type = {}
    for tag in selected_tags:
        tag_type = tag.get("type", "other")
        if tag_type not in tags_by_type:
            tags_by_type[tag_type] = []
        tags_by_type[tag_type].append(tag.get("name", "N/A"))

    for tag_type, tag_names in sorted(tags_by_type.items()):
        lines.append(f"**{tag_type.title()}**: {', '.join(tag_names)}")

    lines.append("")
    return lines


def _generate_references_section(response: Dict[str, Any]) -> list[str]:
    """Generate references section, including page content previews."""
    selected_references = response.get("selected_references", [])
    search_queries = response.get("reference_search_queries", [])

    lines = ["## 📚 References", ""]

    if search_queries:
        lines.extend(["### Search Queries", ""])
        for i, query in enumerate(search_queries, 1):
            lines.append(f"{i}. {query}")
        lines.append("")

    if not selected_references:
        lines.extend(["### Selected References", "", "*No references selected*", ""])
        return lines

    lines.extend(["### Selected References", ""])

    for i, ref in enumerate(selected_references, 1):
        title = ref.get("title", "N/A")
        url = ref.get("url", "N/A")
        content = ref.get("page_content", "").strip()
        preview = content[:300] + "..." if len(content) > 300 else content

        lines.append(f"{i}. **{title}**")
        lines.append(f"   - URL: {url}")
        if preview:
            lines.append(
                f"   - Page Preview:\n\n     > {preview.replace('\n', '\n     > ')}"
            )
        lines.append("")

    return lines


def _generate_debug_section(response: Dict[str, Any]) -> list[str]:
    """Generate debug information section."""
    lines = [
        "## 🔧 Debug Information",
        "",
        "### Workflow Status",
        "",
        f"- **Needs Revision**: {response.get('needs_revision', False)}",
        f"- **Revision Round**: {response.get('revision_round', 0)}",
        f"- **Max Revisions**: {response.get('max_revisions', 0)}",
        "",
    ]

    # Feedback information
    feedbacks = []
    if response.get("title_feedback"):
        feedbacks.append(f"**Title**: {response['title_feedback']}")
    if response.get("tldr_feedback"):
        feedbacks.append(f"**TL;DR**: {response['tldr_feedback']}")
    if response.get("references_feedback"):
        feedbacks.append(f"**References**: {response['references_feedback']}")

    if feedbacks:
        lines.extend(
            [
                "### Latest Feedback",
                "",
            ]
        )
        lines.extend(feedbacks)
        lines.append("")

    # Tag generation statistics
    lines.extend(
        [
            "### Tag Generation Statistics",
            "",
            f"- **LLM Tags**: {len(response.get('llm_tags', []))}",
            f"- **SpaCy Tags**: {len(response.get('spacy_tags', []))}",
            f"- **Gazetteer Tags**: {len(response.get('gazetteer_tags', []))}",
            f"- **Candidate Tags**: {len(response.get('candidate_tags', []))}",
            f"- **Selected Tags**: {len(response.get('selected_tags', []))}",
            "",
        ]
    )

    # Manager brief (if available)
    manager_brief = response.get("manager_brief", "")
    if manager_brief:
        lines.extend(
            [
                "### Manager Brief",
                "",
                f"> {manager_brief}",
                "",
            ]
        )

    return lines


def _generate_input_section(response: Dict[str, Any]) -> list[str]:
    """Generate input text section."""
    input_text = response.get("input_text", "")
    if not input_text:
        return ["## 📄 Input Text", "", "*No input text available*", ""]

    return [
        "## 📄 Input Text",
        "",
        "```",
        input_text,
        "```",
        "",
    ]


def save_tag_generation_response_to_markdown(
    response: Dict[str, Any],
    output_dir: str = "outputs",
    filename: Optional[str] = None,
) -> str:
    """
    Save tag generation response to a markdown file.

    Args:
        response: The response dictionary from tag generation system
        output_dir: Directory to save the markdown file
        filename: Optional custom filename

    Returns:
        Path to the saved markdown file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tag_generation_output_{timestamp}.md"

    if not filename.endswith(".md"):
        filename += ".md"

    filepath = os.path.join(output_dir, filename)

    # Generate markdown content for tag generation
    lines = [
        "# Tag Generation Output",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Input preview
    input_text = response.get("input_text", "")
    if input_text:
        preview = input_text[:200] + "..." if len(input_text) > 200 else input_text
        lines.extend(
            [
                "## 📄 Input Text Preview",
                "",
                f"> {preview}",
                "",
            ]
        )

    # Tag sections
    tag_sections = [
        ("llm_tags", "🤖 LLM Generated Tags"),
        ("spacy_tags", "🔍 SpaCy Extracted Tags"),
        ("gazetteer_tags", "📚 Gazetteer Found Tags"),
        ("candidate_tags", "🔄 All Candidate Tags"),
        ("selected_tags", "⭐ Final Selected Tags"),
    ]

    for tag_key, section_title in tag_sections:
        tags = response.get(tag_key, [])
        lines.extend(
            [
                f"## {section_title}",
                "",
            ]
        )

        if tags:
            for tag in tags:
                name = tag.get("name", "N/A")
                tag_type = tag.get("type", "N/A")
                lines.append(f"- **{name}** ({tag_type})")
        else:
            lines.append("*No tags found*")

        lines.append("")

    # Summary statistics
    llm_count = len(response.get("llm_tags", []))
    spacy_count = len(response.get("spacy_tags", []))
    gazetteer_count = len(response.get("gazetteer_tags", []))
    candidate_count = len(response.get("candidate_tags", []))
    selected_count = len(response.get("selected_tags", []))

    lines.extend(
        [
            "## 📊 Summary Statistics",
            "",
            f"- **LLM Generated**: {llm_count} tags",
            f"- **SpaCy Extracted**: {spacy_count} tags",
            f"- **Gazetteer Found**: {gazetteer_count} tags",
            f"- **Total Candidates**: {candidate_count} tags",
            f"- **Final Selected**: {selected_count} tags",
            "",
        ]
    )

    # Full input text at the end
    if input_text:
        lines.extend(
            [
                "## 📄 Full Input Text",
                "",
                "```",
                input_text,
                "```",
            ]
        )

    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath
