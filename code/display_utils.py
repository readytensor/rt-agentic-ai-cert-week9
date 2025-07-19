from pprint import pprint
from typing import Dict, Any

try:
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def print_a3_response(
    response: Dict[str, Any], title: str = "A3-SYSTEM DEMO", use_rich: bool = True
) -> Dict[str, Any]:
    """
    Pretty print A3 system response in a formatted manner.

    Args:
        response: The response dictionary from A3 system
        title: Optional title for the output (default: "A3-SYSTEM DEMO")
        use_rich: Whether to use Rich formatting if available (default: True)

    Returns:
        The same response dictionary (for chaining)
    """

    if use_rich and RICH_AVAILABLE:
        return _print_a3_response_rich(response, title)
    else:
        return _print_a3_response_plain(response, title)


def _print_a3_response_rich(response: Dict[str, Any], title: str) -> Dict[str, Any]:
    """Rich-formatted version of A3 response printer."""
    console = Console()

    # Main title panel
    console.print(Panel.fit(f"🔍 {title}", style="bold blue"))

    # Manager brief
    manager_brief = response.get("manager_brief", "(Not generated)")
    console.print(
        Panel(
            manager_brief,
            title="[bold green]📋 Manager Brief[/bold green]",
            border_style="green",
        )
    )

    # Title
    title_content = response.get("title", "(Not generated)")
    console.print(
        Panel(
            title_content,
            title="[bold yellow]📝 Title[/bold yellow]",
            border_style="yellow",
        )
    )

    # TL;DR
    tldr_content = response.get("tldr", "(Not generated)")
    console.print(
        Panel(
            tldr_content, title="[bold cyan]📄 TL;DR[/bold cyan]", border_style="cyan"
        )
    )

    # Tags
    selected_tags = response.get("selected_tags", [])
    if selected_tags:
        tags_text = "\n".join(
            f"• [bold]{tag.get('name', 'N/A')}[/bold] ([italic]{tag.get('type', 'N/A')}[/italic])"
            for tag in selected_tags
        )
        console.print(
            Panel(
                tags_text,
                title="[bold magenta]🔖 Tags[/bold magenta]",
                border_style="magenta",
            )
        )
    else:
        console.print(
            Panel(
                "(No tags generated)",
                title="[bold magenta]🔖 Tags[/bold magenta]",
                border_style="magenta",
            )
        )

    # Search queries
    queries = response.get("reference_search_queries", [])
    if queries:
        queries_text = "\n".join(f"• {query}" for query in queries)
        console.print(
            Panel(
                queries_text,
                title="[bold blue]🔍 Search Queries[/bold blue]",
                border_style="blue",
            )
        )
    else:
        console.print(
            Panel(
                "(No search queries generated)",
                title="[bold blue]🔍 Search Queries[/bold blue]",
                border_style="blue",
            )
        )

    # References
    selected_references = response.get("selected_references", [])
    refs_content = f"[bold]Count:[/bold] {len(selected_references)} references\n\n"

    if selected_references:
        for i, ref in enumerate(selected_references, 1):
            title_ref = ref.get("title", "N/A")
            url = ref.get("url", "N/A")
            refs_content += f"[bold]{i}.[/bold] {title_ref}\n"
            refs_content += f"   [dim]🔗 {url}[/dim]\n\n"
    else:
        refs_content += "[dim](No references selected)[/dim]"

    console.print(
        Panel(
            refs_content, title="[bold red]📚 References[/bold red]", border_style="red"
        )
    )

    return response


def _print_a3_response_plain(response: Dict[str, Any], title: str) -> Dict[str, Any]:
    """Plain text fallback version."""

    def print_section_header(text: str, char: str = "=", width: int = 80):
        """Print a section header with specified character and width."""
        print(char * width)
        print(f"🔍 {text}")
        print(char * width)

    def print_subsection(label: str, content: Any, width: int = 80):
        """Print a subsection with label and content."""
        print(f"{label}:")
        if content is None:
            print("(Not generated)")
        elif isinstance(content, (list, dict)):
            pprint(content)
        else:
            print(content)
        print("=" * width)

    # Main header
    print_section_header(title)

    # Manager brief
    print_subsection("Manager brief", response.get("manager_brief"))

    # Title
    print_subsection("Title", response.get("title"))

    # TL;DR
    print_subsection("TL;DR", response.get("tldr"))

    # Tags
    print_subsection("Tags", response.get("selected_tags"))

    # Search queries
    print_subsection("Search queries", response.get("reference_search_queries"))

    # References (special formatting)
    print("References:")
    selected_references = response.get("selected_references", [])
    print(f"Selected # of references: {len(selected_references)}")

    if selected_references:
        print("-" * 40)
        for ref in selected_references:
            print(f"Title: {ref.get('title', 'N/A')}")
            print(f"URL: {ref.get('url', 'N/A')}")
            print("-" * 40)
    else:
        print("(No references selected)")
        print("-" * 40)

    print("=" * 80)

    return response


def print_a3_response_compact(
    response: Dict[str, Any], use_rich: bool = True
) -> Dict[str, Any]:
    """
    Print A3 response in a more compact format.

    Args:
        response: The response dictionary from A3 system
        use_rich: Whether to use Rich formatting if available

    Returns:
        The same response dictionary (for chaining)
    """

    if use_rich and RICH_AVAILABLE:
        console = Console()
        console.print("\n[bold blue]🎯 A3 System Results:[/bold blue]")
        console.print("─" * 50)

        if response.get("title"):
            console.print(f"[green]📝 Title:[/green] {response['title']}")

        if response.get("tldr"):
            console.print(f"[cyan]📄 TL;DR:[/cyan] {response['tldr']}")

        selected_tags = response.get("selected_tags", [])
        if selected_tags:
            tag_names = [
                tag.get("name", "") for tag in selected_tags if tag.get("name")
            ]
            console.print(f"[magenta]🏷️  Tags:[/magenta] {', '.join(tag_names)}")

        selected_references = response.get("selected_references", [])
        if selected_references:
            console.print(
                f"[red]📚 References:[/red] {len(selected_references)} selected"
            )

        console.print("─" * 50)
    else:
        print("\n🎯 A3 System Results:")
        print("-" * 50)

        # Essential outputs only
        if response.get("title"):
            print(f"📝 Title: {response['title']}")

        if response.get("tldr"):
            print(f"📄 TL;DR: {response['tldr']}")

        selected_tags = response.get("selected_tags", [])
        if selected_tags:
            tag_names = [
                tag.get("name", "") for tag in selected_tags if tag.get("name")
            ]
            print(f"🏷️  Tags: {', '.join(tag_names)}")

        selected_references = response.get("selected_references", [])
        if selected_references:
            print(f"📚 References: {len(selected_references)} selected")

        print("-" * 50)

    return response


def print_a3_response_detailed(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Print A3 response with additional debug information.

    Args:
        response: The response dictionary from A3 system

    Returns:
        The same response dictionary (for chaining)
    """

    # Standard output first
    print_a3_response(response, "A3-SYSTEM DETAILED RESULTS")

    # Additional debug info
    print("🔧 DEBUG INFORMATION")
    print("=" * 80)

    # Revision information
    revision_round = response.get("revision_round", 0)
    needs_revision = response.get("needs_revision", False)
    print(f"Revision round: {revision_round}")
    print(f"Needs revision: {needs_revision}")

    # Component approval status
    print("\nComponent Approval Status:")
    print(f"  Title approved: {response.get('title_approved', False)}")
    print(f"  TL;DR approved: {response.get('tldr_approved', False)}")
    print(f"  References approved: {response.get('references_approved', False)}")

    # Feedback (if any)
    feedbacks = []
    if response.get("title_feedback"):
        feedbacks.append(f"Title: {response['title_feedback']}")
    if response.get("tldr_feedback"):
        feedbacks.append(f"TL;DR: {response['tldr_feedback']}")
    if response.get("references_feedback"):
        feedbacks.append(f"References: {response['references_feedback']}")

    if feedbacks:
        print("\nLatest Feedback:")
        for feedback in feedbacks:
            print(f"  {feedback}")

    # Tag generation details
    print("\nTag Generation Details:")
    print(f"  LLM tags: {len(response.get('llm_tags', []))}")
    print(f"  SpaCy tags: {len(response.get('spacy_tags', []))}")
    print(f"  Gazetteer tags: {len(response.get('gazetteer_tags', []))}")
    print(f"  Candidate tags: {len(response.get('candidate_tags', []))}")
    print(f"  Selected tags: {len(response.get('selected_tags', []))}")

    print("=" * 80)

    return response
