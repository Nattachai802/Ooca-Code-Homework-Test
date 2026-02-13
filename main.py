from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from agent import AgentResponse, TriageAgent

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
console = Console()


def load_sample_tickets() -> list[dict]:
    tickets_path = DATA_DIR / "sample_tickets.json"
    with open(tickets_path, encoding="utf-8") as f:
        return json.load(f)


def print_tool_traces(response: AgentResponse) -> None:
    """Display the agent's reasoning: what tools were called and what they returned."""
    console.print()
    console.print(
        Panel(
            f"[bold]Agent completed in {response.rounds} round(s), "
            f"called {len(response.tool_traces)} tool(s)[/bold]",
            title="Agent Reasoning Trace",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    for i, trace in enumerate(response.tool_traces, 1):
        # Tool call header
        console.print(
            f"\n  [bold cyan]Tool Call #{i}:[/bold cyan] "
            f"[yellow]{trace.tool_name}[/yellow]"
            f"({json.dumps(trace.arguments, ensure_ascii=False)})"
        )

        # Tool result — show key info only
        if trace.tool_name == "fetch_customer_data":
            data = trace.result
            if "error" in data:
                console.print(f"    [red]Error: {data['error']}[/red]")
            else:
                details = data.get("plan_details", {})
                info_table = Table(
                    show_header=False, box=box.SIMPLE, padding=(0, 2),
                    title="Customer Data Retrieved",
                    title_style="bold green",
                )
                info_table.add_column("Field", style="dim")
                info_table.add_column("Value")
                info_table.add_row("Name", data.get("name", "N/A"))
                info_table.add_row("Email", data.get("email", "N/A"))
                info_table.add_row("Plan", f"{details.get('label', 'N/A')} (priority: {details.get('priority', 'N/A')})")
                info_table.add_row("Region", data.get("region", "N/A"))
                info_table.add_row("Seats", str(data.get("seats", "N/A")))
                info_table.add_row("Tenure", f"{data.get('tenure_months', 'N/A')} months")
                info_table.add_row("SLA", f"{details.get('sla_hours', 'N/A')} hours")
                info_table.add_row("Auto-escalate", str(details.get("auto_escalate", False)))
                info_table.add_row("Previous Tickets", str(data.get("previous_tickets", 0)))
                console.print(info_table)

        elif trace.tool_name == "query_knowledge_base":
            results = trace.result
            if isinstance(results, dict) and "error" in results:
                console.print(f"    [red]Error: {results['error']}[/red]")
            elif not results or not isinstance(results, list):
                console.print("    [dim]No KB articles found.[/dim]")
            else:
                kb_table = Table(
                    title="KB Articles Found",
                    title_style="bold green",
                    box=box.SIMPLE,
                    padding=(0, 1),
                )
                kb_table.add_column("ID", style="cyan", width=8)
                kb_table.add_column("Topic", style="white")
                kb_table.add_column("Action", style="yellow")
                kb_table.add_column("Score", style="green", width=6)
                for article in results:
                    if not isinstance(article, dict):
                        continue
                    guideline = article.get("guideline", {})
                    score = article.get("relevance_score")
                    score_str = f"{score:.2f}" if score is not None else "N/A"
                    kb_table.add_row(
                        article.get("id", ""),
                        article.get("topic", ""),
                        guideline.get("action", "") if isinstance(guideline, dict) else str(guideline),
                        score_str,
                    )
                console.print(kb_table)


def print_result(response: AgentResponse) -> None:
    """Pretty-print the triage result using rich."""
    result = response.result

    # --- Header ---
    console.print()
    console.rule(f"[bold white] TRIAGE RESULT — {result.ticket_id} [/bold white]", style="bold blue")

    # --- Tool Traces (Agent Reasoning) ---
    print_tool_traces(response)

    # --- Analysis ---
    a = result.analysis
    urgency_colors = {"critical": "red bold", "high": "yellow bold", "medium": "cyan", "low": "green"}
    sentiment_colors = {"angry": "red", "frustrated": "yellow", "neutral": "white", "positive": "green"}

    analysis_table = Table(
        show_header=False, box=box.ROUNDED, border_style="blue",
        title="Analysis", title_style="bold white",
        padding=(0, 2),
    )
    analysis_table.add_column("Field", style="dim", width=14)
    analysis_table.add_column("Value")

    urgency_text = Text(a.urgency.upper(), style=urgency_colors.get(a.urgency, "white"))
    sentiment_text = Text(a.sentiment, style=sentiment_colors.get(a.sentiment, "white"))

    analysis_table.add_row("Urgency", urgency_text)
    analysis_table.add_row("Sentiment", sentiment_text)
    analysis_table.add_row("Issue Type", a.issue_type)
    analysis_table.add_row("Product Area", a.product_area)
    analysis_table.add_row("Language", a.language)
    analysis_table.add_row("Summary", a.summary)

    console.print()
    console.print(analysis_table)

    # --- Action ---
    act = result.action
    action_styles = {
        "auto_respond": ("green", "AUTO-RESPOND"),
        "route_to_specialist": ("yellow", "ROUTE TO SPECIALIST"),
        "escalate_to_human": ("red bold", "ESCALATE TO HUMAN"),
    }
    style, label = action_styles.get(act.action, ("white", act.action.upper()))

    action_content = f"[{style}]{label}[/{style}]\n"
    action_content += f"[dim]Reason:[/dim] {act.reason}\n"
    action_content += f"[dim]Priority:[/dim] [bold]{act.priority_score}/10[/bold]"

    if act.routing_department:
        action_content += f"\n[dim]Route To:[/dim] [yellow]{act.routing_department}[/yellow]"
    if act.escalation_notes:
        action_content += f"\n[dim]Escalation Notes:[/dim] {act.escalation_notes}"

    console.print()
    console.print(Panel(
        action_content,
        title="Action Decision",
        border_style=style.split()[0],  # use the base color
        box=box.ROUNDED,
    ))

    # --- Suggested Reply (always shown) ---
    if act.suggested_reply:
        console.print()
        console.print(Panel(
            act.suggested_reply,
            title="Suggested Reply to Customer",
            border_style="bright_green",
            box=box.ROUNDED,
        ))

    # --- Auto Response Draft ---
    if act.auto_response:
        console.print()
        console.print(Panel(
            act.auto_response,
            title="Draft Auto-Response",
            border_style="green",
            box=box.ROUNDED,
        ))

    # --- Customer Context ---
    console.print()
    console.print(Panel(
        result.customer_context,
        title="Customer Context",
        border_style="blue",
        box=box.ROUNDED,
    ))

    # --- KB Articles ---
    if result.kb_articles_used:
        console.print(
            f"\n  [dim]KB Articles Used:[/dim] [cyan]{', '.join(result.kb_articles_used)}[/cyan]"
        )

    # --- Token Usage ---
    console.print(
        f"\n  [dim]Token Usage:[/dim] "
        f"prompt={response.prompt_tokens}, "
        f"completion={response.completion_tokens}, "
        f"total=[bold]{response.total_tokens}[/bold]"
    )

    console.print()
    console.rule(style="blue")


def main() -> None:
    tickets = load_sample_tickets()
    agent = TriageAgent()

    console.print()
    console.print(
        Panel(
            "[bold white]Support Ticket Triage Agent[/bold white]\n"
            "[dim]AI-powered ticket analysis and routing[/dim]",
            title="Triage Agent",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )

    while True:
        console.print()
        ticket_table = Table(
            title="Available Tickets",
            box=box.ROUNDED,
            border_style="dim",
        )
        ticket_table.add_column("#", style="bold cyan", width=4)
        ticket_table.add_column("Ticket ID", style="white")
        ticket_table.add_column("Subject", style="dim")
        for i, ticket in enumerate(tickets, 1):
            ticket_table.add_row(str(i), ticket["ticket_id"], ticket.get("subject", "N/A"))

        console.print(ticket_table)
        console.print("  [bold cyan][A][/bold cyan] Process ALL tickets    [bold cyan][Q][/bold cyan] Quit")

        choice = console.input("\n[bold]Select ticket to process:[/bold] ").strip().upper()

        if choice == "Q":
            console.print("\n[bold]Goodbye![/bold]\n")
            break
        elif choice == "A":
            for ticket in tickets:
                _process_and_print(agent, ticket)
        elif choice.isdigit() and 1 <= int(choice) <= len(tickets):
            _process_and_print(agent, tickets[int(choice) - 1])
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")


def _process_and_print(agent: TriageAgent, ticket: dict) -> None:
    with console.status(f"[bold cyan]Processing {ticket['ticket_id']}...", spinner="dots"):
        try:
            response = agent.process_ticket(ticket)
        except Exception as e:
            console.print(f"[red]Error processing {ticket['ticket_id']}: {e}[/red]")
            return

    print_result(response)


if __name__ == "__main__":
    main()