from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from logger import AgentIteration, AgentMetadata

COLORS = {
    "primary": "#7AA2F7",
    "secondary": "#BB9AF7",
    "success": "#9ECE6A",
    "warning": "#E0AF68",
    "error": "#F7768E",
    "text": "#A9B1D6",
    "muted": "#565F89",
    "accent": "#7DCFFF",
    "border": "#3B4261",
}


class VerbosePrinter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.console = Console() if enabled else None

    def print_metadata(self, metadata: AgentMetadata) -> None:
        if not self.enabled:
            return
        title = Text()
        title.append("◆ ", style=Style(color=COLORS["accent"]))
        title.append("Agent", style=Style(color=COLORS["primary"], bold=True))

        table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2))
        table.add_column("key", style=Style(color=COLORS["muted"]))
        table.add_column("value", style=Style(color=COLORS["text"]))
        table.add_row("Lead", Text(metadata.lead_model, style=Style(color=COLORS["accent"])))
        table.add_row("Sub", Text(metadata.sub_model, style=Style(color=COLORS["secondary"])))
        table.add_row("Query", Text(metadata.query))

        panel = Panel(table, title=title, title_align="left", border_style=COLORS["border"], padding=(0, 1))
        self.console.print()
        self.console.print(panel)

    def print_iteration(self, iteration: AgentIteration) -> None:
        if not self.enabled:
            return
        if iteration.event == "start":
            self._print_start(iteration)
        elif iteration.event == "tool":
            self._print_tool(iteration)
        elif iteration.event == "finish":
            self._print_finish(iteration)

    def _print_start(self, iteration: AgentIteration) -> None:
        agent_label = f"{iteration.agent_type}:{iteration.agent_name}"
        rule = Rule(Text(f" {agent_label} ", style=Style(color=COLORS["primary"])), style=COLORS["border"])
        self.console.print(rule)

    def _print_tool(self, iteration: AgentIteration) -> None:
        data = iteration.data or {}
        tool_name = data.get("tool", "unknown")
        duration_ms = data.get("duration_ms", 0)

        header = Text()
        header.append("▸ ", style=Style(color=COLORS["success"]))
        header.append(tool_name, style=Style(color=COLORS["success"], bold=True))
        header.append(f"  ({duration_ms:.0f}ms)", style=Style(color=COLORS["muted"]))

        result = data.get("result")
        result_str = str(result)[:200] + "..." if result and len(str(result)) > 200 else str(result)

        content = Text()
        if iteration.error:
            content.append(f"Error: {iteration.error}", style=Style(color=COLORS["error"]))
        else:
            content.append(result_str, style=Style(color=COLORS["text"]))

        panel = Panel(content, title=header, title_align="left", border_style=COLORS["success"], padding=(0, 1))
        self.console.print(panel)

    def _print_finish(self, iteration: AgentIteration) -> None:
        data = iteration.data or {}
        duration_ms = data.get("duration_ms", 0)

        header = Text()
        header.append("★ ", style=Style(color=COLORS["warning"]))
        header.append("Finish", style=Style(color=COLORS["warning"], bold=True))
        header.append(f"  ({duration_ms:.0f}ms)", style=Style(color=COLORS["muted"]))

        answer = data.get("answer") or data.get("result")
        if iteration.agent_type == "lead":
            answer_str = str(answer) if answer else "None"
        else:
            answer_str = str(answer)[:500] + "..." if answer and len(str(answer)) > 500 else str(answer)

        content = Text()
        if iteration.error:
            content.append(f"Error: {iteration.error}", style=Style(color=COLORS["error"]))
        else:
            content.append(answer_str, style=Style(color=COLORS["text"]))

        panel = Panel(content, title=header, title_align="left", border_style=COLORS["warning"], padding=(0, 1))
        self.console.print(panel)
        self.console.print()
