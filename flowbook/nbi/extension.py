"""FlowBook NBI Extension — registers FlowBook tools with Notebook Intelligence.

When activated, disables NBI's built-in notebook-edit and notebook-execute toolsets
(which destroy cell identity) and provides FlowBook's identity-safe replacements.
"""

from notebook_intelligence.api import NotebookIntelligenceExtension, Toolset, Host

from flowbook.nbi.tools import create_tools, FLOWBOOK_INSTRUCTIONS
from flowbook.nbi.session import FlowBookSession


class FlowBookNBIExtension(NotebookIntelligenceExtension):
    """NBI extension that provides FlowBook reproducibility tools."""

    @property
    def id(self) -> str:
        return "flowbook"

    @property
    def name(self) -> str:
        return "FlowBook Reproducibility"

    @property
    def provider(self) -> str:
        return "FlowBook"

    @property
    def url(self) -> str:
        return ""

    def activate(self, host: Host) -> None:
        # Disable NBI's cell tools that destroy cell identity (delete+insert)
        host.disable_builtin_toolset("nbi-notebook-edit")
        host.disable_builtin_toolset("nbi-notebook-execute")

        session = FlowBookSession()
        toolset = Toolset(
            id="flowbook-reproducibility",
            name="FlowBook Reproducibility",
            description="Notebook reproducibility tracking, cell editing, execution, and refactoring",
            provider=self,
            tools=create_tools(session),
            instructions=FLOWBOOK_INSTRUCTIONS,
        )
        host.register_toolset(toolset)
