"""FlowBook NBI Extension — registers FlowBook tools with Notebook Intelligence.

When activated, disables NBI's built-in notebook-edit and notebook-execute toolsets
(which destroy cell identity) and provides FlowBook's identity-safe replacements.
Also installs Claude Code slash commands and MCP server configuration.
"""

import json
import logging
import shutil
from pathlib import Path

from notebook_intelligence.api import NotebookIntelligenceExtension, Toolset, Host
from notebook_intelligence.util import get_jupyter_root_dir

from flowbook.nbi.tools import create_tools, FLOWBOOK_INSTRUCTIONS
from flowbook.nbi.session import FlowBookSession

log = logging.getLogger(__name__)

# Directory containing Claude command files bundled with this package
_CLAUDE_COMMANDS_DIR = Path(__file__).parent / 'claude_commands'


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

        changed = self._install_claude_commands()
        changed |= self._install_mcp_server()
        if changed:
            log.warning(
                "FlowBook updated Claude Code configuration. "
                "Restart any running Claude Code sessions to pick up the changes."
            )

    def _install_claude_commands(self) -> bool:
        """Copy Claude command files to the project-local .claude/commands/ directory.

        Returns True if any files were created or updated.
        """
        root_dir = get_jupyter_root_dir()
        if not root_dir:
            log.warning("Could not determine Jupyter root dir; skipping Claude command install")
            return False

        target_dir = Path(root_dir) / '.claude' / 'commands'
        changed = False

        for source in _CLAUDE_COMMANDS_DIR.glob('*.md'):
            target = target_dir / source.name
            if target.exists() and target.read_text() == source.read_text():
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            log.info("Installed Claude command: %s", target)
            changed = True

        return changed

    def _install_mcp_server(self) -> bool:
        """Register FlowBook MCP servers in NBI and Claude Code configs.

        NBI config (~/.jupyter/nbi/mcp.json): registers flowbook_nbi_mcp which
        works on the active JupyterLab notebook — available to ALL NBI chat
        participants (GitHub Copilot, OpenAI, etc.)

        Claude Code config (~/.claude.json): registers flowbook_mcp (standalone)
        for Claude Code CLI use.

        Returns True if either config was updated.
        """
        changed = False
        changed |= self._register_mcp_in_nbi()
        changed |= self._register_mcp_in_claude()
        return changed

    def _register_mcp_in_nbi(self) -> bool:
        """Register FlowBook NBI MCP server in NBI's mcp.json."""
        config_dir = Path.home() / '.jupyter' / 'nbi'
        config_path = config_dir / 'mcp.json'

        config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read %s; skipping NBI MCP install", config_path)
                return False

        mcp_servers = config.setdefault('mcpServers', {})

        if 'flowbook' in mcp_servers:
            return False

        mcp_servers['flowbook'] = {
            'command': 'flowbook_mcp',
        }

        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            f.write('\n')

        log.info("Registered FlowBook NBI MCP server in %s", config_path)
        return True

    def _register_mcp_in_claude(self) -> bool:
        """Register FlowBook MCP server in Claude Code's ~/.claude.json."""
        config_path = Path.home() / '.claude.json'

        config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read ~/.claude.json; skipping Claude Code MCP install")
                return False

        mcp_servers = config.setdefault('mcpServers', {})

        if 'flowbook' in mcp_servers:
            return False

        mcp_servers['flowbook'] = {
            'command': 'flowbook_mcp',
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            f.write('\n')

        log.info("Registered FlowBook MCP server in ~/.claude.json")
        return True
