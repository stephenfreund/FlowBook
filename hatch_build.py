"""Custom hatch build hook to install Claude Code commands and MCP server globally."""

import json
import shutil
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Commands to install from .claude/commands/ to ~/.claude/commands/
CLAUDE_COMMANDS = [
    'flowbook-nb-fix.md',
]


class CustomBuildHook(BuildHookInterface):
    """Install FlowBook Claude Code commands and MCP server config globally."""

    PLUGIN_NAME = 'custom'

    def initialize(self, version, build_data):
        editable = version == 'editable'
        self._install_claude_commands(editable=editable)
        self._install_mcp_server()

    def _install_claude_commands(self, editable=False):
        """Copy (or symlink for editable) command files to ~/.claude/commands/."""
        source_dir = Path(self.root) / '.claude' / 'commands'
        target_dir = Path.home() / '.claude' / 'commands'

        for filename in CLAUDE_COMMANDS:
            source = source_dir / filename
            if not source.exists():
                continue

            target = target_dir / filename

            # Already symlinked to the right place — nothing to do
            if target.is_symlink() and target.resolve() == source.resolve():
                continue

            target_dir.mkdir(parents=True, exist_ok=True)

            if target.exists() or target.is_symlink():
                target.unlink()

            if editable:
                target.symlink_to(source.resolve())
            else:
                shutil.copy2(source, target)

    def _install_mcp_server(self):
        """Register FlowBook MCP server in ~/.claude.json."""
        config_path = Path.home() / '.claude.json'

        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        mcp_servers = config.setdefault('mcpServers', {})

        # Don't overwrite existing user config
        if 'flowbook' in mcp_servers:
            return

        mcp_servers['flowbook'] = {
            'command': 'flowbook_mcp',
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            f.write('\n')
