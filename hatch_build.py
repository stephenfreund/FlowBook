"""Custom hatch build hook — placeholder for future build-time tasks."""

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Placeholder build hook. Claude commands and MCP server registration
    are now handled by the NBI extension at activation time."""

    PLUGIN_NAME = 'custom'

    def initialize(self, version, build_data):
        pass
