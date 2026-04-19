"""FlowBook NBI chat participant — exposes @flowbook with discovered /commands.

Slash commands and their system prompts are loaded from a directory of markdown
files at activation time (one file per command, filename = command name). Each
file has YAML frontmatter with a `description` followed by the prompt body.

The built-in `/clear` command is always appended (NBI handles it).

Routing note: NBI forces the participant to Claude-Code when Claude-Code mode is on
(ai_service_manager.py). This participant therefore activates in GitHub Copilot /
OpenAI-compatible / LiteLLM / Ollama modes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from notebook_intelligence.api import (
    ChatCommand,
    ChatRequest,
    ChatResponse,
    Tool,
)
from notebook_intelligence.base_chat_participant import BaseChatParticipant, ICON_URL

from flowbook.nbi.tools import FLOWBOOK_BACKGROUND, FLOWBOOK_INSTRUCTIONS


log = logging.getLogger(__name__)


@dataclass
class _Command:
    name: str
    description: str
    prompt: str


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a `---`-delimited YAML frontmatter block off the body. Returns
    ({} , text) when the file has no frontmatter. Values are string-typed; single
    and double quotes are stripped if they surround the whole value."""
    if not text.startswith('---\n'):
        return {}, text
    end = text.find('\n---\n', 4)
    if end == -1:
        return {}, text
    fm_block = text[4:end]
    body = text[end + 5:]
    meta: dict[str, str] = {}
    for line in fm_block.splitlines():
        key, sep, value = line.partition(':')
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            meta[key] = value
    return meta, body


def load_commands(commands_dir: Path) -> dict[str, _Command]:
    """Discover *.md command files. Returns {name: _Command}, sorted by filename."""
    commands: dict[str, _Command] = {}
    if not commands_dir.is_dir():
        log.warning("FlowBook commands dir %s does not exist; no commands loaded", commands_dir)
        return commands
    for path in sorted(commands_dir.glob('*.md')):
        text = path.read_text(encoding='utf-8')
        meta, body = _parse_frontmatter(text)
        name = path.stem
        commands[name] = _Command(
            name=name,
            description=meta.get('description', name),
            prompt=body.lstrip('\n'),
        )
    return commands


class FlowBookChatParticipant(BaseChatParticipant):
    """@flowbook chat participant. Commands are discovered from a markdown dir."""

    def __init__(self, tools: list[Tool], commands_dir: Path):
        super().__init__()
        self._tools_list = tools
        self._commands_dir = Path(commands_dir)
        self._commands = load_commands(self._commands_dir)

    @property
    def id(self) -> str:
        return "flowbook"

    @property
    def name(self) -> str:
        return "FlowBook"

    @property
    def description(self) -> str:
        return "Reproducibility assistant for notebooks running flowbook_kernel"

    @property
    def icon_path(self) -> str:
        return ICON_URL

    @property
    def commands(self) -> list[ChatCommand]:
        cmds = [
            ChatCommand(name=c.name, description=c.description)
            for c in self._commands.values()
        ]
        cmds.append(ChatCommand(name='clear', description='Clear chat history'))
        return cmds

    @property
    def tools(self) -> list[Tool]:
        return self._tools_list

    async def handle_chat_request(
        self,
        request: ChatRequest,
        response: ChatResponse,
        options: dict = {},
    ) -> None:
        self._current_chat_request = request

        command = self._commands.get(request.command)
        if command is not None:
            system_prompt = FLOWBOOK_BACKGROUND + "\n\n" + command.prompt
        else:
            system_prompt = FLOWBOOK_INSTRUCTIONS

        system_prompt = self._inject_rules_into_system_prompt(system_prompt, request)

        merged_options = {**options, "system_prompt": system_prompt}
        await self.handle_chat_request_with_tools(request, response, merged_options)
