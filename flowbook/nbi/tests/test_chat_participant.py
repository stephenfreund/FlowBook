"""Tests for FlowBookChatParticipant and command discovery."""

from pathlib import Path
from textwrap import dedent

import pytest
from unittest.mock import MagicMock

pytest.importorskip("notebook_intelligence")

from notebook_intelligence.api import ChatCommand
from notebook_intelligence.base_chat_participant import BaseChatParticipant

from flowbook.nbi.chat_participant import (
    FlowBookChatParticipant,
    _parse_frontmatter,
    load_commands,
)
from flowbook.nbi.tools import FLOWBOOK_BACKGROUND, FLOWBOOK_INSTRUCTIONS


_REAL_COMMANDS_DIR = Path(__file__).parent.parent / 'commands'


def _make_commands_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    d = tmp_path / 'commands'
    d.mkdir()
    for name, content in files.items():
        (d / name).write_text(content, encoding='utf-8')
    return d


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("Hello world\n")
        assert meta == {}
        assert body == "Hello world\n"

    def test_empty_string(self):
        meta, body = _parse_frontmatter("")
        assert meta == {}
        assert body == ""

    def test_extracts_description(self):
        text = dedent("""\
            ---
            description: 'Fix things'
            ---

            Body here
            """)
        meta, body = _parse_frontmatter(text)
        assert meta == {'description': 'Fix things'}
        assert body.strip() == "Body here"

    def test_strips_double_quotes(self):
        text = '---\ndescription: "Fix things"\n---\nbody'
        meta, _ = _parse_frontmatter(text)
        assert meta['description'] == 'Fix things'

    def test_leaves_unquoted_value_alone(self):
        text = '---\ndescription: Fix things\n---\nbody'
        meta, _ = _parse_frontmatter(text)
        assert meta['description'] == 'Fix things'

    def test_multiple_keys(self):
        text = '---\na: 1\nb: two\n---\nbody'
        meta, _ = _parse_frontmatter(text)
        assert meta == {'a': '1', 'b': 'two'}

    def test_malformed_frontmatter_returned_as_body(self):
        text = '---\nno closing delimiter\nbody here'
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text


# ---------------------------------------------------------------------------
# Command discovery
# ---------------------------------------------------------------------------

class TestLoadCommands:
    def test_empty_dir(self, tmp_path):
        commands = load_commands(_make_commands_dir(tmp_path, {}))
        assert commands == {}

    def test_missing_dir_returns_empty(self, tmp_path):
        commands = load_commands(tmp_path / 'does-not-exist')
        assert commands == {}

    def test_discovers_by_filename(self, tmp_path):
        d = _make_commands_dir(tmp_path, {
            'alpha.md': "---\ndescription: 'A'\n---\nalpha body",
            'beta.md':  "---\ndescription: 'B'\n---\nbeta body",
        })
        commands = load_commands(d)
        assert set(commands) == {'alpha', 'beta'}
        assert commands['alpha'].description == 'A'
        assert commands['alpha'].prompt == 'alpha body'
        assert commands['beta'].prompt == 'beta body'

    def test_ignores_non_md_files(self, tmp_path):
        d = _make_commands_dir(tmp_path, {
            'alpha.md': "body",
            'README.txt': "not a command",
        })
        assert set(load_commands(d)) == {'alpha'}

    def test_missing_frontmatter_uses_name_as_description(self, tmp_path):
        d = _make_commands_dir(tmp_path, {'bare.md': "just a prompt body"})
        cmd = load_commands(d)['bare']
        assert cmd.description == 'bare'
        assert cmd.prompt == "just a prompt body"

    def test_prompt_leading_newlines_stripped(self, tmp_path):
        d = _make_commands_dir(tmp_path, {
            'x.md': "---\ndescription: 'x'\n---\n\n\n# heading\n",
        })
        assert load_commands(d)['x'].prompt.startswith('# heading')

    def test_sorted_order(self, tmp_path):
        d = _make_commands_dir(tmp_path, {
            'zeta.md': "body", 'alpha.md': "body", 'mu.md': "body",
        })
        assert list(load_commands(d)) == ['alpha', 'mu', 'zeta']


# ---------------------------------------------------------------------------
# Real shipped commands directory
# ---------------------------------------------------------------------------

class TestShippedCommands:
    def test_commands_dir_exists(self):
        assert _REAL_COMMANDS_DIR.is_dir()

    def test_fix_and_status_are_present(self):
        commands = load_commands(_REAL_COMMANDS_DIR)
        assert 'fix' in commands
        assert 'status' in commands

    def test_fix_prompt_has_workflow(self):
        fix = load_commands(_REAL_COMMANDS_DIR)['fix']
        assert "Fix Loop" in fix.prompt
        assert "mcp__nbi__" in fix.prompt

    def test_status_prompt_is_read_only(self):
        status = load_commands(_REAL_COMMANDS_DIR)['status']
        lowered = status.prompt.lower()
        assert "read-only" in lowered
        assert "do not" in lowered
        assert "get_status" in status.prompt
        assert "read_cell" in status.prompt

    def test_status_prompt_specifies_output_format(self):
        status = load_commands(_REAL_COMMANDS_DIR)['status']
        assert "Summary" in status.prompt
        assert "Cells" in status.prompt
        assert "| Cell |" in status.prompt

    def test_descriptions_are_short(self):
        # ChatCommand descriptions show in the NBI chat UI — keep them compact.
        for cmd in load_commands(_REAL_COMMANDS_DIR).values():
            assert 0 < len(cmd.description) <= 80, cmd


# ---------------------------------------------------------------------------
# Participant identity
# ---------------------------------------------------------------------------

def _participant(tmp_path=None, **files) -> FlowBookChatParticipant:
    if tmp_path is None:
        commands_dir = _REAL_COMMANDS_DIR
    else:
        commands_dir = _make_commands_dir(tmp_path, files) if files else tmp_path
    return FlowBookChatParticipant(tools=[], commands_dir=commands_dir)


class TestIdentity:
    def test_id(self):
        assert _participant().id == "flowbook"

    def test_name(self):
        assert _participant().name == "FlowBook"

    def test_description_mentions_reproducibility(self):
        assert "reproducibility" in _participant().description.lower()

    def test_icon_is_data_url(self):
        assert _participant().icon_path.startswith("data:image/svg+xml;base64,")

    def test_extends_base_chat_participant(self):
        assert issubclass(FlowBookChatParticipant, BaseChatParticipant)


# ---------------------------------------------------------------------------
# Commands list (dynamic)
# ---------------------------------------------------------------------------

class TestCommandsProperty:
    def test_real_dir_yields_fix_status_clear(self):
        names = [c.name for c in _participant().commands]
        assert names == ['fix', 'status', 'clear']

    def test_descriptions_come_from_frontmatter(self):
        cmds = {c.name: c for c in _participant().commands}
        assert "violation" in cmds['fix'].description.lower()
        assert "status" in cmds['status'].description.lower()

    def test_items_are_chatcommand_instances(self):
        for cmd in _participant().commands:
            assert isinstance(cmd, ChatCommand)

    def test_clear_appended_last(self):
        names = [c.name for c in _participant().commands]
        assert names[-1] == 'clear'

    def test_custom_dir_drives_commands(self, tmp_path):
        p = _participant(
            tmp_path,
            **{
                'foo.md': "---\ndescription: 'Foo it'\n---\nfoo prompt",
                'bar.md': "---\ndescription: 'Bar it'\n---\nbar prompt",
            },
        )
        names = [c.name for c in p.commands]
        assert names == ['bar', 'foo', 'clear']

    def test_empty_dir_still_offers_clear(self, tmp_path):
        p = _participant(tmp_path / 'empty-dir')
        (tmp_path / 'empty-dir').mkdir()
        assert [c.name for c in p.commands] == ['clear']

    def test_stable_across_calls(self):
        p = _participant()
        a = [(c.name, c.description) for c in p.commands]
        b = [(c.name, c.description) for c in p.commands]
        assert a == b


# ---------------------------------------------------------------------------
# Tools pass-through
# ---------------------------------------------------------------------------

class TestTools:
    def test_tools_passed_through(self):
        tool_a, tool_b = MagicMock(), MagicMock()
        participant = FlowBookChatParticipant(tools=[tool_a, tool_b], commands_dir=_REAL_COMMANDS_DIR)
        assert participant.tools == [tool_a, tool_b]


# ---------------------------------------------------------------------------
# Background prompt content
# ---------------------------------------------------------------------------

class TestBackground:
    def test_has_four_predicates(self):
        for predicate in ("NoReadAndWrite", "WriteBeforeRead", "NoReadBeforeWrite", "NoWriteAfterRead"):
            assert predicate in FLOWBOOK_BACKGROUND

    def test_mentions_rerun_consistency(self):
        assert "rerun consistency" in FLOWBOOK_BACKGROUND.lower()

    def test_mentions_staleness(self):
        t = FLOWBOOK_BACKGROUND.lower()
        assert "stale" in t
        assert "forward staleness" in t
        assert "backward staleness" in t

    def test_describes_unrecoverable_mutation(self):
        assert "UNRECOVERABLE_MUTATION" in FLOWBOOK_BACKGROUND

    def test_lists_fix_tools(self):
        for tool in ("alpha_rename", "insert_deepcopy", "remove_inplace",
                     "mark_diagnostic", "merge_cells", "move_cell"):
            assert tool in FLOWBOOK_BACKGROUND

    def test_mentions_clean_state(self):
        assert "CLEAN" in FLOWBOOK_BACKGROUND

    def test_mentions_cell_addressing(self):
        assert "@A" in FLOWBOOK_BACKGROUND and "@B" in FLOWBOOK_BACKGROUND

    def test_mentions_flowbook_kernel(self):
        assert "flowbook_kernel" in FLOWBOOK_BACKGROUND

    def test_instructions_begins_with_background(self):
        assert FLOWBOOK_INSTRUCTIONS.startswith(FLOWBOOK_BACKGROUND)

    def test_instructions_has_fix_algorithm_section(self):
        assert "Fix algorithm" in FLOWBOOK_INSTRUCTIONS
        assert "run_actionable_cells" in FLOWBOOK_INSTRUCTIONS

    def test_background_not_duplicated(self):
        assert FLOWBOOK_INSTRUCTIONS.count(FLOWBOOK_BACKGROUND) == 1


# ---------------------------------------------------------------------------
# Dispatch behavior
# ---------------------------------------------------------------------------

def _patch_dispatch(monkeypatch, participant):
    """Patch the tool-call loop and rule injection; return a dict that receives
    whatever `options` the participant eventually passes through."""
    captured: dict = {}

    async def fake_with_tools(request, response, options):
        captured['options'] = options

    monkeypatch.setattr(participant, 'handle_chat_request_with_tools', fake_with_tools)
    monkeypatch.setattr(participant, '_inject_rules_into_system_prompt', lambda p, r: p)
    return captured


class TestDispatch:
    @pytest.mark.asyncio
    async def test_known_command_uses_its_prompt(self, monkeypatch, tmp_path):
        p = _participant(
            tmp_path,
            **{'alpha.md': "---\ndescription: 'alpha cmd'\n---\nALPHA PROMPT BODY"},
        )
        cap = _patch_dispatch(monkeypatch, p)
        await p.handle_chat_request(MagicMock(command='alpha'), MagicMock())

        sp = cap['options']['system_prompt']
        assert "ALPHA PROMPT BODY" in sp
        assert FLOWBOOK_BACKGROUND in sp

    @pytest.mark.asyncio
    async def test_fix_uses_fix_prompt(self, monkeypatch):
        p = _participant()
        cap = _patch_dispatch(monkeypatch, p)
        await p.handle_chat_request(MagicMock(command='fix'), MagicMock())
        sp = cap['options']['system_prompt']
        assert "Fix Loop" in sp
        assert FLOWBOOK_BACKGROUND in sp

    @pytest.mark.asyncio
    async def test_status_uses_status_prompt(self, monkeypatch):
        p = _participant()
        cap = _patch_dispatch(monkeypatch, p)
        await p.handle_chat_request(MagicMock(command='status'), MagicMock())
        sp = cap['options']['system_prompt']
        assert "read-only" in sp.lower()
        assert FLOWBOOK_BACKGROUND in sp

    @pytest.mark.asyncio
    async def test_unknown_command_uses_full_instructions(self, monkeypatch):
        p = _participant()
        cap = _patch_dispatch(monkeypatch, p)
        await p.handle_chat_request(MagicMock(command=''), MagicMock())
        assert cap['options']['system_prompt'] == FLOWBOOK_INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_clear_falls_through_to_default(self, monkeypatch):
        p = _participant()
        cap = _patch_dispatch(monkeypatch, p)
        await p.handle_chat_request(MagicMock(command='clear'), MagicMock())
        # `clear` has no .md file — should take the default path.
        assert cap['options']['system_prompt'] == FLOWBOOK_INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_sets_current_chat_request(self, monkeypatch):
        p = _participant()
        _patch_dispatch(monkeypatch, p)
        req = MagicMock(command='status')
        await p.handle_chat_request(req, MagicMock())
        assert p._current_chat_request is req

    @pytest.mark.asyncio
    async def test_rule_injection_applied(self, monkeypatch):
        p = _participant()
        captured = {}

        async def fake_with_tools(request, response, options):
            captured['system_prompt'] = options.get('system_prompt')

        monkeypatch.setattr(p, 'handle_chat_request_with_tools', fake_with_tools)
        monkeypatch.setattr(
            p,
            '_inject_rules_into_system_prompt',
            lambda prompt, req: prompt + "\n<RULE-MARKER>",
        )

        for cmd in ('fix', 'status', ''):
            captured.clear()
            await p.handle_chat_request(MagicMock(command=cmd), MagicMock())
            assert "<RULE-MARKER>" in captured['system_prompt']

    @pytest.mark.asyncio
    async def test_caller_options_preserved(self, monkeypatch):
        p = _participant()
        cap = _patch_dispatch(monkeypatch, p)
        await p.handle_chat_request(MagicMock(command='fix'), MagicMock(), {"foo": 42})
        assert cap['options']['foo'] == 42
        assert cap['options']['system_prompt']

    @pytest.mark.asyncio
    async def test_caller_options_not_mutated(self, monkeypatch):
        p = _participant()
        _patch_dispatch(monkeypatch, p)
        caller_options = {"unrelated": "preserve-me"}
        before = dict(caller_options)
        await p.handle_chat_request(MagicMock(command='fix'), MagicMock(), caller_options)
        assert caller_options == before


# ---------------------------------------------------------------------------
# Extension wiring
# ---------------------------------------------------------------------------

class TestExtensionRegistersParticipant:
    def test_activate_registers_chat_participant(self):
        from flowbook.nbi.extension import FlowBookNBIExtension, _COMMANDS_DIR

        ext = FlowBookNBIExtension()
        host = MagicMock()
        ext.activate(host)

        host.register_chat_participant.assert_called_once()
        participant = host.register_chat_participant.call_args[0][0]
        assert isinstance(participant, FlowBookChatParticipant)
        assert participant.id == "flowbook"
        # Participant reuses the toolset's tool list
        toolset = host.register_toolset.call_args[0][0]
        assert participant.tools == toolset.tools
        # Participant reads from the shipped commands directory
        assert participant._commands_dir == _COMMANDS_DIR
        assert {'fix', 'status'}.issubset(participant._commands.keys())


class TestClaudeInstall:
    def test_install_prefixes_filenames(self, tmp_path, monkeypatch):
        """Claude Code install copies commands/*.md → .claude/commands/flowbook-*.md."""
        from flowbook.nbi import extension as ext_mod

        monkeypatch.setattr(ext_mod, 'get_jupyter_root_dir', lambda: str(tmp_path))
        ext = ext_mod.FlowBookNBIExtension()
        changed = ext._install_claude_commands()

        target_dir = tmp_path / '.claude' / 'commands'
        assert changed
        assert (target_dir / 'flowbook-fix.md').is_file()
        assert (target_dir / 'flowbook-status.md').is_file()
        # Raw names must NOT leak through without the prefix.
        assert not (target_dir / 'fix.md').exists()
        assert not (target_dir / 'status.md').exists()

    def test_install_is_idempotent(self, tmp_path, monkeypatch):
        from flowbook.nbi import extension as ext_mod

        monkeypatch.setattr(ext_mod, 'get_jupyter_root_dir', lambda: str(tmp_path))
        ext = ext_mod.FlowBookNBIExtension()
        assert ext._install_claude_commands() is True
        assert ext._install_claude_commands() is False  # no changes on second run
