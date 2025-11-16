import re
import textwrap
import json
from typing import Any, Union, Tuple, Optional, Dict, List


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


# ANSI color code mappings
ANSI_COLORS = {
    '30': 'black',
    '31': 'red',
    '32': 'green',
    '33': 'yellow',
    '34': 'blue',
    '35': 'magenta',
    '36': 'cyan',
    '37': 'white',
    '90': 'bright-black',
    '91': 'bright-red',
    '92': 'bright-green',
    '93': 'bright-yellow',
    '94': 'bright-blue',
    '95': 'bright-magenta',
    '96': 'bright-cyan',
    '97': 'bright-white',
}


def parse_ansi_codes(code_str: str) -> Optional[Dict[str, str]]:
    """
    Parse ANSI escape code string and extract color/style information.

    Args:
        code_str: The numeric part of the ANSI code (e.g., "31" for red)

    Returns:
        Dictionary with color/style info, or None if not a color code
    """
    codes = code_str.split(';')
    result = {}

    for code in codes:
        if code in ANSI_COLORS:
            result['color'] = ANSI_COLORS[code]
        elif code == '1':
            result['bold'] = True
        elif code == '0':
            # Reset code
            return None

    return result if result else None


def parse_ansi_text(text: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Parse text with ANSI codes and extract the first color/style found.

    Args:
        text: Text potentially containing ANSI escape codes

    Returns:
        Tuple of (stripped_text, style_metadata)
        where style_metadata is a dict with 'color' and/or 'bold' keys, or None
    """
    # Pattern to match ANSI escape sequences
    ansi_pattern = r'\x1B\[([0-9;]+)m'

    # Find the first ANSI code
    match = re.search(ansi_pattern, text)
    metadata = None

    if match:
        code_str = match.group(1)
        metadata = parse_ansi_codes(code_str)

    # Strip all ANSI codes
    stripped = strip_ansi(text)

    return stripped, metadata


def transform_json(obj: Any) -> Any:
    """
    Recursively transform:
      1. Strings containing valid JSON → parsed & transformed
      2. Strings with '\\n' → list of lines each ending in '\\n'
    """
    # Handle strings
    if isinstance(obj, str):
        s = obj
        trimmed = s.strip()

        # Rule #1: try JSON parse if it looks like an object/array
        if trimmed.startswith("{") or trimmed.startswith("["):
            try:
                parsed = json.loads(s)
                return transform_json(parsed)
            except json.JSONDecodeError:
                pass

        # Rule #2: split on newlines (keep line endings)
        if "\n" in s:
            # splitlines(True) keeps the newline characters
            return s.splitlines(True)

        # otherwise unchanged
        return s

    # Handle lists/tuples: transform each element
    if isinstance(obj, list):
        return [transform_json(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(transform_json(item) for item in obj)

    # Handle dicts: transform each value
    if isinstance(obj, dict):
        return {key: transform_json(val) for key, val in obj.items()}

    # Other primitives (int, float, bool, None): unchanged
    return obj


def wrap_markdown(text: str, width: int = 100) -> str:
    """
    Wrap long lines in markdown text while respecting markdown formatting.

    Handles:
    - Code blocks (fenced with ``` or indented with 4 spaces)
    - Lists (ordered and unordered, with proper indentation)
    - Headings (kept on single line)
    - Blockquotes (preserves > markers)
    - Normal paragraphs (wrapped to specified width)
    - Blank lines (preserved)
    - Tabs are converted to 8 spaces

    Args:
        text: The markdown text to wrap
        width: Maximum line width (default: 100)

    Returns:
        Wrapped markdown text
    """
    if not text:
        return text

    # Replace tabs with 8 spaces for consistent rendering
    text = text.replace('\t', ' ' * 8)

    lines = text.split('\n')
    result_lines: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for fenced code block start
        if line.strip().startswith('```'):
            # Copy code block verbatim until closing fence
            result_lines.append(line)
            i += 1
            while i < len(lines):
                result_lines.append(lines[i])
                if lines[i].strip().startswith('```'):
                    i += 1
                    break
                i += 1
            continue

        # Check for indented code block (4+ spaces at start)
        if line.startswith('    '):
            # Copy indented code block verbatim
            result_lines.append(line)
            i += 1
            # Continue while lines are indented
            while i < len(lines) and (lines[i].startswith('    ') or lines[i].strip() == ''):
                result_lines.append(lines[i])
                i += 1
            continue

        # Check for blank line
        if line.strip() == '':
            result_lines.append(line)
            i += 1
            continue

        # Check for heading
        if line.lstrip().startswith('#'):
            # Keep headings on single line
            result_lines.append(line)
            i += 1
            continue

        # Check for blockquote
        if line.lstrip().startswith('>'):
            # Handle blockquote
            quote_match = re.match(r'^(\s*)(>+\s*)(.*)', line)
            if quote_match:
                indent = quote_match.group(1)
                marker = quote_match.group(2)
                content = quote_match.group(3)

                # Collect continuation lines
                block_lines = [content]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    if next_line.lstrip().startswith('>'):
                        next_match = re.match(r'^(\s*)(>+\s*)(.*)', next_line)
                        if next_match:
                            block_lines.append(next_match.group(3))
                            j += 1
                        else:
                            break
                    elif next_line.strip() == '':
                        break
                    else:
                        # Continuation line without >
                        block_lines.append(next_line)
                        j += 1

                # Wrap the blockquote content
                full_content = ' '.join(line.strip() for line in block_lines if line.strip())
                wrapped = textwrap.fill(
                    full_content,
                    width=width - len(indent) - len(marker),
                    break_long_words=False,
                    break_on_hyphens=False
                )
                for wrapped_line in wrapped.split('\n'):
                    result_lines.append(f"{indent}{marker}{wrapped_line}")

                i = j
                continue

        # Check for list item (unordered: -, *, +; ordered: 1., 2., etc.)
        list_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.+)', line)
        if list_match:
            indent = list_match.group(1)
            marker = list_match.group(2)
            content = list_match.group(3)

            # Collect continuation lines (indented more than the marker)
            continuation_indent = len(indent) + len(marker) + 1
            # Code blocks need extra indentation - be lenient and accept 2+ extra spaces
            # (strict CommonMark requires 4, but LLMs often output with less)
            code_block_indent = continuation_indent + 2

            # Collect all parts of the list item, preserving order
            # Each element is either ('text', [lines]) or ('blank', '') or ('code', line)
            parts = [('text', [content])]
            j = i + 1

            while j < len(lines):
                next_line = lines[j]

                # Check if it's a new list item
                if re.match(r'^(\s*)([-*+]|\d+\.)\s+', next_line):
                    # New list item - stop here
                    break

                # Check if line is blank
                if next_line.strip() == '':
                    # Blank line within list item - look ahead to see if there's more content
                    k = j + 1
                    has_more_content = False
                    while k < len(lines):
                        peek_line = lines[k]
                        if peek_line.strip() == '':
                            k += 1
                            continue
                        # Check if it's continuation of this list item
                        peek_indent = len(peek_line) - len(peek_line.lstrip())
                        if peek_indent >= continuation_indent and not re.match(r'^(\s*)([-*+]|\d+\.)\s+', peek_line):
                            has_more_content = True
                        break

                    if not has_more_content:
                        # No more content, end the list item
                        break

                    # Blank line with more content after - preserve it
                    parts.append(('blank', ''))
                    j += 1
                    continue

                # Check indentation level
                line_indent = len(next_line) - len(next_line.lstrip())

                if line_indent < continuation_indent:
                    # Not indented enough to be part of this list item
                    break
                elif line_indent >= code_block_indent:
                    # Code block within list item (indented 4+ spaces beyond continuation)
                    # Preserve it verbatim with proper indentation
                    parts.append(('code', next_line))
                    j += 1
                else:
                    # Regular continuation text (indented at continuation level)
                    # Add to current text section or start a new one
                    if parts and parts[-1][0] == 'text':
                        parts[-1][1].append(next_line.strip())
                    else:
                        parts.append(('text', [next_line.strip()]))
                    j += 1

            # Now output the list item, processing parts in order
            first_line = True
            wrapper = textwrap.TextWrapper(
                width=width - len(indent) - len(marker) - 1,
                subsequent_indent=' ' * continuation_indent,
                break_long_words=False,
                break_on_hyphens=False
            )

            for part_type, part_content in parts:
                if part_type == 'text':
                    # Wrap this text section
                    full_text = ' '.join(line for line in part_content if line)
                    if full_text:
                        wrapped = wrapper.fill(full_text)
                        wrapped_lines = wrapped.split('\n')

                        if first_line:
                            # First output line includes the marker
                            result_lines.append(f"{indent}{marker} {wrapped_lines[0]}")
                            first_line = False
                            for wrapped_line in wrapped_lines[1:]:
                                result_lines.append(wrapped_line)
                        else:
                            # Subsequent text sections just get continuation indent
                            for wrapped_line in wrapped_lines:
                                result_lines.append(' ' * continuation_indent + wrapped_line)

                elif part_type == 'blank':
                    result_lines.append('')
                    if first_line:
                        # If we haven't output the marker yet, we need to do it
                        # This shouldn't normally happen, but handle it
                        first_line = False

                elif part_type == 'code':
                    # Preserve code line exactly as-is
                    result_lines.append(part_content)
                    if first_line:
                        # This also shouldn't normally happen (code before any text)
                        first_line = False

            i = j
            continue

        # Regular paragraph - collect until blank line or special element
        para_lines = [line]
        j = i + 1

        while j < len(lines):
            next_line = lines[j]
            # Stop at blank line, code block, heading, list, or blockquote
            if (next_line.strip() == '' or
                next_line.strip().startswith('```') or
                next_line.lstrip().startswith('#') or
                next_line.startswith('    ') or
                next_line.lstrip().startswith('>') or
                re.match(r'^\s*([-*+]|\d+\.)\s+', next_line)):
                break
            para_lines.append(next_line)
            j += 1

        # Join and wrap paragraph
        para_text = ' '.join(line.strip() for line in para_lines if line.strip())
        if para_text:
            wrapped = textwrap.fill(
                para_text,
                width=width,
                break_long_words=False,
                break_on_hyphens=False
            )
            result_lines.append(wrapped)

        i = j

    return '\n'.join(result_lines)
