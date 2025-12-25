"""Display helpers for kernel output formatting."""

import pprint
from typing import Optional

from IPython.display import Markdown, display

from data_ferret.kernel.checkpoint import Checkpoint


DEFAULT_DIV_STYLE = "padding-left: 1em; font-size: 0.8em; background-color: #f0f0f8; margin-bottom: 0em;"


class DisplayHelper:
    """Helper class for displaying formatted output in Jupyter cells."""

    def __init__(self, div_style: str = DEFAULT_DIV_STYLE):
        self.div_style = div_style

    def display_cell_id(self, cell_id: Optional[str]) -> None:
        """Display the current cell ID."""
        display(
            Markdown(
                f"<div style='{self.div_style}'>"
                f"<b>Cell {cell_id}</b>"
                f"</div>"
            )
        )

    def display_icon_and_text(
        self,
        icon: str,
        text: str,
        contents: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Display an icon with text, optionally with expandable contents.

        Creates display data with both text/markdown and text/plain representations.
        """
        # Build plain text version
        # if contents is None:
        plain_text = f"{icon} {text}"
        # else:
        #     plain_text = f"{icon} {text}\n{contents}"

        # Build markdown version
        if contents is None:
            markdown_text = f"<div style='{self.div_style}'>{icon} {text}</div>"
        else:
            markdown_text = (
                f"<div style='{self.div_style}'>"
                f"<details style='display: inline-block; text-align: left;'>"
                f"<summary>{icon} {text}</summary>\n\n"
                f"<pre style='margin: 0;'><code>{contents}</code></pre>\n\n"
                f"</details>"
                f"</div>"
            )

        # Display with both MIME types
        display(
            {"text/markdown": markdown_text, "text/plain": plain_text},
            raw=True,
            metadata=metadata,
        )

    def display_checkpoint_diff(self, old: Checkpoint, new: Checkpoint) -> None:
        """Display the diff between two checkpoints."""
        diffs = Checkpoint.diff(old, new)
        contents = pprint.pformat(diffs, indent=2)
        if diffs:
            self.display_icon_and_text(
                "↔️", f"Changed: {', '.join(sorted(diffs.keys()))}", contents=contents
            )
        else:
            self.display_icon_and_text("↔️", "No changes")
