from data_ferret.lsp.lsp_agent import apply
from pathlib import Path
from nbformat import read
from data_ferret.lsp.notebook_diff import notebook_diff, print_colored_diff
from data_ferret.lsp.output import set_debug
import sys


async def main():
    set_debug(True)
    # sys.args is the name of the notebook to run followed by the prompt
    if len(sys.argv) < 4:
        print("Usage: python a.py <notebook> <cell id> <prompt>")
        sys.exit(1)
    notebook_path = sys.argv[1]
    cell_id = sys.argv[2]
    prompt = " ".join(sys.argv[3:])
    notebook, response_text = await apply(
        Path(notebook_path).absolute(), cell_id, prompt
    )
    print(response_text)
    print("-------")

    diff = notebook_diff(read(notebook_path, as_version=4), notebook)
    print_colored_diff(diff)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
