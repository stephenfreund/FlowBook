import sys
from typing import List
from flowbook.util.output import timer
import argparse
import asyncio
from pathlib import Path
from flowbook.kernel_support.experimental_client import FlowbookClient


async def handle_time(args):
    from nbformat import read

    with timer(message=f"Timing {args.path}"):
        nb = read(args.path, as_version=4)
        client = FlowbookClient(
            nb,
            kernel_name="flowbook_kernel",
            allow_errors=False,
            timeout=60,
        )
        await client.async_execute()


def make_parser():
    parser = argparse.ArgumentParser(description="FlowBook - A data analysis tool")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Base Model to use")
    parser.add_argument("path", help="Path to the notebook or directory of notebooks")

    return parser


async def async_main():
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    await handle_time(args)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
