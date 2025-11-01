# debug_client.py
import re
from nbclient import NotebookClient
from nbformat import NotebookNode
import sys
from pathlib import Path
import uuid
import asyncio
from jupyter_client import BlockingKernelClient
from dataclasses import dataclass
from ipykernel.compiler import get_file_name
from typing import List

from data_ferret.util.output import timer


@dataclass
class ExceptionData:
    etype: str
    evalue: str
    traceback: str
    globals: str


@dataclass
class DebugCommandData:
    cmd: str
    ok: bool
    result: str


@dataclass
class TestCodeData:
    ok: bool
    result: str


@dataclass
class FileRecord:
    cell_id: str
    source: str | List[str]


class FerretClient(NotebookClient):
    def __init__(self, nb: NotebookNode, **kwargs):
        super().__init__(nb, **kwargs)
        self.file_to_cell = {}

    def start_kernel(self, **kwargs):
        # 1) Start the normal NBClient kernel+client
        super().start_kernel(**kwargs)

        # 2) Spin up a SECOND BlockingKernelClient for RPCs
        self.debug_kc = BlockingKernelClient()
        # load the same connection file the main manager is using:
        self.debug_kc.load_connection_file(self.km.connection_file)
        self.debug_kc.start_channels()
        self.debug_kc.wait_for_ready(timeout=self.startup_timeout)

    async def async_start_new_kernel_client(self):
        await super().async_start_new_kernel_client()
        self.debug_kc = BlockingKernelClient()
        self.debug_kc.load_connection_file(self.km.connection_file)
        self.debug_kc.start_channels()
        self.debug_kc.wait_for_ready(timeout=self.startup_timeout)

    def handle_comm_msg(self, outputs, msg, cell_index):
        hdr = msg["header"]
        if hdr["msg_type"] == "comm_msg":
            data = msg["content"]["data"]
            if "etype" in data and "traceback" in data:
                exception_data = ExceptionData(
                    etype=data["etype"],
                    evalue=data["evalue"],
                    traceback=self.replace_filenames(data["traceback"]),
                    globals=data["globals"],
                )
                self.handle_exception(exception_data)
                return

        # otherwise, let NBClient do its normal thing
        return super().handle_comm_msg(outputs, msg, cell_index)

    def handle_exception(self, exception_data: ExceptionData):
        # 2) print the exception info
        print(f"\n⚙️  Debug event: {exception_data.etype}: {exception_data.evalue}")
        print("--- traceback ---")
        for line in self.replace_filenames(exception_data.traceback):
            print(line, end="")
        print("\n")
        print("--- globals ---")
        for line in self.replace_filenames(exception_data.globals):
            print(line, end="")
        print("\n")

    def debug_command(self, cmd: str) -> DebugCommandData:
        comm_id = uuid.uuid4().hex
        # build and send the comm_open for debug_command
        content = {
            "comm_id": comm_id,
            "target_name": "debug_command",
            "target_module": "",
            "data": {"cmd": cmd},
        }
        open_msg = self.debug_kc.session.msg("comm_open", content)
        self.debug_kc.shell_channel.send(open_msg)

        # now pull messages off debug_kc's IOPub until we see our reply
        while True:
            reply = self.debug_kc.get_iopub_msg(timeout=5)
            if (
                reply["header"]["msg_type"] == "comm_msg"
                and reply["content"].get("comm_id") == comm_id
            ):
                data = reply["content"]["data"]
                break

        result = data.get("result") if data.get("ok") else data.get("error")
        return DebugCommandData(
            cmd=cmd, ok=data.get("ok"), result=self.replace_filenames(result)
        )

    def test_code(self) -> TestCodeData:
        """Send test_code comm message to kernel and return random string response."""
        comm_id = uuid.uuid4().hex

        # Build and send the comm_open for test_code
        content = {
            "comm_id": comm_id,
            "target_name": "test_code",
            "target_module": "",
            "data": {},  # No input data needed
        }
        open_msg = self.debug_kc.session.msg("comm_open", content)
        self.debug_kc.shell_channel.send(open_msg)

        # Pull messages off debug_kc's IOPub until we see our reply
        while True:
            reply = self.debug_kc.get_iopub_msg(timeout=5)
            if (
                reply["header"]["msg_type"] == "comm_msg"
                and reply["content"].get("comm_id") == comm_id
            ):
                data = reply["content"]["data"]
                break

        result = data.get("result") if data.get("ok") else data.get("error")
        return TestCodeData(ok=data.get("ok"), result=result)

    async def async_execute_cell(
        self,
        cell: NotebookNode,
        cell_index: int,
        execution_count: int | None = None,
        store_history: bool = True,
    ) -> NotebookNode:
        if cell.cell_type == "code":
            filename = get_file_name(cell.source)
            name = Path(filename).name
            self.file_to_cell[name] = FileRecord(cell.id, cell.source)
        with timer(message=f"Executing cell {cell_index}:{cell.id}"):
            return await super().async_execute_cell(
                cell, cell_index, execution_count, store_history
            )

    def get_cell_by_id(self, cell_id: str) -> NotebookNode | None:
        for cell in self.nb.cells:
            if cell.id == cell_id:
                return cell
        return None

    def get_cell_by_filename(self, filename: str) -> NotebookNode | None:
        record = self.file_to_cell.get(filename)
        if record is not None:
            return self.get_cell_by_id(record.cell_id)
        else:
            return None

    def replace_filenames(self, text: str) -> str:
        # replace all occurences of the filenames if self.file_to_cell with Cell `id` if the cell source matches
        for filename in self.file_to_cell.keys():
            cell = self.get_cell_by_filename(filename)
            if cell:
                pattern = re.compile(r"/\S*?" + re.escape(filename))
                text = pattern.sub(f"In Cell `{cell.id}` ", text)
        return text


if __name__ == "__main__":
    import sys
    import asyncio

    class TestDebugClient(FerretClient):
        def __init__(self, nb: NotebookNode, **kwargs):
            super().__init__(nb, **kwargs)

        def handle_exception(self, exception_data: ExceptionData):
            super().handle_exception(exception_data)

            commands = [
                "list(locals().keys())",
                "locals().get('z')",
                "1 + 1",
                "'done inspecting'",
                "x",
                "z",
                "1 + z",
                "3/0",
                "locals()",
            ]

            for cmd in commands:
                print(">>> ", cmd)
                data = self.debug_command(cmd)
                result = self.replace_filenames(data.result)
                if data.ok:
                    print(result, "\n")
                else:
                    print("Error:", result, "\n")

    async def main():
        from nbformat import read

        nb = read(sys.argv[1], as_version=4)
        client = TestDebugClient(
            nb,
            kernel_name="ferret_kernel",
            allow_errors=False,
            timeout=60,
        )
        print("Executing notebook...")
        try:
            await client.async_execute()
        except Exception as e:
            print("Error:", e)

    asyncio.run(main())
