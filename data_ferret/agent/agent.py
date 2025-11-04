"""
Agent implementation for DataFerret.
"""

import asyncio
from dataclasses import dataclass
import json
import os
import textwrap
import time
import traceback
from typing import Any, Generic, List, Optional, Tuple, Type, TypeVar
from typing import List

from agents import Agent, Runner, Tool, Usage
from agents.extensions.models.litellm_model import LitellmModel
from agents.items import TResponseInputItem
from agents.lifecycle import RunHooks
from agents.memory import SQLiteSession
from agents.model_settings import ModelSettings
from agents.run_context import RunContextWrapper
from pydantic import BaseModel

from data_ferret.agent.llm_cost import cost, full_name
from data_ferret.util.output import error
from data_ferret.util.text import transform_json


class FerretContext:
    """Context for tracking Ferret agent execution."""

    def __init__(self):
        self.start: Optional[float] = None
        self.time: Optional[float] = None
        self.usage: Optional[Usage] = None


@dataclass
class FerretStats:
    """Statistics about Ferret agent execution."""

    model: str
    time: float
    usage: Usage
    cost: float
    log_path: str

    def __init__(
        self, model: str | LitellmModel, time: float, usage: Usage, log_path: str
    ):
        self.model = model if isinstance(model, str) else model.model
        self.time = time
        self.usage = usage
        if "/" in self.model:
            model = self.model.split("/")[1]
        else:
            model = self.model
        self.cost = cost(model, usage)
        self.log_path = log_path


T = TypeVar("T", bound=BaseModel | str)

class FerretAgent(Agent, Generic[T], RunHooks[FerretContext]):

    counters = {}

    def __init__(
        self,
        key: str,
        model: str,
        instructions: str,
        output_type: Type[T] | None = None,
        tools: List[Tool] = [],
        log_dir: str = "agent-logs",
    ):

        full_model = full_name(model)

        super().__init__(
            name=key,
            model=LitellmModel(full_model),
            instructions=instructions,
            output_type=output_type,
            tools=tools,
            model_settings=ModelSettings(
                timeout=120,
            ),
        )
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    @classmethod
    def make_unique(cls, key: str) -> str:
        # generate sequential ints to append to key
        cls.counters[key] = cls.counters.get(key, 0) + 1
        return f"{key}-{cls.counters[key]}"

    def transform_and_dump(self, obj: Any) -> Any:
        return json.dumps(transform_json(obj), indent=2)

    def session_log(self, items: List[TResponseInputItem]) -> str:
        session_text = []
        for item in items:
            if item.get("type", "message") == "message":
                role = item["role"].upper()
                session_text.append(f"{role}")
                session_text.append("-" * len(role))
                content = item["content"]
                if isinstance(content, str):
                    try:
                        as_json = json.loads(content)
                        session_text.extend(
                            textwrap.indent(
                                json.dumps(as_json, indent=2), "  "
                            ).splitlines()
                        )
                        session_text.append("")
                    except:
                        text = textwrap.indent(content, "  ")
                        session_text.extend(text.splitlines())
                        session_text.append("")
                    continue
                else:
                    for content_item in content:
                        if content_item.get("type", "input_text") in [
                            "input_text",
                            "output_text",
                        ]:
                            c = content_item.get("text", "")
                            try:
                                as_json = json.loads(c)
                                session_text.extend(
                                    textwrap.indent(
                                        self.transform_and_dump(as_json), "  "
                                    ).splitlines()
                                )
                                session_text.append("")
                            except:
                                session_text.extend(
                                    textwrap.indent(c, "  ").splitlines()
                                )
                                session_text.append("")
                    continue
            if item.get("type", "") == "function_call":
                session_text.append("Function call")
                session_text.append("-" * len("Function call"))
                session_text.extend(
                    textwrap.indent(self.transform_and_dump(item), "  ").splitlines()
                )
                session_text.append("")
                continue
            if item.get("type", "") == "function_call_output":
                session_text.append("Function call output")
                session_text.append("-" * len("Function call output"))
                session_text.extend(
                    textwrap.indent(self.transform_and_dump(item), "  ").splitlines()
                )
                session_text.append("")
                continue
            assert False, f"bad item {item}"
        return "\n".join(session_text)

    async def run(self, input: str) -> Tuple[T, FerretStats]:
        context = FerretContext()

        key = FerretAgent.make_unique(self.name)
        session = SQLiteSession(key)

        result = await Runner.run(
            self,
            input=input,
            context=context,
            hooks=self,
            session=session,
            max_turns=30,
        )

        assert context.time is not None, "Execution time must be set"
        assert context.usage is not None, "Usage must be set"

        full_text = self.session_log(await session.get_items())

        log_path = f"{self.log_dir}/{key}.txt"
        os.makedirs(self.log_dir, exist_ok=True)
        
        with open(log_path, "w") as f:
            f.write(full_text)

        return result.final_output, FerretStats(
            model=self.model, time=context.time, usage=context.usage, log_path=log_path
        )

    async def on_agent_start(
        self, context: RunContextWrapper[FerretContext], agent
    ) -> None:
        context.context.start = time.time()

    async def on_agent_end(
        self, context: RunContextWrapper[FerretContext], agent, output
    ) -> None:
        assert context.context.start is not None, "Start time must be set"
        context.context.time = time.time() - context.context.start
        context.context.usage = context.usage



    @staticmethod
    async def make_and_run_agent(
        key: str,
        model: str = "gpt-4o-mini",
        instructions: str = "Answer the question.",
        output_type: Type[T] | None = None,
        tools: List[Tool] = [],
        log_dir: str = "agent-logs",
        input: str = "",
        max_retries: int = 3,
    ) -> Tuple[T, FerretStats]:
        i = 0
        while True:
            try:
                agent = FerretAgent(key, model, instructions, output_type, tools, log_dir)
                return await agent.run(input)
            except Exception as e:
                if i >= max_retries:
                    raise e
                i += 1
                error(f"Error running agent: {e}")
                error(traceback.format_exc())
                time.sleep(2**i)

def main():
    agent = FerretAgent(
        key="test-agent",
        model="gpt-4o-mini",
        instructions="You are a test agent.",
    )
    response, stats = asyncio.run(agent.run("What is the capital of France?"))
    print()
    print(response)
    print("cost: ", stats.cost)
    print("total tokens: ", stats.usage.total_tokens)
    print()

    ####

    class TestOutput(BaseModel):
        """Test output model for testing."""
        joke: str
        punchline: str


    agent = FerretAgent(
        key="test-agent",
        model="gpt-4o-mini",
        instructions="You are a test agent.",
        output_type=TestOutput,
    )
    response, stats = asyncio.run(agent.run("Tell me a joke."))
    print(response)
    print("cost: ", stats.cost)
    print("total tokens: ", stats.usage.total_tokens)
    print()


if __name__ == "__main__":
    main()
