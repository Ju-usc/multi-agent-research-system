"""Agent execution logger (RLM-style JSONL).

One file per run:
  - first line: metadata
  - remaining: iteration events (start/tool/finish)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any, Literal
import uuid

import dspy
from dspy.utils.callback import BaseCallback


@dataclass
class AgentMetadata:
    run_id: str
    started_at: str
    lead_model: str
    sub_model: str
    query: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentIteration:
    agent_type: Literal["lead", "subagent"]
    agent_name: str
    event: Literal["start", "tool", "finish"]
    data: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentLogger:
    """Write high-signal domain events to JSONL."""

    def __init__(
        self,
        log_dir: str,
        lead_model: str,
        sub_model: str,
        query: str,
    ) -> None:
        self._lock = threading.Lock()
        self._iteration = 0

        run_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_started_at = datetime.now().isoformat()

        self.metadata = AgentMetadata(
            run_id=run_id,
            started_at=run_started_at,
            lead_model=lead_model,
            sub_model=sub_model,
            query=query,
        )

        log_dir = Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"agent_{timestamp}_{run_id}.jsonl"

        # No lock needed - constructor runs in single thread
        record = {"type": "metadata", "timestamp": run_started_at, **self.metadata.to_dict()}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log(self, iteration: AgentIteration) -> None:
        """Log an iteration event."""
        with self._lock:
            self._iteration += 1
            record = {
                "type": "iteration",
                "timestamp": datetime.now().isoformat(),
                "iteration": self._iteration,
                **iteration.to_dict(),
            }
            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")


class AgentLoggingCallback(BaseCallback):
    """Emit AgentIteration events for ReAct modules and tools."""

    def __init__(self, agent_logger: AgentLogger, verbose_printer=None) -> None:
        self._agent_logger = agent_logger
        self._verbose = verbose_printer
        self._tool_starts: dict[str, tuple[float, str, dict[str, Any], str, str]] = {}
        self._module_starts: dict[str, tuple[float, str, str]] = {}

    def _emit(self, iteration: AgentIteration) -> None:
        self._agent_logger.log(iteration)
        if self._verbose:
            self._verbose.print_iteration(iteration)

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        if not isinstance(instance, dspy.ReAct):
            return

        agent_type = dspy.settings.agent_type
        agent_name = dspy.settings.agent_name
        module_inputs = inputs["kwargs"]

        if agent_type == "lead":
            data = {"query": module_inputs["query"]}
        else:
            data = {"task": module_inputs["task"].model_dump(exclude_none=True)}

        self._emit(AgentIteration(agent_type=agent_type, agent_name=agent_name, event="start", data=data))
        self._module_starts[call_id] = (time.perf_counter(), agent_type, agent_name)

    def on_module_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        start = self._module_starts.pop(call_id, None)
        if start is None:
            return

        t0, agent_type, agent_name = start
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        data: dict[str, Any] = {"duration_ms": duration_ms}
        if agent_type == "lead":
            data["answer"] = outputs.answer if outputs else None
        else:
            final_result = outputs.final_result if outputs else None
            data["result"] = (
                final_result.model_dump(exclude_none=True)
                if final_result
                else None
            )

        self._emit(AgentIteration(
            agent_type=agent_type,
            agent_name=agent_name,
            event="finish",
            data=data,
            error=str(exception) if exception else None,
        ))

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        if instance.name == "finish":
            return

        tool_args = inputs["kwargs"]
        agent_type = dspy.settings.agent_type
        agent_name = dspy.settings.agent_name

        self._tool_starts[call_id] = (time.perf_counter(), instance.name, tool_args, agent_type, agent_name)

    def on_tool_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        start = self._tool_starts.pop(call_id, None)
        if start is None:
            return

        t0, tool_name, tool_args, agent_type, agent_name = start
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        self._emit(AgentIteration(
            agent_type=agent_type,
            agent_name=agent_name,
            event="tool",
            data={"tool": tool_name, "args": tool_args, "result": outputs, "duration_ms": duration_ms},
            error=str(exception) if exception else None,
        ))
