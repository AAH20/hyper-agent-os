"""
CrewAI-Style Swarm Orchestrator Adapter.

Implements role-based persona swarms (Agents, Tasks, Crews) capable of
hierarchical and sequential multi-agent collaboration with structured handoffs.
"""

from __future__ import annotations
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ProcessType(enum.Enum):
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"


AgentCallable = Callable[[str, Dict[str, Any]], str]


@dataclass
class SwarmAgent:
    """An autonomous role-based agent in a swarm."""
    role: str
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    execute_fn: Optional[AgentCallable] = None
    agent_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:6]}")

    def execute(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute a task prompt using the assigned persona."""
        if self.execute_fn:
            return self.execute_fn(prompt, context or {})
        # Default mock persona output if no external LLM hook is attached
        return (
            f"[{self.role}] Completed task: '{prompt[:40]}...'\n"
            f"Aligned with Goal: {self.goal}"
        )


@dataclass
class SwarmTask:
    """A discrete unit of work assigned to an agent."""
    description: str
    expected_output: str
    agent: Optional[SwarmAgent] = None
    output: Optional[str] = None
    status: str = "PENDING"
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:6]}")


@dataclass
class SwarmCrew:
    """A collection of agents cooperating on a set of tasks."""
    agents: List[SwarmAgent]
    tasks: List[SwarmTask]
    process: ProcessType = ProcessType.SEQUENTIAL
    manager: Optional[SwarmAgent] = None
    verbose: bool = False

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute all tasks in the crew according to the configured process.
        """
        context = dict(inputs or {})
        task_results = []

        if self.process == ProcessType.SEQUENTIAL:
            for idx, task in enumerate(self.tasks):
                agent = task.agent or (self.agents[idx % len(self.agents)])
                full_prompt = f"{task.description}\nExpected Output: {task.expected_output}"
                if task_results:
                    full_prompt += f"\nPrevious Output: {task_results[-1]['output']}"

                task.status = "RUNNING"
                out = agent.execute(full_prompt, context)
                task.output = out
                task.status = "COMPLETED"

                task_results.append({
                    "task_id": task.task_id,
                    "agent_role": agent.role,
                    "output": out,
                })
                context[f"task_{task.task_id}_output"] = out

        elif self.process == ProcessType.HIERARCHICAL:
            manager = self.manager or self.agents[0]
            for task in self.tasks:
                # Manager delegates task to the best matching agent
                assigned_agent = task.agent or self._delegate(task, manager)
                task.status = "RUNNING"
                out = assigned_agent.execute(task.description, context)
                task.output = out
                task.status = "COMPLETED"
                task_results.append({
                    "task_id": task.task_id,
                    "agent_role": assigned_agent.role,
                    "delegated_by": manager.role,
                    "output": out,
                })
                context[f"task_{task.task_id}_output"] = out

        final_output = task_results[-1]["output"] if task_results else "No tasks executed."
        return {
            "status": "SUCCESS",
            "results": task_results,
            "final_output": final_output,
            "tasks_completed": len(task_results),
        }

    def _delegate(self, task: SwarmTask, manager: SwarmAgent) -> SwarmAgent:
        """Heuristic or LLM-based delegation to worker agents."""
        workers = [a for a in self.agents if a.agent_id != manager.agent_id]
        if not workers:
            return manager
        # Match keywords in task description with agent roles
        for worker in workers:
            if worker.role.lower() in task.description.lower():
                return worker
        return workers[0]
