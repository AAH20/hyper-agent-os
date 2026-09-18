"""
StateGraph: Cyclical State Machine & DAG Orchestrator.

Implements LangGraph-style workflow orchestration with state checkpoints,
cyclic execution (feedback loops), dynamic conditional edges, and
interruption capabilities for human-in-the-loop validation.
"""

from __future__ import annotations
import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class GraphState:
    """State envelope passed across nodes in the state graph."""
    values: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_node: str = "START"
    iteration: int = 0
    interrupted: bool = False
    interrupt_reason: Optional[str] = None

    def update(self, updates: Dict[str, Any], node_name: str) -> None:
        """Update state values and log snapshot to history."""
        self.values.update(updates)
        self.current_node = node_name
        self.history.append({
            "node": node_name,
            "timestamp": time.time(),
            "snapshot": copy.deepcopy(self.values),
        })


NodeCallable = Callable[[GraphState], Dict[str, Any]]
RouterCallable = Callable[[GraphState], str]


@dataclass
class StateNode:
    """A processing step in the agent workflow."""
    name: str
    action: NodeCallable
    interrupt_before: bool = False


@dataclass
class StateEdge:
    """Transition between two nodes."""
    source: str
    target: str


@dataclass
class RouterCondition:
    """Dynamic routing logic based on state inspection."""
    source: str
    router_fn: RouterCallable


class StateGraph:
    """
    Cyclical Directed Graph Engine executing multi-step agent reasoning loops.
    """

    START = "__START__"
    END = "__END__"

    def __init__(self):
        self.nodes: Dict[str, StateNode] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, RouterCallable] = {}
        self.entry_point: Optional[str] = None

    def add_node(
        self,
        name: str,
        action: NodeCallable,
        interrupt_before: bool = False,
    ) -> StateGraph:
        """Register an execution node."""
        self.nodes[name] = StateNode(
            name=name,
            action=action,
            interrupt_before=interrupt_before,
        )
        return self

    def set_entry_point(self, node_name: str) -> StateGraph:
        """Define starting node."""
        if node_name not in self.nodes and node_name != self.START:
            raise ValueError(f"Entry point {node_name} not found in registered nodes.")
        self.entry_point = node_name
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        """Add deterministic transition edge."""
        self.edges[source] = target
        return self

    def add_conditional_edges(
        self, source: str, router_fn: RouterCallable
    ) -> StateGraph:
        """Add dynamic conditional routing based on state."""
        self.conditional_edges[source] = router_fn
        return self

    def compile(self) -> CompiledGraph:
        """Finalize and validate graph structure."""
        if not self.entry_point:
            raise ValueError("Graph has no entry point defined.")
        return CompiledGraph(self)


class CompiledGraph:
    """Executable compiled instance of a StateGraph."""

    def __init__(self, graph: StateGraph):
        self.graph = graph

    def invoke(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        max_iterations: int = 25,
        resume_from: Optional[GraphState] = None,
    ) -> GraphState:
        """
        Execute the compiled graph until reaching END, maximum iterations, or interrupt.
        """
        if resume_from:
            state = resume_from
            state.interrupted = False
            state.interrupt_reason = None
            current_node_name = state.current_node
            is_resuming = True
        else:
            state = GraphState(values=copy.deepcopy(initial_state or {}))
            current_node_name = self.graph.entry_point
            is_resuming = False

        while current_node_name != StateGraph.END and state.iteration < max_iterations:
            state.iteration += 1

            if current_node_name not in self.graph.nodes:
                raise RuntimeError(f"Unknown node '{current_node_name}' reached.")

            node = self.graph.nodes[current_node_name]

            # Check human-in-the-loop interruption
            if node.interrupt_before and not is_resuming:
                state.interrupted = True
                state.interrupt_reason = f"Interrupted before execution of node: {node.name}"
                state.current_node = current_node_name
                return state
            is_resuming = False

            # Execute node logic
            mutations = node.action(state)
            if mutations:
                state.update(mutations, node.name)

            # Determine next node: conditional router takes precedence over direct edge
            if current_node_name in self.graph.conditional_edges:
                router = self.graph.conditional_edges[current_node_name]
                current_node_name = router(state)
            elif current_node_name in self.graph.edges:
                current_node_name = self.graph.edges[current_node_name]
            else:
                # Default terminates if no outgoing edge is specified
                current_node_name = StateGraph.END

        state.current_node = current_node_name
        return state
