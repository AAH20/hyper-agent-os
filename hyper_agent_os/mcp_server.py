"""
Model Context Protocol (MCP) Server for Hyper-Agent OS.

Implements the official JSON-RPC 2.0 stdio protocol, exposing Hyper-Agent OS
Super-Memory, Swarms, Safety Governors, and Benchmarks as native tools to
Cursor, Windsurf, Claude Desktop, Antigravity, and other MCP clients.
"""

from __future__ import annotations
import json
import sys
from typing import Any, Dict, List, Optional

from hyper_agent_os.memory import (
    BiTemporalKnowledgeGraph,
    SpatialMultimodalIndex,
    UnifiedRetriever,
)
from hyper_agent_os.safety import (
    ControlBarrierFunction,
    SafetyEnvelope,
    ActuatorCommand,
    AegisAttestationBridge,
)
from hyper_agent_os.swarm import (
    StateGraph,
    SwarmAgent,
    SwarmTask,
    SwarmCrew,
    ProcessType,
)
from hyper_agent_os.benchmarks import (
    SWEBenchTask,
    SWEBenchRunner,
    MLEBenchCompetition,
    MLEBenchRunner,
)


class HyperAgentMCPServer:
    """
    Standard JSON-RPC 2.0 Model Context Protocol Server over stdio.
    """

    def __init__(self):
        self.kg = BiTemporalKnowledgeGraph()
        self.spatial = SpatialMultimodalIndex()
        self.retriever = UnifiedRetriever(self.kg, self.spatial)
        self.cbf = ControlBarrierFunction()
        self.aegis = AegisAttestationBridge()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the catalog of available MCP tools."""
        return [
            {
                "name": "hyper_query_memory",
                "description": "Query the Bi-Temporal Knowledge Graph and Multimodal Memory for facts, relations, and citations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The natural language query."},
                        "subject": {"type": "string", "description": "Optional starting subject node for multi-hop graph traversal."},
                        "max_hops": {"type": "integer", "description": "Maximum relational graph traversal depth (default 2).", "default": 2},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "hyper_assert_fact",
                "description": "Commit a verified, persistent bi-temporal fact into the agent knowledge graph.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Subject entity."},
                        "predicate": {"type": "string", "description": "Relational verb or predicate."},
                        "target": {"type": "string", "description": "Target entity or value."},
                        "confidence": {"type": "number", "description": "Confidence score 0.0 to 1.0 (default 1.0).", "default": 1.0},
                    },
                    "required": ["subject", "predicate", "target"],
                },
            },
            {
                "name": "hyper_run_swarm",
                "description": "Execute a collaborative multi-agent swarm task using LangGraph and CrewAI patterns.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_description": {"type": "string", "description": "High-level goal or task prompt."},
                        "roles": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Agent roles to spawn (e.g. ['Planner', 'Executor', 'Auditor']).",
                        },
                    },
                    "required": ["task_description"],
                },
            },
            {
                "name": "hyper_evaluate_cbf_safety",
                "description": "Verify and mathematically filter proposed actuator actions using Control Barrier Functions (CBF).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target_id": {"type": "string", "description": "Actuator or critical device ID."},
                        "vx": {"type": "number", "description": "Proposed linear X velocity (m/s)."},
                        "vy": {"type": "number", "description": "Proposed linear Y velocity (m/s)."},
                        "force": {"type": "number", "description": "Proposed force/torque (N)."},
                        "obstacle_dist": {"type": "number", "description": "Distance to nearest obstacle (m)."},
                    },
                    "required": ["target_id", "vx", "vy", "force", "obstacle_dist"],
                },
            },
        ]

    def handle_call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch and execute an MCP tool invocation."""
        if tool_name == "hyper_query_memory":
            query = args["query"]
            subject = args.get("subject")
            max_hops = args.get("max_hops", 2)
            result = self.retriever.retrieve(query, subject=subject, max_hops=max_hops)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Execution Time: {result.execution_time_ms:.2f}ms\n\n{result.synthesized_context}",
                    }
                ]
            }

        elif tool_name == "hyper_assert_fact":
            fact = self.kg.assert_fact(
                subject=args["subject"],
                predicate=args["predicate"],
                target=args["target"],
                confidence=args.get("confidence", 1.0),
                source="mcp_tool_call",
            )
            # Create cryptographic audit receipt
            receipt = self.aegis.create_receipt(
                action_name="hyper_assert_fact",
                input_data=args,
                output_data={"fact_id": fact.id},
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Fact committed: [{fact.id}] {fact.subject} {fact.predicate} {fact.target}\nAegis Receipt: {receipt.receipt_id} (Hash: {receipt.receipt_hash[:16]}...)",
                    }
                ]
            }

        elif tool_name == "hyper_run_swarm":
            task_desc = args["task_description"]
            roles = args.get("roles", ["Planner", "Executor", "Auditor"])

            agents = [
                SwarmAgent(
                    role=r,
                    goal=f"Achieve success as {r}",
                    backstory=f"Autonomous {r} agent",
                    execute_fn=lambda prompt, ctx, r=r: f"[{r}] Processed: {prompt[:30]}...",
                )
                for r in roles
            ]
            tasks = [
                SwarmTask(
                    description=f"{task_desc} - Phase for {a.role}",
                    expected_output=f"Output from {a.role}",
                    agent=a,
                )
                for a in agents
            ]
            crew = SwarmCrew(agents=agents, tasks=tasks, process=ProcessType.SEQUENTIAL)
            out = crew.kickoff()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Swarm Execution Completed:\nStatus: {out['status']}\nTasks Completed: {out['tasks_completed']}\nFinal Output:\n{out['final_output']}",
                    }
                ]
            }

        elif tool_name == "hyper_evaluate_cbf_safety":
            cmd = ActuatorCommand(
                target_id=args["target_id"],
                vx=float(args["vx"]),
                vy=float(args["vy"]),
                omega=0.0,
                force=float(args["force"]),
                nearest_obstacle_distance=float(args["obstacle_dist"]),
            )
            filtered = self.cbf.filter_action(cmd)
            receipt = self.aegis.create_receipt(
                action_name="evaluate_cbf_safety",
                input_data=args,
                output_data={"was_modified": filtered.was_modified, "safe_vx": filtered.safe_command.vx},
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Safety Check Completed:\n"
                            f"- Was Modified: {filtered.was_modified}\n"
                            f"- Barrier Value h(x): {filtered.barrier_value:.3f}\n"
                            f"- Safe Output: vx={filtered.safe_command.vx:.2f}, vy={filtered.safe_command.vy:.2f}, force={filtered.safe_command.force:.2f}\n"
                            f"- Rejection Reasons: {filtered.rejection_reasons or 'None (Command Safe)'}\n"
                            f"- Attestation Receipt: {receipt.receipt_id}"
                        ),
                    }
                ]
            }

        raise ValueError(f"Unknown tool: {tool_name}")

    def handle_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single JSON-RPC 2.0 request."""
        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "hyper-agent-os",
                        "version": "0.1.0",
                    },
                    "capabilities": {
                        "tools": {},
                    },
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": self.get_tool_definitions(),
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                res = self.handle_call_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": res,
                }
            except Exception as err:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32603,
                        "message": str(err),
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not found",
            },
        }

    def run_stdio(self):
        """Run the JSON-RPC stdio listening loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def main():
    server = HyperAgentMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
