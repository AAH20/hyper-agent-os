"""Unit tests for LangGraph StateGraph, CrewAI Swarms, and Byzantine Quorum."""

import unittest
from hyper_agent_os.swarm import (
    StateGraph,
    SwarmAgent,
    SwarmTask,
    SwarmCrew,
    ProcessType,
    ByzantineQuorum,
    VoteEnvelope,
)


class TestSwarmSubsystem(unittest.TestCase):

    def test_state_graph_cyclical_flow(self):
        graph = StateGraph()

        def increment_node(state):
            count = state.values.get("count", 0) + 1
            return {"count": count}

        def router(state):
            if state.values.get("count", 0) >= 3:
                return StateGraph.END
            return "loop_node"

        graph.add_node("loop_node", increment_node)
        graph.set_entry_point("loop_node")
        graph.add_conditional_edges("loop_node", router)

        compiled = graph.compile()
        final_state = compiled.invoke({"count": 0}, max_iterations=10)

        self.assertEqual(final_state.values["count"], 3)
        self.assertEqual(final_state.iteration, 3)

    def test_state_graph_human_interrupt(self):
        graph = StateGraph()
        graph.add_node("safe_step", lambda s: {"step1": True})
        graph.add_node("critical_tool", lambda s: {"executed": True}, interrupt_before=True)
        graph.set_entry_point("safe_step")
        graph.add_edge("safe_step", "critical_tool")
        graph.add_edge("critical_tool", StateGraph.END)

        compiled = graph.compile()
        state = compiled.invoke({})

        # Should pause right before executing 'critical_tool'
        self.assertTrue(state.interrupted)
        self.assertIn("critical_tool", state.interrupt_reason)
        self.assertNotIn("executed", state.values)

        # Resume execution
        resumed = compiled.invoke(resume_from=state)
        self.assertFalse(resumed.interrupted)
        self.assertTrue(resumed.values.get("executed"))

    def test_crew_sequential_execution(self):
        a1 = SwarmAgent(
            role="Researcher",
            goal="Find facts",
            backstory="Experienced investigator",
            execute_fn=lambda prompt, ctx: "Report: System stable.",
        )
        a2 = SwarmAgent(
            role="Summarizer",
            goal="Condense facts",
            backstory="Editor",
            execute_fn=lambda prompt, ctx: f"Summary: {ctx.get('task_t1_output', '')}",
        )

        t1 = SwarmTask("Gather status", "status", agent=a1, task_id="t1")
        t2 = SwarmTask("Summarize status", "summary", agent=a2, task_id="t2")

        crew = SwarmCrew(agents=[a1, a2], tasks=[t1, t2], process=ProcessType.SEQUENTIAL)
        output = crew.kickoff()

        self.assertEqual(output["status"], "SUCCESS")
        self.assertEqual(output["tasks_completed"], 2)
        self.assertIn("Report: System stable.", output["final_output"])

    def test_byzantine_quorum_consensus(self):
        nodes = {"agent_a", "agent_b", "agent_c", "agent_d"}
        quorum = ByzantineQuorum(authorized_nodes=nodes)
        self.assertEqual(quorum.quorum_threshold, 3)  # floor(8/3) + 1 = 3

        # 3 honest approve, 1 honest reject
        votes = [
            VoteEnvelope("agent_a", "prop_01", True),
            VoteEnvelope("agent_b", "prop_01", True),
            VoteEnvelope("agent_c", "prop_01", True),
            VoteEnvelope("agent_d", "prop_01", False),
        ]
        for v in votes:
            v.sign()

        decision = quorum.evaluate_proposal("prop_01", votes)
        self.assertTrue(decision.decided)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.votes_for, 3)
        self.assertEqual(len(decision.byzantine_nodes), 0)

    def test_byzantine_fault_detection(self):
        nodes = {"agent_a", "agent_b", "agent_c", "agent_d"}
        quorum = ByzantineQuorum(authorized_nodes=nodes)

        # Byzantine attempt: agent_x unauthorized, agent_a double-voting
        v1 = VoteEnvelope("agent_a", "prop_02", True)
        v2 = VoteEnvelope("agent_a", "prop_02", False)  # double voting
        v3 = VoteEnvelope("agent_x", "prop_02", True)   # unauthorized node
        for v in [v1, v2, v3]:
            v.sign()

        decision = quorum.evaluate_proposal("prop_02", [v1, v2, v3])
        self.assertFalse(decision.approved)
        self.assertGreater(len(decision.byzantine_nodes), 0)


if __name__ == "__main__":
    unittest.main()
