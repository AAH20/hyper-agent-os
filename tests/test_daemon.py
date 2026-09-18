"""Unit tests for 24/7 Durable Daemons, Event Bus, and Health Monitoring."""

import time
import unittest
from hyper_agent_os.daemon import (
    DurableEngine,
    DurableWorkflow,
    WorkflowStatus,
    EventBus,
    AgentEvent,
    HealthMonitor,
)


class TestDaemonSubsystem(unittest.TestCase):

    def test_durable_workflow_execution_and_replay(self):
        engine = DurableEngine()
        wf = DurableWorkflow("sync-pipeline", engine)

        executions = {"step1": 0, "step2": 0}

        def step1_fn(state):
            executions["step1"] += 1
            return "step1_done"

        def step2_fn(state):
            executions["step2"] += 1
            return "step2_done"

        wf.add_step("s1", step1_fn)
        wf.add_step("s2", step2_fn)

        # Initial execution
        res1 = wf.execute({"initial": 123})
        wf_id = res1["workflow_id"]
        self.assertEqual(executions["step1"], 1)
        self.assertEqual(executions["step2"], 1)
        self.assertEqual(res1["status"], WorkflowStatus.COMPLETED.value)

        # Replay should NOT execute step functions again
        res2 = wf.execute({"initial": 123}, workflow_id=wf_id)
        self.assertEqual(executions["step1"], 1)
        self.assertEqual(executions["step2"], 1)
        self.assertEqual(res2["state"]["s1"], "step1_done")
        self.assertEqual(res2["state"]["s2"], "step2_done")

    def test_event_bus_pub_sub_wildcard(self):
        bus = EventBus()
        received = []

        def handler(event: AgentEvent):
            received.append(event)

        # Subscribe to pattern
        bus.subscribe("swarm.*.telemetry", handler)

        # Non-matching event
        bus.publish("cloud.cost.alert", {"cost": 50})
        self.assertEqual(len(received), 0)

        # Matching event
        bus.publish("swarm.agent_01.telemetry", {"cpu": 0.45})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["cpu"], 0.45)

    def test_health_monitor_watchdog_timeout(self):
        restarted_agents = []

        def restart_cb(agent_id: str):
            restarted_agents.append(agent_id)

        monitor = HealthMonitor(timeout_seconds=0.1, on_recovery=restart_cb)

        # Emit initial heartbeat
        monitor.record_heartbeat("worker_alpha")
        self.assertEqual(len(monitor.check_health()), 0)

        # Wait past timeout
        time.sleep(0.15)
        alerts = monitor.check_health()
        self.assertGreater(len(alerts), 0)
        self.assertIn("worker_alpha", restarted_agents)


if __name__ == "__main__":
    unittest.main()
