"""Unit tests for Model Context Protocol (MCP) Server."""

import unittest
from hyper_agent_os.mcp_server import HyperAgentMCPServer


class TestMCPServer(unittest.TestCase):

    def setUp(self):
        self.server = HyperAgentMCPServer()

    def test_initialize_protocol(self):
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
        resp = self.server.handle_request(req)
        self.assertEqual(resp["result"]["serverInfo"]["name"], "hyper-agent-os")
        self.assertIn("tools", resp["result"]["capabilities"])

    def test_tools_list(self):
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        resp = self.server.handle_request(req)
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertIn("hyper_query_memory", tool_names)
        self.assertIn("hyper_assert_fact", tool_names)
        self.assertIn("hyper_run_swarm", tool_names)
        self.assertIn("hyper_evaluate_cbf_safety", tool_names)

    def test_tools_call_assert_and_query(self):
        # 1. Assert fact via MCP
        assert_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "hyper_assert_fact",
                "arguments": {
                    "subject": "SystemA",
                    "predicate": "connects_to",
                    "target": "DatabaseB",
                },
            },
        }
        assert_resp = self.server.handle_request(assert_req)
        self.assertIn("Fact committed", assert_resp["result"]["content"][0]["text"])

        # 2. Query memory via MCP
        query_req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "hyper_query_memory",
                "arguments": {
                    "query": "What does SystemA connect to?",
                    "subject": "SystemA",
                },
            },
        }
        query_resp = self.server.handle_request(query_req)
        self.assertIn("DatabaseB", query_resp["result"]["content"][0]["text"])

    def test_tools_call_safety_filter(self):
        safe_req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "hyper_evaluate_cbf_safety",
                "arguments": {
                    "target_id": "robot_arm",
                    "vx": 0.5,
                    "vy": 0.5,
                    "force": 10.0,
                    "obstacle_dist": 1.5,
                },
            },
        }
        resp = self.server.handle_request(safe_req)
        self.assertIn("Was Modified: False", resp["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
