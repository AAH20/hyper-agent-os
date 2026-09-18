"""Unit tests for Cognee & MemGPT migration adapters."""

import unittest
from hyper_agent_os.adapters import CogneeCompat


class TestAdapters(unittest.TestCase):

    def test_cognee_compat_sync_pipeline(self):
        cognee = CogneeCompat()

        # 1. Add data
        cognee.add_sync([
            "Alice leads ProjectTitan.",
            "ProjectTitan builds AutonomousAgents.",
        ])

        # 2. Cognify
        res = cognee.cognify_sync()
        self.assertEqual(res["status"], "COMPLETED")
        self.assertGreaterEqual(res["facts_indexed"], 2)

        # 3. Search
        search_res = cognee.search_sync("Who leads ProjectTitan?", subject="Alice")
        self.assertGreaterEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["subject"], "Alice")
        self.assertEqual(search_res[0]["relationship"], "leads")

    def test_cognee_compat_prune(self):
        cognee = CogneeCompat()
        cognee.add_sync("Temporary uncommitted text.")
        self.assertEqual(len(cognee._staging_data), 1)

        cognee._staging_data.clear()
        self.assertEqual(len(cognee._staging_data), 0)


if __name__ == "__main__":
    unittest.main()
