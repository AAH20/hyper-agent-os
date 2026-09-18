"""
Cognee & MemGPT Drop-in Migration Adapter.

Provides seamless backward compatibility with Cognee's `add()`, `cognify()`,
and `search()` API while transparently routing data into Hyper-Agent OS's
Bi-Temporal Knowledge Graph and Unified Hybrid Retriever.
"""

from __future__ import annotations
import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Union

from hyper_agent_os.memory import (
    BiTemporalKnowledgeGraph,
    SpatialMultimodalIndex,
    UnifiedRetriever,
    TemporalQuery,
)


class CogneeCompat:
    """
    Drop-in replacement for the Cognee memory engine.
    
    Usage:
        from hyper_agent_os.adapters import CogneeCompat
        
        cognee = CogneeCompat()
        await cognee.add(["Alice works at QuantumCorp.", "QuantumCorp is located in Zurich."])
        await cognee.cognify()
        results = await cognee.search("Where does Alice work?")
    """

    def __init__(
        self,
        knowledge_graph: Optional[BiTemporalKnowledgeGraph] = None,
        spatial_index: Optional[SpatialMultimodalIndex] = None,
    ):
        self.graph = knowledge_graph or BiTemporalKnowledgeGraph()
        self.spatial = spatial_index or SpatialMultimodalIndex()
        self.retriever = UnifiedRetriever(self.graph, self.spatial)
        self._staging_data: List[str] = []

    async def add(self, data: Union[str, List[str]], dataset_name: str = "default_dataset") -> None:
        """Stage raw documents or text snippets for cognification."""
        if isinstance(data, str):
            self._staging_data.append(data)
        elif isinstance(data, list):
            self._staging_data.extend([str(item) for item in data])

    def add_sync(self, data: Union[str, List[str]], dataset_name: str = "default_dataset") -> None:
        """Synchronous variant of add()."""
        if isinstance(data, str):
            self._staging_data.append(data)
        elif isinstance(data, list):
            self._staging_data.extend([str(item) for item in data])

    async def cognify(self) -> Dict[str, Any]:
        """
        Execute the 'Extract, Cognify, Load' (ECL) pipeline.
        Extracts semantic triples (subject, predicate, object) from staged text
        and commits them as bi-temporal facts into the knowledge graph.
        """
        return self.cognify_sync()

    def cognify_sync(self) -> Dict[str, Any]:
        """Synchronous variant of cognify()."""
        facts_created = 0
        now = time.time()

        for text in self._staging_data:
            # High-performance rule-based triple extraction heuristic
            # (Matches: [Subject] [verb/relation] [Object])
            sentences = re.split(r"[.!?]\s*", text)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue

                words = s.split()
                if len(words) >= 3:
                    subject = words[0].strip('",\'')
                    predicate = words[1].lower().strip('",\'')
                    target = " ".join(words[2:]).strip('",\'')

                    self.graph.assert_fact(
                        subject=subject,
                        predicate=predicate,
                        target=target,
                        valid_from=now,
                        source="cognee_ecl_pipeline",
                    )
                    facts_created += 1

        self._staging_data.clear()
        return {
            "status": "COMPLETED",
            "facts_indexed": facts_created,
            "graph_summary": self.graph.export_graph_summary(),
        }

    async def search(
        self,
        query_text: str,
        query_type: str = "HYBRID",
        subject: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query the memory substrate using unified graph + vector retrieval.
        """
        return self.search_sync(query_text, query_type=query_type, subject=subject, top_k=top_k)

    def search_sync(
        self,
        query_text: str,
        query_type: str = "HYBRID",
        subject: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Synchronous variant of search()."""
        # Infer subject if not explicitly specified
        inferred_subject = subject
        if not inferred_subject:
            words = query_text.split()
            for w in words:
                clean = w.strip("?,.'\"")
                if clean in self.graph._subject_index:
                    inferred_subject = clean
                    break

        res = self.retriever.retrieve(
            query=query_text,
            subject=inferred_subject,
            max_hops=2,
        )

        formatted_results = []
        for f in res.facts[:top_k]:
            formatted_results.append({
                "id": f["id"],
                "subject": f["subject"],
                "relationship": f["predicate"],
                "target": f["target"],
                "confidence": f["confidence"],
                "source": f["source"],
            })

        for p in res.graph_paths[:top_k]:
            formatted_results.append({
                "type": "multi_hop_path",
                "hops": p["hops"],
                "target": p["target"],
                "path": p["path"],
            })

        return formatted_results

    async def prune(self) -> Dict[str, Any]:
        """Reset and wipe staged data."""
        self._staging_data.clear()
        return {"status": "SUCCESS", "message": "Staged memory cleared."}
