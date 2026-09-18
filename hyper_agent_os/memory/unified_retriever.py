"""
Unified Multi-Hop Hybrid Retriever.

Combines knowledge graph relational traversal with vector semantic similarity
and temporal filters to produce cited, verifiable context for reasoning agents.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from .bi_temporal_graph import BiTemporalKnowledgeGraph, TemporalQuery
from .spatial_multimodal_index import SpatialMultimodalIndex


@dataclass
class Citation:
    """Provenance citation for retrieved evidence."""
    source_type: str  # "graph_fact", "spatial_object", "vector_chunk"
    identifier: str
    summary: str
    confidence: float
    timestamp: float


@dataclass
class RetrievalResult:
    """Output from unified multi-hop retrieval."""
    query: str
    facts: List[Dict[str, any]]
    graph_paths: List[Dict[str, any]]
    spatial_entities: List[Dict[str, any]]
    citations: List[Citation]
    synthesized_context: str
    execution_time_ms: float


class UnifiedRetriever:
    """
    Orchestrates hybrid retrieval across relational graph, temporal facts,
    and spatial embeddings.
    """

    def __init__(
        self,
        knowledge_graph: BiTemporalKnowledgeGraph,
        spatial_index: Optional[SpatialMultimodalIndex] = None,
    ):
        self.graph = knowledge_graph
        self.spatial = spatial_index or SpatialMultimodalIndex()

    def retrieve(
        self,
        query: str,
        subject: Optional[str] = None,
        max_hops: int = 2,
        as_of_time: Optional[float] = None,
        query_vector: Optional[List[float]] = None,
        top_k_spatial: int = 3,
    ) -> RetrievalResult:
        """
        Execute unified hybrid multi-hop retrieval.
        """
        start_time = time.time()
        citations: List[Citation] = []
        retrieved_facts: List[Dict[str, any]] = []
        graph_paths: List[Dict[str, any]] = []
        spatial_results: List[Dict[str, any]] = []

        now = as_of_time if as_of_time is not None else time.time()

        # 1. Fact matching by subject or predicate
        if subject:
            facts = self.graph.query(
                TemporalQuery(
                    as_of_tx_time=now,
                    as_of_valid_time=now,
                    subject=subject,
                )
            )
            for f in facts:
                fact_dict = {
                    "id": f.id,
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "target": f.target,
                    "confidence": f.confidence,
                    "source": f.source,
                }
                retrieved_facts.append(fact_dict)
                citations.append(
                    Citation(
                        source_type="graph_fact",
                        identifier=f.id,
                        summary=f"{f.subject} -> {f.predicate} -> {f.target}",
                        confidence=f.confidence,
                        timestamp=f.tx_from,
                    )
                )

            # 2. Multi-hop traversal
            paths = self.graph.traverse(
                start_subject=subject,
                max_hops=max_hops,
                as_of_time=now,
            )
            graph_paths.extend(paths)

        # 3. Spatial & Multimodal Vector Search if vector provided
        if query_vector:
            sim_objects = self.spatial.query_vector_similarity(
                query_vector=query_vector, top_k=top_k_spatial
            )
            for ent, sim_score in sim_objects:
                spatial_dict = {
                    "id": ent.id,
                    "name": ent.name,
                    "category": ent.category,
                    "similarity": sim_score,
                    "center": ent.bbox.center,
                }
                spatial_results.append(spatial_dict)
                citations.append(
                    Citation(
                        source_type="spatial_object",
                        identifier=ent.id,
                        summary=f"{ent.name} ({ent.category}) similarity: {sim_score:.3f}",
                        confidence=sim_score,
                        timestamp=now,
                    )
                )

        # 4. Synthesize context paragraph
        context_lines = [f"### Context for query: '{query}'"]
        if retrieved_facts:
            context_lines.append("\n**Active Knowledge Graph Facts:**")
            for f in retrieved_facts:
                context_lines.append(
                    f"- [{f['id']}] {f['subject']} {f['predicate']} {f['target']} (confidence: {f['confidence']:.2f})"
                )

        if graph_paths:
            context_lines.append("\n**Multi-Hop Relational Traversal:**")
            for p in graph_paths:
                chain = " -> ".join(
                    f"{step['from']} --({step['predicate']})--> {step['to']}"
                    for step in p["path"]
                )
                context_lines.append(f"- Path: {chain} ({p['hops']} hops)")

        if spatial_results:
            context_lines.append("\n**Spatial/Multimodal Perceptions:**")
            for s in spatial_results:
                context_lines.append(
                    f"- {s['name']} [{s['category']}] at {s['center']} (score: {s['similarity']:.2f})"
                )

        duration_ms = (time.time() - start_time) * 1000.0

        return RetrievalResult(
            query=query,
            facts=retrieved_facts,
            graph_paths=graph_paths,
            spatial_entities=spatial_results,
            citations=citations,
            synthesized_context="\n".join(context_lines),
            execution_time_ms=duration_ms,
        )
