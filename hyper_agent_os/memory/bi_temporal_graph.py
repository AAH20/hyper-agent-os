"""
Bi-Temporal Knowledge Graph Memory Substrate.

Enables agents to maintain persistent facts across time while explicitly
separating validity time (when a fact was true in the real world) from
transaction time (when the agent learned/asserted it).

This allows exact point-in-time reconstruction, retroactive corrections,
and clean rollback of hallucinations without corrupting history.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Entity:
    """An entity node in the knowledge graph."""
    id: str
    name: str
    entity_type: str
    properties: Dict[str, any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class Relation:
    """A semantic relationship between two entities."""
    id: str
    subject_id: str
    predicate: str
    object_id: str
    properties: Dict[str, any] = field(default_factory=dict)


@dataclass
class Fact:
    """
    A bi-temporal fact representation.
    
    Attributes:
        id: Unique identifier for the assertion.
        subject: Subject entity ID or name.
        predicate: Relation or predicate string.
        target: Object entity ID or value.
        valid_from: Epoch timestamp when the fact became true in reality.
        valid_to: Epoch timestamp when the fact ceased being true (None if still active).
        tx_from: Epoch timestamp when this record was written into memory.
        tx_to: Epoch timestamp when this record was superseded/retracted (None if currently believed).
        confidence: Certainty score (0.0 to 1.0).
        source: Provenance attribution (e.g., tool ID, user prompt, sensor).
        is_active: Whether currently believed.
    """
    id: str
    subject: str
    predicate: str
    target: str
    valid_from: float
    valid_to: Optional[float] = None
    tx_from: float = field(default_factory=time.time)
    tx_to: Optional[float] = None
    confidence: float = 1.0
    source: str = "agent"
    is_active: bool = True

    def is_valid_at(self, timestamp: float) -> bool:
        """Check if fact was valid in reality at a given timestamp."""
        if self.valid_from > timestamp:
            return False
        if self.valid_to is not None and self.valid_to <= timestamp:
            return False
        return True

    def was_known_at(self, tx_timestamp: float) -> bool:
        """Check if system knew this fact at a given transaction timestamp."""
        if self.tx_from > tx_timestamp:
            return False
        if self.tx_to is not None and self.tx_to <= tx_timestamp:
            return False
        return True


@dataclass
class TemporalQuery:
    """Parameters for bi-temporal graph queries."""
    as_of_valid_time: Optional[float] = None
    as_of_tx_time: Optional[float] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    target: Optional[str] = None


class BiTemporalKnowledgeGraph:
    """
    In-memory, persistent-capable Bi-Temporal Knowledge Graph for AI Agents.
    """

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.facts: Dict[str, Fact] = {}
        self._subject_index: Dict[str, Set[str]] = {}
        self._predicate_index: Dict[str, Set[str]] = {}
        self._target_index: Dict[str, Set[str]] = {}
        self._adjacency: Dict[str, Set[Tuple[str, str, str]]] = {}  # subject -> (predicate, target, fact_id)

    def add_entity(self, name: str, entity_type: str, properties: Optional[Dict] = None) -> Entity:
        """Register or retrieve an entity node."""
        for ent in self.entities.values():
            if ent.name == name and ent.entity_type == entity_type:
                if properties:
                    ent.properties.update(properties)
                return ent

        ent_id = f"ent_{uuid.uuid4().hex[:8]}"
        entity = Entity(
            id=ent_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
        )
        self.entities[ent_id] = entity
        return entity

    def assert_fact(
        self,
        subject: str,
        predicate: str,
        target: str,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        confidence: float = 1.0,
        source: str = "agent",
    ) -> Fact:
        """
        Record a new fact assertion with bi-temporal attribution.
        Automatically supersedes conflicting active facts with the same subject and predicate.
        """
        now = time.time()
        vf = valid_from if valid_from is not None else now
        fact_id = f"fact_{uuid.uuid4().hex[:8]}"

        # Close out previous conflicting active facts if needed
        existing_facts = self.query(
            TemporalQuery(
                as_of_tx_time=now,
                as_of_valid_time=vf,
                subject=subject,
                predicate=predicate,
            )
        )
        for old_fact in existing_facts:
            if old_fact.target != target and old_fact.is_active:
                old_fact.tx_to = now
                old_fact.valid_to = min(old_fact.valid_to or float("inf"), vf)
                old_fact.is_active = False

        fact = Fact(
            id=fact_id,
            subject=subject,
            predicate=predicate,
            target=target,
            valid_from=vf,
            valid_to=valid_to,
            tx_from=now,
            confidence=confidence,
            source=source,
            is_active=True,
        )

        self.facts[fact_id] = fact
        self._subject_index.setdefault(subject, set()).add(fact_id)
        self._predicate_index.setdefault(predicate, set()).add(fact_id)
        self._target_index.setdefault(target, set()).add(fact_id)
        self._adjacency.setdefault(subject, set()).add((predicate, target, fact_id))

        return fact

    def retract_fact(self, fact_id: str, reason: str = "hallucination_correction") -> bool:
        """
        Retract a fact immediately. Records transaction retraction time.
        """
        if fact_id not in self.facts:
            return False
        fact = self.facts[fact_id]
        now = time.time()
        fact.tx_to = now
        fact.is_active = False
        return True

    def query(self, query: TemporalQuery) -> List[Fact]:
        """
        Execute a bi-temporal query over facts.
        """
        now = time.time()
        tx_time = query.as_of_tx_time if query.as_of_tx_time is not None else now
        val_time = query.as_of_valid_time if query.as_of_valid_time is not None else now

        candidate_ids: Optional[Set[str]] = None

        if query.subject is not None:
            candidate_ids = set(self._subject_index.get(query.subject, []))
        if query.predicate is not None:
            preds = self._predicate_index.get(query.predicate, set())
            candidate_ids = preds if candidate_ids is None else candidate_ids & preds
        if query.target is not None:
            targs = self._target_index.get(query.target, set())
            candidate_ids = targs if candidate_ids is None else candidate_ids & targs

        if candidate_ids is None:
            candidate_ids = set(self.facts.keys())

        results = []
        for fid in candidate_ids:
            fact = self.facts[fid]
            if fact.was_known_at(tx_time) and fact.is_valid_at(val_time):
                results.append(fact)

        return results

    def traverse(
        self,
        start_subject: str,
        max_hops: int = 2,
        as_of_time: Optional[float] = None,
    ) -> List[Dict[str, any]]:
        """
        Perform a multi-hop graph traversal from a start node.
        Returns paths with predicates, destinations, and fact provenance.
        """
        target_time = as_of_time if as_of_time is not None else time.time()
        visited = set()
        paths: List[Dict[str, any]] = []

        def dfs(current: str, depth: int, current_path: List[Dict]):
            if depth > max_hops or current in visited:
                return
            visited.add(current)

            edges = self._adjacency.get(current, set())
            for pred, obj, fid in edges:
                fact = self.facts[fid]
                if fact.was_known_at(target_time) and fact.is_valid_at(target_time):
                    step = {
                        "from": current,
                        "predicate": pred,
                        "to": obj,
                        "fact_id": fid,
                        "confidence": fact.confidence,
                        "source": fact.source,
                    }
                    new_path = current_path + [step]
                    paths.append({"path": new_path, "target": obj, "hops": depth})
                    dfs(obj, depth + 1, new_path)

        dfs(start_subject, 1, [])
        return paths

    def export_graph_summary(self) -> Dict[str, any]:
        """Return high-level statistics of the knowledge graph."""
        active_facts = [f for f in self.facts.values() if f.is_active]
        return {
            "entity_count": len(self.entities),
            "total_facts": len(self.facts),
            "active_facts": len(active_facts),
            "predicates": list(self._predicate_index.keys()),
        }
