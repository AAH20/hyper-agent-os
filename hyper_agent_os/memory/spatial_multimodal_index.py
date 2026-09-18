"""
Spatial & Multimodal Memory Index.

Manages 3D bounding boxes, coordinate frames, spatial relationships
(e.g., inside, adjacent_to, distance_to), and multimodal vector embeddings
for physical, robotic, VLA, and simulation agents.
"""

from __future__ import annotations
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BoundingBox3D:
    """3D Axis-Aligned Bounding Box (AABB)."""
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )

    @property
    def volume(self) -> float:
        dx = max(0.0, self.max_x - self.min_x)
        dy = max(0.0, self.max_y - self.min_y)
        dz = max(0.0, self.max_z - self.min_z)
        return dx * dy * dz

    def contains(self, point: Tuple[float, float, float]) -> bool:
        px, py, pz = point
        return (
            self.min_x <= px <= self.max_x
            and self.min_y <= py <= self.max_y
            and self.min_z <= pz <= self.max_z
        )

    def intersects(self, other: BoundingBox3D) -> bool:
        return (
            self.min_x <= other.max_x
            and self.max_x >= other.min_x
            and self.min_y <= other.max_y
            and self.max_y >= other.min_y
            and self.min_z <= other.max_z
            and self.max_z >= other.min_z
        )

    def distance_to(self, point: Tuple[float, float, float]) -> float:
        cx, cy, cz = self.center
        px, py, pz = point
        return math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2)


@dataclass
class VectorEmbedding:
    """Multimodal vector representation (e.g., text, visual, tactile)."""
    vector: List[float]
    modality: str = "multimodal"  # "text", "vision", "audio", "vla"

    def cosine_similarity(self, other: VectorEmbedding) -> float:
        if len(self.vector) != len(other.vector):
            return 0.0
        dot = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = math.sqrt(sum(a * a for a in self.vector))
        norm_b = math.sqrt(sum(b * b for b in other.vector))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


@dataclass
class SpatialEntity:
    """An entity situated in physical or 3D coordinate space."""
    id: str
    name: str
    category: str
    bbox: BoundingBox3D
    embedding: Optional[VectorEmbedding] = None
    attributes: Dict[str, any] = field(default_factory=dict)


class SpatialMultimodalIndex:
    """Spatial scene graph and vector similarity index for VLA and 3D agents."""

    def __init__(self):
        self.entities: Dict[str, SpatialEntity] = {}

    def register_spatial_object(
        self,
        name: str,
        category: str,
        bbox: BoundingBox3D,
        embedding: Optional[List[float]] = None,
        modality: str = "vision",
        attributes: Optional[Dict] = None,
    ) -> SpatialEntity:
        """Add an object situated in 3D space with optional vector embeddings."""
        ent_id = f"spat_{uuid.uuid4().hex[:8]}"
        vec_emb = VectorEmbedding(vector=embedding, modality=modality) if embedding else None
        entity = SpatialEntity(
            id=ent_id,
            name=name,
            category=category,
            bbox=bbox,
            embedding=vec_emb,
            attributes=attributes or {},
        )
        self.entities[ent_id] = entity
        return entity

    def query_radius(
        self, center_point: Tuple[float, float, float], max_radius: float
    ) -> List[Tuple[SpatialEntity, float]]:
        """Find all objects within a radial sphere around a 3D coordinate."""
        results = []
        for ent in self.entities.values():
            dist = ent.bbox.distance_to(center_point)
            if dist <= max_radius:
                results.append((ent, dist))
        results.sort(key=lambda x: x[1])
        return results

    def query_vector_similarity(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[SpatialEntity, float]]:
        """Find most semantically similar objects via cosine similarity."""
        q_emb = VectorEmbedding(vector=query_vector)
        scored = []
        for ent in self.entities.values():
            if ent.embedding:
                score = q_emb.cosine_similarity(ent.embedding)
                scored.append((ent, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def compute_spatial_relationships(
        self, entity_a_id: str, entity_b_id: str
    ) -> Dict[str, any]:
        """Compute relative relationships between two spatial entities."""
        if entity_a_id not in self.entities or entity_b_id not in self.entities:
            return {"status": "error", "message": "Entities not found"}

        a = self.entities[entity_a_id]
        b = self.entities[entity_b_id]

        dist = a.bbox.distance_to(b.bbox.center)
        intersects = a.bbox.intersects(b.bbox)
        a_contains_b = a.bbox.contains(b.bbox.center)
        b_contains_a = b.bbox.contains(a.bbox.center)

        return {
            "entity_a": a.name,
            "entity_b": b.name,
            "center_distance": dist,
            "colliding": intersects,
            "a_contains_b": a_contains_b,
            "b_contains_a": b_contains_a,
        }
