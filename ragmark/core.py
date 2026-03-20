"""
RagMark — Core engine.

Contains the Node data model and the main RagMark class that implements
context bookmarking, retrieval-strategy decisions, and context updates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .config import RagMarkConfig, load_config
from .utils import cosine_similarity

logger = logging.getLogger("ragmark")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A minimal retrieval node carrying an ID and its embedding vector."""

    id: str
    embedding: List[float] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Internal bookmark entry (node + cached similarity)
# ---------------------------------------------------------------------------

@dataclass
class _Bookmark:
    node: Node
    similarity: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RagMark:
    """In-memory context cursor for RAG pipelines.

    Parameters
    ----------
    config_path : str, optional
        Path to a ``ragmark.yaml`` file.  If omitted the instance is
        created with default settings.
    config : RagMarkConfig, optional
        Pre-built config object (takes precedence over *config_path*).

    Examples
    --------
    >>> rm = RagMark(config_path="ragmark.yaml")
    >>> scope = rm.get_scope([0.1, 0.2, 0.3])
    >>> rm.update([Node("a", [0.1, 0.2, 0.3])], [0.1, 0.2, 0.3])
    >>> rm.reset()
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[RagMarkConfig] = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        elif config_path is not None:
            self._cfg = load_config(config_path)
        else:
            self._cfg = RagMarkConfig()  # all defaults

        self._bookmarks: List[_Bookmark] = []

        if self._cfg.debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        logger.debug(
            "RagMark initialised | enabled=%s | rag_door=%s | "
            "max_bookmarks=%d | threshold=%.3f",
            self._cfg.enabled,
            self._cfg.rag_door,
            self._cfg.max_bookmarks,
            self._cfg.similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> RagMarkConfig:
        """Return the active configuration (read-only view)."""
        return self._cfg

    @property
    def bookmark_count(self) -> int:
        """Number of active bookmarks."""
        return len(self._bookmarks)

    @property
    def bookmark_ids(self) -> List[str]:
        """IDs of all currently stored bookmarks."""
        return [bm.node.id for bm in self._bookmarks]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_scope(self, query_embedding: List[float]) -> Dict[str, Any]:
        """Decide retrieval strategy for the given query.

        Returns
        -------
        dict
            ``strategy``  – ``"local"`` or ``"global"``
            ``confidence`` – highest similarity to any bookmark (0.0 if none)
            ``node_ids``   – list of bookmark IDs (empty when global)
        """
        # --- pass-through cases ---
        if not self._cfg.enabled or self._cfg.rag_door == "off":
            logger.debug("Pass-through (enabled=%s, rag_door=%s)", self._cfg.enabled, self._cfg.rag_door)
            return {"strategy": "global", "confidence": 0.0, "node_ids": []}

        if self._cfg.rag_door == "post":
            # Post-retrieval mode: no decision-making, only tracking.
            logger.debug("Post mode — returning global (tracking only)")
            return {"strategy": "global", "confidence": 0.0, "node_ids": []}

        # --- pre-retrieval decision ---
        if not self._bookmarks:
            logger.debug("No bookmarks yet — global search")
            return {"strategy": "global", "confidence": 0.0, "node_ids": []}

        best_sim = 0.0
        for bm in self._bookmarks:
            try:
                sim = cosine_similarity(query_embedding, bm.node.embedding)
            except ValueError:
                sim = 0.0
            if sim > best_sim:
                best_sim = sim

        if best_sim >= self._cfg.similarity_threshold:
            node_ids = [bm.node.id for bm in self._bookmarks]
            logger.debug("LOCAL search — confidence=%.4f, nodes=%s", best_sim, node_ids)
            return {"strategy": "local", "confidence": best_sim, "node_ids": node_ids}

        logger.debug("GLOBAL search — best similarity=%.4f < threshold=%.3f", best_sim, self._cfg.similarity_threshold)
        return {"strategy": "global", "confidence": best_sim, "node_ids": []}

    def update(
        self,
        nodes: List[Node],
        query_embedding: List[float],
    ) -> None:
        """Update the bookmark context with new retrieval nodes.

        Merges *nodes* into the existing bookmarks, (re-)ranks every
        bookmark by similarity to *query_embedding*, and keeps the
        top-K entries (``max_bookmarks``).
        """
        if not self._cfg.enabled or self._cfg.rag_door == "off":
            logger.debug("Update skipped (disabled / off)")
            return

        # Merge new nodes (replace duplicates by id)
        existing_ids = {bm.node.id for bm in self._bookmarks}
        for node in nodes:
            if node.id in existing_ids:
                # Replace the old bookmark for this id
                self._bookmarks = [bm for bm in self._bookmarks if bm.node.id != node.id]
            self._bookmarks.append(_Bookmark(node=node))
            existing_ids.add(node.id)

        # Re-score all bookmarks against the latest query
        for bm in self._bookmarks:
            try:
                bm.similarity = cosine_similarity(query_embedding, bm.node.embedding)
            except ValueError:
                bm.similarity = 0.0

        # Sort descending by similarity and keep top-K
        self._bookmarks.sort(key=lambda bm: bm.similarity, reverse=True)
        self._bookmarks = self._bookmarks[: self._cfg.max_bookmarks]

        logger.debug(
            "Bookmarks updated — kept %d: %s",
            len(self._bookmarks),
            [(bm.node.id, round(bm.similarity, 4)) for bm in self._bookmarks],
        )

    def reset(self) -> None:
        """Clear all bookmarks."""
        self._bookmarks.clear()
        logger.debug("Bookmarks cleared")
