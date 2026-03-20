"""
RagMark — Lightweight, configurable context bookmarking for RAG pipelines.

Public API
----------
RagMark        Main class — initialise, get_scope, update, reset.
Node           Minimal retrieval-node data model.
RagMarkConfig  Configuration dataclass.
load_config    Helper to read a ragmark.yaml file.
"""

__version__ = "1.1.0"

from .config import RagMarkConfig, load_config
from .core import Node, RagMark

__all__ = ["RagMark", "Node", "RagMarkConfig", "load_config", "__version__"]
