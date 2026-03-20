"""
RagMark — Configuration loader and validator.

Reads ragmark.yaml and returns a validated RagMarkConfig dataclass.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml


_VALID_RAG_DOORS = {"pre", "post", "off"}


@dataclass
class RagMarkConfig:
    """Holds all RagMark configuration parameters."""

    enabled: bool = True
    max_bookmarks: int = 5
    similarity_threshold: float = 0.75
    rag_door: str = "pre"
    debug: bool = False

    def __post_init__(self) -> None:
        # --- type coercions & validations ---
        self.enabled = bool(self.enabled)
        self.debug = bool(self.debug)

        self.max_bookmarks = int(self.max_bookmarks)
        if self.max_bookmarks < 1:
            raise ValueError(f"max_bookmarks must be >= 1, got {self.max_bookmarks}")

        self.similarity_threshold = float(self.similarity_threshold)
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {self.similarity_threshold}"
            )

        self.rag_door = str(self.rag_door).lower().strip()
        if self.rag_door not in _VALID_RAG_DOORS:
            raise ValueError(
                f"rag_door must be one of {_VALID_RAG_DOORS}, got '{self.rag_door}'"
            )


def load_config(path: Union[str, Path]) -> RagMarkConfig:
    """Load and validate a RagMark YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    RagMarkConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If required keys are missing or values are out of range.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict) or "ragmark" not in raw:
        raise ValueError("YAML must contain a top-level 'ragmark' key.")

    section = raw["ragmark"]
    if not isinstance(section, dict):
        raise ValueError("'ragmark' key must map to a dictionary of settings.")

    return RagMarkConfig(**{k: v for k, v in section.items() if k in RagMarkConfig.__dataclass_fields__})
