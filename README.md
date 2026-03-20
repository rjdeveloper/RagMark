# RagMark

**Lightweight, configurable context bookmarking for RAG pipelines.**

RagMark is an in-memory layer that adds *context awareness* to any Retrieval-Augmented Generation system — enabling follow-up queries to continue from where they left off instead of restarting every time.

## Features

- ⚡ **< 1 ms** decision latency
- 🧠 Context-aware retrieval (local vs global search)
- 🔌 Plug-and-play — works with any RAG pipeline
- ⚙️ Fully configurable via a single YAML file
- 🚫 Zero disruption to existing pipelines

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Configure (`ragmark.yaml`)

```yaml
ragmark:
  enabled: true
  max_bookmarks: 5
  similarity_threshold: 0.75
  rag_door: pre        # pre | post | off
  debug: false
```

### 2. Use

```python
from ragmark import RagMark, Node

rm = RagMark(config_path="ragmark.yaml")

# Get retrieval strategy
scope = rm.get_scope(query_embedding)

if scope["strategy"] == "local":
    results = vector_db.search_near(scope["node_ids"])
else:
    results = vector_db.search(query_embedding)

# Update context with results
nodes = [Node(id=r.id, embedding=r.embedding) for r in results]
rm.update(nodes, query_embedding)
```

## Configuration Options

| Key                    | Type   | Default | Description                          |
|------------------------|--------|---------|--------------------------------------|
| `enabled`              | bool   | `true`  | Global on/off switch                 |
| `max_bookmarks`        | int    | `5`     | Max active context nodes             |
| `similarity_threshold` | float  | `0.75`  | Threshold for local search decision  |
| `rag_door`             | string | `pre`   | Pipeline placement: `pre`/`post`/`off` |
| `debug`                | bool   | `false` | Enable debug logging                 |

## API

| Method                          | Description                                  |
|---------------------------------|----------------------------------------------|
| `RagMark(config_path=...)`      | Initialise with a YAML config file           |
| `get_scope(query_embedding)`    | Returns strategy, confidence, and node IDs   |
| `update(nodes, query_embedding)`| Update bookmarks with new retrieval results  |
| `reset()`                       | Clear all bookmarks                          |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Author

**Rishabh**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/rishabh)

Built with ❤️ as a lightweight open-source contribution to the RAG ecosystem.

## License

MIT
