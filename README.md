# Agent Memory System

A simple, production-inspired agent memory system built in Python with SQLite.

Inspired by [OpenClaw's memory architecture](https://github.com/openclaw/openclaw) and Google's *"Context Engineering: Sessions & Memory"* whitepaper (Nov 2025).

## Architecture

```
LogActivity()
     │
     ▼
 Raw activity stored
     │
     ▼
Claude extracts memory facts     ← LLM (Anthropic)
     │
     ▼
Consolidation check              ← FTS search + LLM decision
  (update existing or create new)
     │
     ├── SQLite memories table
     ├── FTS5 virtual table      ← BM25 keyword index
     └── Embedding table         ← cosine similarity index
                                    (all-MiniLM-L6-v2, local)
```

### Memory Taxonomy (Google / OpenClaw)
| Type | What it stores | Example |
|------|---------------|---------|
| **episodic** | Events and interactions | "Attended standup on Mar 1" |
| **semantic** | Facts, preferences, knowledge | "Team uses Python for agents" |
| **procedural** | Workflows and routines | "Deploy: run tests → build → push" |

### Hybrid Search
Retrieval combines **BM25** (SQLite FTS5 with Porter stemmer) and **vector cosine similarity** (sentence-transformers embeddings), weighted 40/60 — same approach as OpenClaw's `mergeHybridResults`.

## API

```python
from memory import MemorySystem

mem = MemorySystem()

# Log an activity — extracts and stores memory facts
result = mem.log_activity(datetime.now(), "Decided to use Python for new agent module.")

# Summarise memories on a topic
summary = mem.summarize_memory("coding")

# Summarise all topics
overview = mem.summarize_memory()

# Retrieve memories relevant to a context
memories = mem.context_memory("authentication and secrets management")
# Returns: [{"id", "content", "topic", "memory_type", "score"}, ...]
```

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY

# Run the CLI demo
python cli.py demo

# Or use individual commands
python cli.py log "Team agreed to use SQLite for agent state storage."
python cli.py summarize coding
python cli.py context "What do I know about security?"
python cli.py stats
```

## Requirements

- Python 3.10+
- Anthropic API key (Claude for extraction, consolidation, summarisation)
- No embedding API key needed — uses local `sentence-transformers` model (~90 MB, downloads on first run)

## Design Decisions

- **SQLite only** — no external services required; WAL mode for concurrent access
- **Local embeddings** — `all-MiniLM-L6-v2` via sentence-transformers; fast, free, 384-dim
- **FTS5 + Porter stemmer** — BM25 ranking built into SQLite, no extra deps
- **LLM consolidation** — Claude checks for duplicates before creating new memories; prevents drift
- **Memory taxonomy** — episodic / semantic / procedural mirrors OpenClaw and Google's whitepaper
