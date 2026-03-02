"""
Agent Memory System
===================
Inspired by OpenClaw's memory architecture and Google's "Context Engineering:
Sessions & Memory" whitepaper (Nov 2025).

Memory taxonomy (Google / OpenClaw):
  - episodic:    specific events and interactions ("what happened")
  - semantic:    facts, preferences, knowledge ("what I know")
  - procedural:  workflows, routines, patterns ("how to do something")

Storage:  SQLite + FTS5 (keyword) + cosine similarity (vector)
LLM:      Anthropic Claude (extraction, consolidation, summarization)
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, no API key)

Public API:
  log_activity(log_time, activity)   → store + extract memories
  summarize_memory(topic="")         → LLM summary by topic or all topics
  context_memory(context)            → hybrid-search relevant memories
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DB      = os.environ.get("MEMORY_DB",         "memory.db")
DEFAULT_MODEL   = os.environ.get("ANTHROPIC_MODEL",   "claude-sonnet-4-5-20250929")
EMBED_MODEL     = os.environ.get("EMBEDDING_MODEL",   "all-MiniLM-L6-v2")

# Hybrid search weights (must sum to 1.0)
WEIGHT_BM25     = 0.40
WEIGHT_VECTOR   = 0.60

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS activities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    log_time    TEXT    NOT NULL,
    activity    TEXT    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memories (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    content             TEXT    NOT NULL,
    topic               TEXT    NOT NULL,
    memory_type         TEXT    NOT NULL
                        CHECK (memory_type IN ('episodic','semantic','procedural')),
    source_activity_id  INTEGER REFERENCES activities(id),
    created_at          TEXT    DEFAULT (datetime('now')),
    updated_at          TEXT    DEFAULT (datetime('now'))
);

-- Full-text search index (BM25 via FTS5 + Porter stemmer)
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, topic,
    content     = 'memories',
    content_rowid = 'id',
    tokenize    = 'porter unicode61'
);

-- Keep FTS in sync with memories table
CREATE TRIGGER IF NOT EXISTS trg_mem_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, topic)
    VALUES (new.id, new.content, new.topic);
END;

CREATE TRIGGER IF NOT EXISTS trg_mem_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, topic)
    VALUES ('delete', old.id, old.content, old.topic);
    INSERT INTO memories_fts(rowid, content, topic)
    VALUES (new.id, new.content, new.topic);
END;

CREATE TRIGGER IF NOT EXISTS trg_mem_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, topic)
    VALUES ('delete', old.id, old.content, old.topic);
END;

-- Vector embeddings (stored as raw float32 bytes)
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id   INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding   BLOB    NOT NULL
);
"""


# ── MemorySystem ──────────────────────────────────────────────────────────────

class MemorySystem:
    """
    Simple agent memory system with hybrid search.

    Parameters
    ----------
    db_path : path to SQLite database file (created if missing)
    api_key : Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
    model   : Anthropic model ID
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.db_path = db_path
        self.model   = model
        self.client  = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._embedder: SentenceTransformer | None = None
        self._setup_db()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executescript(SCHEMA)

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print(f"Loading embedding model '{EMBED_MODEL}'…")
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    # ── LLM helper ───────────────────────────────────────────────────────────

    def _llm(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip()

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Return L2-normalised embedding (cosine similarity = dot product)."""
        return self.embedder.encode(text, normalize_embeddings=True).astype(np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def log_activity(
        self,
        log_time: datetime | str,
        activity: str,
    ) -> dict:
        """
        Log an activity and extract + store key memory facts from it.

        Parameters
        ----------
        log_time : when the activity occurred (datetime or ISO string)
        activity : free-text description of the activity

        Returns
        -------
        dict with 'activity_id' and 'memories' (list of extracted/updated entries)
        """
        ts = log_time.isoformat() if isinstance(log_time, datetime) else str(log_time)

        # 1. Persist raw activity
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "INSERT INTO activities (log_time, activity) VALUES (?, ?)", (ts, activity)
            )
            activity_id = cur.lastrowid

        # 2. Extract memory facts via LLM
        extracted = self._extract_memories(activity, ts)

        # 3. Consolidate or store each fact
        stored = []
        for mem in extracted:
            mem_id = self._consolidate_or_store(mem, activity_id)
            stored.append({"id": mem_id, **mem})

        return {"activity_id": activity_id, "memories": stored}

    # ─────────────────────────────────────────────────────────────────────────

    def summarize_memory(self, topic: str = "") -> str:
        """
        Summarise stored memories.

        Parameters
        ----------
        topic : keyword to focus on. If blank, summarises all topics.

        Returns
        -------
        Human-readable summary string.
        """
        if topic.strip():
            memories = self._hybrid_search(topic, limit=20)
            if not memories:
                return f"No memories found related to '{topic}'."

            mem_text = "\n".join(
                f"- [{m['memory_type']}] {m['content']}" for m in memories
            )
            return self._llm(
                "You are a memory summariser. Given memory entries, write a concise, "
                "factual summary focused on the specified topic. 3-5 sentences max.",
                f"Topic: {topic}\n\nMemories:\n{mem_text}",
                max_tokens=512,
            )

        else:
            # Summarise all topics
            with sqlite3.connect(self.db_path) as con:
                topics = [
                    row[0]
                    for row in con.execute(
                        "SELECT DISTINCT topic FROM memories ORDER BY topic"
                    ).fetchall()
                ]

            if not topics:
                return "No memories stored yet."

            parts = []
            for t in topics:
                mems = self._fts_search(t, limit=10)
                if not mems:
                    continue
                entries = "\n".join(f"- {m['content']}" for m in mems)
                summary = self._llm(
                    "Summarise these memory entries in 1-3 sentences. Be concise and factual.",
                    f"Topic '{t}':\n{entries}",
                    max_tokens=256,
                )
                parts.append(f"### {t.title()}\n{summary}")

            return "\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────

    def context_memory(self, context: str, max_results: int = 5) -> list[dict]:
        """
        Return memories relevant to the given context using hybrid search.

        Parameters
        ----------
        context     : free-text context / query
        max_results : max number of memories to return

        Returns
        -------
        List of dicts: {id, content, topic, memory_type, score}
        Sorted by relevance (highest first).
        """
        if not context.strip():
            return []
        return self._hybrid_search(context, limit=max_results)

    # =========================================================================
    # INTERNALS
    # =========================================================================

    def _extract_memories(self, activity: str, log_time: str) -> list[dict]:
        """Use Claude to extract key facts from an activity description."""
        system = (
            "You are a memory extraction system for an AI agent.\n"
            "Given an activity log entry, extract the key facts worth remembering long-term.\n\n"
            "Classify each memory as:\n"
            "  episodic:   a specific event or interaction that happened\n"
            "  semantic:   a fact, preference, or piece of knowledge\n"
            "  procedural: a workflow, process, or learned routine\n\n"
            "Return ONLY a JSON array. Each element:\n"
            '  {"content": "concise fact", "topic": "1-3 word keyword", '
            '"memory_type": "episodic|semantic|procedural"}\n\n'
            "Topic examples: coding, preferences, security, meetings, deployment, etc.\n"
            "Skip trivial or transient details. Return [] if nothing is worth keeping."
        )
        raw = self._llm(system, f"Activity at {log_time}:\n\n{activity}")
        try:
            start, end = raw.find("["), raw.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def _consolidate_or_store(self, mem: dict, activity_id: int) -> int:
        """
        Check if a similar memory already exists.
        If so, update it (consolidation). Otherwise create a new entry.
        Returns the memory ID.
        """
        content     = mem.get("content", "")
        topic       = mem.get("topic", "general")
        memory_type = mem.get("memory_type", "episodic")

        # Look for candidates via FTS
        candidates = self._fts_search(content, limit=5)

        if candidates:
            existing = "\n".join(f"[{c['id']}] {c['content']}" for c in candidates)
            decision_raw = self._llm(
                "You are a memory consolidation system.\n"
                "Decide whether the new memory UPDATES/DUPLICATES an existing one or is genuinely NEW.\n"
                'Return ONLY JSON: {"action": "update", "id": <int>}  or  {"action": "create"}',
                f"New memory: {content}\n\nExisting similar memories:\n{existing}",
                max_tokens=64,
            )
            try:
                start, end = decision_raw.find("{"), decision_raw.rfind("}") + 1
                decision = json.loads(decision_raw[start:end])
                if decision.get("action") == "update" and decision.get("id"):
                    mem_id = int(decision["id"])
                    with sqlite3.connect(self.db_path) as con:
                        con.execute(
                            "UPDATE memories SET content=?, updated_at=datetime('now') WHERE id=?",
                            (content, mem_id),
                        )
                    self._store_embedding(mem_id, content)
                    return mem_id
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Create new memory
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "INSERT INTO memories (content, topic, memory_type, source_activity_id) "
                "VALUES (?, ?, ?, ?)",
                (content, topic, memory_type, activity_id),
            )
            mem_id = cur.lastrowid

        self._store_embedding(mem_id, content)
        return mem_id

    def _store_embedding(self, memory_id: int, text: str) -> None:
        emb = self._embed(text)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)",
                (memory_id, emb.tobytes()),
            )

    # ── Search ────────────────────────────────────────────────────────────────

    def _fts_search(self, query: str, limit: int = 10) -> list[dict]:
        """BM25 keyword search via SQLite FTS5."""
        try:
            with sqlite3.connect(self.db_path) as con:
                rows = con.execute(
                    """SELECT m.id, m.content, m.topic, m.memory_type, (-rank) AS bm25
                       FROM memories_fts f
                       JOIN memories m ON m.id = f.rowid
                       WHERE memories_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (query, limit),
                ).fetchall()
            return [
                {"id": r[0], "content": r[1], "topic": r[2], "memory_type": r[3], "score": r[4]}
                for r in rows
            ]
        except Exception:
            return []

    def _vector_search(self, query: str, limit: int = 10) -> list[dict]:
        """Cosine similarity search over all stored embeddings."""
        q_emb = self._embed(query)

        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                """SELECT m.id, m.content, m.topic, m.memory_type, e.embedding
                   FROM memory_embeddings e
                   JOIN memories m ON m.id = e.memory_id"""
            ).fetchall()

        if not rows:
            return []

        scored = [
            {
                "id": r[0],
                "content": r[1],
                "topic": r[2],
                "memory_type": r[3],
                "score": self._cosine(q_emb, np.frombuffer(r[4], dtype=np.float32)),
            }
            for r in rows
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _hybrid_search(self, query: str, limit: int = 10) -> list[dict]:
        """
        Combine BM25 + vector search with weighted scoring.
        Mirrors OpenClaw's mergeHybridResults approach.
        """
        bm25 = self._fts_search(query, limit=limit * 2)
        vec  = self._vector_search(query, limit=limit * 2)

        # Normalise BM25 scores to [0, 1]
        bm25_max  = max((r["score"] for r in bm25), default=1.0) or 1.0
        bm25_norm = {r["id"]: r["score"] / bm25_max for r in bm25}
        vec_norm  = {r["id"]: r["score"] for r in vec}  # already [0,1]

        all_ids   = set(bm25_norm) | set(vec_norm)
        lookup    = {r["id"]: r for r in bm25 + vec}

        merged = []
        for mid in all_ids:
            combined = WEIGHT_BM25 * bm25_norm.get(mid, 0.0) + WEIGHT_VECTOR * vec_norm.get(mid, 0.0)
            entry    = lookup[mid].copy()
            entry["score"] = round(combined, 4)
            merged.append(entry)

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:limit]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return summary statistics about stored memories."""
        with sqlite3.connect(self.db_path) as con:
            n_activities = con.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
            n_memories   = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            n_topics     = con.execute("SELECT COUNT(DISTINCT topic) FROM memories").fetchone()[0]
            by_type      = dict(
                con.execute(
                    "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
                ).fetchall()
            )
        return {
            "activities_logged": n_activities,
            "memories_stored":   n_memories,
            "topics":            n_topics,
            "by_type":           by_type,
        }

    def close(self) -> None:
        """Clean up resources and close connections."""
        # Clear the cached embedder to free up resources
        if self._embedder is not None:
            del self._embedder
            self._embedder = None
        
        # Ensure all SQLite connections are closed by opening and closing one more time
        # This forces SQLite to release any remaining file handles
        try:
            with sqlite3.connect(self.db_path) as con:
                con.execute("PRAGMA optimize")
        except Exception:
            pass  # Ignore errors during cleanup
