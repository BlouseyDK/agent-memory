"""
Integration test + demo for the Agent Memory System.

Logs realistic activities from a software team building an agentic ERP system,
then demonstrates memory retrieval across a range of contexts and topics.

Run directly:
    python test_memory.py

Or with pytest (skips if ANTHROPIC_API_KEY not set):
    pytest test_memory.py -v -s
"""

from __future__ import annotations

import os
import shutil
import tempfile
import textwrap
from datetime import datetime

import pytest

from memory import MemorySystem


# ── Sample activities ─────────────────────────────────────────────────────────
# Realistic activities from a team building an agentic ERP product on Azure.
# Mix of episodic (events), semantic (facts/decisions), procedural (workflows).

ACTIVITIES = [
    (
        "2026-02-10T09:00:00",
        "Sprint planning for the agent orchestration module. Team agreed: the purchase "
        "order workflow will be fully autonomous — the agent can approve POs under €5,000 "
        "without human review. Above that threshold it escalates to a manager via Teams. "
        "Maria will own the approval routing logic.",
    ),
    (
        "2026-02-10T14:30:00",
        "Architecture decision: we're standardising on Azure OpenAI (GPT-4o) for all "
        "LLM calls in the ERP agent. Claude will be used for document parsing and "
        "summarisation tasks only. All API keys go through Azure Key Vault — no plaintext "
        "secrets in code or config files.",
    ),
    (
        "2026-02-11T10:00:00",
        "Incident post-mortem: the invoice matching agent ran into an infinite retry loop "
        "on Tuesday when the supplier API returned HTTP 429. The agent kept retrying every "
        "2 seconds for 45 minutes, racking up 1,350 failed calls and a €23 API bill. "
        "Fix: add exponential backoff with jitter, cap retries at 5, alert on failure.",
    ),
    (
        "2026-02-12T09:15:00",
        "Reviewed vendor contracts. Learned that our ERP data must stay within EU data "
        "residency boundaries — no data can be sent to US-based AI services without "
        "explicit DPA agreements. Azure Sweden Central is approved. OpenAI US endpoints "
        "are NOT approved for production customer data.",
    ),
    (
        "2026-02-12T15:00:00",
        "Deployed the goods receipt agent to staging. It reads inbound delivery "
        "notifications, matches them against open purchase orders in Business Central, "
        "and posts the receipt automatically. Accuracy in testing: 97.3% on standard "
        "lines, drops to 81% on partial deliveries. Partial deliveries need human review.",
    ),
    (
        "2026-02-13T11:00:00",
        "Performance review meeting. The invoice processing pipeline currently takes "
        "~4 seconds per invoice end-to-end. Target is under 2 seconds. Bottleneck is "
        "the PDF parsing step — using GPT-4o Vision which is slow. "
        "Plan: switch to a local document intelligence model for extraction, "
        "use LLM only for the matching/decision step.",
    ),
    (
        "2026-02-14T09:30:00",
        "Pair programming with Lars on the agent memory module. We agreed the memory "
        "system should use hybrid search — BM25 for keyword recall and vector embeddings "
        "for semantic similarity. Chose SQLite as the store: no infra, works locally, "
        "FTS5 built in. Sentence-transformers for embeddings so there's no API key "
        "dependency. Lars prefers explicit type annotations everywhere.",
    ),
    (
        "2026-02-17T10:00:00",
        "Security review with the CISO. Key findings: agent-to-agent calls must use "
        "short-lived tokens (TTL ≤ 1 hour), not static API keys. All agent actions "
        "must be logged to an immutable audit trail in Azure Monitor. Human-in-the-loop "
        "checkpoints are mandatory for any financial transaction above €10,000.",
    ),
    (
        "2026-02-18T14:00:00",
        "Customer demo at NordRetail. They were impressed by the automatic PO matching "
        "but had two blockers: (1) they need the agent to speak Danish — currently "
        "responses are English only, (2) their ERP is still on BC 2023 Wave 1, "
        "which doesn't have the REST APIs we depend on. Both need to be addressed "
        "before they'll sign the contract.",
    ),
    (
        "2026-02-19T09:00:00",
        "Standup. Decided the deployment runbook for the agent cluster: "
        "1) run smoke tests against staging, "
        "2) deploy to prod via blue-green with 10% traffic split, "
        "3) monitor error rate for 30 minutes, "
        "4) promote to 100% if error rate stays below 0.5%, "
        "5) rollback immediately if error rate exceeds 1%. "
        "This runbook applies to all agent services going forward.",
    ),
    (
        "2026-02-20T11:30:00",
        "Researched LLM cost optimisation. Key insight: for structured extraction tasks "
        "(parsing invoices, extracting line items), a fine-tuned smaller model is 10x "
        "cheaper than GPT-4o with near-identical accuracy on our data. "
        "Decision: fine-tune GPT-4o-mini on 500 labelled invoices. Maria to prepare "
        "the dataset by end of sprint.",
    ),
    (
        "2026-02-21T14:00:00",
        "Reviewed the BC ALAppExtensions commit log. Noticed the E-Invoicing module "
        "(XRechnung/ZUGFeRD) had a fix for incorrect IBAN in DE exports and a VAT "
        "registration number fix for Public Sector entities. These affect our German "
        "customer pipeline — need to update the agent's document parsing rules to "
        "match the new field mappings.",
    ),
    (
        "2026-02-24T09:00:00",
        "Retrospective on the agent incident last week. Three action items: "
        "(1) every agent must implement circuit breaker pattern — open after 3 failures, "
        "half-open after 60 seconds, "
        "(2) agent cost budgets must be enforced in code, not just monitored, "
        "(3) all agents must support a dry-run mode for testing without side effects.",
    ),
    (
        "2026-02-25T10:00:00",
        "Technical interview with a candidate for the agent engineering role. "
        "She had strong Python and LLM experience. Weaker on distributed systems "
        "and error handling patterns. Decision: offer the role — the team can mentor "
        "on distributed systems, the LLM depth is harder to teach. "
        "Start date: March 17th.",
    ),
    (
        "2026-02-28T09:00:00",
        "Weekly sync with the Business Central team. They confirmed that BC 2025 Wave 1 "
        "(releasing April) will expose the new Copilot Extension APIs we need for "
        "deep agent integration. The team should plan the integration sprint for May. "
        "Also: the ALAppExtensions Quality Management module got significant updates "
        "this week — worth reviewing for our inspection workflow agent.",
    ),
]


# ── Context queries and expected relevance ────────────────────────────────────
# Each tuple: (context_query, description_of_what_should_surface)

CONTEXT_QUERIES = [
    (
        "API rate limiting and retry logic",
        "Should surface the invoice matching incident (infinite retry loop, €23 bill) "
        "and ideally the circuit breaker retrospective action item.",
    ),
    (
        "data residency and GDPR compliance for AI services",
        "Should surface the EU data residency constraint, Azure Sweden Central approval, "
        "and the restriction on US-based AI services for customer data.",
    ),
    (
        "deployment process and rollback procedure",
        "Should surface the blue-green deployment runbook with traffic splitting "
        "and the error rate thresholds.",
    ),
    (
        "LLM cost reduction and model selection",
        "Should surface the fine-tuning decision (GPT-4o-mini, 10x cheaper), "
        "the PDF parsing bottleneck, and the model selection architecture.",
    ),
    (
        "security requirements for agent-to-agent communication",
        "Should surface the CISO review findings: short-lived tokens, audit trail, "
        "HITL checkpoints for transactions above €10,000.",
    ),
    (
        "customer blockers NordRetail",
        "Should surface Danish language requirement and BC 2023 Wave 1 API compatibility issue.",
    ),
]

# ── Topics to summarise ───────────────────────────────────────────────────────

SUMMARY_TOPICS = ["security", "deployment", "performance", "cost"]


# =========================================================================
# Pytest fixtures + tests
# =========================================================================

@pytest.fixture(scope="module")
def mem():
    """Create a MemorySystem backed by a temp DB, pre-loaded with sample data."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    md_dir = tempfile.mkdtemp(suffix="_memory_notes")

    system = MemorySystem(db_path=db_path, md_dir=md_dir)
    print(f"\n\n{'='*60}")
    print("LOADING SAMPLE ACTIVITIES")
    print(f"{'='*60}")

    for ts, activity in ACTIVITIES:
        result = system.log_activity(datetime.fromisoformat(ts), activity)
        n = len(result["memories"])
        print(f"\n  📝 {ts[:10]} → {n} memory/memories extracted")
        for m in result["memories"]:
            print(f"       [{m.get('memory_type','?'):10s}] ({m.get('topic','?')}) {m.get('content','')}")

    s = system.stats()
    print(f"\n  ── Stats: {s['activities_logged']} activities → {s['memories_stored']} memories "
          f"across {s['topics']} topics ──")

    yield system

    # Clean up resources before deleting the file (Windows requirement)
    system.close()
    os.unlink(db_path)
    shutil.rmtree(md_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────

def wrapped_print(text, width=100):
    """
    Prints text wrapped to the given width.
    
    :param text: The string to print.
    :param width: Maximum characters per line.
    """
    if not isinstance(text, str):
        text = str(text)  # Convert non-string input to string
    
    # Wrap the text into a list of lines
    wrapped_lines = textwrap.wrap(text, width=width)
    
    # Print each wrapped line
    for line in wrapped_lines:
        print(line)

def test_activities_loaded(mem):
    """All sample activities were stored."""
    s = mem.stats()
    assert s["activities_logged"] == len(ACTIVITIES), (
        f"Expected {len(ACTIVITIES)} activities, got {s['activities_logged']}"
    )
    print(f"\n  ✓ {s['activities_logged']} activities logged")


def test_memories_extracted(mem):
    """At least one memory was extracted per activity on average."""
    s = mem.stats()
    assert s["memories_stored"] >= len(ACTIVITIES), (
        f"Expected at least {len(ACTIVITIES)} memories, got {s['memories_stored']}"
    )
    print(f"\n  ✓ {s['memories_stored']} memories extracted from {s['activities_logged']} activities")


def test_memory_types_present(mem):
    """All three memory types are represented."""
    s = mem.stats()
    by_type = s["by_type"]
    print(f"\n  Memory type breakdown: {by_type}")
    assert "episodic"   in by_type, "No episodic memories found"
    assert "semantic"   in by_type, "No semantic memories found"
    assert "procedural" in by_type, "No procedural memories found"
    print("  ✓ All three memory types present (episodic, semantic, procedural)")


@pytest.mark.parametrize("topic", SUMMARY_TOPICS)
def test_summarize_topic(mem, topic):
    """summarize_memory(topic) returns a non-empty string."""
    print(f"\n\n{'─'*60}")
    print(f"  📖 SUMMARY — '{topic}'")
    print(f"{'─'*60}")
    result = mem.summarize_memory(topic)
    print(f"\n{result}\n")
    assert isinstance(result, str), "Summary should be a string"
    assert len(result) > 20, f"Summary for '{topic}' is too short: {result!r}"


def test_summarize_all_topics(mem):
    """summarize_memory() with no topic returns grouped summaries."""
    print(f"\n\n{'─'*60}")
    print("  📖 SUMMARY — all topics")
    print(f"{'─'*60}")
    result = mem.summarize_memory()
    print(f"\n{result}\n")
    assert isinstance(result, str)
    assert len(result) > 50


@pytest.mark.parametrize("context,description", CONTEXT_QUERIES)
def test_context_memory(mem, context, description):
    """context_memory() returns relevant results for each query."""
    print(f"\n\n{'─'*60}")
    print(f"  🔍 CONTEXT: '{context}'")
    print(f"  Expected: {description}")
    print(f"{'─'*60}")

    results = mem.context_memory(context, max_results=5)

    print(f"\n  Top {len(results)} result(s):")
    for i, r in enumerate(results, 1):
        bar = "█" * int(r["score"] * 20)
        print(f"    {i}. score={r['score']:.3f} {bar}")
        print(f"       [{r['memory_type']:10s}] ({r['topic']}) {r['content']}")

    assert len(results) > 0, f"No results returned for context: '{context}'"
    assert results[0]["score"] > 0.1, f"Top result score too low: {results[0]['score']}"
    print(f"\n  ✓ {len(results)} relevant memories returned, top score: {results[0]['score']:.3f}")


def test_context_memory_returns_scored_list(mem):
    """context_memory() result structure is correct."""
    results = mem.context_memory("purchase order approval workflow")
    assert isinstance(results, list)
    if results:
        r = results[0]
        assert "content"     in r
        assert "topic"       in r
        assert "memory_type" in r
        assert "score"       in r
        assert r["memory_type"] in ("episodic", "semantic", "procedural")
        assert 0.0 <= r["score"] <= 1.0
    print(f"\n  ✓ Result schema valid — {len(results)} result(s)")


def test_no_context_returns_empty(mem):
    """Empty context string returns empty list."""
    assert mem.context_memory("") == []
    assert mem.context_memory("   ") == []
    print("\n  ✓ Empty context returns []")


# =========================================================================
# Standalone runner (not pytest)
# =========================================================================

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌  ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        raise SystemExit(1)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    md_dir = tempfile.mkdtemp(suffix="_memory_notes")

    system = MemorySystem(db_path=db_path, md_dir=md_dir)

    # ── Load activities ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PHASE 1 — LOGGING ACTIVITIES")
    print(f"{'='*60}")

    for ts, activity in ACTIVITIES:
        result = system.log_activity(datetime.fromisoformat(ts), activity)
        n = len(result["memories"])
        print(f"\n📝 {ts[:10]}  ({n} memories)")
        print (f"Activity")
        wrapped_print(activity)
        for m in result["memories"]:
            print(f"   [{m.get('memory_type','?'):10s}] ({m.get('topic','?')}) {m.get('content','')}")

    s = system.stats()
    print(f"\n{'─'*60}")
    print(f"  {s['activities_logged']} activities → {s['memories_stored']} memories "
          f"across {s['topics']} topics")
    print(f"  Types: {s['by_type']}")

    # ── Topic summaries ───────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  PHASE 2 — TOPIC SUMMARIES")
    print(f"{'='*60}")

    for topic in SUMMARY_TOPICS:
        print(f"\n📖  Summary — '{topic}':\n")
        print(system.summarize_memory(topic))

    # print(f"\n📖  Summary — all topics:\n")
    # print(system.summarize_memory())

    # ── Context retrieval ─────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  PHASE 3 — CONTEXT MEMORY RETRIEVAL")
    print(f"{'='*60}")

    for context, description in CONTEXT_QUERIES:
        print(f"\n🔍  Context: '{context}'")
        print(f"    Expected: {description}\n")
        results = system.context_memory(context, max_results=4)
        for i, r in enumerate(results, 1):
            bar = "█" * int(r["score"] * 20)
            print(f"  {i}. score={r['score']:.3f} {bar}")
            print(f"     [{r['memory_type']:10s}] ({r['topic']}) {r['content']}")

    print(f"\n{'='*60}")
    print("  Done.")

    # Clean up resources before deleting the file (Windows requirement)
    system.close()
    os.unlink(db_path)
    shutil.rmtree(md_dir, ignore_errors=True)
