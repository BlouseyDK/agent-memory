"""
CLI demo for the Agent Memory System.

Usage examples:
  python cli.py log "Attended standup. Team decided to use Python for the new agent module."
  python cli.py log "2026-03-01T09:00:00" "Pair-programmed with Alice on the auth service refactor."
  python cli.py summarize
  python cli.py summarize coding
  python cli.py context "What do I know about the authentication service?"
  python cli.py stats
  python cli.py demo   # runs a full end-to-end demo with sample data
"""

import sys
from datetime import datetime

from memory import MemorySystem


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    mem = MemorySystem()
    cmd = sys.argv[1].lower()

    # ── log ──────────────────────────────────────────────────────────────────
    if cmd == "log":
        if len(sys.argv) == 3:
            log_time = datetime.now()
            activity = sys.argv[2]
        elif len(sys.argv) == 4:
            log_time = datetime.fromisoformat(sys.argv[2])
            activity = sys.argv[3]
        else:
            print("Usage: python cli.py log [<iso-datetime>] <activity>")
            return

        print(f"\n📝 Logging activity…")
        result = mem.log_activity(log_time, activity)
        print(f"   Activity ID: {result['activity_id']}")
        print(f"   Extracted {len(result['memories'])} memory/memories:")
        for m in result["memories"]:
            print(f"     [{m.get('memory_type','?')}] ({m.get('topic','?')}) {m.get('content','')}")

    # ── summarize ─────────────────────────────────────────────────────────────
    elif cmd in ("summarize", "summary", "sum"):
        topic = sys.argv[2] if len(sys.argv) > 2 else ""
        label = f"'{topic}'" if topic else "all topics"
        print(f"\n📖 Summarising memories — {label}…\n")
        print(mem.summarize_memory(topic))

    # ── context ───────────────────────────────────────────────────────────────
    elif cmd in ("context", "ctx", "search"):
        if len(sys.argv) < 3:
            print("Usage: python cli.py context <query>")
            return
        query = " ".join(sys.argv[2:])
        print(f"\n🔍 Context search: '{query}'\n")
        results = mem.context_memory(query)
        if not results:
            print("No relevant memories found.")
        for i, r in enumerate(results, 1):
            bar = "█" * int(r["score"] * 20)
            print(f"  {i}. [{r['memory_type']:10s}] ({r['topic']}) score={r['score']:.3f} {bar}")
            print(f"     {r['content']}\n")

    # ── stats ─────────────────────────────────────────────────────────────────
    elif cmd == "stats":
        s = mem.stats()
        print(f"\n📊 Memory stats")
        print(f"   Activities logged:  {s['activities_logged']}")
        print(f"   Memories stored:    {s['memories_stored']}")
        print(f"   Topics:             {s['topics']}")
        for mtype, count in s["by_type"].items():
            print(f"     {mtype}: {count}")

    # ── demo ──────────────────────────────────────────────────────────────────
    elif cmd == "demo":
        _run_demo(mem)

    else:
        print(__doc__)


def _run_demo(mem: MemorySystem) -> None:
    """Run a full end-to-end demo."""
    ACTIVITIES = [
        (
            "2026-03-01T09:00:00",
            "Morning standup. The team decided to use Python for the new agent module "
            "instead of TypeScript. Alice will lead the implementation. The sprint goal "
            "is to have a working prototype by Friday.",
        ),
        (
            "2026-03-01T11:30:00",
            "Reviewed security findings. Discovered that the backup system was missing "
            "token refresh logic. Fixed the Azure client credentials to use the secrets "
            "module instead of reading from the token blob directly. Pushed fix to GitHub.",
        ),
        (
            "2026-03-01T14:00:00",
            "Architecture meeting. Agreed on using SQLite for local agent state storage "
            "and OneDrive for backups. The team prefers DPAPI for secrets on Windows. "
            "Bob mentioned he prefers dark mode in all dev tools.",
        ),
        (
            "2026-03-02T09:15:00",
            "Pair programming session with Alice on the memory module. Implemented "
            "hybrid BM25 + vector search. Decided to use sentence-transformers with "
            "all-MiniLM-L6-v2 for local embeddings — no API key required.",
        ),
        (
            "2026-03-02T16:00:00",
            "Retrospective. The team highlighted that the security review process "
            "is working well. Agreed to keep the nightly automated review running. "
            "Also noted: Alice prefers pair programming over async code reviews.",
        ),
    ]

    print("\n🎬  Running end-to-end demo\n" + "=" * 52)

    for ts, activity in ACTIVITIES:
        print(f"\n📝 Logging: {ts}")
        print(f"   {activity[:80]}…")
        result = mem.log_activity(datetime.fromisoformat(ts), activity)
        print(f"   → {len(result['memories'])} memory/memories extracted")
        for m in result["memories"]:
            print(f"     [{m.get('memory_type','?'):10s}] ({m.get('topic','?')}) {m.get('content','')}")

    print("\n" + "=" * 52)
    print("\n📖 Summary — 'security':\n")
    print(mem.summarize_memory("security"))

    print("\n" + "=" * 52)
    print("\n📖 Summary — all topics:\n")
    print(mem.summarize_memory())

    print("\n" + "=" * 52)
    print("\n🔍 Context search: 'authentication and secrets management'\n")
    results = mem.context_memory("authentication and secrets management")
    for i, r in enumerate(results, 1):
        bar = "█" * int(r["score"] * 20)
        print(f"  {i}. [{r['memory_type']:10s}] score={r['score']:.3f} {bar}")
        print(f"     {r['content']}\n")

    print("=" * 52)
    s = mem.stats()
    print(f"\n📊 Final stats: {s['activities_logged']} activities → {s['memories_stored']} memories across {s['topics']} topics")


if __name__ == "__main__":
    main()
