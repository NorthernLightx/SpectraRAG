"""Corpus tools for the DCI agent: in-memory lexical search/grep/read over raw text.

The corpus is a directory of `<doc_id>.txt` files, loaded into memory once. These
tools are the entire interface the agent has to the corpus — no embeddings, no
vector index. Faithful to the DCI thesis: the model supplies the intelligence
(which terms to search, what to read, when to stop); the tools expose the raw
text losslessly via lexical scan.

Pure Python (no ripgrep subprocess) so it runs anywhere. Holding the raw lines in
RAM is not a semantic index — it's the corpus itself, scanned linearly per query.
"""

from __future__ import annotations

import builtins
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TERM_RE = re.compile(r"[A-Za-z0-9]+")
_MAX_PATTERN_TERMS = 24  # cap alternation so a verbose query can't build a huge regex

# Whitelisted builtins for the SCRIPT sandbox. No __import__, open, exec, eval,
# input, etc. — the script can compute over the corpus but cannot touch the
# filesystem, network, or import modules. Not hardened against a deliberate
# sandbox escape (attribute-walking to os); adequate here because the agent is
# cooperative, not adversarial — the realistic failure is a buggy/looping script,
# which the thread timeout bounds.
_SAFE_BUILTIN_NAMES = (
    "len sum sorted min max range enumerate list dict set tuple str int float bool "
    "any all zip map filter abs round reversed print repr isinstance bytes frozenset"
)
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES.split()}


@dataclass(frozen=True)
class DocHit:
    """A corpus document that matched, with its match count and a snippet."""

    doc_id: str
    score: int
    snippet: str


@dataclass(frozen=True)
class LineHit:
    """A single matching line, for exact-grep inspection."""

    doc_id: str
    line: int
    text: str


class CorpusTools:
    """In-memory lexical view over a directory of `<doc_id>.txt` documents."""

    def __init__(self, docs: dict[str, str]) -> None:
        self._lines: dict[str, list[str]] = {d: t.splitlines() for d, t in docs.items()}
        self._lower: dict[str, str] = {d: t.lower() for d, t in docs.items()}

    @classmethod
    def from_dir(cls, corpus_dir: Path) -> CorpusTools:
        """Build from a directory of `<doc_id>.txt` files (toy / small corpora)."""
        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"corpus dir not found: {corpus_dir}")
        docs = {
            p.stem: p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(corpus_dir.glob("*.txt"))
        }
        return cls(docs)

    @staticmethod
    def _terms(query: str) -> list[str]:
        seen: list[str] = []
        for tok in _TERM_RE.findall(query.lower()):
            if len(tok) > 1 and tok not in seen:
                seen.append(tok)
        return seen[:_MAX_PATTERN_TERMS]

    def _compile(self, query: str) -> re.Pattern[str] | None:
        terms = self._terms(query)
        if not terms:
            return None
        return re.compile("|".join(re.escape(t) for t in terms))

    # ---- public tools -------------------------------------------------------
    def search(self, query: str, *, top_k: int = 10) -> list[DocHit]:
        """Rank documents by how many times any content term in `query` occurs.

        Recall-oriented: the agent narrows with `grep` once it sees candidates.
        Returns up to `top_k` DocHits sorted by match count (desc)."""
        rx = self._compile(query)
        if rx is None:
            return []
        scored: list[tuple[str, int]] = []
        for doc_id, low in self._lower.items():
            n = len(rx.findall(low))
            if n:
                scored.append((doc_id, n))
        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return [
            DocHit(doc_id=d, score=n, snippet=self._first_match(d, rx)) for d, n in scored[:top_k]
        ]

    def _first_match(self, doc_id: str, rx: re.Pattern[str]) -> str:
        for ln in self._lines[doc_id]:
            if rx.search(ln.lower()):
                return ln.strip()[:300]
        return ""

    def grep(
        self, pattern: str, *, top_k: int = 20, ignore_case: bool = True, fixed: bool = False
    ) -> list[LineHit]:
        """Exact pattern grep. `fixed=True` treats `pattern` as a literal string
        (phrase / exact-constraint search); else it's a regex. Returns matching
        lines with doc id + line number, capped at `top_k`."""
        flags = re.IGNORECASE if ignore_case else 0
        try:
            rx = re.compile(re.escape(pattern) if fixed else pattern, flags)
        except re.error as exc:
            raise ValueError(f"bad regex: {exc}") from exc
        hits: list[LineHit] = []
        for doc_id in sorted(self._lines):
            for i, ln in enumerate(self._lines[doc_id], start=1):
                if rx.search(ln):
                    hits.append(LineHit(doc_id=doc_id, line=i, text=ln.strip()[:300]))
                    if len(hits) >= top_k:
                        return hits
        return hits

    def filter_all(self, terms: list[str], *, top_k: int = 10) -> list[DocHit]:
        """Conjunction (chained grep): rank docs that contain ALL given terms.

        The precision counterpart to `search` — `grep t1 | grep t2 | ...`. Score
        is the total occurrence count across the required terms. Returns up to
        `top_k` DocHits, or [] if no doc satisfies every term."""
        norm = [t.lower() for t in terms if len(t) > 1]
        if not norm:
            return []
        rxs = [re.compile(re.escape(t)) for t in norm]
        scored: list[tuple[str, int]] = []
        for doc_id, low in self._lower.items():
            counts = [len(rx.findall(low)) for rx in rxs]
            if all(counts):  # every required term present
                scored.append((doc_id, sum(counts)))
        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        joined = re.compile("|".join(re.escape(t) for t in norm))
        return [
            DocHit(doc_id=d, score=n, snippet=self._first_match(d, joined))
            for d, n in scored[:top_k]
        ]

    def count(self, term: str) -> tuple[int, int]:
        """Selectivity of a term: (documents containing it, total occurrences).

        Lets the agent pick discriminative terms — `grep -c` across the corpus."""
        t = term.lower()
        if len(t) < 2:
            return (0, 0)
        rx = re.compile(re.escape(t))
        docs = occ = 0
        for low in self._lower.values():
            n = len(rx.findall(low))
            if n:
                docs += 1
                occ += n
        return (docs, occ)

    def run_script(self, code: str, *, timeout: float = 5.0) -> str:
        """Execute a mini Python script over the corpus in a restricted sandbox.

        The script gets helper functions — search(q,k), grep(p,k), count(t),
        text(doc_id), and all_ids — plus `re`, and should assign `result`. No
        imports / filesystem / network (see `_SAFE_BUILTINS`). Runs in a daemon
        thread with a wall-clock timeout to bound runaway loops."""
        api: dict[str, Any] = {
            "search": lambda q, k=20: [h.doc_id for h in self.search(q, top_k=k)],
            "grep": lambda p, k=50: [h.doc_id for h in self.grep(p, top_k=k)],
            "count": lambda t: self.count(t)[0],
            "text": lambda doc_id: "\n".join(self._lines.get(doc_id, [])),
            "all_ids": list(self._lines),
            "re": re,
        }
        ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, **api}
        err: dict[str, str] = {}

        def target() -> None:
            try:
                exec(compile(code, "<dci-script>", "exec"), ns)  # sandboxed: _SAFE_BUILTINS only
            except Exception as exc:  # report to the agent, don't crash the loop
                err["e"] = f"{type(exc).__name__}: {exc}"

        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            return f"(script timed out after {timeout}s — simplify or bound your loops)"
        if "e" in err:
            return f"(script error: {err['e']})"
        if "result" not in ns:
            return "(script ran but did not assign `result`)"
        return repr(ns["result"])[:600]

    def read(self, doc_id: str, *, start: int = 1, end: int | None = None) -> str:
        """Read lines [start, end] (1-based, inclusive) of a document."""
        if doc_id not in self._lines:
            return f"(no such document: {doc_id})"
        lines = self._lines[doc_id]
        lo = max(1, start)
        hi = len(lines) if end is None else min(len(lines), end)
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(lo, hi + 1))
        return body or f"(document {doc_id} has no lines in [{start}, {end}])"

    def stats(self) -> dict[str, int]:
        return {"documents": len(self._lines)}
