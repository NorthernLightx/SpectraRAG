"""The DCI agent: a ReAct loop that interacts with a raw corpus via lexical tools.

The model is given three tools — SEARCH (rank docs by term matches), GREP (find
exact lines), READ (open a doc span) — and must reach either a final ANSWER (QA)
or a final RANK of doc ids (retrieval). Actions are emitted as plain text and
parsed here, so the loop works with any instruct model regardless of whether the
provider supports native tool-calling.

This is the agentic-search core of DCI (arXiv 2605.05242): no embedding index, no
top-k vector step — the model decides what to search, reads what it finds, and
combines lexical clues across turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.dci.tools import CorpusTools
from src.llm.protocol import LLMClient, Message

Mode = Literal["qa", "retrieval"]
Toolset = Literal["readgrep", "fullbash"]

_ACTION_RE = re.compile(
    r"ACTION:\s*(SEARCH|FILTER|COUNT|GREP|READ|SCRIPT|ANSWER|RANK)\b[ \t]*(.*)",
    re.IGNORECASE | re.DOTALL,
)
_READ_RE = re.compile(r"^(\S+?)(?:\s+(\d+))?(?:\s+(\d+))?\s*$")


@dataclass
class DciStep:
    action: str
    arg: str
    observation: str


@dataclass
class DciResult:
    question: str
    mode: Mode
    answer: str | None
    ranked_doc_ids: list[str]
    steps: list[DciStep] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    stopped: str = "budget"  # "final" | "budget" | "error"


def _system_prompt(
    n_docs: int, mode: Mode, top_k: int, toolset: Toolset, exemplars: str = ""
) -> str:
    final = (
        f"ACTION: RANK <doc_id>, <doc_id>, ...   (the {top_k} most relevant doc ids, best first)"
        if mode == "retrieval"
        else "ACTION: ANSWER <your final answer>   (concise; state the answer directly)"
    )
    goal = (
        "find and rank the documents relevant to the question"
        if mode == "retrieval"
        else "answer the question using evidence you find in the corpus"
    )
    read_grep = (
        "  ACTION: SEARCH <terms>         rank docs by how often ANY term appears (broad recall)\n"
        '  ACTION: GREP "<phrase>"         find exact lines containing a phrase\n'
        "  ACTION: GREP <regex>            find exact lines matching a regex\n"
        "  ACTION: READ <doc_id> <start> <end>   read lines start..end of a document\n"
    )
    full_bash = (
        "  ACTION: SEARCH <terms>         rank docs by how often ANY term appears (broad recall)\n"
        "  ACTION: FILTER <t1>, <t2>, ... rank docs containing ALL of these terms (precise; chained grep)\n"
        "  ACTION: COUNT <term>           how many docs contain a term (gauge how discriminative it is)\n"
        '  ACTION: GREP "<phrase>"         find exact lines containing a phrase\n'
        "  ACTION: GREP <regex>            find exact lines matching a regex\n"
        "  ACTION: READ <doc_id> <start> <end>   read lines start..end of a document\n"
        "  ACTION: SCRIPT <python>        run a mini script; helpers search(q),grep(p),count(t),text(id),\n"
        "                                 all_ids, re; assign `result` (e.g. set-intersect two searches)\n"
    )
    strategy = (
        "Strategy: SEARCH broadly, then FILTER on 2-3 discriminative terms to pin the docs that satisfy\n"
        "EVERY constraint (use COUNT to find which terms are rare/specific). READ or GREP to verify, and\n"
        if toolset == "fullbash"
        else "Strategy: SEARCH broadly to find candidates, then READ or GREP to verify. And\n"
    )
    tools = full_bash if toolset == "fullbash" else read_grep
    return (
        f"You are a research agent searching a corpus of {n_docs} text documents to {goal}.\n"
        "You have NO search index — only these lexical tools, one action per turn:\n"
        f"{tools}"
        f"  {final}\n\n"
        f"{strategy}"
        "for multi-step questions use what you learn to search again for the next clue. The best documents\n"
        "often share few surface words with the question, so reason about the underlying concept and search\n"
        "for THAT. Don't guess — verify in the text before you finish.\n\n"
        f"{exemplars}"
        "Format every turn as exactly:\n"
        "THOUGHT: <one line of reasoning>\n"
        "ACTION: <one action from the list above>\n"
    )


def _format_hits(observation_lines: list[str], header: str) -> str:
    if not observation_lines:
        return f"{header}: (no matches)"
    return f"{header}:\n" + "\n".join(observation_lines)


class DciAgent:
    def __init__(
        self,
        tools: CorpusTools,
        llm: LLMClient,
        model: str,
        *,
        max_steps: int = 12,
        search_k: int = 8,
        grep_k: int = 12,
        temperature: float = 0.0,
        max_tokens: int = 512,
        toolset: Toolset = "fullbash",
        exemplars: str = "",
    ) -> None:
        self._tools = tools
        self._llm = llm
        self._model = model
        self._max_steps = max_steps
        self._search_k = search_k
        self._grep_k = grep_k
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._toolset = toolset
        self._exemplars = exemplars

    async def run(self, question: str, *, mode: Mode = "qa", top_k: int = 10) -> DciResult:
        n = self._tools.stats()["documents"]
        messages = [
            Message(
                role="system",
                content=_system_prompt(n, mode, top_k, self._toolset, self._exemplars),
            ),
            Message(role="user", content=f"Question: {question}\n\nBegin."),
        ]
        result = DciResult(question=question, mode=mode, answer=None, ranked_doc_ids=[])
        discovered: list[str] = []  # doc ids seen via SEARCH/GREP, in order, for RANK padding
        last_key: tuple[str, str] | None = None  # detect a model stuck repeating one action
        repeats = 0

        for step in range(self._max_steps):
            resp = await self._llm.chat(
                messages, self._model, temperature=self._temperature, max_tokens=self._max_tokens
            )
            result.tokens_in += resp.tokens_in
            result.tokens_out += resp.tokens_out
            text = resp.text.strip()
            verb, arg = self._parse(text)

            if verb is None:
                obs = "Could not parse an ACTION. Reply with exactly:\nTHOUGHT: ...\nACTION: SEARCH <terms>"
                messages += [
                    Message(role="assistant", content=text),
                    Message(role="user", content=obs),
                ]
                result.steps.append(DciStep(action="(unparsed)", arg="", observation=obs))
                continue

            if verb == "ANSWER":
                result.answer = arg.strip()
                result.ranked_doc_ids = self._pad(result.ranked_doc_ids, discovered, top_k)
                result.stopped = "final"
                result.steps.append(DciStep(action="ANSWER", arg=arg.strip(), observation=""))
                return result
            if verb == "RANK":
                ids = [d.strip() for d in re.split(r"[,\s]+", arg.strip()) if d.strip()]
                result.ranked_doc_ids = self._pad(ids, discovered, top_k)
                result.stopped = "final"
                result.steps.append(DciStep(action="RANK", arg=arg.strip(), observation=""))
                return result

            # Loop-breaker: a model that repeats the exact same action makes no
            # progress and burns the budget. Nudge it; force-finalize if it persists.
            key = (verb, " ".join(arg.split()).lower()[:120])
            if key == last_key:
                repeats += 1
                if repeats >= 2 and discovered:
                    result.ranked_doc_ids = self._pad(result.ranked_doc_ids, discovered, top_k)
                    result.stopped = "looped"
                    return result
                obs = (
                    "You already ran that exact action and the results are unchanged. Do something "
                    "DIFFERENT: FILTER on 2-3 specific terms, READ a candidate to verify, or output "
                    "your final RANK / ANSWER now."
                )
            else:
                repeats = 0
                obs = self._execute(verb, arg, discovered)
            last_key = key
            # Force convergence: many models explore until the budget runs out and
            # never RANK. Demand a final answer in the last couple of turns.
            if self._max_steps - step <= 2:
                want = "RANK <doc ids, best first>" if mode == "retrieval" else "ANSWER <answer>"
                obs += f"\n\n(Only {self._max_steps - step - 1} turn(s) left — reply with ACTION: {want} NOW.)"
            messages += [Message(role="assistant", content=text), Message(role="user", content=obs)]
            result.steps.append(DciStep(action=verb, arg=arg.strip(), observation=obs[:600]))

        # budget exhausted — fall back to discovery order so retrieval still scores
        result.ranked_doc_ids = self._pad(result.ranked_doc_ids, discovered, top_k)
        return result

    def _parse(self, text: str) -> tuple[str | None, str]:
        matches = list(_ACTION_RE.finditer(text))
        if not matches:
            return None, ""
        m = matches[-1]  # last ACTION wins if the model emitted several
        return m.group(1).upper(), m.group(2).strip()

    def _execute(self, verb: str, arg: str, discovered: list[str]) -> str:
        if self._toolset == "readgrep" and verb in {"FILTER", "COUNT", "SCRIPT"}:
            return f"{verb} is not available. Use SEARCH, GREP, or READ."
        if verb == "SEARCH":
            hits = self._tools.search(arg, top_k=self._search_k)
            for h in hits:
                if h.doc_id not in discovered:
                    discovered.append(h.doc_id)
            return _format_hits(
                [f"- {h.doc_id} | {h.score} matches | {h.snippet}" for h in hits],
                f'SEARCH "{arg}" (doc_id | matches | snippet)',
            )
        if verb == "FILTER":
            terms = [
                t.strip().strip('"')
                for t in re.split(r",|\bAND\b", arg, flags=re.IGNORECASE)
                if t.strip()
            ]
            hits = self._tools.filter_all(terms, top_k=self._search_k)
            for h in hits:
                if h.doc_id not in discovered:
                    discovered.append(h.doc_id)
            return _format_hits(
                [f"- {h.doc_id} | {h.score} | {h.snippet}" for h in hits],
                f"FILTER (docs with ALL of {terms}) (doc_id | matches | snippet)",
            )
        if verb == "COUNT":
            docs, occ = self._tools.count(arg.strip().strip('"'))
            return f'COUNT "{arg.strip()}": {docs} docs contain it ({occ} total occurrences)'
        if verb == "SCRIPT":
            out = self._tools.run_script(arg)
            for tok in re.findall(r"\bd\d+\b", out):  # surface doc ids a script returned
                if tok not in discovered:
                    discovered.append(tok)
            return f"SCRIPT result: {out}"
        if verb == "GREP":
            phrase = arg.strip()
            fixed = False
            if len(phrase) >= 2 and phrase[0] == '"' and phrase[-1] == '"':
                phrase, fixed = phrase[1:-1], True
            try:
                lines = self._tools.grep(phrase, top_k=self._grep_k, fixed=fixed)
            except ValueError as exc:
                return f"GREP error: {exc}. Try a simpler pattern or a quoted phrase."
            for lh in lines:
                if lh.doc_id not in discovered:
                    discovered.append(lh.doc_id)
            return _format_hits(
                [f"- {h.doc_id}:{h.line} | {h.text}" for h in lines],
                f"GREP {arg.strip()} (doc_id:line | text)",
            )
        # READ
        rm = _READ_RE.match(arg.strip())
        if rm is None:
            return "READ usage: ACTION: READ <doc_id> <start> <end>"
        doc_id = rm.group(1)
        start = int(rm.group(2)) if rm.group(2) else 1
        end = int(rm.group(3)) if rm.group(3) else start + 30
        return f"READ {doc_id} [{start}-{end}]:\n" + self._tools.read(doc_id, start=start, end=end)

    @staticmethod
    def _pad(primary: list[str], discovered: list[str], top_k: int) -> list[str]:
        out = list(dict.fromkeys(primary))  # dedupe, keep order
        for d in discovered:
            if len(out) >= top_k:
                break
            if d not in out:
                out.append(d)
        return out[:top_k]
