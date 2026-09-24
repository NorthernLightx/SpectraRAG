"""Rerank the visual leg's top-N pages with a page-image reranker.

Reads the visual leg a run recorded (`per_query[].leg_chunk_ids["visual"]`, or
`retrieved_chunk_ids` of a visual-only run with `--leg retrieved`), scores each
query's top-N candidate pages with a pointwise scorer, and writes two ordinary
run JSONs over the same queries and the same candidates:

  <stem>[.sub<n>-s<seed>].visual-top<N>.json            the leg's own order
  <stem>[.sub<n>-s<seed>].visual-top<N>.<scorer>.json   the reranker's order

The pair differs only in order, so recall@N is equal by construction (the
ceiling any reranker can reach) and `scripts/experiments/paired_arm_compare.py`
reads the two as they are. Metrics are page-level and paper-aware
(`scripts/rescore_mmlb_pages.rescore`, the scoring `derive_arms` uses), plus
recall@5 and recall@N.

Scores append to a JSONL cache keyed by (query_id, page_id) and stamped with the
scorer fingerprint, so a killed run resumes where it stopped, and a cache from
another model, revision, pixel cap or instruction fails the run instead of
mixing in.

The default scorer is Qwen3-VL-Reranker-2B (arXiv 2601.04720): the pair is
rendered with the model's own `reranker` chat template and scored as
logit("yes") - logit("no") at the last position, projected in fp32. That
orders pages the same way as the model card's sigmoid score, without its
saturation or bf16's rounding ties. Page images are
resized under `--max-pixels`; an uncapped Qwen2-VL processor fed 150-DPI
MMLongBench pages at ~2.7k tokens and ~11k ViT patches each, which is what
filled the 8 GB card in May.

Run (the ADR 0012 2026-09-24 amendment; drop --subset for all 1,127 queries):
  .venv/Scripts/python.exe -m scripts.experiments.visual_rerank_probe \\
      --run data/eval/mmdocir-depth50-legs.json.gz \\
      --golden data/golden/mmdocir-v1.yaml --pages-dir data/mmdocir/pages \\
      --out-dir <dir> --depth 20 --subset 300 --seed 20260924
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

import yaml

from scripts.derive_arms import read_run
from scripts.derive_arms import run_stem as _file_stem
from scripts.experiments.paired_arm_compare import _bootstrap_ci, _sign_test
from scripts.rescore_mmlb_pages import _dedup_pages_in_rank, _recall_at_k, rescore
from src.eval.metrics_retrieval import ndcg_at_k

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_PAGE_RE = re.compile(r"^(?P<paper>.+?)::p(?P<page>\d+)(?:::|$)")
_IMAGE_SUFFIXES = (".jpg", ".png", ".jpeg")
_COMPARE_METRICS = ("recall_at_5", "ndcg_at_5", "ndcg_at_5_standard", "mrr", "recall_at_10")

DEFAULT_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
# The model's own image cap (preprocessor_config.json, reference scorer): 1280
# merged tokens of 32x32 px. MMDocIR pages (median 612x792) sit below it.
DEFAULT_MAX_PIXELS = 1280 * 32 * 32
DEFAULT_MIN_PIXELS = 4 * 32 * 32
# The instruction the model's reranker template falls back to; fixed rather
# than tuned on the eval queries.
DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."


class PageScorer(Protocol):
    """Scores page images against one query; higher means more relevant."""

    fingerprint: str
    config: dict[str, Any]

    def score(self, query: str, images: Sequence[Path]) -> list[float]: ...

    def stats(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Candidate:
    query_id: str
    category: str
    text: str
    pages: list[str]


def page_id_of(chunk_id: str) -> str | None:
    """`paper::pN::cM` or `paper::pN::page` -> `paper::pN::page`."""
    match = _PAGE_RE.match(chunk_id)
    if match is None:
        return None
    return f"{match['paper']}::p{int(match['page'])}::page"


def candidate_pages(ranked_ids: Iterable[str], depth: int) -> list[str]:
    """The first `depth` distinct pages in rank order."""
    seen: set[str] = set()
    pages: list[str] = []
    for chunk_id in ranked_ids:
        page = page_id_of(chunk_id)
        if page is None or page in seen:
            continue
        seen.add(page)
        pages.append(page)
        if len(pages) == depth:
            break
    return pages


def merge_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One run from query slices of the same eval, in the order given.

    Slices must share the golden set and the retrieval fingerprint and must not
    repeat a query; otherwise the merged legs would not come from one stack.
    """
    if len(runs) == 1:
        return runs[0]
    first = runs[0]
    for run in runs[1:]:
        for key in ("golden_set_name", "golden_set_version"):
            if run[key] != first[key]:
                raise SystemExit(f"slices disagree on {key}: {first[key]!r} vs {run[key]!r}")
        fps = {(r.get("config") or {}).get("retrieval_fingerprint") for r in (first, run)}
        if len(fps) != 1:
            raise SystemExit(f"slices disagree on retrieval_fingerprint: {sorted(map(str, fps))}")
    per_query = [pq for run in runs for pq in run["per_query"]]
    ids = [pq["query_id"] for pq in per_query]
    if len(set(ids)) != len(ids):
        dup = next(q for q in ids if ids.count(q) > 1)
        raise SystemExit(f"query {dup} appears in more than one slice")
    return {
        **first,
        "run_id": "+".join(r["run_id"] for r in runs),
        "started_at": min(r["started_at"] for r in runs),
        "finished_at": max(r["finished_at"] for r in runs),
        "per_query": per_query,
    }


def load_candidates(run: dict[str, Any], *, leg: str, depth: int) -> list[Candidate]:
    """One Candidate per query in run order; a query without the leg fails the run."""
    out: list[Candidate] = []
    missing: list[str] = []
    for pq in run["per_query"]:
        if leg == "retrieved":
            ranked = pq.get("retrieved_chunk_ids")
        else:
            ranked = (pq.get("leg_chunk_ids") or {}).get(leg)
        if not ranked:
            missing.append(pq["query_id"])
            continue
        out.append(
            Candidate(
                query_id=pq["query_id"],
                category=pq["category"],
                text=pq["text"],
                pages=candidate_pages(ranked, depth),
            )
        )
    if missing:
        raise SystemExit(
            f"{len(missing)} queries have no {leg!r} ranking (first: {missing[0]}). "
            "Record the run with eval_run --router --force-route hybrid, or pass "
            "--leg retrieved for a visual-only run."
        )
    return out


def stratified_subset(items: Sequence[tuple[str, str]], n: int, seed: int) -> list[str]:
    """`n` query ids drawn per category in proportion to its size.

    `items` is (query_id, category). Quotas are floors of the proportional
    share with the remainder going to the largest fractional parts; each
    category is then sampled from its sorted ids with one seeded generator.
    """
    if n >= len(items):
        return [qid for qid, _ in items]
    by_cat: dict[str, list[str]] = {}
    for qid, cat in items:
        by_cat.setdefault(cat, []).append(qid)
    total = len(items)
    shares = {cat: n * len(ids) / total for cat, ids in by_cat.items()}
    quotas = {cat: int(share) for cat, share in shares.items()}
    leftover = n - sum(quotas.values())
    for cat in sorted(shares, key=lambda c: (-(shares[c] - quotas[c]), c))[:leftover]:
        quotas[cat] += 1
    rng = random.Random(seed)
    chosen: set[str] = set()
    for cat in sorted(by_cat):
        chosen.update(rng.sample(sorted(by_cat[cat]), quotas[cat]))
    return [qid for qid, _ in items if qid in chosen]


def rerank(pages: Sequence[str], scores: dict[str, float]) -> list[str]:
    """Pages by descending score; equal scores keep the first-stage order."""
    order = {page: rank for rank, page in enumerate(pages)}
    return sorted(pages, key=lambda p: (-scores[p], order[p]))


def resolve_image(pages_dir: Path, page_id: str) -> Path:
    """`paper::pN::page` -> `<pages_dir>/<paper>/<paper>_pN.<ext>`, the
    `src.ingestion.visual.render_pages` layout the page index was built from."""
    match = _PAGE_RE.match(page_id)
    if match is None:
        raise ValueError(f"not a page id: {page_id}")
    paper, page = match["paper"], int(match["page"])
    for suffix in _IMAGE_SUFFIXES:
        path = pages_dir / paper / f"{paper}_p{page}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"no page image for {page_id} under {pages_dir / paper}")


class ScoreCache:
    """Append-only JSONL of (query_id, page_id) -> score for one scorer.

    A line torn by a crash mid-write is cut off on load, so the next append
    starts on a clean line. The append handle stays open between batches:
    reopening the file costs time that grows with its size on this box (124 ms
    per append at 3 MB, 2 ms for a small file), which kept the GPU idle.
    """

    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self._scores: dict[tuple[str, str], float] = {}
        self._ms: dict[tuple[str, str], float] = {}
        self._fh: IO[str] | None = None
        self._load()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            cut = raw.rfind(b"\n") + 1
            raw = raw[:cut]
            self.path.write_bytes(raw)
        for lineno, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["scorer"] != self.fingerprint:
                raise SystemExit(
                    f"{self.path}:{lineno} was scored by {rec['scorer']!r}, not "
                    f"{self.fingerprint!r}. Use another --cache for this scorer."
                )
            key = (rec["query_id"], rec["page_id"])
            self._scores[key] = float(rec["score"])
            self._ms[key] = float(rec["ms"])

    def __contains__(self, key: object) -> bool:
        return key in self._scores

    def __len__(self) -> int:
        return len(self._scores)

    def score(self, query_id: str, page_id: str) -> float:
        return self._scores[(query_id, page_id)]

    def ms(self, query_id: str, page_id: str) -> float:
        return self._ms[(query_id, page_id)]

    def add(self, query_id: str, pages: Sequence[str], scores: Sequence[float], ms: float) -> None:
        if len(pages) != len(scores):
            raise ValueError(f"{len(pages)} pages but {len(scores)} scores")
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        fh = self._fh
        for page, score in zip(pages, scores, strict=True):
            rec = {
                "query_id": query_id,
                "page_id": page,
                "score": score,
                "ms": round(ms, 2),
                "scorer": self.fingerprint,
            }
            fh.write(json.dumps(rec) + "\n")
            self._scores[(query_id, page)] = float(score)
            self._ms[(query_id, page)] = ms
        fh.flush()
        os.fsync(fh.fileno())


@dataclass
class SlowPairGuard:
    """Stops the run when the rolling median seconds per pair exceeds a limit.

    On Windows an over-full card spills to shared system memory instead of
    raising, so a VRAM problem shows up only as a slow run.
    """

    limit_s: float
    warmup: int = 8
    window: int = 40
    _seen: list[float] = field(default_factory=list)

    def observe(self, sec_per_pair: float, n_pairs: int) -> None:
        self._seen.extend([sec_per_pair] * n_pairs)
        recent = self._seen[self.warmup :][-self.window :]
        if len(recent) >= min(self.window, 10) and statistics.median(recent) > self.limit_s:
            raise SystemExit(
                f"median {statistics.median(recent):.2f} s/pair over the last {len(recent)} "
                f"pairs exceeds {self.limit_s:.2f} s. VRAM may be spilling; stopping."
            )


def score_all(
    candidates: Sequence[Candidate],
    scorer: PageScorer,
    cache: ScoreCache,
    pages_dir: Path,
    *,
    batch_size: int,
    guard: SlowPairGuard | None = None,
    progress_every: int = 25,
) -> int:
    """Scores every (query, page) pair not in the cache; returns pairs scored."""
    todo_total = sum(1 for c in candidates for p in c.pages if (c.query_id, p) not in cache)
    done = 0
    start = time.perf_counter()
    for qi, cand in enumerate(candidates, start=1):
        todo = [p for p in cand.pages if (cand.query_id, p) not in cache]
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            images = [resolve_image(pages_dir, p) for p in batch]
            t0 = time.perf_counter()
            scores = scorer.score(cand.text, images)
            dt = time.perf_counter() - t0
            cache.add(cand.query_id, batch, scores, ms=1000.0 * dt / len(batch))
            done += len(batch)
            if guard is not None:
                guard.observe(dt / len(batch), len(batch))
        if todo and (qi % progress_every == 0 or qi == len(candidates)):
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta_min = (todo_total - done) / rate / 60 if rate > 0 else float("nan")
            print(
                f"  q {qi}/{len(candidates)}  pairs {done}/{todo_total}  "
                f"{elapsed / max(done, 1):.2f} s/pair  eta {eta_min:.0f} min  {scorer.stats()}",
                flush=True,
            )
    return done


def _extra_metrics(run: dict[str, Any], golden: dict[str, Any], depth: int) -> None:
    """Adds recall@5, recall@depth and a standard nDCG@5 next to the rescored
    metrics, in place.

    `rescore` takes nDCG's ideal from the gold pages it retrieved, while the
    committed MMDocIR baselines hold `ndcg_at_k`, whose ideal counts every gold
    page up to k. The two differ on multi-gold queries with a gold page missing
    from the top 5, so both are kept: `ndcg_at_5` pairs with `derive_arms`
    arms, `ndcg_at_5_standard` with the baselines.
    """
    relevant = {
        q["query_id"]: {(q["paper_id"], p) for p in q.get("relevant_pages") or []}
        for q in golden["queries"]
    }
    for pq in run["per_query"]:
        gold = relevant.get(pq["query_id"], set())
        pages = _dedup_pages_in_rank(pq["retrieved_chunk_ids"])
        for k in sorted({5, depth}):
            pq["retrieval"][f"recall_at_{k}"] = _recall_at_k(pages, gold, k=k) if gold else None
        page_keys = [f"{paper}::p{n}" for paper, n in pages]
        gold_keys = [f"{paper}::p{n}" for paper, n in gold]
        pq["retrieval"]["ndcg_at_5_standard"] = (
            ndcg_at_k(gold_keys, page_keys, k=5) if gold else None
        )


def build_arm(
    source: dict[str, Any],
    golden: dict[str, Any],
    ordered: dict[str, list[str]],
    candidates: Sequence[Candidate],
    *,
    arm: str,
    config: dict[str, Any],
    latency_ms: dict[str, int] | None = None,
) -> dict[str, Any]:
    """A run JSON in the `derive_arms` shape, scored page-level and paper-aware."""
    src_config = source.get("config") or {}
    per_query = [
        {
            "query_id": c.query_id,
            "category": c.category,
            "text": c.text,
            "retrieved_chunk_ids": ordered[c.query_id],
            "retrieval": {"ndcg_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0},
            "latency_ms": (latency_ms or {}).get(c.query_id, 0),
        }
        for c in candidates
    ]
    run = {
        "run_id": f"{source['run_id']}-{arm}",
        "started_at": source["started_at"],
        "finished_at": source["finished_at"],
        "golden_set_name": source["golden_set_name"],
        "golden_set_version": source["golden_set_version"],
        "config": {
            "derived_from": source["run_id"],
            "arm": arm,
            **config,
            "retrieval_fingerprint": src_config.get("retrieval_fingerprint"),
            "retrieval_config": src_config.get("retrieval_config"),
        },
        "per_query": per_query,
    }
    scored = rescore(run, golden)
    _extra_metrics(scored, golden, config["depth"])
    return scored


def _values(run: dict[str, Any], metric: str) -> dict[str, tuple[str, float]]:
    return {
        pq["query_id"]: (pq["category"], float(pq["retrieval"][metric]))
        for pq in run["per_query"]
        if (pq.get("retrieval") or {}).get(metric) is not None
    }


def paired_table(a: dict[str, Any], b: dict[str, Any], metrics: Sequence[str]) -> list[str]:
    """Rows of A-minus-B paired deltas per metric, overall and per category."""
    rows = [
        f"{'metric':<20}{'subset':<9}{'n':>5}{'A':>8}{'B':>8}{'delta':>9}"
        f"{'95% CI':>20}{'A>B':>5}{'B>A':>5}{'tie':>5}{'sign p':>9}"
    ]
    for metric in metrics:
        va, vb = _values(a, metric), _values(b, metric)
        shared = sorted(set(va) & set(vb))
        cats = sorted({va[q][0] for q in shared})
        for subset in ["all", *cats]:
            qs = [q for q in shared if subset == "all" or va[q][0] == subset]
            if not qs:
                continue
            deltas = [va[q][1] - vb[q][1] for q in qs]
            wins = sum(d > 0 for d in deltas)
            losses = sum(d < 0 for d in deltas)
            lo, hi = _bootstrap_ci(deltas)
            rows.append(
                f"{metric:<20}{subset:<9}{len(qs):>5}"
                f"{statistics.fmean(va[q][1] for q in qs):>8.4f}"
                f"{statistics.fmean(vb[q][1] for q in qs):>8.4f}"
                f"{statistics.fmean(deltas):>+9.4f}"
                f"{f'[{lo:+.4f}, {hi:+.4f}]':>20}"
                f"{wins:>5}{losses:>5}{len(qs) - wins - losses:>5}"
                f"{_sign_test(wins, losses):>9.3g}"
            )
    return rows


def ceiling_lines(first: dict[str, Any], depth: int) -> list[str]:
    """What any reordering of the candidates can reach."""
    per = [pq["retrieval"] for pq in first["per_query"] if pq["retrieval"]["mrr"] is not None]
    n = len(per)
    hit = sum(1 for r in per if (r[f"recall_at_{depth}"] or 0) > 0)
    top1 = sum(1 for r in per if r["mrr"] == 1.0)
    return [
        f"queries scored               {n}",
        f"recall@{depth} (ceiling for recall@5)  {statistics.fmean(r[f'recall_at_{depth}'] for r in per):.4f}",
        f"any gold in top-{depth}          {hit / n:.4f}  ({hit})",
        f"gold already at rank 1        {top1 / n:.4f}  ({top1})",
        f"reorderable headroom          {(hit - top1) / n:.4f}  ({hit - top1} queries with a gold page below rank 1)",
    ]


class Qwen3VLRerankerScorer:
    """Qwen3-VL-Reranker as a pointwise page scorer on one CUDA device.

    Renders each pair with the checkpoint's `reranker` chat template and scores
    the last position's hidden state against lm_head[yes] - lm_head[no], which
    is logit("yes") - logit("no"). The projection runs in fp32: bf16 logits
    land on a grid of about 0.06, which tied pages that the fp32 dot product
    tells apart.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL,
        revision: str = DEFAULT_REVISION,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        instruction: str = DEFAULT_INSTRUCTION,
        device: str = "cuda",
        vram_budget_gb: float | None = 6.0,
    ) -> None:
        import torch
        import transformers
        from huggingface_hub import hf_hub_download

        processor_cls: Any = transformers.AutoProcessor
        model_cls: Any = transformers.Qwen3VLForConditionalGeneration
        self._torch = torch
        self.device = device
        if device.startswith("cuda") and vram_budget_gb is not None:
            total = torch.cuda.get_device_properties(0).total_memory
            # Makes this process raise OOM past the budget rather than grow into
            # the desktop's share and push the card into system-memory fallback.
            torch.cuda.set_per_process_memory_fraction(min(1.0, vram_budget_gb * 2**30 / total))
        self.processor = processor_cls.from_pretrained(model_id, revision=revision)
        self.processor.tokenizer.padding_side = "left"
        image_processor = self.processor.image_processor
        image_processor.size = {"longest_edge": max_pixels, "shortest_edge": min_pixels}
        for attr, value in (("max_pixels", max_pixels), ("min_pixels", min_pixels)):
            if hasattr(image_processor, attr):
                setattr(image_processor, attr, value)
        template_path = hf_hub_download(
            model_id, "additional_chat_templates/reranker.jinja", revision=revision
        )
        self._template = Path(template_path).read_text(encoding="utf-8")
        vocab = self.processor.tokenizer.get_vocab()
        self._yes, self._no = vocab["yes"], vocab["no"]
        # device_map loads the weights straight onto the device, with no full
        # host-RAM copy on the way.
        self.model = model_cls.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        ).eval()
        head = self.model.lm_head.weight
        self._direction = (head[self._yes].float() - head[self._no].float()).detach()
        self.config = {
            "model": model_id,
            "revision": revision,
            "max_pixels": max_pixels,
            "min_pixels": min_pixels,
            "instruction": instruction,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "score": "logit(yes) - logit(no), last position, fp32 projection",
        }
        self.fingerprint = scorer_fingerprint(self.config)
        self._instruction = instruction
        self._peak_device_used = 0.0
        self._max_image_tokens = 0

    def prompt(self, query: str) -> str:
        messages = [
            {"role": "system", "content": self._instruction},
            {"role": "query", "content": [{"type": "text", "text": query}]},
            {"role": "document", "content": [{"type": "image"}]},
        ]
        text: str = self.processor.apply_chat_template(
            messages, chat_template=self._template, tokenize=False, add_generation_prompt=True
        )
        return text

    def score(self, query: str, images: Sequence[Path]) -> list[float]:
        from PIL import Image

        torch = self._torch
        pil = []
        for path in images:
            with Image.open(path) as im:
                pil.append(im.convert("RGB"))
        prompt = self.prompt(query)
        inputs = self.processor(
            text=[prompt] * len(pil), images=pil, return_tensors="pt", padding=True
        ).to(self.device)
        grid = inputs["image_grid_thw"]
        self._max_image_tokens = max(
            self._max_image_tokens, int((grid.prod(dim=-1) // 4).max().item())
        )
        with torch.inference_mode():
            hidden = self.model.model(**inputs).last_hidden_state[:, -1, :].float()
        scores: list[float] = (hidden @ self._direction).tolist()
        if self.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            self._peak_device_used = max(self._peak_device_used, (total - free) / 2**30)
        return scores

    def stats(self) -> dict[str, Any]:
        torch = self._torch
        out: dict[str, Any] = {"max_image_tokens": self._max_image_tokens}
        if self.device.startswith("cuda"):
            out["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            out["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            out["peak_device_used_gib"] = round(self._peak_device_used, 2)
        return out


def scorer_fingerprint(config: dict[str, Any]) -> str:
    """Model and revision in the clear, the rest hashed: equal fingerprints mean
    the same weights, image cap, instruction and score definition."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{config['model']}@{str(config['revision'])[:12]}#{digest}"


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model_id.rsplit("/", 1)[-1]).strip("-").lower()


def _latency(candidates: Sequence[Candidate], cache: ScoreCache) -> dict[str, int]:
    return {c.query_id: round(sum(cache.ms(c.query_id, p) for p in c.pages)) for c in candidates}


def _pair_ms(candidates: Sequence[Candidate], cache: ScoreCache) -> list[float]:
    return [cache.ms(c.query_id, p) for c in candidates for p in c.pages]


def run_probe(
    *,
    run: dict[str, Any],
    run_stem: str,
    golden: dict[str, Any],
    scorer: PageScorer,
    pages_dir: Path,
    out_dir: Path,
    depth: int,
    leg: str,
    subset: int | None,
    seed: int,
    batch_size: int,
    cache_path: Path | None = None,
    guard: SlowPairGuard | None = None,
) -> tuple[Path, Path]:
    """Scores, reranks and writes the first-stage and reranked arms."""
    candidates = load_candidates(run, leg=leg, depth=depth)
    tag = ""
    if subset is not None:
        keep = set(stratified_subset([(c.query_id, c.category) for c in candidates], subset, seed))
        candidates = [c for c in candidates if c.query_id in keep]
        tag = f".sub{subset}-s{seed}"
    for cand in candidates:
        for page in cand.pages:
            resolve_image(pages_dir, page)
    slug = _slug(scorer.config["model"])
    cache = ScoreCache(
        cache_path or out_dir / f"{run_stem}.visual.{slug}.scores.jsonl", scorer.fingerprint
    )
    n_pairs = sum(len(c.pages) for c in candidates)
    print(
        f"{len(candidates)} queries, {n_pairs} pairs, {len(cache)} cached, scorer {scorer.fingerprint}"
    )
    try:
        scored_now = score_all(
            candidates, scorer, cache, pages_dir, batch_size=batch_size, guard=guard
        )
    finally:
        cache.close()

    first_order = {c.query_id: list(c.pages) for c in candidates}
    reranked = {
        c.query_id: rerank(c.pages, {p: cache.score(c.query_id, p) for p in c.pages})
        for c in candidates
    }
    base_cfg: dict[str, Any] = {
        "top_k": depth,
        "depth": depth,
        "leg": leg,
        "subset": None if subset is None else {"n": subset, "seed": seed},
    }
    pair_ms = _pair_ms(candidates, cache)
    first = build_arm(
        run, golden, first_order, candidates, arm=f"visual-top{depth}", config=base_cfg
    )
    second = build_arm(
        run,
        golden,
        reranked,
        candidates,
        arm=f"visual-top{depth}-{slug}",
        config={
            **base_cfg,
            "scorer": scorer.config,
            "scorer_fingerprint": scorer.fingerprint,
            "scorer_stats": {
                **scorer.stats(),
                "pairs_scored_this_session": scored_now,
                "ms_per_pair_p50": round(statistics.median(pair_ms), 1) if pair_ms else None,
                "ms_per_pair_p95": round(sorted(pair_ms)[int(0.95 * (len(pair_ms) - 1))], 1)
                if pair_ms
                else None,
                "batch_size": batch_size,
            },
        },
        latency_ms=_latency(candidates, cache),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    first_path = out_dir / f"{run_stem}{tag}.visual-top{depth}.json"
    second_path = out_dir / f"{run_stem}{tag}.visual-top{depth}.{slug}.json"
    for path, arm_run in ((first_path, first), (second_path, second)):
        path.write_text(json.dumps(arm_run, indent=2), encoding="utf-8", newline="\n")

    print("\nCeiling (first-stage candidates):")
    for line in ceiling_lines(first, depth):
        print(f"  {line}")
    print(f"\nPaired, A = reranked ({slug}), B = first-stage order, same candidates:")
    for line in paired_table(second, first, _COMPARE_METRICS):
        print(f"  {line}")
    print(f"\nscorer stats {second['config']['scorer_stats']}")
    print(f"Wrote {first_path}\nWrote {second_path}")
    return first_path, second_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        type=Path,
        nargs="+",
        required=True,
        help="run JSON with the visual leg; several are merged as query slices of one eval",
    )
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, default=Path("data/mmdocir/pages"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=20, help="candidates reranked per query")
    parser.add_argument(
        "--leg",
        default="visual",
        help="leg_chunk_ids key to rerank, or 'retrieved' for retrieved_chunk_ids",
    )
    parser.add_argument("--subset", type=int, help="stratified query subset size")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache", type=Path, help="score cache JSONL (default under --out-dir)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--vram-budget-gb", type=float, default=6.0)
    parser.add_argument(
        "--max-sec-per-pair",
        type=float,
        default=3.0,
        help="stop when the rolling median exceeds this (0 disables)",
    )
    args = parser.parse_args()

    run = merge_runs([read_run(p) for p in args.run])
    first = _file_stem(args.run[0])
    stem = first if len(args.run) == 1 else f"{first}+{len(args.run) - 1}"
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    # Bad legs or missing page renders fail here, before the model loads.
    for cand in load_candidates(run, leg=args.leg, depth=args.depth):
        for page in cand.pages:
            resolve_image(args.pages_dir, page)
    scorer = Qwen3VLRerankerScorer(
        model_id=args.model,
        revision=args.revision,
        max_pixels=args.max_pixels,
        vram_budget_gb=args.vram_budget_gb,
    )
    run_probe(
        run=run,
        run_stem=stem,
        golden=golden,
        scorer=scorer,
        pages_dir=args.pages_dir,
        out_dir=args.out_dir,
        depth=args.depth,
        leg=args.leg,
        subset=args.subset,
        seed=args.seed,
        batch_size=args.batch_size,
        cache_path=args.cache,
        guard=SlowPairGuard(args.max_sec_per_pair) if args.max_sec_per_pair > 0 else None,
    )


if __name__ == "__main__":
    main()
