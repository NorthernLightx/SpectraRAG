"""ADR 0024 route-by-fit resolver: whole-doc page images when a paper fits the budget."""

from __future__ import annotations

from pathlib import Path

from src.rag.page_budget import resolve_whole_doc_pages


def _make_doc(pages_dir: Path, paper_id: str, n_pages: int) -> None:
    """Render n_pages stub PNGs in the layout the resolver scans."""
    paper_dir = pages_dir / paper_id
    paper_dir.mkdir(parents=True)
    for p in range(1, n_pages + 1):
        (paper_dir / f"{paper_id}_p{p}.png").write_bytes(b"\x89PNG\r\n")


def test_fits_budget_returns_all_pages_as_visual_results(tmp_path: Path) -> None:
    _make_doc(tmp_path, "paperA", 3)
    out = resolve_whole_doc_pages("paperA", tmp_path, budget=5)
    assert out is not None
    assert len(out) == 3
    assert [r.page_numbers[0] for r in out] == [1, 2, 3]  # sorted
    assert all(r.source == "visual" for r in out)
    assert all(r.chunk_id == f"paperA::p{r.page_numbers[0]}::page" for r in out)
    assert all(r.paper_id == "paperA" for r in out)
    # score 1.0 so a whole-doc feed never trips the generator's refusal gate.
    assert all(r.score == 1.0 for r in out)


def test_boundary_equal_to_budget_fits(tmp_path: Path) -> None:
    # Closed interval: page_count == budget routes to whole-doc (ADR 0024).
    _make_doc(tmp_path, "paperA", 4)
    out = resolve_whole_doc_pages("paperA", tmp_path, budget=4)
    assert out is not None
    assert len(out) == 4


def test_over_budget_returns_none(tmp_path: Path) -> None:
    _make_doc(tmp_path, "paperA", 6)
    assert resolve_whole_doc_pages("paperA", tmp_path, budget=5) is None


def test_missing_doc_returns_none(tmp_path: Path) -> None:
    # No directory for this paper -> None, so the caller falls back to RAG.
    assert resolve_whole_doc_pages("ghost", tmp_path, budget=5) is None


def test_empty_doc_dir_returns_none(tmp_path: Path) -> None:
    (tmp_path / "paperA").mkdir()
    assert resolve_whole_doc_pages("paperA", tmp_path, budget=5) is None


def test_prefix_collision_does_not_leak_other_doc_pages(tmp_path: Path) -> None:
    # A paper whose id is a prefix of another must not absorb the other's pages.
    _make_doc(tmp_path, "2310.05", 2)
    # plant a stray file from a longer-id doc inside the short-id dir
    (tmp_path / "2310.05" / "2310.05634_p9.png").write_bytes(b"\x89PNG\r\n")
    out = resolve_whole_doc_pages("2310.05", tmp_path, budget=10)
    assert out is not None
    assert [r.page_numbers[0] for r in out] == [1, 2]  # the p9 of the other doc is excluded


def test_unsorted_filesystem_order_is_sorted(tmp_path: Path) -> None:
    # Page 10 must not sort before page 2 (numeric, not lexicographic).
    _make_doc(tmp_path, "paperA", 12)
    out = resolve_whole_doc_pages("paperA", tmp_path, budget=20)
    assert out is not None
    assert [r.page_numbers[0] for r in out] == list(range(1, 13))


# --- Security: paper_id is untrusted request input used as a path component ---


def test_path_traversal_paper_id_rejected(tmp_path: Path) -> None:
    # A traversal id must not escape pages_dir even when the target exists.
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "secret_p1.png").write_bytes(b"\x89PNG\r\n")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # `../secret` resolves to a real dir with a matching PNG, but is outside corpus.
    assert resolve_whole_doc_pages("../secret", corpus, budget=5) is None


def test_dotdot_and_separators_rejected(tmp_path: Path) -> None:
    for bad in ["..", "../x", "a/b", "a\\b", ".hidden", "", "a/../b"]:
        assert resolve_whole_doc_pages(bad, tmp_path, budget=5) is None


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    # A symlinked paper dir pointing outside pages_dir is rejected by the
    # resolved-containment check, even though the id itself is allowlist-clean.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil_p1.png").write_bytes(b"\x89PNG\r\n")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    try:
        (corpus / "evil").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        import pytest

        pytest.skip("symlink creation not permitted on this platform")
    assert resolve_whole_doc_pages("evil", corpus, budget=5) is None


def test_legitimate_arxiv_and_slug_ids_still_resolve(tmp_path: Path) -> None:
    # The allowlist must not reject real corpus ids (arXiv, hashes, slugs).
    for good in ["2310.05634v2", "05-03-18-political-release", "0b85477387a9d0cc"]:
        _make_doc(tmp_path, good, 2)
        out = resolve_whole_doc_pages(good, tmp_path, budget=5)
        assert out is not None and len(out) == 2
