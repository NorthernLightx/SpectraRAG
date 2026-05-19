"""Page-text hygiene: header detection/stripping and number-soup filtering (ADR 0017)."""

from __future__ import annotations

from src.ingestion.clean import detect_running_header, is_soup, strip_page_furniture


def test_detect_running_header_finds_shared_first_line() -> None:
    pages = [
        "Preprint. Under review.\nIntro body",
        "Preprint. Under review.\nMethod body",
        "Preprint. Under review.\nResults body",
    ]
    assert detect_running_header(pages) == "Preprint. Under review."


def test_detect_running_header_none_when_no_dominant_line() -> None:
    assert detect_running_header(["Alpha\nx", "Beta\ny", "Gamma\nz"]) is None


def test_detect_running_header_none_with_too_few_pages() -> None:
    assert detect_running_header(["Header\na", "Header\nb"]) is None


def test_strip_page_furniture_removes_header_and_page_numbers() -> None:
    text = "Header X\n12\nReal body line\nmore body\n7"
    assert strip_page_furniture(text, "Header X") == "Real body line\nmore body"


def test_strip_page_furniture_keeps_deep_numeric_line() -> None:
    # A bare number that is not leading/trailing furniture (e.g. a real datum)
    # must survive — only page-edge numbers are page numbers.
    text = "Header X\nThe model has\n2048\nhidden units"
    assert strip_page_furniture(text, "Header X") == "The model has\n2048\nhidden units"


def test_strip_page_furniture_no_header_detected_is_noop_on_body() -> None:
    assert strip_page_furniture("Body only, no furniture", None) == "Body only, no furniture"


def test_is_soup_true_for_number_grid() -> None:
    assert is_soup("64 128 256 384 512 0.000122 0.000173 2.127 2.126 2.134 2.142 2.150")


def test_is_soup_true_for_empty() -> None:
    assert is_soup("   \n  ")


def test_is_soup_false_for_prose() -> None:
    assert not is_soup(
        "The method uses a transformer to encode inputs and then retrieves "
        "relevant passages from the corpus before generating an answer."
    )


def test_is_soup_false_for_equation_bearing_prose() -> None:
    # Equation-dense but meaningful: the prose-hint guard must keep it.
    assert not is_soup(
        "We approximate the posterior with a Gaussian mixture where the mean "
        "is theta and the covariance Sigma controls how the basins spread."
    )
