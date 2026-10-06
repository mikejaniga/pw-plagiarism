"""DataLoader module tests."""

from pathlib import Path

from plagiarism_detector.data_loader import DataLoader


def test_clean_text() -> None:
    loader = DataLoader()
    raw = "To   jest   tekst.\r\n\r\n\n\nNowy    akapit."
    cleaned = loader.clean_text(raw)
    assert cleaned == "To jest tekst.\n\nNowy akapit."


def test_abbreviations_do_not_split_sentences() -> None:
    loader = DataLoader(min_sentence_length=10)
    text = "Prof. Jan Kowalski spotkał dr. Nowaka oraz mgr. Wiśniewskiego w Krakowie."
    sentences = loader.split_into_sentences(
        text, doc_id="test_doc", doc_name="test.txt"
    )

    assert len(sentences) == 1
    assert sentences[0].text == text


def test_period_after_regular_word_splits_sentence() -> None:
    loader = DataLoader(min_sentence_length=10)
    text = "To jest pierwsze zdanie. A to jest drugie zdanie."
    sentences = loader.split_into_sentences(
        text, doc_id="test_doc", doc_name="test.txt"
    )

    assert len(sentences) == 2
    assert sentences[0].text == "To jest pierwsze zdanie."
    assert sentences[1].text == "A to jest drugie zdanie."


def test_min_sentence_length_filtering() -> None:
    loader = DataLoader(min_sentence_length=30)
    text = "Krótkie zdanie. To zdanie jest zdecydowanie dłuższe i powinno przejść filtr długości."
    sentences = loader.split_into_sentences(
        text, doc_id="test_doc", doc_name="test.txt"
    )

    assert len(sentences) == 1
    assert "powinno przejść filtr" in sentences[0].text


def test_load_document(tmp_path: Path) -> None:
    loader = DataLoader(min_sentence_length=15)
    sample_file = tmp_path / "artykul_testowy.txt"
    sample_file.write_text(
        "To jest pierwszy pełny akapit testowy do weryfikacji. A to jest drugie poprawne zdanie.",
        encoding="utf-8",
    )

    doc = loader.load_document(sample_file)
    assert doc.doc_id == "artykul_testowy"
    assert doc.title == "Artykul Testowy"
    assert len(doc.sentences) == 2
