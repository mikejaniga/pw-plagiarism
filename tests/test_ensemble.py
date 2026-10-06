"""EnsemblePlagiarismDetector module and similarity computations tests."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from plagiarism_detector.data_loader import Document, SentenceSegment
from plagiarism_detector.models.embedder import TextEmbedder
from plagiarism_detector.models.ensemble import (
    EnsemblePlagiarismDetector,
    PairwiseDocResult,
)


def test_compute_similarity_matrix_normalized() -> None:
    a: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]])
    b: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim: np.ndarray = TextEmbedder.compute_similarity_matrix(a, b)

    assert sim.shape == (2, 2)
    assert np.isclose(sim[0, 0], 1.0)
    assert np.isclose(sim[0, 1], 0.0)
    assert np.isclose(sim[1, 0], 0.0)
    assert np.isclose(sim[1, 1], 1.0)


def _make_embedder(text_to_vector: dict[str, list[float]]) -> MagicMock:
    mock: MagicMock = MagicMock()

    def _embed(texts: list[str], **_kwargs: object) -> np.ndarray:
        return np.array([text_to_vector[t] for t in texts], dtype=np.float32)

    mock.embed_texts.side_effect = _embed
    return mock


def _make_document(doc_id: str, sentence_texts: list[str]) -> Document:
    sentences = [
        SentenceSegment(
            doc_id=doc_id,
            doc_name=f"{doc_id}.txt",
            sentence_idx=i,
            text=t,
            char_start=0,
            char_end=len(t),
        )
        for i, t in enumerate(sentence_texts)
    ]
    return Document(
        doc_id=doc_id,
        file_path=Path(f"{doc_id}.txt"),
        title=doc_id,
        raw_text=doc_id,
        clean_text=doc_id,
        sentences=sentences,
    )


def _compare_two_docs(vectors: dict[str, list[float]]) -> PairwiseDocResult:
    embedder = _make_embedder(vectors)
    detector = EnsemblePlagiarismDetector(
        embedder_1=embedder,
        embedder_2=embedder,
        alpha=0.5,
        sentence_threshold=0.5,
    )
    results, _, _ = detector.compare_documents(
        [_make_document("DOC_A", ["A1", "A2"]), _make_document("DOC_B", ["B1", "B2"])]
    )
    return results[0]


def test_identical_documents_are_flagged_as_very_high_risk() -> None:
    result = _compare_two_docs(
        {
            "DOC_A": [1.0, 0.0],
            "DOC_B": [1.0, 0.0],
            "A1": [1.0, 0.0],
            "A2": [1.0, 0.0],
            "B1": [1.0, 0.0],
            "B2": [1.0, 0.0],
        }
    )
    assert "Very high" in result.risk_level


def test_unrelated_documents_are_flagged_as_low_risk() -> None:
    result = _compare_two_docs(
        {
            "DOC_A": [1.0, 0.0],
            "DOC_B": [0.0, 1.0],
            "A1": [1.0, 0.0],
            "A2": [1.0, 0.0],
            "B1": [0.0, 1.0],
            "B2": [0.0, 1.0],
        }
    )
    assert "Low" in result.risk_level


def test_compare_documents_ensemble_fusion() -> None:
    mock_emb1: MagicMock = MagicMock()
    mock_emb2: MagicMock = MagicMock()

    mock_emb1.embed_texts.side_effect = lambda texts, **_kwargs: np.array(
        [[1.0, 0.0]] * len(texts)
    )
    mock_emb2.embed_texts.side_effect = lambda texts, **_kwargs: np.array(
        [[1.0, 0.0] if "D1" in t else [0.0, 1.0] for t in texts]
    )

    detector: EnsemblePlagiarismDetector = EnsemblePlagiarismDetector(
        embedder_1=mock_emb1,
        embedder_2=mock_emb2,
        alpha=0.6,
        sentence_threshold=0.5,
    )

    doc_a: Document = Document(
        doc_id="D1",
        file_path=Path("doc1.txt"),
        title="Dokument 1",
        raw_text="Tekst D1",
        clean_text="Tekst D1",
        sentences=[],
    )

    doc_b: Document = Document(
        doc_id="D2",
        file_path=Path("doc2.txt"),
        title="Dokument 2",
        raw_text="Tekst D2",
        clean_text="Tekst D2",
        sentences=[],
    )

    pairwise_results, _, matrices = detector.compare_documents([doc_a, doc_b])

    assert len(pairwise_results) == 1
    assert np.isclose(matrices["model_1"][0, 1], 1.0)
    assert np.isclose(matrices["model_2"][0, 1], 0.0)
    assert np.isclose(matrices["ensemble"][0, 1], 0.6)
    assert np.isclose(pairwise_results[0].doc_sim_ensemble, 0.6)
