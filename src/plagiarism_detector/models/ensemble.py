"""Module for ensembling the results of two language models and comparing their effectiveness."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from plagiarism_detector.data_loader import Document
from plagiarism_detector.models.embedder import TextEmbedder


@dataclass
class SentenceMatch:
    doc_a_name: str
    doc_a_id: str
    sent_a_idx: int
    sent_a_text: str

    doc_b_name: str
    doc_b_id: str
    sent_b_idx: int
    sent_b_text: str

    score_model_1: float
    score_model_2: float
    score_ensemble: float
    divergence: float  # |score_1 - score_2|


@dataclass
class PairwiseDocResult:
    doc_a_name: str
    doc_b_name: str
    doc_a_id: str
    doc_b_id: str

    # Global similarity of the entire texts
    doc_sim_model_1: float
    doc_sim_model_2: float
    doc_sim_ensemble: float

    # Plagiarism coverage metric (percentage of sentences in A strongly overlapping with B)
    coverage_score: float

    # Number of detected suspicious sentences
    matched_sentences_count: int

    # Mean similarity of the detected sentences
    mean_match_similarity: float

    # Plagiarism risk level
    risk_level: str


class EnsemblePlagiarismDetector:
    """Manages two models, fuses their results, and generates a comparative analysis."""

    def __init__(
        self,
        embedder_1: TextEmbedder,
        embedder_2: TextEmbedder,
        alpha: float = 0.55,
        sentence_threshold: float = 0.72,
        high_threshold: float = 0.85,
    ) -> None:
        self.embedder_1 = embedder_1
        self.embedder_2 = embedder_2
        self.alpha = alpha
        self.sentence_threshold = sentence_threshold
        self.high_threshold = high_threshold

    def _determine_risk_level(
        self, ensemble_score: float, coverage_score: float
    ) -> str:
        """Categorizes the degree of plagiarism risk."""
        combined = 0.5 * ensemble_score + 0.5 * coverage_score
        if combined >= 0.80 or coverage_score >= 0.65:
            return "Very high (Probable plagiarism/copy)"
        elif combined >= 0.65 or coverage_score >= 0.40:
            return "High (Significant borrowing / paraphrase)"
        elif combined >= 0.45 or coverage_score >= 0.20:
            return "Moderate (Common sources / quotations)"
        else:
            return "Low (Independent texts)"

    def compare_documents(
        self, documents: List[Document]
    ) -> Tuple[List[PairwiseDocResult], List[SentenceMatch], Dict[str, np.ndarray]]:
        """Performs a full pairwise comparison of all documents."""
        n_docs = len(documents)

        # 1. Embedding entire documents
        clean_docs = [doc.clean_text for doc in documents]
        doc_emb_1 = self.embedder_1.embed_texts(clean_docs)
        doc_emb_2 = self.embedder_2.embed_texts(clean_docs)

        matrix_doc_1 = TextEmbedder.compute_similarity_matrix(doc_emb_1, doc_emb_1)
        matrix_doc_2 = TextEmbedder.compute_similarity_matrix(doc_emb_2, doc_emb_2)
        matrix_doc_ensemble = (
            self.alpha * matrix_doc_1 + (1.0 - self.alpha) * matrix_doc_2
        )

        # 2. Embedding sentences for each document
        doc_sent_emb_1 = []
        doc_sent_emb_2 = []
        for doc in documents:
            s_texts = [s.text for s in doc.sentences]
            emb_1 = self.embedder_1.embed_texts(s_texts)
            emb_2 = self.embedder_2.embed_texts(s_texts)
            doc_sent_emb_1.append(emb_1)
            doc_sent_emb_2.append(emb_2)

        pairwise_results: List[PairwiseDocResult] = []
        all_sentence_matches: List[SentenceMatch] = []

        # 3. Pairwise comparative analysis (i < j)
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                doc_a = documents[i]
                doc_b = documents[j]

                s_emb_a1 = doc_sent_emb_1[i]
                s_emb_b1 = doc_sent_emb_1[j]
                s_emb_a2 = doc_sent_emb_2[i]
                s_emb_b2 = doc_sent_emb_2[j]

                pair_matches: List[SentenceMatch] = []

                if len(doc_a.sentences) > 0 and len(doc_b.sentences) > 0:
                    sim_s1 = TextEmbedder.compute_similarity_matrix(s_emb_a1, s_emb_b1)
                    sim_s2 = TextEmbedder.compute_similarity_matrix(s_emb_a2, s_emb_b2)
                    sim_s_ens = self.alpha * sim_s1 + (1.0 - self.alpha) * sim_s2

                    # Searching for sentence pairs exceeding the threshold
                    for sa_idx in range(len(doc_a.sentences)):
                        for sb_idx in range(len(doc_b.sentences)):
                            ens_score = float(sim_s_ens[sa_idx, sb_idx])
                            if ens_score >= self.sentence_threshold:
                                sc1 = float(sim_s1[sa_idx, sb_idx])
                                sc2 = float(sim_s2[sa_idx, sb_idx])
                                match = SentenceMatch(
                                    doc_a_name=doc_a.file_path.name,
                                    doc_a_id=doc_a.doc_id,
                                    sent_a_idx=sa_idx,
                                    sent_a_text=doc_a.sentences[sa_idx].text,
                                    doc_b_name=doc_b.file_path.name,
                                    doc_b_id=doc_b.doc_id,
                                    sent_b_idx=sb_idx,
                                    sent_b_text=doc_b.sentences[sb_idx].text,
                                    score_model_1=sc1,
                                    score_model_2=sc2,
                                    score_ensemble=ens_score,
                                    divergence=abs(sc1 - sc2),
                                )
                                pair_matches.append(match)
                                all_sentence_matches.append(match)

                    # Computing coverage: what fraction of sentences has a highly similar counterpart
                    max_sims_a = (
                        np.max(sim_s_ens, axis=1)
                        if sim_s_ens.size > 0
                        else np.array([0.0])
                    )
                    max_sims_b = (
                        np.max(sim_s_ens, axis=0)
                        if sim_s_ens.size > 0
                        else np.array([0.0])
                    )

                    cov_a = np.mean(max_sims_a >= self.sentence_threshold)
                    cov_b = np.mean(max_sims_b >= self.sentence_threshold)
                    coverage = float(0.5 * (cov_a + cov_b))
                else:
                    coverage = 0.0

                mean_sim = (
                    float(np.mean([m.score_ensemble for m in pair_matches]))
                    if pair_matches
                    else 0.0
                )

                ens_doc_sim = float(matrix_doc_ensemble[i, j])
                risk = self._determine_risk_level(ens_doc_sim, coverage)

                pairwise_results.append(
                    PairwiseDocResult(
                        doc_a_name=doc_a.file_path.name,
                        doc_b_name=doc_b.file_path.name,
                        doc_a_id=doc_a.doc_id,
                        doc_b_id=doc_b.doc_id,
                        doc_sim_model_1=float(matrix_doc_1[i, j]),
                        doc_sim_model_2=float(matrix_doc_2[i, j]),
                        doc_sim_ensemble=ens_doc_sim,
                        coverage_score=coverage,
                        matched_sentences_count=len(pair_matches),
                        mean_match_similarity=mean_sim,
                        risk_level=risk,
                    )
                )

        matrices = {
            "model_1": matrix_doc_1,
            "model_2": matrix_doc_2,
            "ensemble": matrix_doc_ensemble,
        }

        # Sorting sentence pairs in descending order by hybrid similarity
        all_sentence_matches.sort(key=lambda x: x.score_ensemble, reverse=True)

        return pairwise_results, all_sentence_matches, matrices
