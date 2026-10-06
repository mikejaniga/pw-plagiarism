"""Configuration of parameters for the plagiarism detection system."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectorConfig:
    # Model 1: Specialized Polish model based on RoBERTa trained on Polish paraphrase pairs
    model_name_1: str = "sdadas/st-polish-paraphrase-from-distilroberta"
    model_label_1: str = "ST-Polish-RoBERTa"

    # Model 2: Proven, multilingual bi-encoder Sentence-Transformers
    model_name_2: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_label_2: str = "Multilingual-MiniLM"

    # Weights in the hybrid model (ensemble): alpha * Model1 + (1 - alpha) * Model2
    ensemble_alpha: float = 0.55

    # Similarity thresholds for fragments/sentences (0.0 - 1.0)
    sentence_similarity_threshold: float = 0.72
    high_similarity_threshold: float = 0.85

    # Minimum sentence length in characters (filtering out noise, headers, single words)
    min_sentence_length: int = 25

    # Output paths
    output_dir: Path = Path("reports")
    excel_report_name: str = "plagiarism_report.xlsx"
    heatmap_image_name: str = "similarity_matrix.png"
    multi_doc_html_name: str = "text_comparison.html"


DEFAULT_CONFIG: DetectorConfig = DetectorConfig()
