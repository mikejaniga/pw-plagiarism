"""Main entry-point script for the plagiarism detection system."""

import argparse
import sys
from pathlib import Path

from plagiarism_detector.config import DEFAULT_CONFIG, DetectorConfig
from plagiarism_detector.data_loader import DataLoader
from plagiarism_detector.models.embedder import TextEmbedder
from plagiarism_detector.models.ensemble import EnsemblePlagiarismDetector
from plagiarism_detector.reporter import ExcelPlagiarismReporter
from plagiarism_detector.visualizer import Visualizer


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    config: DetectorConfig = DEFAULT_CONFIG,
) -> None:
    print("=" * 75)
    print("           ADVANCED PLAGIARISM DETECTION SYSTEM (NLP ENSEMBLE)")
    print("=" * 75)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading files
    print(f"\n[STEP 1/5] Loading files from directory: {input_dir}")
    loader = DataLoader(min_sentence_length=config.min_sentence_length)
    documents = loader.load_directory(input_dir)

    print(f"-> Successfully loaded {len(documents)} documents:")
    for doc in documents:
        print(
            f"   • {doc.file_path.name} ({len(doc.sentences)} sentence segments, {len(doc.clean_text)} characters)"
        )

    if len(documents) < 2:
        print("Error: At least 2 documents are required for comparison.")
        sys.exit(1)

    # 2. Initializing language models
    print(f"\n[STEP 2/5] Initializing transformer models...")
    embedder_1 = TextEmbedder(
        model_name=config.model_name_1, model_label=config.model_label_1
    )
    embedder_2 = TextEmbedder(
        model_name=config.model_name_2, model_label=config.model_label_2
    )

    # 3. Comparative analysis and ensemble fusion
    print(
        f"\n[STEP 3/5] Computing vector representations and fusing results (alpha={config.ensemble_alpha})..."
    )
    detector = EnsemblePlagiarismDetector(
        embedder_1=embedder_1,
        embedder_2=embedder_2,
        alpha=config.ensemble_alpha,
        sentence_threshold=config.sentence_similarity_threshold,
        high_threshold=config.high_similarity_threshold,
    )

    pairwise_results, sentence_matches, matrices = detector.compare_documents(documents)

    print(f"-> Analyzed {len(pairwise_results)} document pairs.")
    print(
        f"-> Detected a total of {len(sentence_matches)} suspicious sentence pairs/paraphrases above the threshold of {config.sentence_similarity_threshold:.0%}."
    )

    # 4. Generating the Excel report
    print(f"\n[STEP 4/5] Creating the multi-sheet Excel report...")
    excel_path = output_dir / config.excel_report_name
    reporter = ExcelPlagiarismReporter(
        model_1_name=config.model_label_1,
        model_2_name=config.model_label_2,
        alpha=config.ensemble_alpha,
    )
    reporter.generate_report(
        output_path=excel_path,
        documents=documents,
        pairwise_results=pairwise_results,
        sentence_matches=sentence_matches,
        matrices=matrices,
    )

    # 5. Generating visualizations
    print(f"\n[STEP 5/5] Generating similarity visualizations...")
    heatmap_path = output_dir / config.heatmap_image_name
    Visualizer.plot_heatmap(
        documents=documents,
        matrices=matrices,
        output_image_path=heatmap_path,
        model_1_label=config.model_label_1,
        model_2_label=config.model_label_2,
    )

    html_path = output_dir / config.multi_doc_html_name
    Visualizer.generate_multi_document_html(
        documents=documents,
        sentence_matches=sentence_matches,
        output_html_path=html_path,
        max_docs=min(len(documents), 4),
    )

    # Presenting results in the console
    print("\n" + "=" * 75)
    print("                     PLAGIARISM ANALYSIS SUMMARY")
    print("=" * 75)
    print(
        f"{'Document A':<22} | {'Document B':<22} | {'Ensemble':<9} | {'Coverage':<8} | Risk Level"
    )
    print("-" * 88)
    for r in pairwise_results:
        print(
            f"{r.doc_a_name[:20]:<22} | "
            f"{r.doc_b_name[:20]:<22} | "
            f"{r.doc_sim_ensemble:>7.1%}  | "
            f"{r.coverage_score:>7.1%}  | "
            f"{r.risk_level}"
        )
    print("-" * 88)

    print("\nGenerated artifacts:")
    print(f" 1. Excel workbook : {excel_path.resolve()}")
    print(f" 2. PNG charts     : {heatmap_path.resolve()}")
    print(f" 3. HTML view      : {html_path.resolve()}")
    print("=" * 75)
    print("Analysis completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plagiarism detection system for text documents with language model fusion."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=Path("data/test_articles"),
        help="Path to the folder containing text files to compare (default: data/test_articles)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_CONFIG.output_dir,
        help="Path to the folder where reports and visualizations will be saved (default: reports)",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=DEFAULT_CONFIG.ensemble_alpha,
        help="Weight of the first model in the ensemble (0.0 - 1.0, default: 0.55)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_CONFIG.sentence_similarity_threshold,
        help="Sentence similarity threshold (0.0 - 1.0, default: 0.72)",
    )

    args = parser.parse_args()

    cfg = DetectorConfig(
        ensemble_alpha=args.alpha,
        sentence_similarity_threshold=args.threshold,
        output_dir=args.output_dir,
    )

    run_pipeline(input_dir=args.input_dir, output_dir=args.output_dir, config=cfg)


if __name__ == "__main__":
    main()
