"""Visualization module: similarity matrix (heatmap) and interactive HTML dashboard for 2-4 texts."""

import html
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plagiarism_detector.data_loader import Document
from plagiarism_detector.models.ensemble import SentenceMatch


class Visualizer:
    @staticmethod
    def plot_heatmap(
        documents: List[Document],
        matrices: Dict[str, np.ndarray],
        output_image_path: Path,
        model_1_label: str,
        model_2_label: str,
    ) -> None:
        """Generates a triple heatmap plot (Model 1, Model 2, Ensemble) in PNG format."""
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        labels: List[str] = [d.file_path.stem[:18] for d in documents]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(
            "Document Similarity Matrix Comparison",
            fontsize=14,
            fontweight="bold",
        )

        plots_data = [
            (f"Model 1: {model_1_label}", matrices["model_1"], "Blues"),
            (f"Model 2: {model_2_label}", matrices["model_2"], "Greens"),
            ("Ensemble (Hybrid)", matrices["ensemble"], "YlOrRd"),
        ]

        for ax, (title, matrix, cmap) in zip(axes, plots_data):
            sns.heatmap(
                matrix,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                xticklabels=labels,
                yticklabels=labels,
                vmin=0.0,
                vmax=1.0,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=35, ha="right", fontsize=9
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.tight_layout()
        plt.savefig(output_image_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"-> Saved heatmap plot: {output_image_path}")

    @staticmethod
    def generate_multi_document_html(
        documents: List[Document],
        sentence_matches: List[SentenceMatch],
        output_html_path: Path,
        max_docs: int = 4,
    ) -> None:
        """Generates an interactive HTML report with a Side-by-Side view for 2 to 4 texts.

        Allows simultaneous browsing of texts, interactive highlighting of matching sentences,
        and exploring overlapping fragments across multiple articles at once.
        """
        if not documents:
            return

        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        target_docs: List[Document] = documents[:max_docs]

        # Build sentence match mapping
        matches_data: List[Dict[str, Any]] = []
        for m in sentence_matches:
            matches_data.append(
                {
                    "doc_a": m.doc_a_name,
                    "idx_a": m.sent_a_idx,
                    "doc_b": m.doc_b_name,
                    "idx_b": m.sent_b_idx,
                    "score_ens": round(m.score_ensemble, 3),
                    "score_m1": round(m.score_model_1, 3),
                    "score_m2": round(m.score_model_2, 3),
                }
            )

        matches_json: str = json.dumps(matches_data, ensure_ascii=False)

        columns_html: List[str] = []
        col_width_pct: float = round(100.0 / len(target_docs), 1)

        for doc in target_docs:
            sentences_html: List[str] = []
            for s in doc.sentences:
                safe_text: str = html.escape(s.text)
                sent_div: str = (
                    f'<div class="sentence-item" '
                    f'data-doc="{html.escape(doc.file_path.name)}" '
                    f'data-idx="{s.sentence_idx}" '
                    f'onclick="handleSentenceClick(this)">'
                    f'<span class="sent-num">#{s.sentence_idx + 1}</span> {safe_text}'
                    f"</div>"
                )
                sentences_html.append(sent_div)

            col: str = f"""
            <div class="document-column" style="width: {col_width_pct}%;">
                <div class="doc-header">
                    <h3>{html.escape(doc.title)}</h3>
                    <span class="doc-filename">{html.escape(doc.file_path.name)}</span>
                    <span class="doc-stats">Sentence count: {len(doc.sentences)}</span>
                </div>
                <div class="doc-body">
                    {"".join(sentences_html)}
                </div>
            </div>
            """
            columns_html.append(col)

        html_content: str = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Text Similarity Visualization (Multi-Document Plagiarism View)</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f6f9;
            color: #2c3e50;
        }}
        header {{
            background: #1a252f;
            color: #ffffff;
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        header h1 {{ margin: 0; font-size: 20px; font-weight: 600; }}
        .legend {{
            display: flex;
            gap: 15px;
            font-size: 13px;
            background: #2c3e50;
            padding: 6px 14px;
            border-radius: 6px;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .badge-color {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
        .color-high {{ background-color: #ff7675; }}
        .color-mid {{ background-color: #ffeaa7; }}
        .color-active {{ background-color: #74b9ff; border: 2px solid #0984e3; }}

        .info-panel {{
            background: #ffffff;
            border-bottom: 1px solid #dcdde1;
            padding: 10px 24px;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #match-details {{
            font-weight: 500;
            color: #2c3e50;
        }}

        .container {{
            display: flex;
            flex-direction: row;
            gap: 12px;
            padding: 16px;
            height: calc(100vh - 125px);
            overflow: hidden;
        }}
        .document-column {{
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            display: flex;
            flex-direction: column;
            border: 1px solid #e1e8ed;
        }}
        .doc-header {{
            padding: 12px 16px;
            background: #f8fafc;
            border-bottom: 1px solid #e1e8ed;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }}
        .doc-header h3 {{
            margin: 0 0 4px 0;
            font-size: 14px;
            color: #1e293b;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .doc-filename {{ font-size: 11px; color: #64748b; display: block; }}
        .doc-stats {{ font-size: 11px; color: #0284c7; font-weight: 500; display: block; margin-top: 2px; }}

        .doc-body {{
            padding: 12px;
            overflow-y: auto;
            flex: 1;
        }}
        .sentence-item {{
            margin-bottom: 8px;
            padding: 6px 8px;
            border-radius: 5px;
            line-height: 1.45;
            font-size: 13px;
            border: 1px solid transparent;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .sentence-item:hover {{
            background-color: #f1f5f9;
        }}
        .sent-num {{
            font-size: 10px;
            color: #94a3b8;
            font-weight: bold;
            margin-right: 4px;
        }}

        /* Highlight styles */
        .highlight-high {{
            background-color: #ffebee !important;
            border-left: 4px solid #e53935 !important;
        }}
        .highlight-mid {{
            background-color: #fffde7 !important;
            border-left: 4px solid #fbc02d !important;
        }}
        .selected-active {{
            background-color: #e0f2fe !important;
            border: 2px solid #0284c7 !important;
            box-shadow: 0 0 8px rgba(2, 132, 199, 0.4);
        }}
        .connected-match {{
            background-color: #fce4ec !important;
            border: 2px solid #d81b60 !important;
            box-shadow: 0 0 8px rgba(216, 27, 96, 0.4);
        }}
    </style>
</head>
<body>
    <header>
        <h1>Plagiarism Detection System — Simultaneous Visualization of {len(target_docs)} Texts</h1>
        <div class="legend">
            <div class="legend-item"><span class="badge-color color-high"></span> High similarity (&gt;=85%)</div>
            <div class="legend-item"><span class="badge-color color-mid"></span> Moderate similarity (72-84%)</div>
            <div class="legend-item"><span class="badge-color color-active"></span> Clicked sentence / Connections</div>
        </div>
    </header>

    <div class="info-panel">
        <span id="match-details">Click any sentence to highlight its counterparts across all texts.</span>
        <button onclick="resetHighlights()" style="padding: 4px 10px; border-radius: 4px; border: 1px solid #ccc; cursor: pointer; background: #fff;">Clear selection</button>
    </div>

    <div class="container">
        {"".join(columns_html)}
    </div>

    <script>
        const matches = {matches_json};

        // Mark sentences that have any match
        document.addEventListener('DOMContentLoaded', () => {{
            matches.forEach(m => {{
                markSentence(m.doc_a, m.idx_a, m.score_ens);
                markSentence(m.doc_b, m.idx_b, m.score_ens);
            }});
        }});

        function markSentence(docName, idx, score) {{
            const el = document.querySelector(`.sentence-item[data-doc="${{docName}}"][data-idx="${{idx}}"]`);
            if (el) {{
                if (score >= 0.85) {{
                    el.classList.add('highlight-high');
                }} else if (!el.classList.contains('highlight-high')) {{
                    el.classList.add('highlight-mid');
                }}
            }}
        }}

        function handleSentenceClick(element) {{
            resetSelectionOnly();

            const docName = element.getAttribute('data-doc');
            const idx = parseInt(element.getAttribute('data-idx'));

            element.classList.add('selected-active');

            // Find connections
            const related = [];
            matches.forEach(m => {{
                if (m.doc_a === docName && m.idx_a === idx) {{
                    related.push({{ targetDoc: m.doc_b, targetIdx: m.idx_b, score: m.score_ens, m1: m.score_m1, m2: m.score_m2 }});
                }} else if (m.doc_b === docName && m.idx_b === idx) {{
                    related.push({{ targetDoc: m.doc_a, targetIdx: m.idx_a, score: m.score_ens, m1: m.score_m1, m2: m.score_m2 }});
                }}
            }});

            const detailsSpan = document.getElementById('match-details');
            if (related.length === 0) {{
                detailsSpan.innerHTML = `Selected sentence has no detected borrowings above the similarity threshold.`;
                return;
            }}

            let infoText = `<strong>Selected sentence has ${{related.length}} connection(s):</strong> `;
            related.forEach(r => {{
                const targetEl = document.querySelector(`.sentence-item[data-doc="${{r.targetDoc}}"][data-idx="${{r.targetIdx}}"]`);
                if (targetEl) {{
                    targetEl.classList.add('connected-match');
                    targetEl.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                }}
                infoText += `[${{r.targetDoc}} #sentence ${{r.targetIdx + 1}} -> Ensemble: <b>${{(r.score * 100).toFixed(1)}}%</b> (M1: ${{(r.m1 * 100).toFixed(1)}}%, M2: ${{(r.m2 * 100).toFixed(1)}}%)] `;
            }});

            detailsSpan.innerHTML = infoText;
        }}

        function resetSelectionOnly() {{
            document.querySelectorAll('.selected-active').forEach(el => el.classList.remove('selected-active'));
            document.querySelectorAll('.connected-match').forEach(el => el.classList.remove('connected-match'));
        }}

        function resetHighlights() {{
            resetSelectionOnly();
            document.getElementById('match-details').innerText = "Click any sentence to highlight its counterparts across all texts.";
        }}
    </script>
</body>
</html>
"""
        output_html_path.write_text(html_content, encoding="utf-8")
        print(f"-> Saved interactive HTML dashboard: {output_html_path}")
