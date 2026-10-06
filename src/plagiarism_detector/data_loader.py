"""Module for loading text files and splitting into segments (sentences)."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

ABBREVIATIONS: Set[str] = {
    "np",
    "itp",
    "itd",
    "prof",
    "dr",
    "hab",
    "mgr",
    "inż",
    "red",
    "fot",
    "tzw",
    "tzn",
    "art",
    "ust",
    "pkt",
    "al",
    "ul",
    "nr",
    "godz",
    "min",
    "sek",
}


@dataclass
class SentenceSegment:
    doc_id: str
    doc_name: str
    sentence_idx: int
    text: str
    char_start: int
    char_end: int


@dataclass
class Document:
    doc_id: str
    file_path: Path
    title: str
    raw_text: str
    clean_text: str
    sentences: List[SentenceSegment]


class DataLoader:
    def __init__(self, min_sentence_length: int = 25) -> None:
        self.min_sentence_length: int = min_sentence_length

    def clean_text(self, text: str) -> str:
        """Cleans text from redundant whitespace and typical editorial noise."""
        # Unify line ending characters and spaces
        cleaned: str = re.sub(r"\r\n|\r", "\n", text)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def split_into_sentences(
        self, text: str, doc_id: str, doc_name: str
    ) -> List[SentenceSegment]:
        """Splits text into sentences while preserving start and end positions."""
        candidate_pattern: re.Pattern[str] = re.compile(
            r"(?<=[.!?…])(\s+)(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9\"„«])",
            re.UNICODE,
        )

        split_positions: List[Tuple[int, int]] = []
        for match in candidate_pattern.finditer(text):
            prefix: str = text[: match.start()]
            last_word_match: Optional[re.Match[str]] = re.search(
                r"(\b[\wĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)\.$", prefix
            )
            if last_word_match:
                word: str = last_word_match.group(1)
                # Skip splitting if the period belongs to a known abbreviation or initial
                if word.lower() in ABBREVIATIONS:
                    continue
                if len(word) == 1 and word.isupper():
                    continue
            split_positions.append((match.start(), match.end()))

        raw_splits: List[Tuple[int, int]] = []
        prev_idx: int = 0
        for start_ws, end_ws in split_positions:
            raw_splits.append((prev_idx, start_ws))
            prev_idx = end_ws
        raw_splits.append((prev_idx, len(text)))

        segments: List[SentenceSegment] = []
        idx: int = 0
        for start, end in raw_splits:
            chunk: str = text[start:end]
            chunk_strip: str = chunk.strip()
            if not chunk_strip:
                continue

            leading_ws: int = len(chunk) - len(chunk.lstrip())
            found_pos: int = start + leading_ws
            end_pos: int = found_pos + len(chunk_strip)

            if len(chunk_strip) >= self.min_sentence_length:
                segments.append(
                    SentenceSegment(
                        doc_id=doc_id,
                        doc_name=doc_name,
                        sentence_idx=idx,
                        text=chunk_strip,
                        char_start=found_pos,
                        char_end=end_pos,
                    )
                )
                idx += 1

        return segments

    def load_document(self, file_path: Path, doc_id: Optional[str] = None) -> Document:
        """Loads a single text file."""
        path: Path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")

        # Read with encoding detection (UTF-8 with fallback to cp1250)
        try:
            raw_text: str = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = path.read_text(encoding="cp1250", errors="replace")

        clean: str = self.clean_text(raw_text)
        identifier: str = doc_id if doc_id else path.stem
        sentences: List[SentenceSegment] = self.split_into_sentences(
            clean, identifier, path.name
        )

        return Document(
            doc_id=identifier,
            file_path=path,
            title=path.stem.replace("_", " ").title(),
            raw_text=raw_text,
            clean_text=clean,
            sentences=sentences,
        )

    def load_directory(
        self, directory_path: Path, extensions: Tuple[str, ...] = (".txt", ".md")
    ) -> List[Document]:
        """Loads all files with specified extensions from the given folder."""
        dir_p: Path = Path(directory_path)
        if not dir_p.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {dir_p}")

        files: List[Path] = sorted(
            [
                f
                for f in dir_p.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            ]
        )
        if not files:
            raise ValueError(
                f"No matching files found in directory: {dir_p}"
            )

        documents: List[Document] = []
        for i, f in enumerate(files, start=1):
            doc_id: str = f"DOC_{i:02d}_{f.stem}"
            doc: Document = self.load_document(f, doc_id=doc_id)
            documents.append(doc)

        return documents
