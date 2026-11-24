#!/usr/bin/env python3
"""Build a high-coverage, deterministic CLAD bank from CrowdQuake logs.

The generated bank keeps recurring masked message shapes as learned templates.
For an unseen shape it returns a generic variable-masked template rather than
``UNKNOWN``. This intentionally optimizes coverage, not parsing accuracy.

Examples (run from clad-tool/):
    python parser/synthesis/build_crowdquake_bank.py \
        --input /path/to/crowdquake/logs --header-delimiter auto
    python parser/synthesis/build_crowdquake_bank.py \
        --input /path/to/zookeeper_content.log \
        --output parser/banks/qwen2.5-7b/CrowdQuake_2k/parser_bank.py

The default output can be loaded by the existing CLI with
``--bank qwen2.5-7b --system CrowdQuake``. The input corpus is not copied into
the generated bank.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "parser/banks/qwen2.5-7b/CrowdQuake_2k/parser_bank.py"
)
INPUT_SUFFIXES = {".csv", ".log", ".txt"}

_ZK_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]")
_MASK_PATTERNS = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?\b"),
    re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"),
    re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?!\w)"),
    re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
    re.compile(r"\b0x[0-9a-fA-F]+\b", re.IGNORECASE),
    re.compile(r"(?<!\S)/[^\s,;)]*"),
    re.compile(r"\b\d+(?:\.\d+)?\b"),
)


def _input_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in INPUT_SUFFIXES
            )
        elif path.suffix.lower() in INPUT_SUFFIXES:
            files.append(path)
        else:
            raise ValueError(f"Unsupported input file type: {path}")
    return sorted(set(files))


def _content_from_line(line: str, mode: str) -> str:
    if mode == "none":
        return line
    if mode == "first":
        return line.split(" - ", 1)[1] if " - " in line else line
    if mode == "second":
        parts = line.split(" - ", 2)
        return parts[2] if len(parts) == 3 else line

    parts = line.split(" - ", 2)
    if len(parts) == 3 and _ZK_TIMESTAMP.match(parts[0]):
        return parts[2]
    if len(parts) >= 2:
        return line.split(" - ", 1)[1]
    return line


def _iter_lines(files: list[Path], header_delimiter: str) -> Iterator[str]:
    for path in files:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or "Content" not in reader.fieldnames:
                    raise ValueError(f"CSV must have a 'Content' column: {path}")
                for row in reader:
                    content = str(row.get("Content") or "").strip()
                    if content:
                        yield content
            continue

        with path.open(encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                content = _content_from_line(raw.rstrip("\r\n"), header_delimiter).strip()
                if content:
                    yield content


def _mask_variables(line: str) -> str:
    normalized = " ".join(line.split())
    for pattern in _MASK_PATTERNS:
        normalized = pattern.sub("<*>", normalized)
    return normalized


def _render_bank(templates: list[str]) -> str:
    return '''"""Generated CrowdQuake parser bank; see parser/synthesis/build_crowdquake_bank.py."""

import re

TEMPLATES = %r
_TEMPLATE_IDS = {template: index for index, template in enumerate(TEMPLATES)}

_MASK_PATTERNS = (
    re.compile(r"\\b\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}:\\d{2}(?:[.,]\\d+)?Z?\\b"),
    re.compile(r"\\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\\b"),
    re.compile(r"(?<![\\w.])(?:\\d{1,3}\\.){3}\\d{1,3}(?::\\d+)?(?!\\w)"),
    re.compile(r"\\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\\b"),
    re.compile(r"\\b0x[0-9a-fA-F]+\\b", re.IGNORECASE),
    re.compile(r"(?<!\\S)/[^\\s,;)]*"),
    re.compile(r"\\b\\d+(?:\\.\\d+)?\\b"),
)


def _mask_variables(log):
    normalized = " ".join(str(log).split())
    for pattern in _MASK_PATTERNS:
        normalized = pattern.sub("<*>", normalized)
    return normalized or "<EMPTY>"


def match_template(log):
    """Return the learned template, or the generic masked fallback."""
    return _mask_variables(log)


def process_log(log):
    template = match_template(log)
    return {
        "template": template,
        "template_id": _TEMPLATE_IDS.get(template, -1),
        "fallback": template not in _TEMPLATE_IDS,
    }
''' % templates


def build_bank(inputs: list[Path], output: Path, header_delimiter: str) -> tuple[int, int]:
    files = _input_files(inputs)
    if not files:
        raise ValueError("No .log, .txt, or .csv input files found")
    line_count = 0
    templates = set()
    for line in _iter_lines(files, header_delimiter):
        line_count += 1
        templates.add(_mask_variables(line))
    if not line_count:
        raise ValueError("No non-empty log messages found in the input")

    templates = sorted(templates)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_bank(templates), encoding="utf-8")
    return line_count, len(templates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", required=True, type=Path,
        help="Log/CSV files or directories; CSV input must contain a Content column.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Generated parser_bank.py (default: {DEFAULT_OUTPUT.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--header-delimiter", choices=("auto", "none", "first", "second"),
        default="auto",
        help="Remove a log header split by ' - '; auto handles ZooKeeper and one-delimiter formats.",
    )
    args = parser.parse_args()

    try:
        line_count, template_count = build_bank(
            args.input, args.output, args.header_delimiter
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Built {args.output}: lines={line_count} templates={template_count} "
        "unseen-line fallback=enabled"
    )


if __name__ == "__main__":
    main()