#!/usr/bin/env python3
"""Lint tutorial prose for lists that break during notebook conversion.

Sphinx-Gallery reads prose from module docstrings and from comment blocks that
follow a gallery separator.  That prose is parsed as reStructuredText for the
HTML documentation, but is converted to Markdown for generated notebooks.
Some list layouts accepted by reStructuredText are interpreted differently by
Pandoc and produce malformed Markdown cells.  This linter detects those
layouts in the source, before a tutorial is built.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from enum import Enum
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence


LINTER_CODE = "TUTORIAL_MARKUP"
GALLERY_SEPARATOR = re.compile(r"^#{20,}\s*$")
LIST_MARKER = re.compile(
    r"^(?P<indent> *)(?P<marker>[-+*]|\d+[.)]|[A-Za-z][.)])\s+(?P<body>.*)$"
)
LIST_TABLE_ROW = re.compile(r"^\s*\*\s+-(?:\s|$)")
SECTION_ADORNMENT = re.compile(r"^\s*([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])\1{2,}\s*$")


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


class ProseLine(NamedTuple):
    text: str
    source_line: int


def _module_docstring(source: str) -> list[ProseLine]:
    """Return a module docstring without normalizing meaningful indentation."""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return []

    if not module.body:
        return []
    expression = module.body[0]
    if not (
        isinstance(expression, ast.Expr)
        and isinstance(expression.value, ast.Constant)
        and isinstance(expression.value.value, str)
    ):
        return []

    value = expression.value.value
    return [
        ProseLine(text, expression.lineno + offset)
        for offset, text in enumerate(value.splitlines())
    ]


def _comment_text(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    return line[1:]


def extract_prose_blocks(source: str) -> list[list[ProseLine]]:
    """Extract the source regions Sphinx-Gallery treats as narrative prose."""
    source_lines = source.splitlines()
    blocks: list[list[ProseLine]] = []

    docstring = _module_docstring(source)
    if docstring:
        blocks.append(docstring)

    index = 0
    while index < len(source_lines):
        if not GALLERY_SEPARATOR.fullmatch(source_lines[index]):
            index += 1
            continue

        index += 1
        block: list[ProseLine] = []
        while index < len(source_lines) and source_lines[index].startswith("#"):
            block.append(ProseLine(_comment_text(source_lines[index]), index + 1))
            index += 1
        if block:
            blocks.append(block)

    return blocks


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _marker(line: str) -> re.Match[str] | None:
    return LIST_MARKER.match(line)


def _has_same_indent_marker_since_blank(
    lines: Sequence[ProseLine], index: int, indent: int
) -> bool:
    for previous in reversed(lines[:index]):
        if not previous.text.strip():
            return False
        match = _marker(previous.text)
        if match and len(match.group("indent")) == indent:
            return True
    return False


def _has_valid_parent_list(
    lines: Sequence[ProseLine], index: int, indent: int
) -> bool:
    """Recognize a conventionally indented child of an active list item."""
    for previous in reversed(lines[:index]):
        if not previous.text.strip():
            return False
        match = _marker(previous.text)
        if not match:
            continue
        parent_indent = len(match.group("indent"))
        if parent_indent >= indent:
            continue
        content_column = parent_indent + len(match.group("marker")) + 1
        return indent == content_column
    return False


def _missing_blank_before_indented_list(
    lines: Sequence[ProseLine], index: int
) -> bool:
    match = _marker(lines[index].text)
    if not match:
        return False

    indent = len(match.group("indent"))
    marker = match.group("marker")
    if index == 0 or indent == 0 or marker.endswith(")"):
        return False

    previous = lines[index - 1].text
    if not previous.strip() or SECTION_ADORNMENT.fullmatch(previous):
        return False

    # Later items in the same list do not need another separating blank line.
    if _has_same_indent_marker_since_blank(lines, index, indent):
        return False

    # A child list aligned to its parent's content is valid reStructuredText
    # and converts cleanly.  The broken examples use an arbitrary indent.
    if _has_valid_parent_list(lines, index, indent):
        return False

    # Do not confuse list-table cells with ordinary list items.
    if LIST_TABLE_ROW.match(previous):
        return False

    # An item aligned with the preceding directive content (for example an
    # ``.. note::`` body) is already separated by the directive structure.
    if indent > 0 and _leading_spaces(previous) == indent:
        return False

    return True


def _bad_list_continuations(lines: Sequence[ProseLine]) -> Iterable[ProseLine]:
    """Yield unindented continuation lines from blank-separated top-level lists."""
    index = 0
    while index < len(lines):
        first_match = _marker(lines[index].text)
        if not first_match or first_match.group("indent"):
            index += 1
            continue
        if index > 0 and lines[index - 1].text.strip():
            index += 1
            continue

        marker_indent = 0
        cursor = index + 1
        first_bad: ProseLine | None = None
        has_next_item = False
        while cursor < len(lines) and lines[cursor].text.strip():
            match = _marker(lines[cursor].text)
            if match and len(match.group("indent")) == marker_indent:
                has_next_item = True
            elif not match and _leading_spaces(lines[cursor].text) <= marker_indent:
                first_bad = first_bad or lines[cursor]
            cursor += 1

        if has_next_item and first_bad is not None:
            yield first_bad
        index = max(cursor, index + 1)


def lint_source(filename: str, source: str) -> list[LintMessage]:
    messages: list[LintMessage] = []

    for block in extract_prose_blocks(source):
        for index, prose_line in enumerate(block):
            if _missing_blank_before_indented_list(block, index):
                messages.append(
                    LintMessage(
                        path=filename,
                        line=prose_line.source_line,
                        char=1,
                        code=LINTER_CODE,
                        severity=LintSeverity.ERROR,
                        name="list missing a preceding blank line",
                        original=None,
                        replacement=None,
                        description=(
                            "This indented list starts immediately after prose. "
                            "It renders as a list in the HTML tutorial, but Pandoc "
                            "can merge or mis-indent it in the generated notebook. "
                            "Add a blank narrative line before the list."
                        ),
                    )
                )

        for prose_line in _bad_list_continuations(block):
            messages.append(
                LintMessage(
                    path=filename,
                    line=prose_line.source_line,
                    char=1,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="unindented list continuation",
                    original=None,
                    replacement=None,
                    description=(
                        "This list item continues at the same indentation as the "
                        "list marker. Pandoc treats the continuation as ordinary "
                        "text in generated notebooks. Indent continuation lines "
                        "to the list item's content column."
                    ),
                )
            )

    return messages


def lint_file(filename: str) -> list[LintMessage]:
    return lint_source(filename, Path(filename).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("filenames", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for filename in args.filenames:
        for message in lint_file(filename):
            print(json.dumps(message._asdict()))


if __name__ == "__main__":
    main()
