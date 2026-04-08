"""Parse Python build logs for DeprecationWarning and FutureWarning messages.

Reads a Sphinx Gallery build log and extracts structured warning information
so downstream tools can report on deprecated API usage in tutorials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Strip ANSI escape sequences and carriage-return progress lines
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Matches standard Python warning output:
#   /path/to/file.py:42: DeprecationWarning: some message
#   <unknown>:2: DeprecationWarning: invalid escape sequence '\s'
# The message may span continuation lines (indented), but we grab the first line.
_WARNING_RE = re.compile(
    r"(?P<path>/?[^\s:]+\.py|<unknown>):(?P<lineno>\d+):\s+"
    r"(?P<category>DeprecationWarning|FutureWarning):\s+"
    r"(?P<message>.+)$"
)

# Docker workdir used by the CI build container
_DOCKER_PREFIX = "/var/lib/workspace/"

# Tutorial source directories (relative to repo root)
_SOURCE_DIRS = (
    "beginner_source/",
    "intermediate_source/",
    "advanced_source/",
    "recipes_source/",
    "unstable_source/",
)


@dataclass
class BuildWarning:
    """A single deprecation/future warning extracted from a build log."""

    file: str
    lineno: int
    category: str  # "DeprecationWarning" or "FutureWarning"
    message: str


def _normalize_path(raw_path: str) -> str:
    """Map an absolute Docker path back to a repo-relative tutorial source path.

    If the path doesn't belong to a known source directory the raw path is
    returned as-is (it may come from a dependency — still useful to log).
    """
    path = raw_path
    if path.startswith(_DOCKER_PREFIX):
        path = path[len(_DOCKER_PREFIX) :]

    # Strip leading "./" if present
    if path.startswith("./"):
        path = path[2:]

    return path


def parse_log(log_path: str | Path) -> List[BuildWarning]:
    """Parse *log_path* and return deduplicated :class:`BuildWarning` objects.

    Deduplication key: ``(file, message)`` — only the first occurrence (by
    line number) is kept so the report highlights unique issues rather than
    repeating the same warning 50 times.
    """
    log_path = Path(log_path)
    try:
        text = log_path.read_text(errors="replace")
    except FileNotFoundError:
        return []

    seen: dict[tuple[str, str], BuildWarning] = {}
    warnings: list[BuildWarning] = []

    for line in text.splitlines():
        # Strip ANSI escapes and split on \r to handle progress-line overwriting
        line = _ANSI_RE.sub("", line)
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
        m = _WARNING_RE.search(line)
        if m is None:
            continue

        rel_path = _normalize_path(m.group("path"))
        message = m.group("message").strip()
        key = (rel_path, message)

        if key in seen:
            continue

        warning = BuildWarning(
            file=rel_path,
            lineno=int(m.group("lineno")),
            category=m.group("category"),
            message=message,
        )
        seen[key] = warning
        warnings.append(warning)

    return warnings


def is_tutorial_source(path: str) -> bool:
    """Return True if *path* belongs to a known tutorial source directory."""
    return any(path.startswith(d) for d in _SOURCE_DIRS)


# Package prefixes that belong to PyTorch core
_PYTORCH_CORE_PACKAGES = (
    "/torch/",
    "torch/",
)

# Package prefixes for PyTorch ecosystem libraries
_PYTORCH_LIB_PACKAGES = (
    "/torchvision/",
    "/torchaudio/",
    "/torchtext/",
    "/torchrl/",
    "/tensordict/",
    "/torchdata/",
    "/torchtune/",
    "/torchtitan/",
    "/functorch/",
    "/torch_xla/",
    "/executorch/",
)


def classify_dependency(path: str) -> str:
    """Classify a non-tutorial warning path into a dependency category.

    Returns one of: ``"pytorch"``, ``"pytorch_libs"``, ``"third_party"``.
    """
    for prefix in _PYTORCH_CORE_PACKAGES:
        if prefix in path:
            return "pytorch"
    for prefix in _PYTORCH_LIB_PACKAGES:
        if prefix in path:
            return "pytorch_libs"
    return "third_party"
