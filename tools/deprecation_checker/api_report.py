"""Generate Markdown deprecation reports and optionally file GitHub Issues.

Usage::

    NOTE: This tool is designed to run in CI where tutorials are fully executed
    by Sphinx Gallery on GPU workers. Running ``make html-noplot`` locally skips
    tutorial execution, so no runtime warnings are emitted. Use these commands
    to re-parse a build.log downloaded from a CI run, or to test with a
    synthetic log.

    # Local report to stdout
    python -m tools.deprecation_checker.api_report --build-log _build/build.log

    # Local report written to a file
    python -m tools.deprecation_checker.api_report --build-log _build/build.log -o _build/api_report.md

    # Create / update a GitHub Issue (requires GITHUB_TOKEN)
    python -m tools.deprecation_checker.api_report --build-log _build/build.log --create-issue
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import List

from .build_warning_parser import BuildWarning, classify_dependency, is_tutorial_source, parse_log

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPO_OWNER = "pytorch"
REPO_NAME = "tutorials"
ISSUE_LABEL = "docs-agent-deprecations"
ISSUE_TITLE = "[CI] Deprecated API usage in tutorials"
ISSUE_CC = "svekars"

# --------------------------------------------------------------------------- #
# Markdown report generation
# --------------------------------------------------------------------------- #


def _summary_table(warnings: List[BuildWarning]) -> str:
    """Build a Markdown table summarising warning counts per file."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for w in warnings:
        counts[w.file][w.category] += 1

    lines = [
        "| Tutorial file | DeprecationWarning | FutureWarning | Total |",
        "|---|---:|---:|---:|",
    ]
    for file in sorted(counts):
        dep = counts[file].get("DeprecationWarning", 0)
        fut = counts[file].get("FutureWarning", 0)
        lines.append(f"| `{file}` | {dep} | {fut} | {dep + fut} |")

    total_dep = sum(c.get("DeprecationWarning", 0) for c in counts.values())
    total_fut = sum(c.get("FutureWarning", 0) for c in counts.values())
    lines.append(f"| **Total** | **{total_dep}** | **{total_fut}** | **{total_dep + total_fut}** |")
    return "\n".join(lines)


def _findings_section(warnings: List[BuildWarning]) -> str:
    """Detailed findings grouped by file, sorted by line number."""
    by_file: dict[str, list[BuildWarning]] = defaultdict(list)
    for w in warnings:
        by_file[w.file].append(w)

    sections: list[str] = []
    for file in sorted(by_file):
        items = sorted(by_file[file], key=lambda w: w.lineno)
        parts = [f"### `{file}`\n"]
        for w in items:
            parts.append(
                f"- **Line {w.lineno}** ({w.category}): {w.message}"
            )
        sections.append("\n".join(parts))

    return "\n\n".join(sections)


def generate_report(warnings: List[BuildWarning]) -> str:
    """Return a full Markdown report string."""
    if not warnings:
        return (
            "# API Deprecation Report\n\n"
            "No `DeprecationWarning` or `FutureWarning` detected in this build. :tada:"
        )

    tutorial_warnings = [w for w in warnings if is_tutorial_source(w.file)]
    other_warnings = [w for w in warnings if not is_tutorial_source(w.file)]

    parts = [
        "# API Deprecation Report",
        "",
        f"**{len(warnings)}** unique deprecation/future warnings found in this build.",
        "",
    ]

    if tutorial_warnings:
        parts += [
            "## Summary (tutorial sources)",
            "",
            _summary_table(tutorial_warnings),
            "",
            "## Findings",
            "",
            _findings_section(tutorial_warnings),
            "",
        ]

    # Classify dependency warnings
    pytorch_warnings = [w for w in other_warnings if classify_dependency(w.file) == "pytorch"]
    pytorch_lib_warnings = [w for w in other_warnings if classify_dependency(w.file) == "pytorch_libs"]
    third_party_warnings = [w for w in other_warnings if classify_dependency(w.file) == "third_party"]

    if pytorch_warnings:
        parts += [
            "## PyTorch warnings",
            "",
            _findings_section(pytorch_warnings),
            "",
        ]

    if pytorch_lib_warnings:
        parts += [
            "## PyTorch libraries warnings",
            "",
            _findings_section(pytorch_lib_warnings),
            "",
        ]

    if third_party_warnings:
        parts += [
            "## Third-party dependency warnings",
            "",
            _findings_section(third_party_warnings),
            "",
        ]

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# GitHub Issue creation / update
# --------------------------------------------------------------------------- #


def _gh_api(
    method: str,
    endpoint: str,
    token: str,
    body: dict | None = None,
) -> dict:
    """Minimal GitHub REST API helper using only stdlib."""
    url = f"https://api.github.com{endpoint}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _ensure_label(token: str) -> None:
    """Create the issue label if it doesn't exist yet."""
    try:
        _gh_api(
            "POST",
            f"/repos/{REPO_OWNER}/{REPO_NAME}/labels",
            token,
            {"name": ISSUE_LABEL, "color": "d93f0b", "description": "Auto-generated deprecation report from CI"},
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 422:
            pass  # label already exists
        else:
            raise


def _find_existing_issue(token: str) -> int | None:
    """Return the issue number of the existing open deprecation issue, or None."""
    results = _gh_api(
        "GET",
        f"/repos/{REPO_OWNER}/{REPO_NAME}/issues?labels={ISSUE_LABEL}&state=open&per_page=1",
        token,
    )
    if results:
        return results[0]["number"]
    return None


def create_or_update_issue(report_body: str, token: str) -> str:
    """Create or update the deprecation GitHub Issue. Returns the issue URL."""
    _ensure_label(token)
    existing = _find_existing_issue(token)

    body = f"cc: @{ISSUE_CC}\n\n{report_body}"

    if existing:
        result = _gh_api(
            "PATCH",
            f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{existing}",
            token,
            {"body": body},
        )
        return result["html_url"]
    else:
        result = _gh_api(
            "POST",
            f"/repos/{REPO_OWNER}/{REPO_NAME}/issues",
            token,
            {
                "title": ISSUE_TITLE,
                "body": body,
                "labels": [ISSUE_LABEL],
            },
        )
        return result["html_url"]


def close_issue_if_open(token: str) -> str | None:
    """Close the deprecation issue if one is open. Returns the URL or None."""
    existing = _find_existing_issue(token)
    if not existing:
        return None
    result = _gh_api(
        "PATCH",
        f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{existing}",
        token,
        {
            "state": "closed",
            "body": f"cc: @{ISSUE_CC}\n\n"
                    "All `DeprecationWarning` and `FutureWarning` issues have been resolved. "
                    "This issue will reopen automatically if new deprecations are detected.",
        },
    )
    return result["html_url"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate an API deprecation report from a Sphinx build log.",
    )
    parser.add_argument(
        "--build-log",
        required=True,
        help="Path to the build log file (e.g. _build/build.log).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the Markdown report to this file instead of stdout.",
    )
    parser.add_argument(
        "--create-issue",
        action="store_true",
        help="Create or update a GitHub Issue with the report (requires GITHUB_TOKEN).",
    )
    args = parser.parse_args(argv)

    warnings = parse_log(args.build_log)
    report = generate_report(warnings)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    if args.create_issue:
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            print("WARNING: GITHUB_TOKEN not set — skipping issue creation.", file=sys.stderr)
            return
        if not warnings:
            url = close_issue_if_open(token)
            if url:
                print(f"All warnings resolved — closed issue: {url}")
            else:
                print("No warnings found and no open issue to close.")
            return
        url = create_or_update_issue(report, token)
        print(f"GitHub Issue: {url}")


if __name__ == "__main__":
    main()
