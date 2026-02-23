---
name: triaging-issues
description: Triages GitHub issues in the pytorch/tutorials repo by applying labels, routing to domain areas, and responding to questions. Use when processing new tutorial issues or when asked to triage an issue.
---

# Tutorials Issue Triage Skill

This skill helps triage GitHub issues in the `pytorch/tutorials` repo by classifying issues, applying labels, and leaving first-line responses.

## Contents
- [MCP Tools Available](#mcp-tools-available)
- [Labels You Must NEVER Add](#labels-you-must-never-add)
- [Issue Triage Steps](#issue-triage-for-each-issue)
  - Step 0: Already Triaged — SKIP
  - Step 1: Question vs Bug/Feature
  - Step 2: Redirect to PyTorch Core
  - Step 3: Classify the Issue
  - Step 4: Label the Issue
  - Step 5: High Priority — REQUIRES HUMAN REVIEW
  - Step 6: Mark Triaged
- [V1 Constraints](#v1-constraints)

**Labels reference:** See [labels.json](labels.json) for the catalog of labels suitable for triage. **ONLY apply labels that exist in this file.** Do not invent or guess label names.

**Response templates:** See [templates.json](templates.json) for standard response messages.

---

## MCP Tools Available

Use these GitHub MCP tools for triage:

| Tool | Purpose |
|------|---------|
| `mcp__github__issue_read` | Get issue details, comments, and existing labels |
| `mcp__github__issue_write` | Apply labels (do NOT close issues) |
| `mcp__github__add_issue_comment` | Add comment (for redirecting questions or requesting info) |
| `mcp__github__search_issues` | Find similar issues for context |

---

## Labels You Must NEVER Add

| Prefix/Category | Reason |
|-----------------|--------|
| Labels not in `labels.json` | Only apply labels that exist in the allowlist |
| `ciflow/*` | CI job triggers for PRs only |
| Version labels (`1.7`, `2.11`, etc.) | Release tracking, added by maintainers |
| `cla signed` | Automated by CLA bot |
| `stale` | Automated by stale bot |
| `do-not-merge` | Requires human decision |
| `approved` | Requires human decision |
| `no-stale` | Requires human decision |

---

## Issue Triage (for each issue)

### 0) Already Triaged — SKIP

**If an issue already has the `triaged` label, SKIP IT entirely.** Do not:
- Add any labels
- Leave comments
- Do any triage work

### 1) Question vs Bug/Feature

- If it is a usage question (e.g., "How do I...", "Why doesn't this work in my setup...") and not a bug in the tutorial itself: add a comment using the `redirect_to_forum` template from `templates.json`. Do NOT close the issue.
- If unclear whether it is a tutorial bug vs a user environment issue: request additional information using the `request_more_info` template and stop.

### 2) Redirect to PyTorch Core

If the issue is actually a PyTorch framework bug (not a tutorial content issue), use the `redirect_to_pytorch` template to point the user to `pytorch/pytorch`. Do NOT transfer or close the issue — only add a comment.

**Indicators of a core PyTorch bug:**
- Error occurs with the user's own code, not tutorial code
- Bug is in `torch.*` API behavior, not tutorial instructions
- Issue involves a PyTorch build/installation problem

### 3) Classify the Issue

Determine the issue type:

| Type | Description | Label |
|------|-------------|-------|
| Tutorial bug | Code errors, outdated API usage, wrong output in a tutorial | `bug` |
| Content issue | Incorrect descriptions, unclear explanations, grammar | `content` |
| Broken link | Dead or incorrect links in tutorials | `incorrect link` |
| Build issue | Tutorial fails to build in CI | `build issue` |
| New tutorial request | Request for a tutorial on a new topic | `new tutorial` or `tutorial-proposal` |
| Enhancement | Improvement to an existing tutorial | `enhancement` |
| Website rendering | Display/formatting issues on pytorch.org/tutorials | `website` |

### 4) Label the Issue

Apply labels from [labels.json](labels.json):

1. **Type label** — One of: `bug`, `content`, `incorrect link`, `build issue`, `new tutorial`, `tutorial-proposal`, `enhancement`, `website`
2. **Domain label** — Based on the tutorial topic area. Common domain labels:
   - `distributed` — distributed training tutorials
   - `torch.compile` — torch.compile related tutorials
   - `core` — core PyTorch functionality
   - `nlp` — NLP tutorials
   - `torchvision`, `torchaudio`, `torchserve`, `torchrec` — domain library tutorials
   - `CUDA`, `mps`, `intel`, `amd` — platform-specific
   - `C++` — C++ frontend tutorials
   - `Mobile` — mobile deployment tutorials
   - `onnx` — ONNX export tutorials
   - `quantization` — quantization tutorials
   - `module: export`, `module: inductor`, `module: profiler` — specific module areas
3. **Difficulty label** (for new tutorial requests only) — `easy`, `medium`, `hard`

### 5) High Priority — REQUIRES HUMAN REVIEW

**CRITICAL:** If you believe an issue is high priority, you MUST:
1. Add `triage review` label (if it exists) or leave a comment noting it needs maintainer attention
2. Do NOT attempt to fix or close the issue

High priority criteria for tutorials:
- Tutorial produces silently wrong results (incorrect code output)
- Tutorial uses a removed or dangerous API
- Build is broken for a popular tutorial
- Security concern (e.g., tutorial instructs users to disable safety features)
- Regression: tutorial that previously worked is now broken

### 6) Mark Triaged

If not flagged for human review, add `triaged`.

---

## V1 Constraints

**DO NOT:**
- Close any issues — only add comments and labels, never close
- Assign issues to users
- Add comments to bug reports or feature requests, except when requesting more info
- Add release version labels

**DO:**
- Comment on usage questions and point to discuss.pytorch.org (per Step 1), but leave the issue open
- Be conservative — when in doubt, leave for human attention
- Apply type labels (`bug`, `content`, `enhancement`, `new tutorial`) when confident
- Apply domain labels when the affected tutorial area is clear
- Add `triaged` label when classification is complete
