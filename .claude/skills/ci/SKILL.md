---
name: ci-context
description: Constraints and environment details for Claude running inside the GitHub Actions workflow on pytorch/tutorials. This skill is automatically loaded in the CI context. Use when Claude is invoked via @claude mentions on issues or PRs.
---

# CI Environment Skill

This skill applies when Claude is running inside the GitHub Actions workflow (`claude-code.yml`) on `pytorch/tutorials`. It defines what is available, what is allowed, and what is forbidden in this context.

## Environment

| Detail | Value |
|--------|-------|
| Runner | `ubuntu-latest` |
| Python | 3.12 (pre-installed via `actions/setup-python`) |
| Lintrunner | 0.12.5 (pre-installed and initialized) |
| Timeout | 60 minutes |
| Model | `claude-opus-4-6-v1` via AWS Bedrock |

**All tools you need are already installed.** Do not run `pip install`, `apt-get`, or any other installation commands. If a tool is missing, state that it is unavailable and move on.

## Permissions

The workflow grants these GitHub token permissions:

| Permission | Level | What it allows |
|------------|-------|----------------|
| `contents` | `read` | Read repo files, checkout code |
| `pull-requests` | `write` | Comment on PRs, post reviews |
| `issues` | `write` | Comment on issues, add/remove labels |
| `id-token` | `write` | OIDC authentication to AWS Bedrock |

## What You MUST NOT Do

- **Commit or push** — You have read-only access to repo contents. Never attempt `git commit`, `git push`, or create branches.
- **Merge or close PRs** — You cannot and should not merge pull requests.
- **Close issues** — Never close issues. Only add comments and labels.
- **Install packages** — Everything needed is pre-installed. Do not run `pip install`, `npm install`, `apt-get`, etc.
- **Modify workflow files** — Do not suggest changes to `.github/workflows/` files in automated comments.
- **Assign users** — Do not assign issues or PRs to specific people.
- **Add labels that require human judgment** — See the triage skill for the full restricted labels list.

## What You CAN Do

- **Read all repo files** — Full checkout is available at the workspace root.
- **Run lintrunner** — `lintrunner -m main` or `lintrunner --all-files` are available.
- **Run make (dry/noplot)** — `make html-noplot` works for RST/Sphinx validation (no GPU).
- **Comment on PRs and issues** — Post review comments, inline suggestions, and general comments.
- **Add/remove labels on issues** — Apply triage labels from the approved list.
- **Search for similar issues** — Use GitHub MCP tools to find duplicates or related issues.

## Available MCP Tools

| Tool | Purpose |
|------|---------|
| `mcp__github__issue_read` | Read issue details, comments, and labels |
| `mcp__github__issue_write` | Add/remove labels on issues (do NOT close) |
| `mcp__github__add_issue_comment` | Post a comment on an issue |
| `mcp__github__search_issues` | Search for similar/duplicate issues |
| `mcp__github__pr_read` | Read PR details, diff, and review comments |
| `mcp__github__pr_comment` | Post a comment or review on a PR |

## Trigger Context

Claude is invoked when a user mentions `@claude` in:
- An issue comment (`issue_comment` event)
- A PR review comment (`pull_request_review_comment` event)
- A new issue body (`issues.opened` event)

The triggering comment or issue body is passed as the prompt. Respond directly to what the user asked — do not perform unrequested actions.

## Interaction Style

- You are responding asynchronously via GitHub comments. There is no interactive terminal session.
- Be concise — GitHub comments should be scannable, not walls of text.
- Use markdown formatting (headers, tables, code blocks) for readability.
- If you cannot complete a request due to permission constraints, explain what you tried and what the user should do instead.
