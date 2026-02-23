---
name: pr-review
description: Review PyTorch tutorials pull requests for content quality, code correctness, build compatibility, and style. Use when reviewing PRs, when asked to review code changes, or when the user mentions "review PR", "code review", or "check this PR".
---

# Tutorials PR Review Skill

Review PyTorch tutorials pull requests for content quality, code correctness, tutorial structure, and Sphinx/RST formatting. CI lintrunner only checks trailing whitespace, tabs, and newlines — it does not validate RST syntax, Python formatting, or Sphinx directives, so those must be reviewed manually.

## Usage Modes

### No Argument

If the user invokes `/pr-review` with no arguments, **do not perform a review**. Instead, ask the user what they would like to review:

> What would you like me to review?
> - A PR number or URL (e.g., `/pr-review 12345`)
> - A local branch (e.g., `/pr-review branch`)

### Local CLI Mode

The user provides a PR number or URL:

```
/pr-review 12345
/pr-review https://github.com/pytorch/tutorials/pull/12345
```

For a detailed review with line-by-line specific comments:

```
/pr-review 12345 detailed
```

Use `gh` CLI to fetch PR data:

```bash
# Get PR details
gh pr view <PR_NUMBER> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits

# Get the diff
gh pr diff <PR_NUMBER>

# Get PR comments
gh pr view <PR_NUMBER> --json comments,reviews
```

### Local Branch Mode

Review changes in the current branch that are not in `main`:

```
/pr-review branch
/pr-review branch detailed
```

Use git commands to get branch changes:

```bash
# Get current branch name
git branch --show-current

# Get list of changed files compared to main
git diff --name-only main...HEAD

# Get full diff compared to main
git diff main...HEAD

# Get commit log for the branch
git log main..HEAD --oneline

# Get diff stats (files changed, insertions, deletions)
git diff --stat main...HEAD
```

For local branch reviews:
- The "Summary" should describe what the branch changes accomplish based on commit messages and diff
- Use the current branch name in the review header instead of a PR number
- All other review criteria apply the same as PR reviews

### GitHub Actions Mode

When invoked via workflow, PR data is passed as context. The PR number or diff will be available in the prompt. See the [CI Environment Skill](../ci/SKILL.md) for environment constraints, available tools, and permissions.

## Review Workflow

### Step 1: Fetch PR Information

For local mode, use `gh` commands to get:
1. PR metadata (title, description, author)
2. List of changed files
3. Full diff of changes
4. Existing comments/reviews

### Step 2: Analyze Changes

Read through the diff systematically:
1. Identify the purpose of the change from title/description
2. Group changes by type (tutorial content, config, build, infra)
3. Note the scope of changes (files affected, lines changed)

### Step 3: Deep Review

Perform thorough analysis using the review checklist. See [review-checklist.md](review-checklist.md) for detailed criteria covering:
- Tutorial content quality and accuracy
- Code correctness in tutorial examples
- Sphinx/RST formatting
- Build compatibility
- Project structure compliance

### Step 4: Formulate Review

Structure your review with actionable feedback organized by category.

## Review Areas

| Area | Focus | Reference |
|------|-------|-----------|
| Content Quality | Accuracy, clarity, learning objectives | [review-checklist.md](review-checklist.md) |
| Code Correctness | Working examples, imports, API usage | [review-checklist.md](review-checklist.md) |
| Structure | File placement, index entries, toctree | [review-checklist.md](review-checklist.md) |
| Formatting | RST/Sphinx syntax, Sphinx Gallery conventions | [review-checklist.md](review-checklist.md) |
| Build | Dependencies, data downloads, CI compat | [review-checklist.md](review-checklist.md) |

## Output Format

Structure your review as follows:

```markdown
## PR Review: #<number>
<!-- Or for local branch reviews: -->
## Branch Review: <branch-name> (vs main)

### Summary
Brief overall assessment of the changes (1-2 sentences).

### Content Quality
[Issues and suggestions, or "No concerns" if none]

### Code Correctness
[Issues with tutorial code examples, imports, API usage, or "No concerns"]

### Structure & Formatting
[File placement, RST/Sphinx issues, index/toctree entries, or "No concerns"]

### Build Compatibility
[Dependency issues, data download concerns, CI compatibility, or "No concerns"]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

[Brief justification for recommendation]
```

### Specific Comments (Detailed Review Only)

**Only include this section if the user requests a "detailed" or "in depth" review.**

**Do not repeat observations already made in other sections.** This section is for additional file-specific feedback that doesn't fit into the categorized sections above.

When requested, add file-specific feedback with line references:

```markdown
### Specific Comments
- `beginner_source/my_tutorial.py:42` - Docstring prose is unclear; rephrase for non-native speakers
- `index.rst:150` - Missing customcarditem entry for the new tutorial
- `requirements.txt:30` - New dependency should be pinned to a specific version
```

## Key Principles

1. **No repetition** - Each observation appears in exactly one section
2. **Focus on what CI cannot check** - Don't comment on trailing whitespace or tab characters (caught by lintrunner). RST syntax, Sphinx directives, and Python code style must still be reviewed
3. **Be specific** - Reference file paths and line numbers
4. **Be actionable** - Provide concrete suggestions, not vague concerns
5. **Be proportionate** - Minor issues shouldn't block, but note them
6. **Audience awareness** - Tutorials are read by beginners; clarity matters more than brevity

## Files to Reference

When reviewing, consult these project files for context:
- `CLAUDE.md` - Project structure and coding style
- `CONTRIBUTING.md` - Submission process, tutorial types, and authoring guidance
- `conf.py` - Sphinx Gallery configuration and extensions
- `requirements.txt` - Approved dependencies
- `index.rst` - Card listings and toctree structure
