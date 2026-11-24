# PyTorch Documentation Release Automation

Automates the mechanical steps of the PyTorch docs release checklist.

## Prerequisites

- Git push access to `pytorch/tutorials` and `pytorch/docs`
- CUDA version suffix from the RC announcement on [dev-discuss.pytorch.org](https://dev-discuss.pytorch.org)
- Optional: `lintrunner` installed for lint checks (skipped if not available)

## Phases

### Phase 1: `enable-nightly` (after first RC)

Run after the first release candidate is available. Creates PRs in both repos.

**Tutorials repo (`pytorch/tutorials`):**
- Enables nightly CI workflow triggers (pull_request + push to main)
- Updates torch RC version in `.jenkins/build.sh`

**Docs repo (`pytorch/docs`):**
- Adds `v{version}.0 (release candidate)` entry to `pytorch-versions.json`
- Bumps the `(unstable)` label to the next version

### Phase 2: `pre-release` (a few days before release)

Single command that creates four PRs across both repos.

**Tutorials repo (`pytorch/tutorials`):**
- Updates `torch==` pin in `requirements.txt`
- Updates torch version in `.jenkins/build.sh`
- Pushes branch `release-{version}` with these changes

**Docs repo (`pytorch/docs`):**
- Promotes RC to `(stable)` with `"preferred": true` in `pytorch-versions.json` (branch: `update-stable-{version}`)
- Updates `stable` symlink to new version (branch: `update-stable-symlink-{version}`)
- Adds noindex tags to previous version docs (branch: `add-noindex-{prev_version}`)

### Phase 3: `post-release` (day after release)

**Tutorials repo (`pytorch/tutorials`):**
- Disables nightly CI workflow triggers
- Updates torch version in `.ci/docker/requirements.txt`

## Usage

```bash
# 1. After first RC is available
python .release/release_docs.py --version 2.12 --cuda 130 --phase enable-nightly

# 2. A few days before release (creates 4 PRs: 1 tutorials + 3 docs)
python .release/release_docs.py --version 2.12 --cuda 130 --prev-version 2.11 --phase pre-release

# 3. Day after release
python .release/release_docs.py --version 2.12 --prev-version 2.11 --phase post-release
```

The cross-repo phases (`update-versions`, `stable-symlink`, `noindex`) can also be run individually if needed:

```bash
python .release/release_docs.py --version 2.12 --phase update-versions
python .release/release_docs.py --version 2.12 --phase stable-symlink
python .release/release_docs.py --version 2.12 --prev-version 2.11 --phase noindex
```

## Options

| Flag | Description |
|------|-------------|
| `--version` | New PyTorch version (e.g., `2.12`) |
| `--prev-version` | Previous version (e.g., `2.11`). Required for `pre-release`, `post-release`, and `noindex` |
| `--cuda` | CUDA version suffix (e.g., `130` for `cu130`). Required for `enable-nightly` and `pre-release`. Get this from the [dev-discuss RC announcement](https://dev-discuss.pytorch.org) |
| `--clean` | Remove existing `pytorch/docs` clone at `/tmp/pytorch-docs` and start fresh |
| `--dry-run` | Show what would change without modifying files |
| `--phase list` | Show all phases with descriptions |

## How it works

- **Tutorials repo changes**: Creates a branch, commits, and pushes. You create the PR via the printed GitHub compare link.
- **Docs repo changes**: Shallow-clones `pytorch/docs` to `/tmp/pytorch-docs` (reuses if already present). Each phase creates a separate branch and pushes it. You create PRs via the printed links.
- **Validation**: `pytorch-versions.json` is validated after every modification (valid JSON, required fields, exactly one preferred entry, no duplicates).
- **Re-runnable**: Safe to run multiple times. Stale local branches are deleted before re-creating, and remote branches are force-pushed.

## What it does NOT do

- Merge any PRs (you review and merge manually)
- Update ecosystem package versions (`torchrl`, `torchao`, etc.) in `requirements.txt`
- Update the "What's New" section in `index.rst`
- Review proposed features or attend meetings

## Full checklist

See the [PyTorch Documentation Release Checklist](https://www.internalfb.com/wiki/PyTorch/Teams/PyTorch_Doc_Engineering/PyTorch_Documentation_Release_Checklist/) for the complete process.
