#!/usr/bin/env python3
"""PyTorch Documentation Release Automation.

Automates the mechanical steps of the PyTorch docs release process:
- Enable/disable nightly CI workflow
- Update torch version pins across the repo
- Prepare the stable symlink PR in pytorch/docs
- Remove old version docs from Google Search
- Update the "What's New" section in index.rst

Usage:
    python .release/release_docs.py --version 2.12 --phase enable-nightly
    python .release/release_docs.py --version 2.12 --phase pre-release --prev-version 2.11
    python .release/release_docs.py --version 2.12 --phase post-release
    python .release/release_docs.py --version 2.12 --phase list

Each phase creates branches and PRs but does NOT merge them.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NIGHTLY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build-tutorials-nightly.yml"
BUILD_SH = REPO_ROOT / ".jenkins" / "build.sh"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
INDEX_RST = REPO_ROOT / "index.rst"


def run(cmd, check=True, capture=True, cwd=None):
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True,
        cwd=cwd or REPO_ROOT
    )
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip() if capture else ""


def validate_versions_json(versions_file):
    """Validate pytorch-versions.json structure and content."""
    print("\nValidating pytorch-versions.json...")
    try:
        versions = json.loads(versions_file.read_text())
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON — {e}")
        return False

    required_fields = {"name", "version", "url"}
    errors = []
    has_preferred = False

    for i, entry in enumerate(versions):
        missing = required_fields - set(entry.keys())
        if missing:
            errors.append(f"  Entry {i}: missing fields: {', '.join(missing)}")
        if entry.get("preferred"):
            if has_preferred:
                errors.append(f"  Entry {i}: multiple entries have 'preferred': true")
            has_preferred = True

    if not has_preferred:
        errors.append("  No entry has 'preferred': true")

    if errors:
        print("  Issues found:")
        for e in errors:
            print(e)
        return False

    print(f"  Valid — {len(versions)} entries, preferred set.")
    return True


def run_linter():
    """Run lintrunner on changed files."""
    print("\nRunning linter...")
    # Check if lintrunner is available
    check = subprocess.run(
        "which lintrunner", shell=True, capture_output=True, cwd=REPO_ROOT
    )
    if check.returncode != 0:
        print("  lintrunner not found. Install with: pip install lintrunner")
        print("  Skipping lint check.")
        return

    result = subprocess.run(
        "lintrunner -m main", shell=True, capture_output=True, text=True,
        cwd=REPO_ROOT
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print("  Linter found issues. Please fix before submitting.")
        if result.stderr:
            print(result.stderr)
    else:
        print("  Linter passed.")


# ---------------------------------------------------------------------------
# Phase: enable-nightly
# ---------------------------------------------------------------------------

def detect_current_cuda():
    """Try to detect the current CUDA version from .jenkins/build.sh."""
    if BUILD_SH.exists():
        match = re.search(r'whl/test/cu(\d+)', BUILD_SH.read_text())
        if match:
            return match.group(1)
    return None


def dedup_versions(versions):
    """Remove duplicate version entries, keeping the first occurrence."""
    seen = set()
    deduped = []
    for entry in versions:
        ver = entry.get("version", "")
        if ver in seen:
            print(f"  Removed duplicate entry for {ver}.")
            continue
        seen.add(ver)
        deduped.append(entry)
    return deduped




def phase_enable_nightly(version, cuda_version):
    """Enable the nightly/RC CI workflow and update the torch version pin."""
    print(f"\n=== Phase: enable-nightly (PyTorch {version}) ===\n")
    changes = []

    # 1. Update nightly workflow triggers
    print("[1/2] Enabling nightly workflow triggers...")
    content = NIGHTLY_WORKFLOW.read_text()
    original = content

    # Uncomment pull_request trigger
    content = content.replace(
        "  # pull_request:",
        "  pull_request:"
    )
    # Uncomment push trigger
    content = content.replace(
        "  # push:\n  #   branches:\n  #    - main",
        "  push:\n    branches:\n     - main"
    )

    if content != original:
        NIGHTLY_WORKFLOW.write_text(content)
        changes.append(str(NIGHTLY_WORKFLOW.relative_to(REPO_ROOT)))
        print("  Enabled pull_request and push triggers.")
    else:
        print("  Triggers already enabled.")

    # 2. Update torch version in .jenkins/build.sh
    print("[2/3] Updating torch version in .jenkins/build.sh...")
    content = BUILD_SH.read_text()
    original = content

    content = re.sub(
        r'pip3 install torch==[\d.]+ torchvision torchaudio --index-url https://download\.pytorch\.org/whl/test/cu\d+',
        f'pip3 install torch=={version}.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu{cuda_version}',
        content
    )

    if content != original:
        BUILD_SH.write_text(content)
        changes.append(str(BUILD_SH.relative_to(REPO_ROOT)))
        print(f"  Updated to torch=={version}.0 with cu{cuda_version}.")
    else:
        print("  Already up to date.")

    # 3. Add RC entry to pytorch-versions.json in pytorch/docs
    print("[3/3] Adding RC entry to pytorch-versions.json in pytorch/docs...")
    # Compute next version for the unstable label (e.g., 2.12 -> 2.13)
    major, minor = version.split(".")
    next_version = f"{major}.{int(minor) + 1}"

    docs_dir = ensure_docs_clone()
    versions_file = docs_dir / "pytorch-versions.json"

    if not versions_file.exists():
        print(f"  ERROR: pytorch-versions.json not found in {docs_dir}")
        sys.exit(1)

    versions = json.loads(versions_file.read_text())

    # Update the main/unstable label to next version
    for entry in versions:
        if entry.get("version") == "main":
            entry["name"] = f"v{next_version}.0 (unstable)"
            print(f"  Updated main entry label to v{next_version}.0 (unstable).")
            break

    # Add RC entry if not already present
    rc_exists = any(e.get("version") == version for e in versions)
    if not rc_exists:
        rc_entry = {
            "name": f"v{version}.0 (release candidate)",
            "version": version,
            "url": f"https://docs.pytorch.org/docs/{version}/"
        }
        # Insert after the main/unstable entry
        insert_idx = 0
        for i, entry in enumerate(versions):
            if entry.get("version") == "main":
                insert_idx = i + 1
                break
        versions.insert(insert_idx, rc_entry)
        print(f"  Added v{version}.0 (release candidate) entry.")
    else:
        print(f"  RC entry for {version} already exists.")

    versions = dedup_versions(versions)
    versions_file.write_text(json.dumps(versions, indent=2) + "\n")
    validate_versions_json(versions_file)

    branch_name = f"add-rc-{version}"
    docs_checkout_branch(branch_name)
    run("git add pytorch-versions.json", cwd=docs_dir)
    run(f'git commit -m "Add {version} RC to pytorch-versions.json"', cwd=docs_dir)
    docs_push_branch(branch_name)

    if changes:
        print(f"\nLocal files modified (tutorials repo): {', '.join(changes)}")
    print("\nNext steps:")
    print(f"  1. Review the changes: git diff")
    print(f"  2. Create a branch and PR for tutorials repo changes")
    print(f"  3. Review the pytorch/docs PR for pytorch-versions.json")

    run_linter()
    return changes


# ---------------------------------------------------------------------------
# Phase: pre-release
# ---------------------------------------------------------------------------

def phase_pre_release(version, prev_version, cuda_version):
    """Prepare PRs for a few days before the release."""
    print(f"\n=== Phase: pre-release (PyTorch {version}, prev: {prev_version}) ===\n")

    # 1. Update requirements.txt with new stable version
    print("[1/5] Updating torch version in requirements.txt...")
    content = REQUIREMENTS.read_text()
    content = content.replace(f"torch=={prev_version}", f"torch=={version}")
    REQUIREMENTS.write_text(content)
    print(f"  Updated torch=={prev_version} -> torch=={version}")

    # 2. Switch build.sh back to stable
    print("[2/5] Switching .jenkins/build.sh back to stable...")
    content = BUILD_SH.read_text()
    content = re.sub(
        r'pip3 install torch==[\d.]+ torchvision torchaudio --index-url https://download\.pytorch\.org/whl/test/cu\d+',
        f'pip3 install torch=={version}.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu{cuda_version}',
        content
    )
    BUILD_SH.write_text(content)
    print(f"  Updated build.sh to torch=={version}.0.")

    files_changed = [
        str(REQUIREMENTS.relative_to(REPO_ROOT)),
        str(BUILD_SH.relative_to(REPO_ROOT)),
    ]

    # 3. Create tutorials repo PR
    print("[3/5] Creating tutorials repo PR...")
    branch_name = f"release-{version}"
    original_branch = run("git rev-parse --abbrev-ref HEAD")
    # Stash any prior uncommitted changes, create branch from main
    run("git stash", check=False)
    run(f"git checkout main")
    run("git pull origin main")
    run(f"git branch -D {branch_name}", check=False)
    run(f"git checkout -b {branch_name}")

    # Re-apply the changes
    content = REQUIREMENTS.read_text()
    content = content.replace(f"torch=={prev_version}", f"torch=={version}")
    REQUIREMENTS.write_text(content)

    content = BUILD_SH.read_text()
    content = re.sub(
        r'pip3 install torch==[\d.]+ torchvision torchaudio --index-url https://download\.pytorch\.org/whl/test/cu\d+',
        f'pip3 install torch=={version}.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu{cuda_version}',
        content
    )
    BUILD_SH.write_text(content)

    for f in files_changed:
        run(f"git add {f}")

    # Check if there are staged changes to commit
    if subprocess.run("git diff --cached --quiet", shell=True, cwd=REPO_ROOT).returncode != 0:
        run(f'git commit -m "Update to PyTorch {version} stable"')
        run(f"git push -u origin {branch_name} --force")
        print(f"  Pushed branch '{branch_name}'.")
        print(f"  Create PR: https://github.com/pytorch/tutorials/compare/main...{branch_name}")
    else:
        print("  No changes to commit — files already match target versions.")

    # Return to original branch
    run(f"git checkout {original_branch}")
    run("git stash pop", check=False)

    # 4-5: Cross-repo PRs in pytorch/docs
    phase_update_versions(version)
    phase_stable_symlink(version)
    phase_noindex(prev_version)

    run_linter()
    return files_changed


# ---------------------------------------------------------------------------
# Phase: stable-symlink (cross-repo helper)
# ---------------------------------------------------------------------------

DOCS_CLONE_DIR = Path("/tmp/pytorch-docs")


def clean_docs_clone():
    """Remove the existing pytorch/docs clone."""
    if DOCS_CLONE_DIR.exists():
        import shutil
        shutil.rmtree(DOCS_CLONE_DIR)
        print(f"  Removed {DOCS_CLONE_DIR}.")
    else:
        print(f"  No clone at {DOCS_CLONE_DIR} to remove.")


def ensure_docs_clone():
    """Clone pytorch/docs to /tmp/pytorch-docs if not already present."""
    if not DOCS_CLONE_DIR.exists():
        print(f"  Cloning pytorch/docs to {DOCS_CLONE_DIR}...")
        run(f"git clone --depth 1 --branch site git@github.com:pytorch/docs.git {DOCS_CLONE_DIR}")
    else:
        print(f"  Using existing clone at {DOCS_CLONE_DIR}.")
        run("git fetch origin", cwd=DOCS_CLONE_DIR)
    # Always start from the site branch
    run("git checkout site", cwd=DOCS_CLONE_DIR)
    run("git pull origin site", cwd=DOCS_CLONE_DIR)
    return DOCS_CLONE_DIR


def docs_checkout_branch(branch_name):
    """Create or reset a branch in the docs clone."""
    # Delete local branch if it exists from a previous run
    run(f"git branch -D {branch_name}", cwd=DOCS_CLONE_DIR, check=False)
    run(f"git checkout -b {branch_name}", cwd=DOCS_CLONE_DIR)


def docs_push_branch(branch_name):
    """Push branch to pytorch/docs and print the PR URL."""
    run(f"git push -u origin {branch_name} --force", cwd=DOCS_CLONE_DIR)
    print(f"  Pushed branch '{branch_name}'.")
    print(f"  Create PR: https://github.com/pytorch/docs/compare/site...{branch_name}")
    # Return to site branch for next operation
    run("git checkout site", cwd=DOCS_CLONE_DIR)


def phase_update_versions(version):
    """Update pytorch-versions.json in pytorch/docs to set preferred version."""
    print(f"\n=== Updating pytorch-versions.json for {version} ===\n")

    docs_dir = ensure_docs_clone()
    versions_file = docs_dir / "pytorch-versions.json"

    if not versions_file.exists():
        print(f"  ERROR: pytorch-versions.json not found in {docs_dir}")
        print("  Check that pytorch/docs has this file on the site branch.")
        sys.exit(1)

    versions = json.loads(versions_file.read_text())
    print(f"  Found {len(versions)} version entries.")

    # Promote the RC entry to stable with preferred, demote the old stable.
    found = False
    for entry in versions:
        ver = entry.get("version", "")
        # Move preferred to the new version
        if ver == version:
            entry["name"] = f"v{version}.0 (stable)"
            entry["url"] = f"https://docs.pytorch.org/docs/{version}/"
            entry["preferred"] = True
            found = True
            print(f"  Set {version} as preferred (stable).")
        elif entry.get("preferred"):
            del entry["preferred"]
            print(f"  Removed preferred from {ver}.")
        # Demote old "(stable)" label
        if "(stable)" in entry.get("name", "") and ver != version:
            entry["name"] = entry["name"].replace(" (stable)", "")
            print(f"  Demoted {ver} from stable label.")

    if not found:
        new_entry = {
            "name": f"v{version}.0 (stable)",
            "version": version,
            "url": f"https://docs.pytorch.org/docs/{version}/",
            "preferred": True
        }
        # Insert after the main/unstable entry (index 1), or at 0 if no main
        insert_idx = 0
        for i, entry in enumerate(versions):
            if entry.get("version") == "main":
                insert_idx = i + 1
                break
        versions.insert(insert_idx, new_entry)
        print(f"  Added {version} as new preferred (stable) entry.")

    # Ensure the main/unstable entry shows the next version, not the current release
    major, minor = version.split(".")
    next_version = f"{major}.{int(minor) + 1}"
    for entry in versions:
        if entry.get("version") == "main":
            if f"v{version}" in entry.get("name", ""):
                entry["name"] = f"v{next_version}.0 (unstable)"
                print(f"  Updated main entry label to v{next_version}.0 (unstable).")
            break

    versions = dedup_versions(versions)
    versions_file.write_text(json.dumps(versions, indent=2) + "\n")
    validate_versions_json(versions_file)

    branch_name = f"update-stable-{version}"
    docs_checkout_branch(branch_name)
    run("git add pytorch-versions.json", cwd=docs_dir)
    run(f'git commit -m "Set {version} as preferred version"', cwd=docs_dir)
    docs_push_branch(branch_name)


def phase_stable_symlink(version):
    """Create the stable symlink PR in pytorch/docs."""
    print(f"\n=== Creating stable symlink PR for {version} ===\n")

    docs_dir = ensure_docs_clone()

    branch_name = f"update-stable-symlink-{version}"
    docs_checkout_branch(branch_name)
    run("rm -f stable", cwd=docs_dir)
    run(f'ln -s "{version}" stable', cwd=docs_dir)
    run("git add stable", cwd=docs_dir)
    run(f'git commit -m "Update stable symlink to {version}"', cwd=docs_dir)
    docs_push_branch(branch_name)


# ---------------------------------------------------------------------------
# Phase: noindex (cross-repo helper)
# ---------------------------------------------------------------------------

def phase_noindex(prev_version):
    """Create the noindex PR in pytorch/docs for the previous version."""
    print(f"\n=== Creating noindex PR for {prev_version} ===\n")

    docs_dir = ensure_docs_clone()

    branch_name = f"add-noindex-{prev_version}"
    docs_checkout_branch(branch_name)
    run(f"bash add_noindex_tag.sh {prev_version}", cwd=docs_dir)
    run("git add .", cwd=docs_dir)
    run(f'git commit -m "Add noindex tags to {prev_version} docs"', cwd=docs_dir)
    docs_push_branch(branch_name)


# ---------------------------------------------------------------------------
# Phase: post-release (disable nightly, clean up)
# ---------------------------------------------------------------------------

def phase_post_release(version, prev_version):
    """Disable the nightly workflow after the release."""
    print(f"\n=== Phase: post-release (PyTorch {version}) ===\n")
    changes = []

    # 1. Disable nightly workflow triggers
    print("[1/2] Disabling nightly workflow triggers...")
    content = NIGHTLY_WORKFLOW.read_text()
    original = content

    # Comment out pull_request trigger (but not if already commented)
    content = re.sub(
        r'^(\s{2})pull_request:',
        r'\1# pull_request:',
        content,
        flags=re.MULTILINE
    )
    # Comment out push trigger block
    content = re.sub(
        r'^(\s{2})push:\n(\s+)branches:\n(\s+)- main',
        r'\1# push:\n\2#   branches:\n\3#    - main',
        content,
        flags=re.MULTILINE
    )

    if content != original:
        NIGHTLY_WORKFLOW.write_text(content)
        changes.append(str(NIGHTLY_WORKFLOW.relative_to(REPO_ROOT)))
        print("  Disabled pull_request and push triggers.")
    else:
        print("  Triggers already disabled.")

    # 2. Update torch version in .ci/docker/requirements.txt
    docker_req = REPO_ROOT / ".ci" / "docker" / "requirements.txt"
    print("[2/2] Updating torch version in .ci/docker/requirements.txt...")
    if docker_req.exists() and prev_version:
        content = docker_req.read_text()
        original = content
        content = content.replace(f"torch=={prev_version}", f"torch=={version}")
        if content != original:
            docker_req.write_text(content)
            changes.append(str(docker_req.relative_to(REPO_ROOT)))
            print(f"  Updated torch=={prev_version} -> torch=={version}.")
        else:
            print("  Already up to date.")
    else:
        print("  Skipped (file not found or --prev-version not provided).")

    if changes:
        print(f"\nFiles modified: {', '.join(changes)}")
        print("\nNext steps:")
        print("  1. Review: git diff")
        print("  2. Create branch and PR")
    else:
        print("\nNo changes needed.")

    return changes


# ---------------------------------------------------------------------------
# Phase: list (show what each phase does)
# ---------------------------------------------------------------------------

def phase_list():
    print("""
PyTorch Docs Release Phases
============================

Tutorials repo (pytorch/tutorials):

  enable-nightly   Enable the RC/nightly CI workflow, pull torch RC version.
                   Run after the first RC is available.
                   Modifies: .github/workflows/build-tutorials-nightly.yml
                             .jenkins/build.sh
  pre-release      Prepare tutorials repo for release + show cross-repo steps.
                   Updates torch version to stable, comments out RC config.
                   Modifies: requirements.txt, .ci/docker/requirements.txt, .jenkins/build.sh

  post-release     Disable the nightly workflow, update docker torch pin.
                   Modifies: .github/workflows/build-tutorials-nightly.yml
                             .ci/docker/requirements.txt
Docs repo (pytorch/docs) — each creates a PR against the 'site' branch:

  update-versions  Set the new version as preferred in pytorch-versions.json.
                   Clones pytorch/docs to /tmp/pytorch-docs.

  stable-symlink   Update the stable symlink to the new version.
                   Reuses /tmp/pytorch-docs clone.

  noindex          Add noindex tags to the previous version's docs.
                   Runs add_noindex_tags.sh, reuses /tmp/pytorch-docs clone.

Common options:
  --version        The new PyTorch version (e.g., 2.12)
  --prev-version   The previous version (e.g., 2.11) — needed for pre-release and noindex
  --cuda           CUDA version suffix (e.g., 130 for cu130). Required for enable-nightly,
                   pre-release, and post-release. Get this from the RC announcement post
                   on dev-discuss.pytorch.org (e.g., the RC announcement post).
  --clean          Remove existing pytorch/docs clone and start fresh
  --dry-run        Show what would change without modifying files

Typical release workflow:
  1. python .release/release_docs.py --version 2.12 --cuda 130 --phase enable-nightly
  2. python .release/release_docs.py --version 2.12 --cuda 130 --prev-version 2.11 --phase pre-release
  3. python .release/release_docs.py --version 2.12 --phase update-versions
  4. python .release/release_docs.py --version 2.12 --phase stable-symlink
  5. python .release/release_docs.py --version 2.12 --prev-version 2.11 --phase noindex
  6. python .release/release_docs.py --version 2.12 --prev-version 2.11 --phase post-release
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyTorch Documentation Release Automation")
    parser.add_argument("--version", help="New PyTorch version (e.g., 2.12)")
    parser.add_argument("--prev-version", help="Previous PyTorch version (e.g., 2.11)")
    parser.add_argument("--cuda",
                        help="CUDA version suffix (e.g., 130 for cu130). "
                             "Get this from the release Workplace post install command.")
    parser.add_argument("--phase", required=True,
                        choices=["enable-nightly", "pre-release", "post-release",
                                 "update-versions", "stable-symlink", "noindex", "list"],
                        help="Release phase to execute")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying files")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing pytorch/docs clone and start fresh")

    args = parser.parse_args()

    if args.phase == "list":
        phase_list()
        return

    if not args.version:
        parser.error("--version is required for all phases except 'list'")

    # Phases that modify CUDA-versioned URLs need --cuda
    cuda_phases = {"enable-nightly", "pre-release"}
    if args.phase in cuda_phases and not args.cuda:
        # Try to detect current CUDA version from build.sh
        current_cuda = detect_current_cuda()
        hint = f" (currently {current_cuda} in .jenkins/build.sh)" if current_cuda else ""
        parser.error(
            f"--cuda is required for {args.phase}. "
            f"Get the CUDA suffix from the RC install command in the "
            f"dev-discuss.pytorch.org announcement post "
            f"(e.g., --cuda 130 for cu130){hint}."
        )

    if args.clean:
        clean_docs_clone()

    if args.dry_run:
        print("DRY RUN MODE — no files will be modified.\n")

    if args.phase == "enable-nightly":
        phase_enable_nightly(args.version, args.cuda)

    elif args.phase == "pre-release":
        if not args.prev_version:
            parser.error("--prev-version is required for pre-release phase")
        phase_pre_release(args.version, args.prev_version, args.cuda)

    elif args.phase == "post-release":
        if not args.prev_version:
            parser.error("--prev-version is required for post-release phase")
        phase_post_release(args.version, args.prev_version)

    elif args.phase == "update-versions":
        if args.dry_run:
            print(f"Would update pytorch-versions.json to set {args.version} as preferred")
            return
        phase_update_versions(args.version)

    elif args.phase == "stable-symlink":
        if args.dry_run:
            print(f"Would create stable -> {args.version} symlink PR in pytorch/docs")
            return
        phase_stable_symlink(args.version)

    elif args.phase == "noindex":
        if not args.prev_version:
            parser.error("--prev-version is required for noindex phase")
        if args.dry_run:
            print(f"Would clone pytorch/docs and create noindex PR for {args.prev_version}")
            return
        phase_noindex(args.prev_version)


if __name__ == "__main__":
    main()
