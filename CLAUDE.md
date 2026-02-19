# Environment

If any tool you're trying to use (make, sphinx-build, lintrunner, python, etc.) is missing, always stop and ask the user if an environment is needed. Do NOT try to find alternatives or install these tools.

# Project Structure

This is the PyTorch Tutorials website (`pytorch.org/tutorials`), built with Sphinx and Sphinx Gallery.

- `beginner_source/`, `intermediate_source/`, `advanced_source/`, `recipes_source/`, `unstable_source/` — tutorial source files (`.py` and `.rst`)
- `index.rst`, `recipes_index.rst` — card listings and toctrees for the website
- `conf.py` — Sphinx configuration (gallery dirs, extensions, theme)
- `_static/` — images, CSS, and thumbnails
- `requirements.txt` — all Python dependencies (Sphinx, tutorial packages)
- `.jenkins/` — CI build scripts, data download logic, post-processing
- `Makefile` — build entry points

Tutorials authored as `.py` files use Sphinx Gallery format: top-level docstrings become RST prose, code blocks become executable cells. These are executed during CI builds and converted to Jupyter notebooks and HTML. Tutorials authored as `.rst` are static and their code is not executed.

# Build

- `make html-noplot` — builds HTML without executing tutorial code. Fast, no GPU needed. Use this for local validation of RST/Sphinx structure.
- `make docs` — full build that downloads data, executes all `.py` tutorials, and produces the final site. Requires a GPU-powered machine with CUDA.
- `GALLERY_PATTERN="my_tutorial.py" make html` — build only a single tutorial by name (regex supported).

The CI build runs inside Docker across 15 GPU-powered shards via `.jenkins/build.sh`. Do not attempt to replicate the full CI build locally unless you have a proper GPU setup.

# Linting

This repo uses `lintrunner`. Do not use `spin`, `flake8`, or other linters directly.

```
pip install lintrunner==0.12.5
lintrunner init
```

- `lintrunner -m main` — lint changes relative to the main branch
- `lintrunner --all-files` — lint all files in the repo

# Testing

There is no unit test suite. Validation is done by building tutorials:

- `make html-noplot` is the quick sanity check for RST and Sphinx errors.
- Full execution of `.py` tutorials is handled by CI (GPU shards). Do not attempt to run all tutorials locally.
- To test a single tutorial locally: `GALLERY_PATTERN="my_tutorial.py" make html`

# Tutorial File Format

- Interactive tutorials are `.py` files using Sphinx Gallery conventions. Filenames should end in `_tutorial.py`.
- Non-interactive tutorials are `.rst` files.
- Data dependencies must be added via `.jenkins/download_data.py`, not the Makefile. Follow the existing patterns in that file.
- New Python package dependencies go in `requirements.txt`.

# Adding a New Tutorial

1. Place the file in the appropriate `*_source/` directory based on difficulty level.
2. Add a `customcarditem` entry in `index.rst` (or `recipes_index.rst` for recipes).
3. Add the tutorial to the corresponding `toctree` in `index.rst`.
4. Add a square, high-resolution thumbnail image to `_static/img/thumbnails/cropped/`.

# Commit Messages

Don't commit unless the user explicitly asks you to.

When writing a commit message, don't make a bullet list of the individual
changes. Instead, if the PR is large, explain the order to review changes
(e.g., the logical progression), or if it's short just omit the bullet list
entirely.

Disclose that the PR was authored with Claude.

Do not use ghstack. It is not supported in this repo.

# Coding Style Guidelines

Follow these rules for all code changes in this repository:

- Minimize comments; be concise; code should be self-explanatory.
- Match existing code style and architectural patterns.
- Tutorial prose should be written for a global audience with clear, easy to understand language. Avoid idioms.
- Use active voice in tutorial instructions.
- If uncertain, choose the simpler, more concise implementation.
