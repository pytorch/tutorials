# PR Review Checklist

This checklist covers areas that CI cannot check. CI lintrunner only catches trailing whitespace, tabs, and missing newlines — it does **not** validate RST syntax, Python formatting, import ordering, or Sphinx directives. Those must all be reviewed manually.

References:
- [CONTRIBUTING.md](../../../CONTRIBUTING.md) — Authoring guidance, submission process, tutorial structure template
- [tutorial_submission_policy.md](../../../tutorial_submission_policy.md) — Acceptance criteria, community submission process, maintenance expectations
- [beginner_source/template_tutorial.py](../../../beginner_source/template_tutorial.py) — Official tutorial template with required structure
- [conf.py](../../../conf.py) — Sphinx Gallery configuration and build settings

## Acceptance Criteria (tutorial_submission_policy.md § "Acceptance Criteria")

For **new tutorial** PRs, verify the submission meets one of the two accepted use cases:

- [ ] **New PyTorch feature tutorial** — Authored by engineers developing the feature for an upcoming release. Typically one tutorial per feature. Does not require the community submission process
- [ ] **Community tutorial (PyTorch + other tools)** — Illustrates innovative uses of PyTorch alongside other open-source projects, models, and tools. Must remain neutral and not promote or endorse proprietary technologies over others

### Community Submission Process (tutorial_submission_policy.md § "Submission Process")

For community-contributed tutorials, verify:

- [ ] **Corresponding issue exists** — An issue was filed in pytorch/tutorials proposing the tutorial using the [Feature request template](https://github.com/pytorch/tutorials/blob/main/.github/ISSUE_TEMPLATE/feature-request.yml), explaining importance and confirming no existing tutorial covers the same topic
- [ ] **Issue has `approved` label** — A maintainer reviewed and approved the proposal. PRs without a corresponding approved issue may take longer to review
- [ ] **PR links the approved issue** — The PR description references the issue where approval was granted

## Tutorial Content Quality

### Learning Objectives (CONTRIBUTING.md § "Learning objectives")

- [ ] **Explicit learning objectives** — Tutorial states what the user will implement by the end. Focus on what the user will *do*, not abstract concepts. Examples: "Create a custom dataset", "Implement greedy-search decoding", "Train encoder and decoder models using mini-batches"
- [ ] **Actionable content** — Tutorials and recipes are always actionable. If the material is purely informative, it belongs in the API docs, not a tutorial
- [ ] **No duplication** — Tutorial demonstrates PyTorch functionality not already covered by existing tutorials. Check existing tutorials at https://pytorch.org/tutorials/

### Tutorial Structure (CONTRIBUTING.md § "Structure")

The recommended structure from CONTRIBUTING.md:

1. [ ] **Introduction** — What the tutorial is about
2. [ ] **Motivation** — Why this topic is important
3. [ ] **Background links** — Links to relevant research papers or background material
4. [ ] **Learning objectives** — Clearly state what the tutorial covers and what users will implement by the end (see the [TensorBoard tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) as a good example)
5. [ ] **Setup and requirements** — Call out any required setup or data downloads upfront
6. [ ] **Step-by-step instructions** — Steps should map back to the learning objectives. Code comments should correspond to these steps
7. [ ] **Links to PyTorch documentation** — Reference relevant [API docs](https://pytorch.org/docs/stable/index.html) to give readers context
8. [ ] **Recap/Conclusion** — Summarize the steps and concepts covered, highlight key takeaways
9. [ ] **Additional resources** — Links for further learning (docs, other tutorials, research)
10. [ ] *(Optional)* **Practice exercises** — For users to test their knowledge (see the [NLP From Scratch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#exercises) for an example)

### Template Compliance (beginner_source/template_tutorial.py)

For `.py` tutorials, verify use of the official template structure:

- [ ] **"What you will learn" grid card** — Uses the `.. grid:: 2` / `.. grid-item-card::` RST directive with `:octicon:\`mortar-board;1em;\`` to list what the user will learn
- [ ] **"Prerequisites" grid card** — Adjacent grid card with `:octicon:\`list-unordered;1em;\`` listing prerequisites (PyTorch version, GPU requirements, etc.)
- [ ] **Overview section** — Describes why the topic is important, links to relevant research
- [ ] **Steps section** — Code with inline explanations
- [ ] **Conclusion section** — Summarizes steps and concepts, highlights key takeaways
- [ ] **Further Reading section** — Links for additional learning

### Voice and Writing Style (CONTRIBUTING.md § "Voice and writing style")

- [ ] **Global audience** — Clear, easy to understand language. No idioms or figures of speech
- [ ] **Active voice** — Instructions are instructive and directive, not passive
- [ ] **Concise** — No extraneous information; content serves the learning objectives
- [ ] **Neutral tone** — Does not disproportionately endorse one technology over another (per submission policy)

## Tutorial Types and File Format (CONTRIBUTING.md § "New Tutorials")

### Interactive Tutorials (`.py` files)

- [ ] **Filename ends in `_tutorial.py`** — Required naming convention (e.g., `cool_pytorch_feature_tutorial.py`)
- [ ] **Not a Jupyter notebook** — `.ipynb` files are not accepted. Must be a `.py` file using Sphinx Gallery format
- [ ] **Python format preferred** — Per submission policy: unless the tutorial involves multi-GPU, parallel/distributed training, or requires extended execution time (25+ minutes), `.py` format is preferred over `.rst`
- [ ] **Top docstring is RST** — First triple-quoted docstring becomes the tutorial title and intro prose. Must use RST formatting with a proper heading (underlined with `===`)
- [ ] **`######################################################################` separators** — Code blocks are separated by comment blocks starting with `#` lines. The `#` comment lines between code blocks become RST prose
- [ ] **Code blocks are logical** — Each code cell does one clear thing
- [ ] **Prose between code blocks** — Comment blocks between code explain the next step, mapping back to learning objectives
- [ ] **No `!pip install` or `%magic` commands** — Notebook magic syntax is not allowed. The `first_notebook_cell` in `conf.py` handles `%matplotlib inline` automatically
- [ ] **Author attribution** — Includes `**Author:** \`Name <url>\`_` in the opening docstring

### Non-Interactive Tutorials (`.rst` files)

- [ ] **Valid RST syntax** — Directives, roles, and cross-references are well-formed
- [ ] **Code blocks have language specified** — Use `.. code-block:: python`, not bare `::`
- [ ] **Used appropriately** — `.rst` format should only be used for tutorials involving multi-GPU, parallel/distributed training, or extended execution time (25+ minutes)

### Recipes (CONTRIBUTING.md § "New Tutorials")

- [ ] **Bite-sized and scoped** — Recipes demonstrate how to use a specific feature, not a full end-to-end workflow
- [ ] **Added to `recipes_source/`** — Recipes go in `recipes_source/`, not the other `*_source/` directories
- [ ] **Listed in `recipes_source/recipes/README.txt`** — Recipe is added to the recipes README

## Project Structure and Index (CONTRIBUTING.md § "Submission Process")

### File Placement

- [ ] **Correct source directory** — File is in the right directory based on difficulty level:
  - `beginner_source/` — Introductory content
  - `intermediate_source/` — Intermediate content
  - `advanced_source/` — Advanced content
  - `recipes_source/` — Recipes (any difficulty)

### Card Entry (CONTRIBUTING.md § "Include Your Tutorial in index.rst")

- [ ] **`customcarditem` added** — Card entry exists in `index.rst` (or `recipes_index.rst` for recipes) with all required fields:
  ```rst
  .. customcarditem::
     :header: Tutorial Title
     :card_description: A short description of the tutorial.
     :image: _static/img/thumbnails/cropped/my-thumbnail.png
     :link: beginner/my_tutorial.html
     :tags: Getting-Started
  ```
- [ ] **Tags are valid** — Tags chosen from existing tags in the file. Multi-word tags are hyphenated (e.g., `Getting-Started`). No whitespace between tag words — incorrect tags will break the build and cards won't display
- [ ] **Link path is correct** — The `link` value matches the tutorial's location (e.g., `beginner/my_tutorial.html` for a file in `beginner_source/`)

### Toctree Entry

- [ ] **Added to the appropriate toctree** — Tutorial listed under the correct topic section in `index.rst`

### Thumbnail Image (CONTRIBUTING.md § "Image")

- [ ] **Image added to `_static/img/thumbnails/cropped/`** — Thumbnail exists
- [ ] **Square dimensions** — Equal `x` and `y` dimensions render best
- [ ] **High resolution** — Image is clear and not pixelated

## Code Correctness

### Python Code in Tutorials

- [ ] **Code runs correctly** — Examples produce the described output
- [ ] **Imports are complete** — All necessary imports are present
- [ ] **API usage is current** — Uses non-deprecated PyTorch APIs
- [ ] **No hardcoded paths** — No absolute paths or user-specific paths in code
- [ ] **Reproducibility** — Random seeds set where results need to be deterministic

## Dependencies and Data (CONTRIBUTING.md § "Managing data" and "Python packages")

### Data

- [ ] **Data downloads via `.jenkins/download_data.py`** — Follow the same pattern as other download functions. Do NOT add download logic to `Makefile` (it incurs overhead for all CI shards)
- [ ] **Data stored on reliable external storage** — Recommended: Amazon S3 or similar. URLs should be long-lived
- [ ] **No large files committed** — Data and models are downloaded at build time, not checked into the repo

### Python Packages

- [ ] **New dependencies added to `requirements.txt`** — Any package not already listed must be added
- [ ] **Mature, well-supported packages only** — Obscure or poorly maintained packages may break with Python/PyTorch updates and lead to tutorial deprecation (per submission policy: tutorials broken for 90+ days may be deleted)
- [ ] **Versions pinned** — New dependencies should have pinned versions for reproducibility

## Build Compatibility

- [ ] **No CI-breaking changes** — Tutorial doesn't introduce untested imports or missing data
- [ ] **GPU requirements noted** — If tutorial needs GPU, it's documented and CI can handle it
- [ ] **Reasonable execution time** — Tutorials requiring 25+ minutes of execution time should be `.rst` format (per submission policy). Keep individual `.py` tutorials well under CI timeout limits
- [ ] **Clean build with `make html-noplot`** — RST structure is valid even without executing code

## Maintenance Expectations (tutorial_submission_policy.md § "Maintaining Tutorials")

When reviewing new tutorials, consider long-term maintainability:

- [ ] **Author can maintain it** — The contributor should be able to keep the tutorial in sync with PyTorch updates
- [ ] **Not fragile** — Tutorial doesn't depend on rapidly-changing external APIs or unstable packages that are likely to break
- [ ] **90-day fix window** — If a tutorial breaks against main, it will be excluded from the build. If not resolved within 90 days, it may be deleted from the repository
- [ ] **Annual review expectation** — Each tutorial should be reviewed at least once a year to ensure relevance

## Common Issues to Flag

- Tutorials that import packages not in `requirements.txt`
- Broken or outdated links to pytorch.org/docs
- Code that only works on specific PyTorch versions without noting the requirement
- Missing or incorrect card tags in `index.rst` (tags must be hyphenated, no spaces between words)
- Tutorials that download data in the Makefile instead of `.jenkins/download_data.py`
- Jupyter notebooks submitted instead of `.py` files
- Copy-pasted code from notebooks with leftover `!pip` or `%magic` commands
- Missing author attribution in the opening docstring
- Tutorials with purely informative content that belongs in API docs instead
- PRs where the author tries to self-merge (only maintainers authorize publishing)
- Community tutorials submitted without a corresponding approved issue
- Tutorials that promote proprietary technologies or are not neutral
- Tutorials duplicating content already covered by existing tutorials
- Missing "What you will learn" / "Prerequisites" grid cards from the official template
