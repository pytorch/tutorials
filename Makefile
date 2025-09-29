# Minimal makefile for Sphinx documentation
#

# Locale
export LC_ALL=C

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PyTorchTutorials
SOURCEDIR     = .
BUILDDIR      = _build
DATADIR       = _data
GH_PAGES_SOURCES = $(SOURCEDIR) Makefile

ZIPOPTS       ?= -qo
TAROPTS       ?=

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile docs

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -v

download:
	# IMPORTANT NOTE: Please make sure your dataset is downloaded to *_source/data folder,
	# otherwise CI might silently break.

	# NOTE: Please consider using the Step1 and one of Step2 for new dataset,
	# [something] should be replaced with the actual value.
	# Step1. DOWNLOAD: wget -nv -N [SOURCE_FILE] -P $(DATADIR)
	# Step2-1. UNZIP: unzip -o $(DATADIR)/[SOURCE_FILE] -d [*_source/data/]
	# Step2-2. UNTAR: tar -xzf $(DATADIR)/[SOURCE_FILE] -C [*_source/data/]
	# Step2-3. AS-IS: cp $(DATADIR)/[SOURCE_FILE] [*_source/data/]

	# Run structured downloads first (will also make directories
	python3 .jenkins/download_data.py

	# data loader tutorial
	wget -nv -N https://download.pytorch.org/tutorial/faces.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/faces.zip -d beginner_source/data/

	wget -nv -N https://download.pytorch.org/models/tutorials/4000_checkpoint.tar -P $(DATADIR)
	cp $(DATADIR)/4000_checkpoint.tar beginner_source/data/

	# neural style images
	rm -rf advanced_source/data/images/ || true
	mkdir -p advanced_source/data/images/
	cp -r _static/img/neural-style/ advanced_source/data/images/

	# Download dataset for beginner_source/hybrid_frontend/introduction_to_hybrid_frontend_tutorial.py
	wget -nv -N https://s3.amazonaws.com/pytorch-tutorial-assets/iris.data -P $(DATADIR)
	cp $(DATADIR)/iris.data beginner_source/data/

	# Download dataset for beginner_source/chatbot_tutorial.py
	wget -nv -N https://s3.amazonaws.com/pytorch-tutorial-assets/cornell_movie_dialogs_corpus_v2.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/cornell_movie_dialogs_corpus_v2.zip -d beginner_source/data/

	# Download PennFudanPed dataset for intermediate_source/torchvision_tutorial.py
	wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P $(DATADIR)
	unzip -o $(DATADIR)/PennFudanPed.zip -d intermediate_source/data/

download-last-reviewed-json:
	@echo "Downloading tutorials-review-data.json..."
	curl -o tutorials-review-data.json https://raw.githubusercontent.com/pytorch/tutorials/refs/heads/last-reviewed-data-json/tutorials-review-data.json
	@echo "Finished downloading tutorials-review-data.json."
docs:
	make download
	make download-last-reviewed-json
	make html
	@python .jenkins/insert_last_verified.py $(BUILDDIR)/html
	rm -rf docs
	cp -r $(BUILDDIR)/html docs
	touch docs/.nojekyll
	rm -rf tutorials-review-data.json

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	# bash .jenkins/remove_invisible_code_block_batch.sh "$(BUILDDIR)/html"
	@echo
	make download-last-reviewed-json
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
	@echo "Running post-processing script to insert 'Last Verified' dates..."
	@python .jenkins/insert_last_verified.py $(BUILDDIR)/html
	rm -rf tutorials-review-data.json

clean-cache:
	make clean
	rm -rf advanced beginner intermediate recipes
	# remove additional python files downloaded for torchvision_tutorial.py
	rm -rf intermediate_source/engine.py intermediate_source/utils.py intermediate_source/transforms.py intermediate_source/coco_eval.py intermediate_source/coco_utils.py
