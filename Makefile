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
	# Step1. DOWNLOAD: wget -N [SOURCE_FILE] -P $(DATADIR)
	# Step2-1. UNZIP: unzip -o $(DATADIR)/[SOURCE_FILE] -d [*_source/data/]
	# Step2-2. UNTAR: tar -xzf $(DATADIR)/[SOURCE_FILE] -C [*_source/data/]
	# Step2-3. AS-IS: cp $(DATADIR)/[SOURCE_FILE] [*_source/data/]

	# make data directories
	mkdir -p $(DATADIR)
	mkdir -p advanced_source/data
	mkdir -p beginner_source/data
	mkdir -p intermediate_source/data
	mkdir -p prototype_source/data

	# transfer learning tutorial data
	wget -N https://download.pytorch.org/tutorial/hymenoptera_data.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/hymenoptera_data.zip -d beginner_source/data/

	# nlp tutorial data
	wget -N https://download.pytorch.org/tutorial/data.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/data.zip -d intermediate_source/  # This will unzip all files in data.zip to intermediate_source/data/ folder

	# data loader tutorial
	wget -N https://download.pytorch.org/tutorial/faces.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/faces.zip -d beginner_source/data/

	wget -N https://download.pytorch.org/models/tutorials/4000_checkpoint.tar -P $(DATADIR)
	cp $(DATADIR)/4000_checkpoint.tar beginner_source/data/

	# neural style images
	rm -rf advanced_source/data/images/ || true
	mkdir -p advanced_source/data/images/
	cp -r _static/img/neural-style/ advanced_source/data/images/

	# Download dataset for beginner_source/dcgan_faces_tutorial.py
	wget -N https://s3.amazonaws.com/pytorch-tutorial-assets/img_align_celeba.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/img_align_celeba.zip -d beginner_source/data/celeba

	# Download dataset for beginner_source/hybrid_frontend/introduction_to_hybrid_frontend_tutorial.py
	wget -N https://s3.amazonaws.com/pytorch-tutorial-assets/iris.data -P $(DATADIR)
	cp $(DATADIR)/iris.data beginner_source/data/

	# Download dataset for beginner_source/chatbot_tutorial.py
	wget -N https://s3.amazonaws.com/pytorch-tutorial-assets/cornell_movie_dialogs_corpus.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/cornell_movie_dialogs_corpus.zip -d beginner_source/data/

	# Download dataset for beginner_source/audio_classifier_tutorial.py
	wget -N https://s3.amazonaws.com/pytorch-tutorial-assets/UrbanSound8K.tar.gz -P $(DATADIR)
	tar $(TAROPTS) -xzf $(DATADIR)/UrbanSound8K.tar.gz -C ./beginner_source/data/

	# Download model for beginner_source/fgsm_tutorial.py
	wget -N https://s3.amazonaws.com/pytorch-tutorial-assets/lenet_mnist_model.pth -P $(DATADIR)
	cp $(DATADIR)/lenet_mnist_model.pth ./beginner_source/data/lenet_mnist_model.pth

	# Download model for advanced_source/dynamic_quantization_tutorial.py
	wget -N https://s3.amazonaws.com/pytorch-tutorial-assets/word_language_model_quantize.pth -P $(DATADIR)
	cp $(DATADIR)/word_language_model_quantize.pth advanced_source/data/word_language_model_quantize.pth

	# Download data for advanced_source/dynamic_quantization_tutorial.py
	wget -N https://s3.amazonaws.com/pytorch-tutorial-assets/wikitext-2.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/wikitext-2.zip -d advanced_source/data/

	# Download model for advanced_source/static_quantization_tutorial.py
	wget -N https://download.pytorch.org/models/mobilenet_v2-b0353104.pth -P $(DATADIR)
	cp $(DATADIR)/mobilenet_v2-b0353104.pth advanced_source/data/mobilenet_pretrained_float.pth


	# Download model for prototype_source/graph_mode_static_quantization_tutorial.py
	wget -N https://download.pytorch.org/models/resnet18-5c106cde.pth -P $(DATADIR)
	cp $(DATADIR)/resnet18-5c106cde.pth prototype_source/data/resnet18_pretrained_float.pth


docs:
	make download
	make html
	rm -rf docs
	cp -r $(BUILDDIR)/html docs
	touch docs/.nojekyll

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	# bash .jenkins/remove_invisible_code_block_batch.sh "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean-cache:
	make clean
	rm -rf advanced beginner intermediate recipes
