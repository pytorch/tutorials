# Minimal makefile for Sphinx documentation
#

# Locale
export LC_ALL=C

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PyTorchTutorials
SOURCEDIR     = .
BUILDDIR      = _build
GH_PAGES_SOURCES = $(SOURCEDIR) Makefile

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

	# transfer learning tutorial data
	wget -N https://download.pytorch.org/tutorial/hymenoptera_data.zip
	unzip -o hymenoptera_data.zip -d beginner_source/data
	
	# nlp tutorial data
	wget -N https://download.pytorch.org/tutorial/data.zip
	unzip -o data.zip -d intermediate_source/  # This will unzip all files in data.zip to intermediate_source/data/ folder
	
	# data loader tutorial
	wget -N https://download.pytorch.org/tutorial/faces.zip
	unzip -o faces.zip -d beginner_source/data

	wget -N https://download.pytorch.org/models/tutorials/4000_checkpoint.tar
	cp 4000_checkpoint.tar beginner_source/data
	
	# neural style images
	rm -rf advanced_source/data/images/ || true
	mkdir -p advanced_source/data/images/
	cp -r _static/img/neural-style/ advanced_source/data/images/

	# Download dataset for beginner_source/dcgan_faces_tutorial.py
	curl https://s3.amazonaws.com/pytorch-tutorial-assets/img_align_celeba.zip --output img_align_celeba.zip
	sudo mkdir -p beginner_source/data/celeba
	sudo chmod -R 0777 beginner_source/data/celeba
	unzip img_align_celeba.zip -d beginner_source/data/celeba > null

	# Download dataset for beginner_source/hybrid_frontend/introduction_to_hybrid_frontend_tutorial.py
	mkdir data/
	curl https://s3.amazonaws.com/pytorch-tutorial-assets/iris.data --output beginner_source/data/iris.data

	# Download dataset for beginner_source/chatbot_tutorial.py
	curl https://s3.amazonaws.com/pytorch-tutorial-assets/cornell_movie_dialogs_corpus.zip --output cornell_movie_dialogs_corpus.zip
	mkdir -p beginner_source/data
	unzip cornell_movie_dialogs_corpus.zip -d beginner_source/data/ > null

	# Download dataset for beginner_source/audio_classifier_tutorial.py
	curl https://s3.amazonaws.com/pytorch-tutorial-assets/UrbanSound8K.tar.gz --output UrbanSound8K.tar.gz
	tar -xzf UrbanSound8K.tar.gz -C ./beginner_source/data

	# Download model for beginner_source/fgsm_tutorial.py
	curl https://s3.amazonaws.com/pytorch-tutorial-assets/lenet_mnist_model.pth --output ./beginner_source/data/lenet_mnist_model.pth

docs:
	make download
	make html
	rm -rf docs
	cp -r $(BUILDDIR)/html docs
	touch docs/.nojekyll

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	bash .jenkins/remove_invisible_code_block_batch.sh "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean-cache:
	make clean
	rm -rf advanced beginner intermediate
