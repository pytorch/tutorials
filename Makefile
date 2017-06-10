# Minimal makefile for Sphinx documentation
#

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
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

download:
	# transfer learning tutorial data
	wget -N https://download.pytorch.org/tutorial/hymenoptera_data.zip
	unzip -o hymenoptera_data.zip -d beginner_source
	
	# nlp tutorial data
	wget -N https://download.pytorch.org/tutorial/data.zip
	unzip -o data.zip -d intermediate_source
	
	# data loader tutorial
	wget -N https://download.pytorch.org/tutorial/faces.zip
	unzip -o faces.zip -d beginner_source
	
	# neural style images
	rm -rf advanced_source/images/
	cp -r _static/img/neural-style/ advanced_source/images/

docs:
	make download
	make html
	rm -rf docs
	cp -r $(BUILDDIR)/html docs
	touch docs/.nojekyll

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean-cache:
	make clean
	rm -rf advanced beginner intermediate
