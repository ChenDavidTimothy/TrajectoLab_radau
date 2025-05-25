# Makefile for Sphinx documentation

SPHINXOPTS    ?= -W --keep-going -j auto
SPHINXBUILD  ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean-api

# Generate API docs before building
api:
	@echo "🔄 Generating API documentation..."
	cd $(SOURCEDIR) && python generate_api_docs.py

# Build HTML with API generation
html: api
	@echo "🔨 Building HTML documentation..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "✓ Documentation built in _build/html/"

# Clean everything including API docs
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@rm -rf $(SOURCEDIR)/api/
	@echo "✓ Cleaned all build files"

# Quick build without API regeneration
html-fast:
	@echo "🔨 Building HTML documentation (fast mode)..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
