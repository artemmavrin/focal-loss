PYTHON := python
PIP := $(PYTHON) -m pip
SETUP := $(PYTHON) setup.py -q
COVERAGE := coverage
DOCS := docs
SPHINXOPTS := '-W'
RM := rm -rf

.PHONY: clean dev docs help install py_info test

help:
	@ echo "Usage:\n"
	@ echo "make install   Install the package using Setuptools."
	@ echo "make dev       Install the package for development using pip."
	@ echo "make test      Run unit tests and check code coverage."
	@ echo "make docs      Generate package documentation using Sphinx"
	@ echo "make clean     Remove auxiliary files."

install: clean py_info
	$(PIP) install --upgrade .

dev: clean py_info
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade setuptools wheel
	$(PIP) install --upgrade --editable .[dev]

test: clean py_info
	$(COVERAGE) run -m pytest
	$(COVERAGE) report --show-missing

docs: clean
	make -C $(DOCS) html SPHINXOPTS=$(SPHINXOPTS)

clean:
	@ $(RM) $(DOCS)/build $(DOCS)/source/generated
	@ $(RM) src/*.egg-info .eggs .pytest_cache .coverage
	@ $(RM) build dist

distribute: clean py_info
	@ $(PIP) install --upgrade twine
	$(SETUP) sdist bdist_wheel
	@ echo "Upload to PyPI using 'twine upload dist/*'"

py_info:
	@ echo "Using $$($(PYTHON) --version) at $$(which $(PYTHON))"
