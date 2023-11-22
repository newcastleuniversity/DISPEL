.PHONY: clean clean-test clean-docs clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

ifeq (, $(shell which snakeviz))
	PROFILE = pytest --profile-svg
	PROFILE_RESULT = prof/combined.svg
	PROFILE_VIEWER = $(BROWSER)
else
    PROFILE = pytest --profile
    PROFILE_RESULT = prof/combined.prof
	PROFILE_VIEWER = snakeviz
endif

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-lint clean-typing clean-test clean-docs ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -path ./venv -prune -false -o -name '*.egg-info' -exec rm -fr {} +
	find . -path ./venv -prune -false -o -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -path ./venv -prune -false -o -name '*.pyc' -exec rm -f {} +
	find . -path ./venv -prune -false -o -name '*.pyo' -exec rm -f {} +
	find . -path ./venv -prune -false -o -name '*~' -exec rm -f {} +
	find . -path ./venv -prune -false -o -name '__pycache__' -exec rm -fr {} +

clean-lint: ## remove lint artifacts
	rm -fr .ruff_cache

clean-typing: ## remove type checking artifacts
	rm -fr .mypy_cache

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr prof/

clean-docs: ## remove docs artifacts
	rm -fr docs/_build
	rm -fr docs/api

lint: ## check style with ruff
	ruff check dispel tests

lint-docs: docs-api ## check docs using pydocstyle and doc8
	doc8 docs *.rst --ignore-path docs/_build --max-line-length 88

format: ## Format code with isort and black
	isort dispel tests
	black dispel tests
	ruff dispel tests --fix

test: ## run tests quickly with the default Python
	pytest

test-typing: ## check static typing using mypy
	mypy dispel

test-docs: docs-api ## run tests defined in documentation
	pytest --doctest-glob="*.rst" --doctest-plus docs dispel

coverage: ## check code coverage quickly with the default Python
	coverage run --source dispel -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

profile:  ## create a profile from test cases
	$(PROFILE) $(TARGET)
	$(PROFILE_VIEWER) $(PROFILE_RESULT)

docs-api:  ## generate the API documentation for Sphinx
	rm -rf docs/api
	sphinx-apidoc -e -M -o docs/api dispel

docs: docs-api ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

dist: clean ## builds source and wheel package
	python -m build
