.PHONY: install install-dev lint fmt test run

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	python -m ruff check .

fmt:
	python -m black .
	python -m ruff check . --fix

test:
	python -m pytest -q

run:
	uvicorn main:app --reload
