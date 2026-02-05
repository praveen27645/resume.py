# Contributing

Thanks for your interest in contributing! Here’s a quick guide to get set up and productive.

## Setup
1. Fork and clone the repo.
2. Install dependencies:
```
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Run locally
```
uvicorn main:app --reload
```

## Tests
```
python -m pytest -q
```

## Style
```
python -m ruff check .
python -m black . --check
```

## Pull Requests
- Keep changes focused and include tests when applicable.
- Update documentation if you change behavior or APIs.
