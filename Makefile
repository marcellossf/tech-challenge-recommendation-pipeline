.PHONY: install lint format test validate download reproduce mlflow train

install:
	poetry install

lint:
	poetry run ruff check src tests scripts
	poetry run ruff format --check src tests scripts

format:
	poetry run ruff check --fix src tests scripts
	poetry run ruff format src tests scripts

test:
	poetry run pytest

validate:
	poetry run validate-env

download:
	poetry run download-data

reproduce:
	poetry run dvc repro

mlflow:
	poetry run mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

train:
	poetry run train-models

