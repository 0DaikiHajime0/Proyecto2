.PHONY: create_environment clean data preprocess train predict run_all

## Delete all compiled python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Make Dataset
data:
	python scripts/run_preprocessing.py

## Preprocess data
preprocess:
	python scripts/run_preprocessing.py

## Train models
train:
	python scripts/run_training.py

## Run predictions
predict:
	python scripts/run_prediction.py

## Run all pipeline
run_all: data preprocess train predict

## Run tests
test:
	python -m pytest tests/

## Lint code
lint:
	python -m flake8 src/ scripts/

## Format code
format:
	python -m black src/ scripts/
