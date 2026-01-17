.PHONY: help install test train run docker-build docker-up docker-down clean

help:
	@echo "Infrastructure Health Predictor - Make commands"
	@echo ""
	@echo "  make install       Install dependencies"
	@echo "  make test          Run tests"
	@echo "  make train         Train model with sample data"
	@echo "  make run           Start API server"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start docker-compose services"
	@echo "  make docker-down   Stop docker-compose services"
	@echo "  make clean         Clean generated files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

train:
	python src/data/generate_sample_data.py
	python src/models/train_model.py --data data/raw/sample_metrics.csv --epochs 30

run:
	python src/api/main.py

docker-build:
	docker build -t infrastructure-health-predictor -f docker/Dockerfile .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -f .coverage
