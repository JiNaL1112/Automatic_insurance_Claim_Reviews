.PHONY: help install generate train serve test lint docker-up docker-down clean

help:
	@echo ""
	@echo "  make install      Install all dependencies"
	@echo "  make generate     Generate synthetic claims data"
	@echo "  make train        Run full ML pipeline (train + evaluate + register)"
	@echo "  make serve        Start BentoML service on :3000"
	@echo "  make app          Start Flask app on :5005"
	@echo "  make test         Run all pytest tests"
	@echo "  make lint         Run ruff linter"
	@echo "  make docker-up    Start all services via docker-compose"
	@echo "  make docker-down  Stop all docker services"
	@echo "  make clean        Remove pycache, model artifacts"
	@echo ""

install:
	pip install -r ml-core/requirements.txt
	pip install -r app/requirements.txt

generate:
	python ml-core/src/data/generate.py

train:
	python ml-core/src/models/pipeline.py

serve:
	cd ml-core && bentoml serve src/serving/service.py --reload

app:
	python app/src/api/flask_app.py

test:
	pytest ml-core/tests/ -v
	pytest app/tests/ -v

lint:
	ruff check ml-core/src/ app/src/

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f ml-core/model.pkl