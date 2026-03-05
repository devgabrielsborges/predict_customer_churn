MODELS := logistic_regression random_forest svm xgboost_ gradient_boosting knn

.PHONY: help up down restart logs preprocess download train-% train-all clean

help:
	@echo "Infrastructure"
	@echo "  make up             Start Postgres + MinIO + MLflow"
	@echo "  make down           Stop all services"
	@echo "  make restart        Restart all services"
	@echo "  make logs           Tail service logs"
	@echo "  make clean          Stop services and delete volumes"
	@echo ""
	@echo "Data"
	@echo "  make download       Download dataset from Kaggle"
	@echo "  make preprocess     Run preprocessing pipeline"
	@echo ""
	@echo "Training"
	@echo "  make train-<model>  Train a single model (e.g. make train-random_forest)"
	@echo "  make train-all      Train all models sequentially"
	@echo ""
	@echo "Available models: $(MODELS)"

up:
	docker compose up -d --build
	@echo ""
	@echo "MLflow UI:      http://localhost:5000"
	@echo "MinIO Console:  http://localhost:9001  (minioadmin / minioadmin)"

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

init:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	uv run --python 3.11 src/utils/download_dataset.py
	@set -a && [ -f .env ] && . ./.env && set +a; \
	uv run --python 3.11 src/preprocessing/preprocess.py

train-%:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	uv run --python 3.11 src/models/$*.py

train-all:
	@for model in $(MODELS); do \
		echo "\n========== Training $$model =========="; \
		set -a && [ -f .env ] && . ./.env && set +a; \
		uv run --python 3.11 src/models/$$model.py; \
	done

clean:
	docker compose down -v
