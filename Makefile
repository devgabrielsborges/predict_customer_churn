.PHONY: help up down restart logs preprocess download train-% train-all clean train-ensemble optimize-ensemble

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
	@echo "Training (TASK_TYPE inferred from .env)"
	@echo "  make train-<model>  Train a single model (e.g. make train-random_forest)"
	@echo "  make train-all      Train all models for the current TASK_TYPE"
	@echo ""
	@echo "Ensemble"
	@echo "  make train-ensemble     Full ensemble pipeline (Optuna + 20-fold)"
	@echo "  make optimize-ensemble  Optuna optimisation only (saves best_config.json)"
	@echo ""
	@set -a && [ -f .env ] && . ./.env && set +a; \
	echo "TASK_TYPE=$$TASK_TYPE  —  available models:"; \
	for f in src/models/$$TASK_TYPE/*.py; do \
		printf "  %s\n" "$$(basename $$f .py)"; \
	done

up:
	@bash scripts/up.sh

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
	uv run --python 3.11 src/models/$$TASK_TYPE/$*.py

train-all:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	for f in src/models/$$TASK_TYPE/*.py; do \
		model=$$(basename "$$f" .py); \
		echo "\n========== Training $$model =========="; \
		uv run --python 3.11 "$$f"; \
	done

train-ensemble:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	uv run --python 3.11 src/ensemble/train.py

optimize-ensemble:
	@set -a && [ -f .env ] && . ./.env && set +a; \
	uv run --python 3.11 src/ensemble/train.py --optimize-only

clean:
	docker compose down -v
