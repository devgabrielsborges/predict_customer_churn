#!/usr/bin/env bash
set -euo pipefail

port_in_use() {
    ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${1}$"
}

echo "Checking services..."

postgres_ext=false
minio_ext=false
mlflow_ext=false

if port_in_use 5432; then
    postgres_ext=true
    echo "  PostgreSQL: port 5432 already in use, reusing existing service"
fi

if port_in_use 9000 || port_in_use 9001; then
    minio_ext=true
    echo "  MinIO: port 9000/9001 already in use, reusing existing service"
fi

if port_in_use 5000; then
    mlflow_ext=true
    echo "  MLflow: port 5000 already in use, reusing existing service"
fi

if $postgres_ext && $minio_ext && $mlflow_ext; then
    echo ""
    echo "All services already running!"
    echo ""
    echo "MLflow UI:      http://localhost:5000"
    echo "MinIO Console:  http://localhost:9001  (minioadmin / minioadmin)"
    exit 0
fi

# --- Start services whose ports are free, in dependency order ---

if ! $postgres_ext; then
    echo "Starting PostgreSQL..."
    docker compose up -d --build --no-deps postgres
    echo "Waiting for PostgreSQL to be ready..."
    until docker compose exec -T postgres pg_isready -U mlflow &>/dev/null; do sleep 1; done
    echo "PostgreSQL ready."
fi

if ! $minio_ext; then
    echo "Starting MinIO..."
    docker compose up -d --build --no-deps minio
    echo "Waiting for MinIO to be ready..."
    until docker compose exec -T minio curl -sf http://localhost:9000/minio/health/live &>/dev/null; do sleep 1; done
    echo "MinIO ready. Running bucket setup..."
    docker compose up -d --no-deps minio-setup
fi

if ! $mlflow_ext; then
    # When dependencies are external, point MLflow at the host network
    if $postgres_ext; then
        export POSTGRES_HOST=host.docker.internal
    fi
    if $minio_ext; then
        export MINIO_HOST=host.docker.internal
    fi

    echo "Starting MLflow..."
    docker compose up -d --build --no-deps mlflow
    echo "MLflow ready."
fi

echo ""
echo "MLflow UI:      http://localhost:5000"
echo "MinIO Console:  http://localhost:9001  (minioadmin / minioadmin)"
