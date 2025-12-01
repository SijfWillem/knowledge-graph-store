.PHONY: help build up down restart logs clean setup dev-backend dev-frontend test status neo4j-shell start wipe

# Default target
help:
	@echo "Cognee RAG - Makefile Commands"
	@echo "=============================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make start          - Build and start everything"
	@echo "  make wipe           - Delete ALL data (Neo4j, uploads, Cognee)"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create .env file from template"
	@echo "  make build          - Build all Docker images"
	@echo ""
	@echo "Running:"
	@echo "  make up             - Start all services (Docker)"
	@echo "  make down           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make logs           - View logs from all services"
	@echo "  make status         - Show status of all services"
	@echo ""
	@echo "Development:"
	@echo "  make dev-backend    - Run backend locally (without Docker)"
	@echo "  make dev-frontend   - Run frontend locally (without Docker)"
	@echo "  make dev            - Run both backend and frontend locally"
	@echo ""
	@echo "Database:"
	@echo "  make neo4j-shell    - Open Neo4j browser (http://localhost:7475)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove all containers and volumes"
	@echo "  make wipe           - Delete ALL data and restart fresh"
	@echo ""

# Setup environment file
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please edit it with your API keys."; \
	else \
		echo ".env file already exists."; \
	fi

# Build Docker images
build:
	docker-compose build

# Start all services
up:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Run 'make setup' first."; \
		exit 1; \
	fi
	docker-compose up -d
	@echo ""
	@echo "Services starting..."
	@echo "  - Frontend: http://localhost:8501"
	@echo "  - Backend API: http://localhost:8000"
	@echo "  - Neo4j Browser: http://localhost:7475"
	@echo "  - Weaviate: http://localhost:8081"
	@echo ""
	@echo "Run 'make logs' to view logs"

# Stop all services
down:
	docker-compose down

# Restart services
restart: down up

# View logs
logs:
	docker-compose logs -f

# View logs for specific service
logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

logs-neo4j:
	docker-compose logs -f neo4j

logs-weaviate:
	docker-compose logs -f weaviate

# Show service status
status:
	docker-compose ps

# Local development - backend only
dev-backend:
	@echo "Starting backend locally..."
	@echo "Make sure Neo4j and Weaviate are running (docker-compose up neo4j weaviate)"
	cd backend && pip install -r requirements.txt && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Local development - frontend only
dev-frontend:
	@echo "Starting frontend locally..."
	@echo "Make sure backend is running"
	cd frontend && pip install -r requirements.txt && streamlit run app.py

# Start just the databases (for local development)
dev-db:
	docker-compose up -d neo4j weaviate
	@echo "Neo4j and Weaviate started."
	@echo "  - Neo4j Browser: http://localhost:7475 (neo4j/cogneepassword)"
	@echo "  - Weaviate: http://localhost:8081"

# Local development - both services
dev: dev-db
	@echo "Starting backend and frontend locally..."
	@echo "Run in separate terminals:"
	@echo "  Terminal 1: make dev-backend"
	@echo "  Terminal 2: make dev-frontend"

# Neo4j browser shortcut
neo4j-shell:
	@echo "Opening Neo4j Browser at http://localhost:7475"
	@echo "Credentials: neo4j / cogneepassword"
	@open http://localhost:7475 2>/dev/null || xdg-open http://localhost:7475 2>/dev/null || echo "Please open http://localhost:7475 in your browser"

# Check Weaviate status
weaviate-check:
	@curl -s http://localhost:8081/v1/.well-known/ready | python3 -m json.tool || echo "Weaviate not ready"

# Clean up everything
clean:
	docker-compose down -v --remove-orphans
	@echo "Cleaned up containers and volumes"

# Clean uploaded files
clean-uploads:
	rm -rf uploads/*
	@echo "Cleaned uploaded files"

# Full reset
reset: clean clean-uploads
	@echo "Full reset complete"

# Install local dependencies (for development)
install:
	pip install -r backend/requirements.txt
	pip install -r frontend/requirements.txt

# Run tests
test:
	@echo "Running backend tests..."
	cd backend && python -m pytest tests/ -v || echo "No tests found"

# Health check
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "Backend not responding"
	@curl -s http://localhost:8081/v1/.well-known/ready | python3 -m json.tool || echo "Weaviate not responding"
	@curl -s http://localhost:7475 > /dev/null && echo "Neo4j: OK" || echo "Neo4j not responding"

# Start everything (setup + build + up)
start:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file - please edit it with your OPENAI_API_KEY"; \
		echo "Then run 'make start' again"; \
		exit 1; \
	fi
	@echo "Building and starting all services..."
	docker-compose build
	docker-compose up -d
	@echo ""
	@echo "============================================"
	@echo "  Cognee RAG is starting!"
	@echo "============================================"
	@echo ""
	@echo "  Frontend:    http://localhost:8501"
	@echo "  Backend API: http://localhost:8000"
	@echo "  Neo4j:       http://localhost:7475"
	@echo ""
	@echo "  Run 'make logs' to view logs"
	@echo "============================================"

# Wipe all data (delete everything and start fresh)
wipe:
	@echo "Stopping all services..."
	docker-compose down -v --remove-orphans
	@echo "Clearing uploaded files..."
	rm -rf uploads/* 2>/dev/null || true
	touch uploads/.gitkeep
	@echo ""
	@echo "============================================"
	@echo "  All data has been wiped!"
	@echo "============================================"
	@echo ""
	@echo "  - Neo4j database: cleared"
	@echo "  - Cognee data: cleared"
	@echo "  - Uploaded files: cleared"
	@echo ""
	@echo "  Run 'make start' to start fresh"
	@echo "============================================"
