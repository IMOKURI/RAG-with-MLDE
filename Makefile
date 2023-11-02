.PHONY: help
.DEFAULT_GOAL := help


build: ## Build container image.
	docker build -t baseimage:latest -f Dockerfile.baseimage .
	docker build -t determined:latest -f Dockerfile.determined .
	docker build -t fastchat:latest -f Dockerfile.fastchat .
	docker build -t streamlit:latest -f Dockerfile.streamlit .


up-determined: ## Start Determined cluster.
	det deploy local cluster-up


batch-inference: ## Run batch inference.
	det experiment create ./determined_config.yaml .


fastchat: ## Start FastChat API Server.
	docker compose --project-name llm --env-file fastchat1.env up -d


rag-app: ## Run RAG-System app.
	docker run -d --rm --name rag-system -p 8501:8501 \
		--gpus all --shm-size=32g \
		--net llm_default \
		-v /home/hpe01/.cache:/root/.cache \
		-v $(pwd):/app \
		streamlit:latest \
		streamlit run rag_system.py


down: down-rag down-fastchat ## Stop all containers.

down-rag: ## Stop RAG-System.
	docker stop rag-system

down-fastchat: ## Stop FastChat API Server.
	docker compose --project-name llm --env-file fastchat1.env down


help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
