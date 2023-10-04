.PHONY: help
.DEFAULT_GOAL := help

NOW = $(shell date '+%Y%m%d-%H%M%S')
IMAGE_TAG = latest


build: ## Build container image.
	docker build -t rag-system:$(IMAGE_TAG) .


batch-inference: ## Run batch inference.
	det experiment create ./determined_config.yaml .


fastchat-controller: ## Start FastChat API Server Controller.
	python3 -m fastchat.serve.controller

fastchat-model-worker: ## Start FastChat API Server Model Worker.
	python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5

fastchat-api-server: ## Start FastChat API Server
	python3 -m fastchat.serve.openai_api_server --host localhost --port 8000


rag-app: ## Run RAG-System app.
	streamlit run rag_system.py


help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
