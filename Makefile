.PHONY: help
.DEFAULT_GOAL := help


build: ## Build container image.
	docker build -t rag-system:latest .


up-determined: ## Start Determined cluster.
	det deploy local cluster-up


run-inference: ## Run batch inference.
	det experiment create ./determined_config.yaml .


up-fastchat: ## Start FastChat API Server.
	docker compose --project-name llm up -d


up-rag: ## Run RAG-System app.
	docker run -d --rm --name rag-system -p 8501:8501 \
		--gpus all --shm-size=32g \
		--net rag-system \
		-v /home/hpe01/.cache:/root/.cache \
		-v $(shell pwd):/app \
		rag-system:latest \
		streamlit run rag_system.py


chat: ## Run chat.
	python3 -m fastchat.serve.cli --num-gpus 1 --model-path cyberagent/calm2-7b-chat



down: down-rag down-fastchat down-determined ## Stop all containers.

down-rag: ## Stop RAG-System.
	docker stop rag-system || :

down-fastchat: ## Stop FastChat API Server.
	docker compose --project-name llm down || :

down-determined: ## Stop Determined cluster.
	det deploy local cluster-down || :


help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
