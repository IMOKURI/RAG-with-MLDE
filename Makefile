.PHONY: help
.DEFAULT_GOAL := help

NOW = $(shell date '+%Y%m%d-%H%M%S')
IMAGE_TAG = latest


build: ## Build container image.
	docker build --build-arg PROXY=$(http_proxy) -t baseimage:$(IMAGE_TAG) -f Dockerfile.baseimage .
	docker build --build-arg PROXY=$(http_proxy) -t determined:$(IMAGE_TAG) -f Dockerfile.determined .
	docker build --build-arg PROXY=$(http_proxy) -t fastchat:$(IMAGE_TAG) -f Dockerfile.fastchat .
	docker build --build-arg PROXY=$(http_proxy) -t streamlit:$(IMAGE_TAG) -f Dockerfile.streamlit .


up-determined: ## Start Determined cluster.
	det deploy local cluster-up


batch-inference: ## Run batch inference.
	det experiment create ./determined_config.yaml .


fastchat: ## Start FastChat API Server.
	docker compose --project-name llm1 --env-file fastchat1.env up -d


rag-app: ## Run RAG-System app.
	docker run -d --rm --name rag-system -p 8501:8501 \
		--gpus '"device=6,7"' --shm-size=32g \
		--net llm1_default \
		-e no_proxy="fastchat-controller,fastchat-llm-worker,fastchat-api-server,localhost,127.0.0.1,ponkots01,16.171.32.68,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16" \
		-v /data/home/sugiyama/.cache:/root/.cache \
		-v /data/home/sugiyama/rag-system:/app/rag-system \
		streamlit:$(IMAGE_TAG) \
		streamlit run rag_system.py


down: down-rag down-fastchat ## Stop all containers.

down-rag: ## Stop RAG-System.
	docker stop rag-system

down-fastchat: ## Stop FastChat API Server.
	docker compose --project-name llm1 --env-file fastchat1.env down


help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
