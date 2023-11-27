.PHONY: help
.DEFAULT_GOAL := help


build: ## Build container image.
	docker build -t rag-system:latest .


up-determined: ## Start Determined cluster.
	det deploy local cluster-up


run-inference: ## Run batch inference.
	det experiment create ./determined_config.yaml .


up-fastchat-controller: ## Start FastChat controller.
	docker run -d --rm --name fastchat-controller -p 20000:20000 \
		--net rag-system \
		-e FASTCHAT_WORKER_API_TIMEOUT=300 \
		rag-system:latest \
		python3 -m fastchat.serve.controller --host 0.0.0.0 --port 20000

up-fastchat-model-worker: ## Start FastChat model worker.
	docker run -d --rm --name fastchat-model-worker \
		--gpus all --shm-size=32g \
		--net rag-system \
		-e FASTCHAT_WORKER_API_TIMEOUT=300 \
		-v /home/hpe01/.cache:/root/.cache \
		rag-system:latest \
		python3 -m fastchat.serve.model_worker \
		--model-names gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002 \
		--model-path lmsys/vicuna-7b-v1.5 \
		--worker-address http://fastchat-model-worker:21000 \
		--controller-address http://fastchat-controller:20000 \
		--host 0.0.0.0 \
		--port 21000 \
		--num-gpus 1

up-fastchat-model-worker-13b: ## Start FastChat model worker. (13B)
	docker run -d --rm --name fastchat-model-worker \
		--gpus all --shm-size=32g \
		--net rag-system \
		-e FASTCHAT_WORKER_API_TIMEOUT=300 \
		-v /home/hpe01/.cache:/root/.cache \
		rag-system:latest \
		python3 -m fastchat.serve.model_worker \
		--model-names gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002 \
		--model-path lmsys/vicuna-13b-v1.5 \
		--worker-address http://fastchat-model-worker:21000 \
		--controller-address http://fastchat-controller:20000 \
		--host 0.0.0.0 \
		--port 21000 \
		--num-gpus 1

up-fastchat-api-server: ## Start FastChat API server.
	docker run -d --rm --name fastchat-api-server -p 8000:8000 \
		--net rag-system \
		-e FASTCHAT_WORKER_API_TIMEOUT=300 \
		rag-system:latest \
		python3 -m fastchat.serve.openai_api_server \
		--controller-address http://fastchat-controller:20000 \
		--host 0.0.0.0 \
		--port 8000


up-rag: ## Run RAG-System app.
	docker run -d --rm --name rag-system -p 8501:8501 \
		--gpus all --shm-size=32g \
		--net rag-system \
		-e USE_LLM=True \
		-v /home/hpe01/.cache:/root/.cache \
		-v $(shell pwd):/app \
		rag-system:latest \
		streamlit run rag_system.py


down: down-rag down-fastchat down-determined ## Stop all containers.

down-rag: ## Stop RAG-System.
	docker stop rag-system || :

down-fastchat: ## Stop FastChat API Server.
	docker stop fastchat-api-server || :
	docker stop fastchat-model-worker || :
	docker stop fastchat-controller || :

down-determined: ## Stop Determined cluster.
	det deploy local cluster-down || :


chat: ## Run chat.
	python3 -m fastchat.serve.cli --num-gpus 1 --model-path lmsys/vicuna-7b-v1.5


release-gpu: ## Release GPU memory.
	kill $(shell lsof -t /dev/nvidia*)


help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
