# RAG-with-MLDE
LLM RAG System with MLDE

## Prerequisite

- Start determined cluster.
- Create python virtual env and install requirements.

## How to Run

### Create Embedding DB

``` bash
make build
make batch-inference
```

### Start LLM

``` bash
make fastchat-controller
make fastchat-model-worker
make fastchat-api-server
```

### Start RAG System

``` bash
make rag-app
```
