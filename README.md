# RAG-with-MLDE
LLM RAG System with MLDE

## Prerequisite

- Start determined cluster

## How to Run

### Create Embedding DB

``` bash
make build
make run
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
