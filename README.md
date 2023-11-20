# LLM-RAG-with-MLDE

LLM RAG System with MLDE


## Architecture

![RAG System Architecture](./images/architecture.drawio.png)

### Index Generation

![Index Generation](./images/index-generation.drawio.png)

### Retrieval Index

![Index Retrieval](./images/retrieval-document-summary-index.drawio.png)


## Showcase

![Screenshot](./images/showcase.png)


## Prerequisite

``` bash
docker create network rag-system
```


## How to Run

### Build container images

``` bash
make build
```

### Start LLM

``` bash
make up-fastchat-controller
make up-fastchat-model-worker
make up-fastchat-api-server
```

### Start determined cluster

``` bash
make up-determined
```

### Create Embedding DB

``` bash
make run-inference
```

### Start RAG System

``` bash
make up-rag
```
