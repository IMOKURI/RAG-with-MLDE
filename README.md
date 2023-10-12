# LLM-RAG-with-MLDE

LLM RAG System with MLDE

## Architecture

![RAG System](./images/RAG-System.drawio.png)

## Showcase

![Screenshot](./images/showcase.png)

## Prerequisite

- determined cluster (>= 0.26)

## How to Run

### Build container images

``` bash
make build
```

### Create Embedding DB

``` bash
make batch-inference
```

### Start LLM

``` bash
make fastchat
```

### Start RAG System

``` bash
make rag-app
```
