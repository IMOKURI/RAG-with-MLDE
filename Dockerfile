FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y \
    git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg daemontools
RUN python3 -m pip install --no-cache-dir --upgrade pip

# PyTorch
RUN python3 -m pip install --no-cache-dir --upgrade torch torchvision torchaudio

# Huggingface
RUN python3 -m pip install --no-cache-dir --upgrade transformers accelerate datasets sentencepiece

# LLM
RUN python3 -m pip install --no-cache-dir --upgrade openai langchain llama-index html2text

# Index DB
RUN python3 -m pip install --no-cache-dir --upgrade faiss-gpu

# Determined AI
RUN python3 -m pip install --no-cache-dir --upgrade determined tensorboard

# FastChat
RUN python3 -m pip install --no-cache-dir --upgrade fschat[model_worker]

# Streamlit
RUN python3 -m pip install --no-cache-dir --upgrade streamlit

WORKDIR /app
