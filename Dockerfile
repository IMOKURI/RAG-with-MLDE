FROM huggingface/transformers-pytorch-deepspeed-latest-gpu:latest

RUN apt-get update && \
    apt-get install -y daemontools

RUN pip install --no-cache -U \
    accelerate \
    auto-gptq \
    bitsandbytes \
    datasets \
    determined \
    faiss-gpu \
    optimum \
    peft \
    protobuf==3.20.* \
    tensorboard-plugin-profile \
    tokenizers

