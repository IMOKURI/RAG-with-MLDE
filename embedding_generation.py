import logging
import os
import pathlib
import shutil
import sqlite3

import faiss
import numpy as np
import torch
from datasets import load_dataset
from determined.pytorch import experimental
from transformers import BertModel, BertTokenizer


class EmbeddingProcessor(experimental.TorchBatchProcessor):
    def __init__(self, context):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

        self.device = context.device

        self.model = context.prepare_model_for_inference(self.model)
        self.index_dim = self.model.pooler.dense.out_features

        self.output = []

        self.context = context
        self.last_index = 0
        self.rank = self.context.get_distributed_rank()

        self.output_dir = "rag-system/processing"
        os.makedirs(self.output_dir, exist_ok=True)

    def process_batch(self, batch, batch_idx) -> None:
        with torch.no_grad():
            tokenized_text = self.tokenizer.batch_encode_plus(
                batch["content"],
                truncation=True,
                padding="max_length",
                max_length=512,
                add_special_tokens=True,
            )
            inputs = torch.tensor(tokenized_text["input_ids"])
            inputs = inputs.to(self.device)
            masks = torch.tensor(tokenized_text["attention_mask"])
            masks = masks.to(self.device)

            outputs = self.model(inputs, masks)

            # To create an embedding vector for each document,
            # 1. we take the hidden states from the last layer (output["hidden_states"][-1]),
            #    which is a tensor of (#examples, #tokens, #hidden_states) size.
            # 2. we calculate the average across the token-dimension, resulting in a tensor of
            #    (#examples, #hidden_states) size.
            outputs = torch.mean(outputs["hidden_states"][-1], dim=1)

            self.output.append(
                {
                    "embeddings": outputs,
                    "id": [int(u.rsplit("/", 2)[1]) for u in batch["url"]],
                    "text": batch["content"],
                }
            )
            self.last_index = batch_idx

    def on_checkpoint_start(self):
        """
        In this function, each worker persists the in-memory embeddings to the file system of the agent machine.
           - Note that our set-up is for demonstration purpose only. Production use cases should save to a
             shared file system directory bind-mounted to all agent machines and experiment containers.
        File names contain rank and batch index information to avoid duplication between:
        - files created by different workers
        - files created by the same worker for different batches of input data
        """
        if len(self.output) == 0:
            return
        file_name = f"embedding_worker_{self.rank}_end_batch_{self.last_index}"
        file_path = pathlib.Path(self.output_dir, file_name)
        torch.save(self.output, file_path)
        self.output = []

    def on_finish(self):
        """
        In this function, the chief worker (rank 0):
        - initializes a Chroma client and creates a Chroma collection. The collection is persisted in the
          directory "/tmp/chroma" of the container. The "/tmp" directory in the container is a bind-mount of the
          "/tmp" directory on the agent machine (see distributed.yaml file).
          - Note that our set-up is for demonstration purpose only. Production use cases should use a
            shared file system directory bind-mounted to all agent machines and experiment containers.
        - reads in and insert embedding files generated from all workers to the Chroma collection
        """
        if self.rank == 0:
            db_dir = "rag-system/db"
            os.makedirs(db_dir, exist_ok=True)
            embedding_db_path = os.path.join(db_dir, "embedding.index")
            document_db_path = os.path.join(db_dir, "document.db")

            if os.path.exists(embedding_db_path):
                # index_cpu = faiss.read_index(embedding_db_path)
                # index = faiss.index_cpu_to_all_gpus(index_cpu)
                index = faiss.read_index(embedding_db_path)
            else:
                # res = faiss.StandardGpuResources()
                # config = faiss.GpuIndexFlatConfig()
                # base_index = faiss.GpuIndexFlatL2(res, self.index_dim, config)
                base_index = faiss.IndexFlatL2(self.index_dim)
                index = faiss.IndexIDMap(base_index)

            conn = sqlite3.connect(document_db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, document TEXT)")

            embeddings = []
            documents = []
            ids = []

            for file in os.listdir(self.output_dir):
                file_path = pathlib.Path(self.output_dir, file)
                batches = torch.load(file_path, map_location="cuda:0")
                for batch in batches:
                    embeddings.append(batch["embeddings"].cpu().numpy())
                    ids += batch["id"]
                    documents += batch["text"]

            embeddings = np.concatenate(embeddings)
            ids = np.array(ids)
            print(f"Enbeddings shape: {embeddings.shape}, Ids shape: {ids.shape}")
            print(f"Ids: {ids}")

            index.add_with_ids(embeddings, ids)

            # index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index, embedding_db_path)

            cursor.executemany(
                "INSERT INTO documents (id,document) VALUES (?,?) "
                + "ON CONFLICT (id) DO UPDATE SET document=EXCLUDED.document",
                [(id, doc) for id, doc in zip(ids, documents)],
            )
            conn.commit()

            logging.info(f"Embedding contains {len(ids)} entries")

            # Clean-up temporary embedding files
            shutil.rmtree(self.output_dir)


if __name__ == "__main__":
    dataset = load_dataset("llm-book/livedoor-news-corpus", split="train")
    # Persisting embeddings can take quite a while on Chroma
    # Adding a limit on dataset size to ensure the example finishes sooner
    dataset = dataset.select(list(range(100)))
    experimental.torch_batch_process(
        EmbeddingProcessor,
        dataset,
        batch_size=64,
        checkpoint_interval=10,
    )
