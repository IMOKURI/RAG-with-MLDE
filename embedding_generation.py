import logging
import os
import pathlib
import shutil
import sqlite3

import faiss
import numpy as np
import torch
from determined.pytorch import experimental
from transformers import AutoModel, AutoTokenizer

import create_dataset as ds


class EmbeddingProcessor(experimental.TorchBatchProcessor):
    def __init__(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-japanese-large")
        self.model = AutoModel.from_pretrained("studio-ousia/luke-japanese-large", output_hidden_states=True)

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
        logging.info(f"Processing batch {batch_idx}")
        with torch.no_grad():
            tokenized_text = self.tokenizer.batch_encode_plus(
                batch["text"],
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
                    "seq_id": batch["seq_id"],
                    "text": batch["text"],
                    "categories": batch["categories"],
                    "filename": batch["filename"],
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
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS documents "
                "(id TEXT PRIMARY KEY, document TEXT, categories TEXT, filename TEXT)"
            )

            embeddings = []
            documents = []
            ids = []
            num_ids = []
            categories = []
            filenames = []

            for file in os.listdir(self.output_dir):
                file_path = pathlib.Path(self.output_dir, file)
                batches = torch.load(file_path, map_location="cuda:0")
                for batch in batches:
                    embeddings.append(batch["embeddings"].cpu().detach().numpy())
                    ids += batch["seq_id"]
                    num_ids += list(map(int, batch["seq_id"]))
                    documents += batch["text"]
                    categories += batch["categories"]
                    filenames += batch["filename"]

            embeddings = np.concatenate(embeddings)
            num_ids = np.array(num_ids)
            logging.info(f"Enbeddings shape: {embeddings.shape}, Ids shape: {num_ids.shape}")

            index.add_with_ids(embeddings, num_ids)

            # index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index, embedding_db_path)

            cursor.executemany(
                "INSERT INTO documents (id,document,categories,filename) VALUES (?,?,?,?) "
                "ON CONFLICT (id) DO UPDATE SET "
                "document=EXCLUDED.document, categories=EXCLUDED.categories, filename=EXCLUDED.filename",
                [(id, doc, cat, file) for id, doc, cat, file in zip(ids, documents, categories, filenames)],
            )
            conn.commit()

            logging.info(f"Embedding contains {len(ids)} entries")

            # Clean-up temporary embedding files
            shutil.rmtree(self.output_dir)


if __name__ == "__main__":
    dataset = ds.DocumentDataset(ds.load_dataset())

    experimental.torch_batch_process(
        EmbeddingProcessor,
        dataset,
        batch_size=64,
        checkpoint_interval=10,
    )
