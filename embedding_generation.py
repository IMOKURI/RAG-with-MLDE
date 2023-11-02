import logging
import os
import pathlib
import shutil

import numpy as np
import torch
from datasets import load_dataset
from determined.pytorch import experimental

from rag_utils import DocumentDB, EmbeddingModel, IndexDB


class EmbeddingProcessor(experimental.TorchBatchProcessor):
    def __init__(self, context):
        self.model_names = [
            # "cl-nagoya/sup-simcse-ja-large",
            # "intfloat/multilingual-e5-large",
            # "pkshatech/GLuCoSE-base-ja",
            "studio-ousia/luke-japanese-large",
        ]

        self.models = [EmbeddingModel(model_name, context) for model_name in self.model_names]

        self.outputs = {model_name: [] for model_name in self.model_names}

        self.last_index = 0
        self.rank = context.get_distributed_rank()

        self.output_dir = "rag-system/processing"
        os.makedirs(self.output_dir, exist_ok=True)

    def process_batch(self, batch, batch_idx) -> None:
        logging.info(f"Processing batch {batch_idx}")

        for model in self.models:
            outputs = model.inference(batch)
            self.outputs[model.model_name].append(
                {
                    "embeddings": outputs,
                    "seq_id": [u.rsplit("/", 2)[1] for u in batch["url"]],
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
        if len(self.outputs) == 0:
            return
        file_name = f"embedding_worker_{self.rank}_end_batch_{self.last_index}"
        file_path = pathlib.Path(self.output_dir, file_name)
        torch.save(self.outputs, file_path)
        self.outputs = {model_name: [] for model_name in self.model_names}

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

            for model in self.models:
                embeddings = []
                documents = []
                ids = []
                num_ids = []

                for file in os.listdir(self.output_dir):
                    file_path = pathlib.Path(self.output_dir, file)
                    batches = torch.load(file_path, map_location="cuda:0")[model.model_name]
                    for batch in batches:
                        embeddings.append(batch["embeddings"].cpu().detach().numpy())
                        ids += batch["seq_id"]
                        num_ids += list(map(int, batch["seq_id"]))
                        documents += batch["text"]

                embeddings = np.concatenate(embeddings)
                num_ids = np.array(num_ids)
                logging.info(f"Enbeddings shape: {embeddings.shape}, Ids shape: {num_ids.shape}")

                model_name = model.model_name.replace("/", "_")

                embedding_db_path = os.path.join(db_dir, f"{model_name}_embedding.index")
                index_db = IndexDB(embedding_db_path, model.index_dim)
                index_db.update(num_ids, embeddings)

                document_db_path = os.path.join(db_dir, f"{model_name}_document.db")
                document_db = DocumentDB(document_db_path)
                document_db.update(ids, documents)

            # Clean-up temporary embedding files
            shutil.rmtree(self.output_dir)


if __name__ == "__main__":
    dataset = load_dataset("llm-book/livedoor-news-corpus", split="train")

    experimental.torch_batch_process(
        EmbeddingProcessor,
        dataset,
        batch_size=32,
        checkpoint_interval=10,
    )
