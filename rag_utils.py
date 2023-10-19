import os
import sqlite3

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    def __init__(self, model_name, context=None):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        if context is not None:
            self.device = context.device
            self.model = context.prepare_model_for_inference(self.model)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.model.eval()

        self.index_dim = self.model.pooler.dense.out_features

    def embedding(self, input_text: str) -> np.ndarray:
        with torch.no_grad():
            tokenized_text = self.tokenizer.encode(input_text)
            tokenized_text = torch.tensor(tokenized_text).to(self.device)
            tokenized_text = tokenized_text.unsqueeze(0)
            output = self.model(tokenized_text)
            output = torch.mean(output["hidden_states"][-1], dim=1)
            output = output.cpu().detach().numpy()

        return output

    def inference(self, batch):
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

        return outputs


class IndexDB:
    def __init__(self, path, dim=1024):
        self.path = path

        if os.path.exists(path):
            # index_cpu = faiss.read_index(path)
            # self.index = faiss.index_cpu_to_all_gpus(index_cpu)
            self.index = faiss.read_index(path)
        else:
            # res = faiss.StandardGpuResources()
            # config = faiss.GpuIndexFlatConfig()
            # base_index = faiss.GpuIndexFlatL2(res, dim, config)
            base_index = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIDMap(base_index)

    def update(self, ids, embeddings):
        self.index.add_with_ids(embeddings, ids)

        # self.index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(self.index, self.path)

    def search(self, embedded_text, k):
        distances, indices = self.index.search(embedded_text, k=k)
        indices = indices.tolist()[0]
        return indices


class DocumentDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS documents " "(id TEXT PRIMARY KEY, document TEXT)")

    def update(self, ids, documents):
        self.cursor.executemany(
            "INSERT INTO documents (id,document) VALUES (?,?) "
            "ON CONFLICT (id) DO UPDATE SET "
            "document=EXCLUDED.document",
            [(id, doc) for id, doc in zip(ids, documents)],
        )
        self.conn.commit()

    def search(self, ids):
        self.cursor.execute("SELECT document FROM documents WHERE id IN ({})".format(",".join("?" * len(ids))), ids)
        documents = self.cursor.fetchall()
        return [doc[0] for doc in documents]
