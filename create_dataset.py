import logging
import os
import re

import openai
from llama_index import GPTListIndex, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from torch.utils.data import Dataset
from transformers import AutoTokenizer

openai_api_key = "dummy"
openai_api_base = "http://fastchat-api-server:8000/v1"

openai.api_key = openai_api_key
openai.api_base = openai_api_base


class DocumentDataset(Dataset):
    def __init__(self, doc_base_dir: str = "/determined/docs", load_ds: bool = True):
        self.doc_base_dir = doc_base_dir
        self.docs = []

        self.tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-japanese-large")

        if load_ds:
            self.load_dataset()

    def filename_to_metadata(self, filename: str) -> dict:
        "ディレクトリ名をカテゴリ名のメタデータにする"
        dirs = re.sub(r".*/determined/(.*)/.*\.rst", r"\1", filename).split("/")
        dirs.remove("docs")
        return {"categories": dirs, "filename": os.path.basename(filename)}

    def tokenize(self, text: str):
        return self.tokenizer(text)["input_ids"]

    def load_dataset(self):
        documents = SimpleDirectoryReader(
            self.doc_base_dir,
            recursive=True,
            required_exts=[".rst"],
            file_metadata=self.filename_to_metadata,
        ).load_data()

        text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=384,  # 1024
            chunk_overlap=20,  # 20
            tokenizer=self.tokenize,
        )
        node_parser = SimpleNodeParser(text_splitter=text_splitter)

        service_context = ServiceContext.from_defaults(node_parser=node_parser)

        list_index = GPTListIndex.from_documents(documents, service_context=service_context)

        exclude_docs = ["_index.rst"]
        docs = []
        for n, node in enumerate(list_index.storage_context.docstore.docs.values()):
            _node_dict = node.__dict__
            if _node_dict["metadata"]["filename"] in exclude_docs:
                continue
            node_dict = {}
            node_dict["seq_id"] = str(n)
            node_dict["id_"] = _node_dict["id_"]
            node_dict["text"] = _node_dict["text"]
            node_dict["categories"] = ",".join(_node_dict["metadata"]["categories"])
            node_dict["filename"] = _node_dict["metadata"]["filename"]
            # logging.info(f"{node_dict['seq_id']}: {len(self.tokenize(node_dict['text']))} {node_dict['filename']}")
            docs.append(node_dict)

        logging.info(f"Number of docs: {len(docs)}")

        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]


def main():
    logging.basicConfig(level=logging.INFO)

    DocumentDataset("/home/sugiyama/github/determined/docs")


if __name__ == "__main__":
    main()
