import logging
import os
import re

import openai
import tiktoken
from llama_index import GPTListIndex, ServiceContext, SimpleDirectoryReader
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from torch.utils.data import Dataset
from transformers import AutoTokenizer

openai_api_key = "dummy"
openai_api_base = "http://fastchat-api-server:8000/v1"

openai.api_key = openai_api_key
openai.api_base = openai_api_base


def filename_to_metadata(filename: str) -> dict:
    "ディレクトリ名をカテゴリ名のメタデータにする"
    dirs = re.sub(r".*/determined/(.*)/.*\.rst", r"\1", filename).split("/")
    dirs.remove("docs")
    return {"categories": dirs, "filename": os.path.basename(filename)}


def load_dataset(base_dir: str = "/determined/docs"):
    documents = SimpleDirectoryReader(
        base_dir,
        recursive=True,
        required_exts=[".rst"],
        file_metadata=filename_to_metadata,
    ).load_data()

    tokenizer=AutoTokenizer.from_pretrained("studio-ousia/luke-japanese-large")
    # tokenizer = tiktoken.get_encoding("cl100k_base").encode

    text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=512,  # 1024
        chunk_overlap=32,  # 20
        tokenizer=tokenizer,
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
        docs.append(node_dict)

    logging.info(f"Number of docs: {len(docs)}")

    return docs


class DocumentDataset(Dataset):
    def __init__(self, docs):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]


def main():
    docs = load_dataset("/home/sugiyama/github/determined/docs")
    dataset = DocumentDataset(docs)
    print(dataset[0])


if __name__ == "__main__":
    main()
