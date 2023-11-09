import logging
import os

from determined.pytorch import experimental
from llama_index import SimpleWebPageReader
import nest_asyncio

from document_summary_index import CustomDocumentSummaryIndex
from web_dataset import WebDocument


class EmbeddingProcessor(experimental.TorchBatchProcessor):
    def __init__(self, context):
        self.rank = context.get_distributed_rank()

        # 非同期処理の有効化
        nest_asyncio.apply()

        self.document_summary_index = CustomDocumentSummaryIndex(openai_api_base="http://172.19.9.24:8000/v1")

        self.outputs = {}
        self.last_index = 0

        self.output_dir = "rag-system"

    def process_batch(self, batch, batch_idx) -> None:
        logging.info(f"Processing batch {batch_idx}")
        # logging.info(f"Processing batch {batch}")

        documents = []
        for doc_id, url in zip(batch["doc_id"], batch["url"]):
            web_doc = SimpleWebPageReader(html_to_text=True).load_data([url])
            web_doc[0].doc_id = doc_id
            documents.extend(web_doc)

        logging.info(f"Loaded {len(documents)} documents ...")

        self.document_summary_index.from_documents(documents)

        dir_name = f"worker_{self.rank}_batch_{batch_idx}"
        self.document_summary_index.persist(os.path.join(self.output_dir, dir_name))

        # self.outputs[batch_idx] = index
        self.last_index = batch_idx

    def on_checkpoint_start(self):
        """
        File names contain rank and batch index information to avoid duplication between:
        - files created by different workers
        - files created by the same worker for different batches of input data
        """
        if len(self.outputs) == 0:
            return

        for idx, index in self.outputs.items():
            ...

        self.outputs = {}

    def on_finish(self):
        """
        チーフワーカー (rank 0) で各ワーカーがチェックポイントで保存したファイルを読み込み、マージする。
        """
        if self.rank == 0:
            ...


if __name__ == "__main__":
    dataset = WebDocument("./document_list.csv", size=5)

    experimental.torch_batch_process(
        EmbeddingProcessor,
        dataset,
        batch_size=32,
        checkpoint_interval=10,
    )
