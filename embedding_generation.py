import logging
import os

from determined.pytorch import experimental
from llama_index import (
    DocumentSummaryIndex,
    OpenAIEmbedding,
    PromptHelper,
    ServiceContext,
    SimpleWebPageReader,
    get_response_synthesizer,
)
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.response_synthesizers import ResponseMode
from llama_index.text_splitter import TokenTextSplitter
import nest_asyncio

import prompt_template as pt
from web_dataset import WebDocument


openai_api_key = "dummy"
openai_api_base = "http://172.19.9.24:8000/v1"


class EmbeddingProcessor(experimental.TorchBatchProcessor):
    def __init__(self, context):
        self.rank = context.get_distributed_rank()

        # 非同期処理の有効化
        nest_asyncio.apply()

        llm = OpenAI(temperature=0, batch_size=1, max_tokens=512, api_key=openai_api_key, api_base=openai_api_base)
        embed_model = OpenAIEmbedding(embed_batch_size=1, api_key=openai_api_key, api_base=openai_api_base)

        text_splitter = TokenTextSplitter(
            separator="。", chunk_size=16384, chunk_overlap=64, backup_separators=["、", " ", "\n"]
        )
        node_parser = SimpleNodeParser(text_splitter=text_splitter)
        prompt_helper = PromptHelper(
            context_window=16384, num_output=512, chunk_overlap_ratio=0.05, chunk_size_limit=None, separator="。"
        )

        self.service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model, node_parser=node_parser, prompt_helper=prompt_helper
        )

        self.response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE,
            service_context=self.service_context,
            use_async=True,
            text_qa_template=pt.CHAT_TEXT_QA_PROMPT,
            summary_template=pt.CHAT_TREE_SUMMARIZE_PROMPT,
        )

        self.outputs = {}
        self.last_index = 0

        self.output_dir = "rag-system"
        os.makedirs(self.output_dir, exist_ok=True)

    def process_batch(self, batch, batch_idx) -> None:
        logging.info(f"Processing batch {batch_idx}")
        # logging.info(f"Processing batch {batch}")

        documents = []
        for doc_id, url in zip(batch["doc_id"], batch["url"]):
            web_doc = SimpleWebPageReader(html_to_text=True).load_data([url])
            web_doc[0].doc_id = doc_id
            documents.extend(web_doc)

        logging.info(f"Loaded {len(documents)} documents ...")

        index = DocumentSummaryIndex.from_documents(
            documents,
            service_context=self.service_context,
            response_synthesizer=self.response_synthesizer,
            show_progress=True,
            summary_query=pt.SUMMARY_QUERY,
        )

        self.outputs[batch_idx] = index
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
            dir_name = f"index_worker_{self.rank}_end_batch_{idx}"
            index.storage_context.persist(os.path.join(self.output_dir, dir_name))

        self.outputs = {}

    def on_finish(self):
        """
        チーフワーカー (rank 0) で各ワーカーがチェックポイントで保存したファイルを読み込み、マージする。
        """
        if self.rank == 0:
            ...


if __name__ == "__main__":
    dataset = WebDocument("./document_list.csv", size=3)

    experimental.torch_batch_process(
        EmbeddingProcessor,
        dataset,
        batch_size=32,
        checkpoint_interval=10,
    )
