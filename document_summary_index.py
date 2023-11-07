"""
参照:
LlamaIndex - Web Page Reader
https://gpt-index.readthedocs.io/en/latest/examples/data_connectors/WebPageDemo.html
"""

import argparse
import logging

import nest_asyncio
import openai
from llama_index import (
    DocumentSummaryIndex,
    OpenAIEmbedding,
    PromptHelper,
    ServiceContext,
    SimpleWebPageReader,
    StorageContext,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.graph_stores import SimpleGraphStore
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.text_splitter import TokenTextSplitter
from llama_index.vector_stores import SimpleVectorStore

import web_dataset as wds
import prompt_template as pt


openai_api_key = "dummy"
openai_api_base = "http://localhost:8000/v1"

openai.api_key = openai_api_key
openai.api_base = openai_api_base



def main():
    logging.basicConfig(level=logging.INFO)

    args = get_args()
    logging.debug(args)

    # 非同期処理の有効化
    nest_asyncio.apply()

    if args.build_index:
        ds = wds.WebDocument("./document_list.csv", size=3)

        logging.info("Loading documents...")
        documents = []
        for doc_data in ds:
            logging.info(f"Processing document: {doc_data['title']}")
            web_doc = SimpleWebPageReader(html_to_text=True).load_data([doc_data["url"]])
            web_doc[0].doc_id = doc_data["doc_id"]
            documents.extend(web_doc)
        logging.info("Loaded %d documents", len(documents))

        llm = OpenAI(temperature=0, batch_size=1, max_tokens=512)
        embed_model = OpenAIEmbedding(embed_batch_size=1)

        text_splitter = TokenTextSplitter(
            separator="。", chunk_size=16384, chunk_overlap=64, backup_separators=["、", " ", "\n"]
        )
        node_parser = SimpleNodeParser(text_splitter=text_splitter)

        prompt_helper = PromptHelper(
            context_window=16384, num_output=512, chunk_overlap_ratio=0.05, chunk_size_limit=None, separator="。"
        )

        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model, node_parser=node_parser, prompt_helper=prompt_helper
        )

        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=True,
            text_qa_template=pt.CHAT_TEXT_QA_PROMPT,
            summary_template=pt.CHAT_TREE_SUMMARIZE_PROMPT,
        )

        logging.info("Building index...")
        index = DocumentSummaryIndex.from_documents(
            documents,
            service_context=service_context,
            response_synthesizer=response_synthesizer,
            show_progress=True,
            summary_query=pt.SUMMARY_QUERY,
        )

        index.storage_context.persist("./rag-system")

    logging.info("Loading index...")
    # TODO: support multiple indices
    docstore = SimpleDocumentStore.from_persist_path("./rag-system/docstore.json")
    indexstore = SimpleIndexStore.from_persist_path("./rag-system/index_store.json")
    vectorstore = SimpleVectorStore.from_persist_path("./rag-system/vector_store.json")
    graphstore = SimpleGraphStore.from_persist_path("./rag-system/graph_store.json")

    storage_context = StorageContext.from_defaults(docstore, indexstore, vectorstore, graphstore)
    doc_summary_index = load_index_from_storage(storage_context)

    logging.info("Getting document relations...")
    logging.info(doc_summary_index.ref_doc_info)

    logging.info("Getting document summary...")
    logging.info(doc_summary_index.get_document_summary("id-002"))

    logging.info("Querying...")
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        use_async=True,
        text_qa_template=pt.CHAT_TEXT_QA_PROMPT,
        summary_template=pt.CHAT_TREE_SUMMARIZE_PROMPT,
    )
    query_engine = doc_summary_index.as_query_engine(
        choice_select_prompt=pt.DEFAULT_CHOICE_SELECT_PROMPT,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query("Swarm Learning とは何ですか？")
    logging.info(response)


def get_args():
    parser = argparse.ArgumentParser(
        description="""
    This is great script!
    """
    )

    parser.add_argument("--build-index", action="store_true", help="Use this flag to build index.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
