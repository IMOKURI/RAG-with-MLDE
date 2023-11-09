import argparse
import logging
import time

from llama_index import SimpleWebPageReader
import nest_asyncio

from document_summary_index import CustomDocumentSummaryIndex
from utils import time_since
import web_dataset as wds


openai_api_key = "dummy"
openai_api_base = "http://localhost:8000/v1"


def main():
    start = time.time()
    logging.basicConfig(level=logging.INFO)

    args = get_args()
    logging.debug(args)

    # 非同期処理の有効化
    nest_asyncio.apply()

    document_summary_index = CustomDocumentSummaryIndex()

    if args.build_index:
        ds = wds.WebDocument("./document_list.csv", size=2)

        documents = []
        for doc_data in ds:
            web_doc = SimpleWebPageReader(html_to_text=True).load_data([doc_data["url"]])
            web_doc[0].doc_id = doc_data["doc_id"]
            documents.extend(web_doc)
        logging.info(f"Loaded {len(documents)} documents ...")

        document_summary_index.from_documents(documents)
        document_summary_index.persist()

    document_summary_index.load()
    document_summary_index.as_retriever()

    logging.info(f"Querying document 1 ... {time_since(start)}")
    logging.info(document_summary_index.query("Swarm Learning とは何ですか？"))

    logging.info(f"Querying document 2 ... {time_since(start)}")
    logging.info(document_summary_index.query("HPEの障害者雇用の取り組みについて教えてください。"))

    logging.info(f"Done ... {time_since(start)}")


def get_args():
    parser = argparse.ArgumentParser(description=""" This is great script! """)

    parser.add_argument("-b", "--build-index", action="store_true", help="Use this flag to build index.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
