import logging
import time

import streamlit as st

from utils import time_since
from document_summary_index import CustomDocumentSummaryIndex


def main():
    logging.basicConfig(level=logging.INFO)

    st.set_page_config(page_title="😏 検索補完生成デモ", layout="wide")

    st.title("😏 検索補完生成デモ")
    st.write(
        "\n"
        "このアプリケーションは、ユーザーの質問に対し、"
        "あらかじめ取り込まれた記事の情報を元に回答する"
        "大規模言語モデルの検索補完生成(RAG)のデモンストレーションです。"
    )

    with st.form("rag"):
        text = st.text_area("Enter text:", "HPE Swarm Learning について教えてください。")
        submitted = st.form_submit_button("Submit")

        if submitted:
            with st.spinner(text="In progress..."):
                query(text)


def query(text):
    start = time.time()

    document_summary_index = CustomDocumentSummaryIndex(openai_api_base="http://fastchat-api-server:8000/v1")
    document_summary_index.load("/app/rag-system/worker_0_batch_0")
    document_summary_index.as_retriever()

    response = document_summary_index.query(text)

    st.write("LLM Response")
    st.info(response)

    st.write(time_since(start))
    logging.info(f"Returned LLM response ... {time_since(start)}")


if __name__ == "__main__":
    main()
