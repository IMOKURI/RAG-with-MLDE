import logging
import time
from PIL import Image

import streamlit as st

from utils import time_since
from document_summary_index import CustomDocumentSummaryIndex


def main():
    logging.basicConfig(level=logging.INFO)

    if "llm_response" not in st.session_state:
        st.session_state["llm_response"] = None
        st.session_state["llm_response_time"] = ""
    if "rag_response" not in st.session_state:
        st.session_state["rag_response"] = None
        st.session_state["rag_response_time"] = ""

    if "architectures" not in st.session_state:
        image_pretreined = Image.open("./images/pre-trained.drawio.png")
        image_rag = Image.open("./images/rag.drawio.png")

        st.session_state["architectures"] = {
            "pre-trained": image_pretreined,
            "rag": image_rag,
        }

    st.set_page_config(page_title="😏 検索補完生成デモ", layout="wide")

    st.title("😏 検索補完生成デモ")
    st.write(
        "\n\n"
        "このアプリケーションは、ユーザーの質問に対し、"
        "あらかじめ取り込まれた記事の情報を元に回答する"
        "大規模言語モデルの検索補完生成(RAG)のデモンストレーションです。"
        "\n"
        "以下のような質問を入力してみてください。"
        "\n\n"
        "- HPE Swarm Learningを構成するコンポーネントについて教えてください。\n"
        "- HPEの障害者雇用の取り組みに関して、最近受賞した賞について教えてください。\n"
        "\n"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.header("LLMに直接質問した場合")
        with st.form("llm"):
            text = st.text_area("Enter text:", "HPE Swarm Learningを構成するコンポーネントについて教えてください。")
            submitted_1 = st.form_submit_button("Submit")

            if submitted_1:
                with st.spinner(text="In progress..."):
                    llm_query(text)

            if st.session_state["llm_response"] is not None:
                st.write("LLM Response")
                st.info(st.session_state["llm_response"])
                st.write(st.session_state["llm_response_time"])

        st.image(st.session_state["architectures"]["pre-trained"], caption="LLM")

    with col2:
        st.header("RAGの仕組みで付加情報を取得した場合")
        with st.form("rag"):
            text = st.text_area("Enter text:", "HPE Swarm Learningを構成するコンポーネントについて教えてください。")
            submitted_2 = st.form_submit_button("Submit")

            if submitted_2:
                with st.spinner(text="In progress..."):
                    rag_query(text)

            if st.session_state["rag_response"] is not None:
                st.write("RAG Response")
                st.info(st.session_state["rag_response"])
                st.write(st.session_state["rag_response_time"])

        st.image(st.session_state["architectures"]["rag"], caption="RAG (Retrieval Augmented Generation)")

        st.write(
            "RAG では、回答を生成する際に以下の情報を参照しています。"
            "\n\n"
            "- [HPE Swarm Learning とは](https://imokuri-com.pages.dev/blog/2022/06/hpe-swarm-learning-intro/)\n"
            "- [HPE、東京都 障害者雇用エクセレントカンパニー賞を受賞](https://prtimes.jp/main/html/rd/p/000000127.000045092.html)\n"
            "\n"
        )


def llm_query(text):
    start = time.time()

    document_summary_index = CustomDocumentSummaryIndex(openai_api_base="http://fastchat-api-server:8000/v1")

    st.session_state["llm_response"] = document_summary_index.llm.complete(text)
    st.session_state["llm_response_time"] = time_since(start)

    logging.info(f"Returned LLM response ... {time_since(start)}")


def rag_query(text):
    start = time.time()

    document_summary_index = CustomDocumentSummaryIndex(openai_api_base="http://fastchat-api-server:8000/v1")
    document_summary_index.load("/app/rag-system/worker_0_batch_0")
    document_summary_index.as_retriever()

    st.session_state["rag_response"] = document_summary_index.query(text)
    st.session_state["rag_response_time"] = time_since(start)

    logging.info(f"Returned RAG response ... {time_since(start)}")


if __name__ == "__main__":
    main()
