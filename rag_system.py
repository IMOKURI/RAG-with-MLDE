import logging
import os
import time

from PIL import Image
import streamlit as st

from document_summary_index import CustomDocumentSummaryIndex
from showcase import llm_response_demo, rag_response_demo
from utils import time_since


def main():
    logging.basicConfig(level=logging.INFO)

    if "use_llm" not in st.session_state:
        st.session_state["use_llm"] = os.getenv("USE_LLM", "False").lower() in ["t", "true", "1"]

    if "llm_response" not in st.session_state:
        st.session_state["llm_response"] = None
        st.session_state["llm_response_time"] = ""
    if "rag_response" not in st.session_state:
        st.session_state["rag_response"] = None
        st.session_state["rag_response_time"] = ""

    if "architectures" not in st.session_state:
        image_pretreined = Image.open("./images/pre-trained.drawio.png")
        image_rag = Image.open("./images/rag.drawio.png")
        image_architecture = Image.open("./images/architecture.drawio.png")
        image_index_generation = Image.open("./images/index-generation.drawio.png")
        image_retrieval_index = Image.open("./images/retrieval-document-summary-index.drawio.png")

        st.session_state["architectures"] = {
            "pre-trained": image_pretreined,
            "rag": image_rag,
            "architecture": image_architecture,
            "index-generation": image_index_generation,
            "retrieval-index": image_retrieval_index,
        }

    st.set_page_config(page_title="📝 検索補完生成デモ", layout="wide")

    st.title("📝 検索補完生成デモ")
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
        "- ライダーカップにはどのような課題があり、HPEは2023年のライダーカップでどのようなサポートをしましたか？\n"
        "\n"
    )

    if not st.session_state["use_llm"]:
        st.warning("この画面の表示は、LLMを使用していません。応答内容はイメージです。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LLMに直接質問した場合")
        st.markdown(
            """
LLMに直接質問することで、素早く応答を返すことができます。
"""
        )
        if st.session_state["use_llm"]:
            with st.form("llm"):
                text = st.text_area("Enter text:", "HPE Swarm Learningを構成するコンポーネントについて教えてください。")
                submitted_1 = st.form_submit_button("Submit")

                if submitted_1:
                    llm_query(text)
                    st.warning(
                        "LLMは、検索結果を生成する際に、付加情報を参照していないため、"
                        "嘘の情報が表示されている可能性があります。 (ハルシネーション)"
                    )
                    st.write(st.session_state["llm_response_time"])

                elif st.session_state["llm_response"] is not None:
                    st.info(st.session_state["llm_response"])
                    st.warning(
                        "LLMは、検索結果を生成する際に、付加情報を参照していないため、"
                        "嘘の情報が表示されている可能性があります。 (ハルシネーション)"
                    )
                    st.write(st.session_state["llm_response_time"])

        else:
            text = st.selectbox(
                "Select text:",
                [
                    "HPE Swarm Learningを構成するコンポーネントについて教えてください。",
                    "HPEの障害者雇用の取り組みに関して、最近受賞した賞について教えてください。",
                    "ライダーカップにはどのような課題があり、HPEは2023年のライダーカップでどのようなサポートをしましたか？",
                ],
                key="llm",
                index=None,
            )
            st.info(llm_response_demo(text))
            st.warning(
                "LLMは、検索結果を生成する際に、付加情報を参照していないため、"
                "嘘の情報が表示されている可能性があります。 (ハルシネーション)"
            )

        st.image(st.session_state["architectures"]["pre-trained"], caption="LLM 概要図")

    with col2:
        st.subheader("RAGの仕組みで付加情報を取得した場合")
        st.markdown(
            """
RAG (Retrieval Augmented Generation) とは、ユーザーからの質問に答えるために必要な文章を検索し、LLMの入力に追加する手法です。

付加情報を利用することで、より正確な回答を生成することができます。
"""
        )
        if st.session_state["use_llm"]:
            with st.form("rag"):
                text = st.text_area("Enter text:", "HPE Swarm Learningを構成するコンポーネントについて教えてください。")
                submitted_2 = st.form_submit_button("Submit")

                if submitted_2:
                    rag_query(text)
                    st.write(st.session_state["rag_response_time"])

                elif st.session_state["rag_response"] is not None:
                    st.info(st.session_state["rag_response"])
                    st.write(st.session_state["rag_response_time"])

        else:
            text = st.selectbox(
                "Select text:",
                [
                    "HPE Swarm Learningを構成するコンポーネントについて教えてください。",
                    "HPEの障害者雇用の取り組みに関して、最近受賞した賞について教えてください。",
                    "ライダーカップにはどのような課題があり、HPEは2023年のライダーカップでどのようなサポートをしましたか？",
                ],
                key="rag",
                index=None,
            )
            st.info(rag_response_demo(text))

        st.image(st.session_state["architectures"]["rag"], caption="RAG (Retrieval Augmented Generation) 概要図")

        st.write(
            "このデモでは、回答を生成する際に以下の情報を参照しています。"
            "\n\n"
            "- [HPE Swarm Learning とは](https://imokuri.com/blog/2022/06/hpe-swarm-learning-intro/)\n"
            "- [HPE、東京都 障害者雇用エクセレントカンパニー賞を受賞](https://prtimes.jp/main/html/rd/p/000000127.000045092.html)\n"
            "- [HPE、革新的なプライベート5GとWi-Fiの統合ネットワークを2023 ライダーカップ会場に導入](https://prtimes.jp/main/html/rd/p/000000126.000045092.html)\n"
            "\n"
        )

    st.subheader("アーキテクチャ紹介")

    st.markdown(
        """
このデモのアーキテクチャをご紹介します。
"""
    )

    st.image(st.session_state["architectures"]["architecture"], caption="デモのアーキテクチャ")

    st.markdown(
        """
#### Index 作成

付加情報として利用したいデータセットを、MLDE (Machine Learning Developement Environment) の バッチ推論 で、
Document Summary Index に登録します。

Document Summary Index は、チャンクに分割した文章の要約を保持しているインデックスです。
"""
    )

    st.image(st.session_state["architectures"]["index-generation"], caption="Index 作成")

    st.markdown(
        """
#### ユーザーの質問に対する回答の生成

ユーザーの質問にマッチする要約文を探します。

その要約文の元文章を、LLM の入力に追加して、回答を生成します。
"""
    )

    st.image(st.session_state["architectures"]["retrieval-index"], caption="ユーザーの質問に対する回答の生成")

    st.markdown(
        """
---
##### 参考: 検証機のスペック

- HW: ProLiant DL380 Gen11
- CPU: Intel(R) Xeon(R) Gold 6416H x2 (2P36C)
- Memory: 256GB
- GPU: NVIDIA H100 PCIe 80GB x1

##### 参考: 検証機のリソース利用状況

- CPU Memory: 約 5GB
- GPU Memory: 約 16GB (RAGシステム稼働時)
- GPU Memory: 約 28GB (Index作成時)

##### 利用しているLLM

- [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)

##### 利用しているライブラリ

- [HPE Machine Learning Developement Environment (Determined AI)](https://hpe-mlde.determined.ai/latest/)
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/)
- [FastChat](https://github.com/lm-sys/FastChat)
- [Streamlit](https://streamlit.io/)
"""
    )


def llm_query(text):
    start = time.time()

    with st.spinner(text="回答生成中 ..."):
        document_summary_index = CustomDocumentSummaryIndex(openai_api_base="http://fastchat-api-server:8000/v1")

        response = document_summary_index.llm.stream_complete(text)
        result_area = st.empty()
        result = ""

        for item in response:
            result = item.text
            result_area.info(result)

    st.session_state["llm_response"] = result
    st.session_state["llm_response_time"] = time_since(start)

    logging.info(f"Returned LLM response ... {time_since(start)}")


def rag_query(text):
    start = time.time()

    with st.spinner(text="付加情報 検索中 ..."):
        document_summary_index = CustomDocumentSummaryIndex(openai_api_base="http://fastchat-api-server:8000/v1")
        document_summary_index.load("/app/rag-system/worker_0_batch_0")
        document_summary_index.as_retriever()

        response = document_summary_index.query(text)

    logging.info(f"Searched addtional infomation ... {time_since(start)}")

    with st.spinner(text="回答生成中 ..."):
        result_area = st.empty()
        result = ""

        for item in response.response_gen:
            result += item
            result_area.info(result)

    st.session_state["rag_response"] = result
    st.session_state["rag_response_time"] = time_since(start)

    logging.info(f"Returned RAG response ... {time_since(start)}")


if __name__ == "__main__":
    main()
