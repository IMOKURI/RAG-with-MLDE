import logging
import math
import os
import time

import openai
import streamlit as st
from langchain.llms import OpenAI

from rag_utils import DocumentDB, EmbeddingModel, IndexDB


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since):
    now = time.time()
    s = now - since
    return f"Run time: {as_minutes(s)}"


def main():
    logging.basicConfig(level=logging.INFO)

    model_names = [
        # "cl-nagoya/sup-simcse-ja-large",
        # "intfloat/multilingual-e5-large",
        # "pkshatech/GLuCoSE-base-ja",
        "studio-ousia/luke-japanese-large",
    ]

    db_dir = "/app/rag-system/db"

    st.set_page_config(page_title="😏 Quickstart App")

    st.title("😏 Quickstart App")
    st.write("Hello world!")

    # openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    openai_api_key = "dummy"
    openai_api_base = "http://fastchat-api-server:8000/v1"

    openai.api_key = openai_api_key
    openai.api_base = openai_api_base

    with st.form("rag_form"):
        text = st.text_area("Enter text:", "ツイッターが最近行った調査について教えてください。")
        submitted = st.form_submit_button("Submit")

        if submitted:
            start = time.time()

            models = []
            index_dbs = []
            document_dbs = []

            for model_name in model_names:
                models.append(EmbeddingModel(model_name))

                model_name = model_name.replace("/", "_")

                embedding_db_path = os.path.join(db_dir, f"{model_name}_embedding.index")
                index_dbs.append(IndexDB(embedding_db_path))

                document_db_path = os.path.join(db_dir, f"{model_name}_document.db")
                document_dbs.append(DocumentDB(document_db_path))
                logging.info(f"Loaded {model_name} ... {time_since(start)}")

            llm = OpenAI(openai_api_key=openai_api_key, openai_api_base=openai_api_base, batch_size=1)
            logging.info(f"Loaded LLM ... {time_since(start)}")

            documents = []
            for model, index_db, document_db in zip(models, index_dbs, document_dbs):
                embedded_text = model.embedding(text)

                indeices = index_db.search(embedded_text, 1)
                documents += document_db.search(indeices)

            prompt = (
                "あなたは世界中で信頼されているQAシステムです。\n"
                "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
                "従うべきいくつかのルール:\n"
                "1. 回答内で指定されたコンテキストを直接参照しないでください。\n"
                "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、"
                "またはそれに類するような記述は避けてください。\n"
                "3. 200文字程度で回答してください。\n"
                "コンテキスト情報は以下のとおりです。\n"
                "---------------------\n" + "\n---------------------\n".join(documents) + "\n---------------------\n"
                "事前知識ではなくコンテキスト情報を考慮して、クエリに答えます。\n"
                "疑問がある場合は、「情報無し」と答えてください。\n"
                f"Query: {text}\n"
                "Answer: "
            )
            st.write("Reference document")
            st.info(documents[0])
            logging.info(f"Searched index ... {time_since(start)}")

            st.write("LLM Response")
            st.info(llm(prompt))

            st.write(time_since(start))
            logging.info(f"Returned answer ... {time_since(start)}")


if __name__ == "__main__":
    main()
