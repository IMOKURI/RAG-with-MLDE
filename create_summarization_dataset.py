"""
参照:
LlamaIndex の DocumentSummaryIndex を試す
https://note.com/npaka/n/n78a1184706d7
"""

import logging
import os
import re

import nest_asyncio
import openai
import tiktoken
from datasets import load_dataset
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from llama_index import (
    GPTListIndex,
    LangchainEmbedding,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    get_response_synthesizer,
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts.base import ChatPromptTemplate
from llama_index.schema import Document
from llama_index.text_splitter import TokenTextSplitter
from torch.utils.data import Dataset
from transformers import AutoTokenizer

embedding_model = "intfloat/multilingual-e5-large"
# embedding_model = "studio-ousia/luke-japanese-large"


openai_api_key = "dummy"
openai_api_base = "http://localhost:8000/v1"

openai.api_key = openai_api_key
openai.api_base = openai_api_base


# QAシステムプロンプト
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "あなたは世界中で信頼されているQAシステムです。\n"
        "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
        "従うべきいくつかのルール:\n"
        "1. 回答内で指定されたコンテキストを直接参照しないでください。\n"
        "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。"
    ),
    role=MessageRole.SYSTEM,
)

# QAプロンプトテンプレートメッセージ
TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "コンテキスト情報は以下のとおりです。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "事前知識ではなくコンテキスト情報を考慮して、クエリに答えます。\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

# チャットQAプロンプト
CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)


# ツリー要約プロンプトメッセージ
TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "複数のソースからのコンテキスト情報を以下に示します。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "予備知識ではなく、複数のソースからの情報を考慮して、質問に答えます。\n"
            "疑問がある場合は、「情報無し」と答えてください。\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

# ツリー要約プロンプト
CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS)


# Summaryクエリ
SUMMARY_QUERY = "提供されたテキストの内容を要約してください。"


def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)


def main():
    logging.basicConfig(level=logging.DEBUG)

    # 非同期処理の有効化
    nest_asyncio.apply()

    dataset = load_dataset("llm-book/livedoor-news-corpus", split="train")

    logging.info("Creating LLM...")
    llm = OpenAI(openai_api_key=openai_api_key, openai_api_base=openai_api_base, batch_size=1)

    # docs = []
    for n, data in enumerate(dataset):
        content = (
            f"タイトル: {data['title']}\n"
            f"カテゴリ: {data['category']}\n"
            f"本文: {data['content']}\n\n"
            "---------------------\n"
            "上記の文章を200字程度で要約してください。\n"
        )

        logging.info(f"n: {n}, title: {data['title']}, summary: {llm(content)}")
        # doc = Document(text=data["content"])
        # doc.doc_id = data["title"]
        # docs.append(doc)
        if n > 2:
            break

    # logging.info("Creating ServiceContext...")
    # service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024)
    #
    # logging.info("Creating ResponseSynthesizer...")
    # response_synthesizer = get_response_synthesizer(
    #     response_mode="tree_summarize",
    #     use_async=True,
    #     text_qa_template=CHAT_TEXT_QA_PROMPT,
    #     summary_template=CHAT_TREE_SUMMARIZE_PROMPT,
    # )

    # logging.info("Creating DocumentSummaryIndex...")
    # doc_summary_index = DocumentSummaryIndex.from_documents(
    #     docs,
    #     service_context=service_context,
    #     response_synthesizer=response_synthesizer,
    #     summary_query=SUMMARY_QUERY,
    #     show_progress=True,
    # )


if __name__ == "__main__":
    main()
