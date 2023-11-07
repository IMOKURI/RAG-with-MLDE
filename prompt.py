"""
参照:
LlamaIndex - Web Page Reader
https://gpt-index.readthedocs.io/en/latest/examples/data_connectors/WebPageDemo.html
"""

import logging

import nest_asyncio
import openai
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
from llama_index.text_splitter import TokenTextSplitter
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

import web_dataset as wds


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

# ChoiceSelectプロンプトテンプレート
DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
    "書類の一覧を以下に示します。各文書の横には文書の要約とともに番号が付いています。"
    "質問に答えるために参照する必要がある文書の番号を、関連性の高い順に答えてください。"
    "関連性スコアは、文書が質問に対してどの程度関連していると思われるかに基づいて1～10の数値で表します。\n\n"
    "必ず以下の書式で記述してください。"
    "それ以外の文章は絶対に記述しないでください。\n\n"
    "Document 1:\n<summary of document 1>\n\n"
    "Document 2:\n<summary of document 2>\n\n"
    "...\n\n"
    "Document 10:\n<summary of document 10>\n\n"
    "Question: <question>\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "では、はじめましょう。\n\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    "Answer:\n"
)

# ChoiceSelectプロンプト
DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT)


def main():
    logging.basicConfig(level=logging.INFO)

    # 非同期処理の有効化
    nest_asyncio.apply()

    ds = wds.WebDocument("./document_list.csv")

    logging.info("Loading documents...")
    web_documents = SimpleWebPageReader(html_to_text=True).load_data([d["url"] for d in ds])
    documents = []
    for doc_data, web_doc in zip(ds, web_documents):
        web_doc.doc_id = doc_data["title"]
        documents.append(web_doc)
    logging.info("Loaded %d documents", len(documents))

    llm = OpenAI(temperature=0, batch_size=1, max_tokens=512)
    embed_model = OpenAIEmbedding(embed_batch_size=1)

    text_splitter = TokenTextSplitter(
        separator="。", chunk_size=4096, chunk_overlap=64, backup_separators=["、", " ", "\n"]
    )
    node_parser = SimpleNodeParser(text_splitter=text_splitter)

    prompt_helper = PromptHelper(
        context_window=4096, num_output=512, chunk_overlap_ratio=0.1, chunk_size_limit=None, separator="。"
    )

    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser, prompt_helper=prompt_helper
    )

    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        use_async=True,
        text_qa_template=CHAT_TEXT_QA_PROMPT,
        summary_template=CHAT_TREE_SUMMARIZE_PROMPT,
    )

    logging.info("Building index...")
    index = DocumentSummaryIndex.from_documents(
        documents,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        show_progress=True,
        summary_query=SUMMARY_QUERY,
    )

    # logging.info(index.get_document_summary("swarm learning 概要"))

    query_engine = index.as_query_engine(
        choice_select_prompt=DEFAULT_CHOICE_SELECT_PROMPT,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query("HPE が提供する予定の LLM を as a Service として提供するサービスは何ですか？")
    logging.info(response)


if __name__ == "__main__":
    main()
