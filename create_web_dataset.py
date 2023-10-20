import logging

import openai
from llama_index import PromptHelper, ServiceContext, SimpleWebPageReader, SummaryIndex
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter

openai_api_key = "dummy"
openai_api_base = "http://localhost:8000/v1"

openai.api_key = openai_api_key
openai.api_base = openai_api_base


def main():
    logging.basicConfig(level=logging.DEBUG)

    documents = SimpleWebPageReader(html_to_text=True).load_data(["http://paulgraham.com/worked.html"])
    logging.info("Loaded %d documents", len(documents))

    llm = OpenAI(batch_size=1, max_tokens=256)
    embed_model = OpenAIEmbedding(embed_batch_size=1)

    text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
    node_parser = SimpleNodeParser(text_splitter=text_splitter)

    prompt_helper = PromptHelper(context_window=1024, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None)

    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser, prompt_helper=prompt_helper
    )

    index = SummaryIndex.from_documents(documents, service_context=service_context, show_progress=True)

    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")

    logging.info(response)


if __name__ == "__main__":
    main()
