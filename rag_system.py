import math
import os
import sqlite3
import time

import faiss
import numpy as np
import openai
import streamlit as st
import torch
from langchain.llms import OpenAI
from transformers import AutoModel, AutoTokenizer

################################################################################
# Initialize StreamLit
################################################################################
st.set_page_config(page_title="😏 Quickstart App")

st.title("😏 Quickstart App")
st.write("Hello world!")

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai_api_key = "dummy"
openai_api_base = "http://fastchat-api-server:8000/v1"

openai.api_key = openai_api_key
openai.api_base = openai_api_base

# st.info(openai.Model.list())


################################################################################
# Initialize Index DB
################################################################################
db_dir = "/app/rag-system/db"
embedding_db_path = os.path.join(db_dir, "embedding.index")
document_db_path = os.path.join(db_dir, "document.db")

index = faiss.read_index(embedding_db_path)

conn = sqlite3.connect(document_db_path)
cursor = conn.cursor()


################################################################################
# Initialize Embedding Model
################################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-japanese-large")
model = AutoModel.from_pretrained("studio-ousia/luke-japanese-large", output_hidden_states=True)
model = model.to(device)
model.eval()


################################################################################
# Functions
################################################################################


def embedding(input_text: str) -> np.ndarray:
    with torch.no_grad():
        tokenized_text = tokenizer.encode(input_text)
        tokenized_text = torch.tensor(tokenized_text).to(device)
        tokenized_text = tokenized_text.unsqueeze(0)
        output = model(tokenized_text)
        output = torch.mean(output["hidden_states"][-1], dim=1)
        output = output.cpu().detach().numpy()

    return output


def search_index(embedded_text: np.ndarray) -> str:
    distances, indices = index.search(embedded_text, k=1)

    cursor.execute("SELECT document FROM documents WHERE id = ?", indices[0])
    document = cursor.fetchone()

    st.write("Reference Document")
    st.info(document[0])

    return document[0]


def generate_response(input_text: str):
    model = "vicuna-13b-v1.5"
    llm = OpenAI(openai_api_key=openai_api_key, openai_api_base=openai_api_base, model=model, batch_size=1)

    st.write("LLM Response")
    st.info(llm(input_text))


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since):
    now = time.time()
    s = now - since
    return f"Run time: {as_minutes(s)}"


with st.form("my_form"):
    text = st.text_area("Enter text:", "ツイッターが最近行った調査について教えてください。")
    submitted = st.form_submit_button("Submit")

    if submitted:
        start = time.time()
        embedded_text = embedding(text)
        document = search_index(embedded_text)

        prompt = f"### Context:\n{document}\n\n### Human:\n{text}\n\n### Assistant:\n"

        generate_response(prompt)

        st.write(time_since(start))
