import os
import sqlite3

import faiss
import numpy as np
import openai
import streamlit as st
import torch
from langchain.llms import OpenAI
from transformers import BertModel, BertTokenizer

################################################################################
# Initialize StreamLit
################################################################################
st.set_page_config(page_title="😏 Quickstart App")

st.title("😏 Quickstart App")
st.write("Hello world!")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai_api_base = "http://localhost:8000/v1"

openai.api_key = openai_api_key
openai.api_base = openai_api_base

# st.info(openai.Model.list())


################################################################################
# Initialize Index DB
################################################################################
db_dir = "/home/sugi/work/rag-system/db"
embedding_db_path = os.path.join(db_dir, "embedding.index")
document_db_path = os.path.join(db_dir, "document.db")

index = faiss.read_index(embedding_db_path)

conn = sqlite3.connect(document_db_path)
cursor = conn.cursor()


################################################################################
# Initialize Embedding Model
################################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
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

    st.write("Reference document")
    st.info(document[0])

    return document[0]


def generate_response(input_text: str):
    llm = OpenAI(openai_api_key=openai_api_key, openai_api_base=openai_api_base, model="vicuna-13b-v1.5", batch_size=1)

    st.write("LLM Response")
    st.info(llm(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "最近、人気のゲームを教えてください。")
    submitted = st.form_submit_button("Submit")

    if submitted:
        embedded_text = embedding(text)
        document = search_index(embedded_text)

        prompt = f"### Context:\n{document}\n\n### Human:\n{text}\n\n### Assistant:\n"

        generate_response(prompt)
