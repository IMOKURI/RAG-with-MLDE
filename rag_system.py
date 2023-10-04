import openai
import streamlit as st
from langchain.llms import OpenAI

st.set_page_config(page_title="😏 Quickstart App")

st.title("😏 Quickstart App")
st.write("Hello world!")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai_api_base = "http://localhost:8000/v1"

openai.api_key = openai_api_key
openai.api_base = openai_api_base

# st.info(openai.Model.list())


def generate_response(input_text: str):
    """docstring for generate_response"""
    llm = OpenAI(openai_api_key=openai_api_key, openai_api_base=openai_api_base, model="vicuna-13b-v1.5", batch_size=1)
    st.info(llm(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "What are the three key pieces of advice for learning how to code?")
    submitted = st.form_submit_button("Submit")

    if submitted:
        generate_response(text)
