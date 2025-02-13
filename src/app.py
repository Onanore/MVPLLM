import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from database import *
from queries import query_with_rag_fusion, query_rag
from embedding import get_embedding

st.set_page_config(page_title="Ollama Chatbot", layout="wide")

st.title("🤖 Local Chatbot with Ollama")
st.markdown("Upload the file and ask questions based on it.")

if "messages" not in st.session_state:
    st.session_state.messages = []

Ollama_model = st.sidebar.selectbox("Model: ",[ "qwen2.5:1.5b", "deepseek-r1:1.5b"])
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.3)
uploaded_file = st.sidebar.file_uploader("Attach file", type=["txt", "pdf"])
rag_fusion = st.sidebar.checkbox("RAG Fusion")

if(st.sidebar.button("Clear database")):
    clear_database()
    st.success("Database cleared!")

if uploaded_file:
    target_directory = "data"

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    file_path = os.path.join(target_directory, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    documents = load_documents(uploaded_file)
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    st.success("File is uploaded!")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if rag_fusion:
        answer = query_with_rag_fusion(prompt, Ollama_model, temperature)
    else:
        answer = query_rag(prompt, Ollama_model, temperature)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)