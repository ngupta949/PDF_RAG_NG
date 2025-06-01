import streamlit as st
import logging
import os
import shutil
import pdfplumber
import ollama
import hashlib
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import HuggingFaceHub
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Ollama PDF RAG Streamlit UI", page_icon="🎈", layout="wide")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_llm(selected_model: str):
    hf_models = ["mistralai/Mistral-7B-Instruct-v0.1"]
    if selected_model in hf_models:
        return HuggingFaceHub(
            repo_id=selected_model,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 512},
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )
    return ChatOllama(model=selected_model, temperature=0.1)

@st.cache_resource(show_spinner=True)
def extract_model_names():
    try:
        res = requests.get("http://localhost:11434/api/tags").json()
        return tuple(m["name"] for m in res.get("models", []) if m.get("name") != "llama2")
    except Exception as e:
        logger.warning(f"Failed to fetch models: {e}")
        return ("mistralai/Mistral-7B-Instruct-v0.1",)

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

def get_pdf_text(pdf_docs):
    return "".join(page.extract_text() or "" for pdf in pdf_docs for page in PdfReader(pdf).pages)

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    try:
        return FAISS.from_texts(text_chunks, embedding=get_embeddings())
    except Exception as e:
        logger.error(f"Vector store error: {e}")
        return None

def compute_hash(files):
    hasher = hashlib.md5()
    for f in files:
        hasher.update(f.getbuffer())
    return hasher.hexdigest()

@st.cache_resource
def process_pdf_to_faiss(files, file_hash):
    raw_text = get_pdf_text(files)
    chunks = get_text_chunks(raw_text)
    return get_vector_store(chunks), extract_all_pages_as_images(files[0])

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    with pdfplumber.open(file_upload) as pdf:
        return [page.to_image().original for page in pdf.pages]

def delete_vector_db():
    for key in ["pdf_pages", "file_upload", "vector_db"]:
        st.session_state.pop(key, None)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    st.success("Vector DB deleted.")
    st.rerun()

def process_question(question: str, vector_db: FAISS, selected_model: str) -> str:
    llm = get_llm(selected_model)
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm,
        prompt=PromptTemplate(input_variables=["question"], template="Original question: {question}")
    )
    template = """Answer the question as detailed as possible from the provided context only.
    Do not guess. If not known, respond with "I don’t know the answer as not sufficient information is provided in the PDF."
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    response = chain.invoke(question).strip()
    return response if response else "I don’t know the answer as not sufficient information is provided in the PDF."

def main():
    st.subheader("🧠 Ollama Chat with PDF RAG -- Nidhi Gupta", divider="gray")

    available_models = extract_model_names()
    col1, col2 = st.columns([1.5, 2])

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_db", None)

    selected_model = col2.selectbox("Pick a model", available_models)
    pdf_docs = col1.file_uploader("Upload PDFs", accept_multiple_files=True)

    col_btns = col1.columns(2)
    with col_btns[0]:
        submit_btn = st.button("Submit & Process")
    with col_btns[1]:
        if st.button("⚠️ Delete collection"):
            delete_vector_db()

    if submit_btn and pdf_docs:
        file_hash = compute_hash(pdf_docs)
        with st.spinner("Processing PDFs..."):
            vector_db, images = process_pdf_to_faiss(pdf_docs, file_hash)
            st.session_state["vector_db"] = vector_db
            st.session_state["pdf_pages"] = images
            st.success("PDF processed.")

    if st.session_state.get("pdf_pages"):
        zoom = col1.slider("Zoom", min_value=100, max_value=1000, value=700, step=50)
        with col1.container(height=410, border=True):
            for img in st.session_state["pdf_pages"]:
                st.image(img, width=zoom)

    with col2:
        chat_box = st.container(height=500, border=True)
        for msg in st.session_state["messages"]:
            avatar = "🤖" if msg["role"] == "assistant" else "😎"
            with chat_box.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about your PDF"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with chat_box.chat_message("user", avatar="😎"):
                st.markdown(prompt)
            with chat_box.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    if st.session_state["vector_db"]:
                        answer = process_question(prompt, st.session_state["vector_db"], selected_model)
                    else:
                        answer = "Please upload and process a PDF file first."
                    st.markdown(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
