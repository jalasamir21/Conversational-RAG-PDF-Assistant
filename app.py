"""
app.py
-------
UI layer only. Talks to Chatbot via the ChatEngine interface — has no
idea whether it's backed by SimpleFAQEngine or RagEngine underneath.
Upgrading from FAQ replies to full RAG (or swapping in a future engine)
never requires touching this file beyond the wiring at the bottom.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from chat_engine import Chatbot, RagEngine, SimpleFAQEngine
from memory import MemoryStore
from rag_pipeline import PDFIndexer, RagChainBuilder

import uuid

load_dotenv()
st.set_page_config(
    page_title="Conversational rAG PDF Assistant",
    page_icon="📑"
)
st.title("📑Chat with PDF RAG")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id


def reset_state():
    for key in st.session_state:
        del st.session_state[key]


# --- API key handling ---
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = st.text_input("Enter your API key", type="password")
    api_key = st.session_state["api_key"]
else:
    if expected_password := os.getenv("PASSWORD"):
        password = st.text_input("What's the secret password?", type="password")
        if password != expected_password:
            api_key = ""
            st.error("Unauthorized access.")
            reset_state()
        else:
            api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.stop()


# --- Cached resources (expensive to create, safe to reuse across reruns) ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_memory_store():
    return MemoryStore()


embeddings = get_embeddings()
memory_store = get_memory_store()

# --- Model + system prompt controls ---
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama-3.1-8b-instant"

model_options = (
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
)
st.session_state["groq_model"] = st.selectbox(
    "Select a model",
    model_options,
    index=model_options.index(st.session_state["groq_model"]),
)

if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = ""
st.text_input("System Prompt", value=st.session_state["system_prompt"], key="system_prompt")

# --- PDF upload + indexing (sidebar, persists across reruns) ---
st.sidebar.header("Upload PDF for RAG")
uploaded_file = st.sidebar.file_uploader("Choose your .pdf file", type="pdf")

if uploaded_file is not None:
    MAX_SIZE = 10 * 1024 * 1024

    if len(uploaded_file.getvalue()) > MAX_SIZE:
        st.sidebar.error("File exceeds 10 MB.")
        st.stop()

    if uploaded_file.type != "application/pdf":
        st.sidebar.error("Only PDF files are allowed.")
        st.stop()

indexer = PDFIndexer(embeddings)

if uploaded_file is not None:
    collection_name = uploaded_file.name.replace(".", "_")
    file_hash = hash(uploaded_file.getvalue())
    if st.session_state.get("indexed_file_hash") != file_hash:
        with st.spinner("Indexing PDF..."):
            try:
                vectorstore, n_chunks = indexer.index_file(uploaded_file.getvalue(), collection_name)

                st.session_state["vectorstore"] = vectorstore
                st.session_state["indexed_file_hash"] = file_hash
            except Exception as e:
                st.error(f"Unable to process PDF: {e}")
                st.stop()

        st.sidebar.success(f"Indexed {n_chunks} chunks from {collection_name}")
    else:
        st.sidebar.info(f"Already indexed: {uploaded_file.name}")

# --- Wire the engine: RAG once a PDF is indexed, FAQ fallback until then ---
if "vectorstore" in st.session_state:
    builder = RagChainBuilder(
        api_key=api_key,
        model_name=st.session_state["groq_model"],
        memory_store=memory_store,
    )
    chain = builder.build(
        st.session_state["vectorstore"], system_prompt=st.session_state["system_prompt"]
    )
    chatbot = Chatbot(RagEngine(chain, memory_store))
else:
    chatbot = Chatbot(SimpleFAQEngine())

# --- Render existing history ---
for role, content in chatbot.history(SESSION_ID):
    with st.chat_message(role):
        st.markdown(content)

# --- Chat input ---
if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.ask(SESSION_ID, prompt)
        st.markdown(response)
