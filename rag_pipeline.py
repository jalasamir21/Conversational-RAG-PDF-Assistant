"""
rag_pipeline.py
----------------
All LangChain-specific plumbing lives here: turning an uploaded PDF into
a persisted Chroma collection, and assembling a history-aware RAG chain
over it using Groq. Nothing in this file knows about Streamlit, and
nothing outside this file needs to know how LangChain's Runnable/prompt
APIs work.
"""

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
from memory import MemoryStore


class PDFIndexer:
    """Extracts + chunks + embeds a PDF into its own persisted Chroma collection."""

    def __init__(self, embeddings: HuggingFaceEmbeddings, persist_root: str = ".chroma"):
        self.embeddings = embeddings
        self.persist_root = persist_root

    def index_file(self, file_bytes: bytes, collection_name: str):
        """Writes the upload to a temp file (PyPDFLoader needs a real path),
        splits it, and persists a fresh Chroma collection for it.
        One collection per file — so a second CV never bleeds into a
        first CV's retrieval results."""
        persist_dir = os.path.join(self.persist_root, collection_name)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            pages = PyPDFLoader(tmp_path).load()
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            ).split_documents(pages)
            print("=" * 80)
            print("Total chunks:", len(chunks))

            for i, chunk in enumerate(chunks):
                print(f"\nChunk {i+1}")
                print(chunk.metadata)
                print(chunk.page_content[:300])

            print("=" * 80)
            if not chunks:
                raise ValueError("No text extracted from PDF. Is it a scanned image?")

            vectorstore = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=persist_dir,
                collection_name=collection_name,
            )

            print("Stored docs:", vectorstore._collection.count())

            stored = vectorstore.get()

            for i, doc in enumerate(stored["documents"]):
                print(f"\nStored Document {i+1}")
                print(doc[:250])
            return vectorstore, len(chunks)
        finally:
            os.unlink(tmp_path)

    def load_existing(self, collection_name: str):
        """Reload a previously-indexed collection from disk, if present."""
        persist_dir = os.path.join(self.persist_root, collection_name)
        if not os.path.exists(persist_dir):
            return None
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )


class RagChainBuilder:
    """Builds a history-aware RAG chain: retriever -> prompt -> Groq LLM,
    wrapped in RunnableWithMessageHistory so it reads/writes through MemoryStore."""

    def __init__(self, api_key: str, model_name: str, memory_store: MemoryStore):
        self.llm = ChatGroq(api_key=api_key, model=model_name)
        self.memory_store = memory_store

    def build(self, vectorstore, system_prompt: str = ""):
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 15,
                "lambda_mult": 0.7,
            },
        )
        base_system = system_prompt or "You are a helpful assistant."

        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the user's latest question into a standalone question "
                    "using the conversation history if necessary. "
                    "Do NOT answer the question."
                ),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )

        question_rewriter = (
            rewrite_prompt
            | self.llm
            | StrOutputParser()
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""{base_system}

        You are an AI assistant that answers questions about the uploaded document.

        Use all of the retrieved context to answer.

        If the answer can be inferred by combining multiple retrieved chunks, do so.

        Only answer "I don't know" if the information is truly absent from the retrieved context.

        If the retrieved context is incomplete, answer with the information that is available.
        Context:
        {{context}}
        """,
                ),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        def build_input(inputs):

            history = inputs.get("history", [])

            if history:
                standalone_question = question_rewriter.invoke(
                    {
                        "question": inputs["question"],
                        "history": history,
                    }
                )
            else:
                standalone_question = inputs["question"]

            docs = retriever.invoke(standalone_question)

            print("=" * 80)
            print("Standalone question:", standalone_question)

            for i, doc in enumerate(docs):
                print(f"\nChunk {i+1}")
                print(doc.page_content[:500])

            print("=" * 80)

            context = "\n\n".join(doc.page_content for doc in docs)

            return {
                "question": inputs["question"],
                "history": history,
                "context": context,
            }

    

        chain = (
            RunnableLambda(build_input)
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        return RunnableWithMessageHistory(
            chain,
            self.memory_store.get,
            input_messages_key="question",
            history_messages_key="history",
        )
