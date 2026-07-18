# 📄 Conversational RAG PDF Assistant

🚀 **Live Demo:** [https://conversational-rager.streamlit.app/]

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions about their content.

The assistant maintains conversational context across multiple interactions, reformulates follow-up questions using chat history, retrieves relevant document chunks from a vector database, and generates grounded responses using a Large Language Model.

---

## Features

- Upload and chat with PDF documents
- Context-aware conversations
- History-aware question reformulation
- Chroma vector database for semantic retrieval
- Hugging Face embedding models
- Groq LLM integration
- Session-based memory
- Persistent vector storage
- Streamlit user interface
- Modular architecture

---

## Architecture

```
                PDF Upload
                     │
                     ▼
             PyPDFLoader
                     │
                     ▼
      RecursiveCharacterTextSplitter
                     │
                     ▼
      HuggingFace Embeddings
                     │
                     ▼
             Chroma Vector DB
                     │
                     ▼
      History-Aware Retriever
                     │
                     ▼
         Retrieved Context
                     │
                     ▼
            Prompt Template
                     │
                     ▼
              Groq LLM
                     │
                     ▼
              Final Answer
```

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Groq API
- Streamlit
- PyPDFLoader

---

## Key AI Concepts

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Dense Vector Embeddings
- Context-Aware Retrieval
- History-Aware Question Reformulation
- Prompt Engineering
- Conversational Memory
- Vector Databases
- Large Language Models (LLMs)

---

## Project Structure

```
.
├── app.py
├── rag_pipeline.py
├── chat_engine.py
├── memory.py
├── requirements.txt
├── .chroma/
└── README.md
```

---

## How It Works

1. Upload a PDF (or multiple PDFs) document.
2. The document is split into semantic chunks.
3. Each chunk is converted into embeddings.
4. Embeddings are stored inside ChromaDB.
5. User questions are rewritten into standalone questions when necessary.
6. Relevant chunks are retrieved.
7. The Groq LLM answers only using the retrieved context.
8. Conversation history is maintained for each session.

---

## Example Questions

- What skills does the applicant have?
- Summarize the projects.
- What certifications are listed?
- Which programming languages are mentioned?
- What university does the applicant attend?
- What is the expected graduation year?

---

## Future Improvements

- Hybrid (keyword + semantic) search
- Reranking retrieved documents
- Citation-based answers
- OCR support for scanned PDFs
- LangGraph memory instead of RunnableWithMessageHistory
- Authentication and user accounts

---

## Installation

```bash
git clone https://github.com/jalasamir21/Conversational-RAG-PDF-Assistant.git

cd Conversational-RAG-PDF-Assistant

pip install -r requirements.txt

streamlit run app.py
```
