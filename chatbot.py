"""
Fast RAG chatbot: Gemma-3-27-B + Gemini embeddings + Pinecone
- Designed for Google AI Studio with ADC authentication
- Index is built once and reused
- No re-indexing on every request
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# Use LangChain first-party Google classes
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chat_models import ChatGooglePalm
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

# Load environment variables (for PROJECT, LOCATION, Pinecone key)
load_dotenv()

# ---------- CONFIG ----------
EMBED_MODEL   = "models/embedding-001"
LLM_MODEL     = "gemma-3-27b-it"
INDEX_NAME    = "langchain-demo"
DIMENSION     = 768
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

@st.cache_resource(show_spinner="Loading knowledge base …")
def _build_chain():
    """Load PDFs once → embed once → return a RetrievalQA chain."""
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # 1️⃣ Create index if missing
    if INDEX_NAME not in pc.list_indexes().names():
        docs: List[Document] = []
        for pdf in Path("./materials").glob("*.pdf"):
            docs.extend(
                CharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                ).split_documents(
                    PyPDFLoader(str(pdf)).load()
                )
            )

        # Use GooglePalmEmbeddings with ADC (no explicit key)
        embeddings = GooglePalmEmbeddings(
            model=EMBED_MODEL
        )

        # Create Pinecone index and upsert docs
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=INDEX_NAME
        )

    # 2️⃣ Load existing index as a retriever
    embeddings = GooglePalmEmbeddings(
        model=EMBED_MODEL
    )
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    # Initialize the Gemini chat model via AI Studio
    llm = ChatGooglePalm(
        model_name=LLM_MODEL,
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer in one sentence:"
        ),
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="J.A.C.K.S.O.N RAG", layout="centered")
st.title("🦾 J.A.C.K.S.O.N RAG Chatbot")

# Custom chat-bubble CSS
st.markdown(
    """
    <style>
    .user-msg, .bot-msg {
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        border-radius: 1.2rem;
        max-width: 80%;
        line-height: 1.4;
    }
    .user-msg {
        background: #d0e6ff;
        color: #003366;
        margin-left: auto;
    }
    .bot-msg {
        background: #e8f5e8;
        color: #004d00;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm J.A.C.K.S.O.N. How can I help?"}
    ]

# Render history
for msg in st.session_state.messages:
    css = "user-msg" if msg["role"] == "user" else "bot-msg"
    html = '<div class="{css}">{content}</div>'.format(css=css, content=msg["content"])
    st.markdown(html, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    html = '<div class="user-msg">{prompt}</div>'.format(prompt=prompt)
    st.markdown(html, unsafe_allow_html=True)

    with st.spinner("Thinking…"):
        answer = _build_chain().run(prompt)
    reply = answer.strip()

    st.session_state.messages.append({"role": "assistant", "content": reply})
    html = '<div class="bot-msg">{reply}</div>'.format(reply=reply)
    st.markdown(html, unsafe_allow_html=True)
