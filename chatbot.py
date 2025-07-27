"""
Fine-tuned RAG chatbot: Gemma-3-27-B + Gemini embeddings + Pinecone
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
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chat_models import ChatGooglePalm
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------- CONFIG ----------
INDEX_NAME     = "langchain-demo"
EMBED_MODEL    = "models/embedding-001"
LLM_MODEL      = "gemma-3-27b-it"
DIMENSION      = 768
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200


@st.cache_resource(show_spinner="Building knowledge base …")
def _build_chain():
    """Load PDFs → embed → store → return QA chain."""
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

    embeddings = GooglePalmEmbeddings(
        model=EMBED_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if INDEX_NAME in pc.list_indexes().names():
        pc.delete_index(INDEX_NAME)  # ⚠️ safe for demo; remove if you need persistence
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    docsearch = PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=INDEX_NAME
    )

    llm = ChatGooglePalm(
        model_name=LLM_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Reply in one short sentence:"
        ),
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="J.A.C.K.S.O.N", layout="centered")

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
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm J.A.C.K.S.O.N. How can I help?"}
    ]

for msg in st.session_state.messages:
    css = "user-msg" if msg["role"] == "user" else "bot-msg"
    st.markdown(
        f'<div class="{css}">{msg["content"]}</div>',
        unsafe_allow_html=True
    )

if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f'<div class="user-msg">{prompt}</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Thinking…"):
        response = _build_chain().invoke(prompt)
        reply = response["result"].strip()
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        st.markdown(
            f'<div class="bot-msg">{reply}</div>',
            unsafe_allow_html=True
        )
