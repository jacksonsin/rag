"""
Fast RAG chatbot: Gemma-3-27B + Gemini embeddings + Pinecone
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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------- CONFIG ----------
INDEX_NAME   = "langchain-demo"
EMBED_MODEL  = "textembedding-gecko@001"   # ← valid Vertex AI embedding model ID
LLM_MODEL    = "gemma-3-27b-it"
DIMENSION    = 768
CHUNK_SIZE   = 512
CHUNK_OVERLAP = 50

@st.cache_resource(show_spinner="Loading knowledge base …")
def _build_chain():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # 1. Build / verify index
    if INDEX_NAME not in pc.list_indexes().names():
        docs: List[Document] = []
        for pdf in Path("./materials").glob("*.pdf"):
            docs.extend(
                CharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                ).split_documents(PyPDFLoader(str(pdf)).load())
            )

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)

    # 2. Load existing index
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        max_output_tokens=512,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Context: {context}\n\nQuestion: {question}\n\nAnswer in one sentence:",
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
