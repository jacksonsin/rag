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

# ---------- config ----------
INDEX_NAME = "langchain-demo"
EMBED_MODEL = "models/embedding-001"
LLM_MODEL = "gemma-3-27b-it"
DIMENSION = 768  # must match Gemini embedding size

@st.cache_resource(show_spinner=False)
def _build_chain():
    # 1. Load **all** PDFs in ./materials/
    pdf_files = Path("./materials").glob("*.pdf")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs: List[Document] = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(splitter.split_documents(loader.load()))

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # ---- create or recreate index with correct dimension ----
    if INDEX_NAME in pc.list_indexes().names():
        pc.delete_index(INDEX_NAME)
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    docsearch = PineconeVectorStore.from_documents(
        docs, embeddings, index_name=INDEX_NAME
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        max_output_tokens=1024,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are J.A.C.K.S.O.N.  \n"
            "Answer ONLY with the concise result (≤ 2 sentences).  \n"
            "Use **bold**, *italics*, bullet lists, or emojis to make it vivid.  \n\n"
            "Context: {context}  \n"
            "Question: {question}  \n"
            "Answer: "
        ),
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )


# ---------- Streamlit UI ----------
st.title("J.A.C.K.S.O.N")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm J.A.C.K.S.O.N. How can I help?"}
    ]

# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything…", key="user_input"):
    # append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # generate & display assistant reply
    with st.chat_message("assistant"):
        with st.spinner("One moment please…"):
            answer = _build_chain().invoke(prompt)
            st.write(answer["result"])

    # save assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": answer["result"]}
    )
