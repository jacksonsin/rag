# ------------------------------------------------------------------
# 1. Imports (Gemma 27 B via Google AI Studio)
# ------------------------------------------------------------------
from __future__ import annotations
import os
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ------------------------------------------------------------------
# 2. Constants
# ------------------------------------------------------------------
INDEX_NAME  = "langchain-demo"
EMBED_MODEL = "models/embedding-001"   # Gemini embedding
DIMENSION   = 768                      # Gemini vectors are 768-d
GEMMA_MODEL = "gemma-3-27b-it"         # Google-AI-Studio hosted Gemma 27 B

class ChatBot:
    def __init__(self) -> None:

        # 2.1 Load PDF
        loader = PyPDFLoader("./materials/torontoTravelAssistant.pdf")
        documents: List[Document] = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # 2.2 Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        docsearch = PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=INDEX_NAME,
        )

        # 2.3 Gemma 27 B (hosted)
        llm = ChatGoogleGenerativeAI(
            model=GEMMA_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        # 2.4 Prompt & chain
        template = """
        You are a Toronto travel assistant. Use the context below to answer the user's question.
        If you don't know, say so. Keep it short and concise (≤ 2 sentences).

        Context: {context}
        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=docsearch.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False,
        )
