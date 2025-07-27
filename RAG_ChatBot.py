"""
RAG chatbot with Gemma (via Ollama) + Pinecone
"""
from __future__ import annotations

import os
from typing import List

from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

GEMMA_MODEL = "gemma:2b"          # or gemma:7b
EMBED_MODEL = "nomic-embed-text"  # lightweight local embedding (384-d)
INDEX_NAME  = "langchain-demo"

class ChatBot:
    def __init__(self) -> None:

        # ------------------------------------------------------------------
        # 1. Load & split documents
        # ------------------------------------------------------------------
        loader = TextLoader("./materials/torontoTravelAssistant.txt", encoding="utf-8")
        documents: List[Document] = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # ------------------------------------------------------------------
        # 2. Local embeddings + Pinecone
        # ------------------------------------------------------------------
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,                # nomic-embed-text dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        docsearch = PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=INDEX_NAME,
        )

        # ------------------------------------------------------------------
        # 3. Local Gemma LLM via Ollama
        # ------------------------------------------------------------------
        llm = Ollama(model=GEMMA_MODEL, temperature=0)

        # ------------------------------------------------------------------
        # 4. Prompt & Chain
        # ------------------------------------------------------------------
        template = """
        You are a Toronto travel assistant. Use the context below to answer the user's question.
        If you don't know, say so. Keep it short and concise (≤ 2 sentences).

        Context: {context}
        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=docsearch.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False,
        )


# --------------------------------------------------------------------------
# Quick test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    bot = ChatBot()
    print(bot.rag_chain.run("Where can I eat dim sum in Toronto?"))
